# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import rasterio

from rasterio.windows import Window

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

AVIRIS = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.dat"
)

STEP14A = ROOT / (
    "09_Mineral_Mapping/"
    "Step14A_Lithology_Training_Data_Preparation"
)

STEP14D = ROOT / (
    "09_Mineral_Mapping/"
    "Step14D_Broad_Lithology_Probability_Uncertainty"
)

FIELD_SPECTRA = STEP14A / (
    "AVIRIS_Step14A_Field_Spectra_3x3Median_Wide.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step14E_HighConfidence_Subtype_Overlay"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# INPUT RASTERS
# ============================================================

PROB_FILES = {
    "ANORTHOSITE_HOST":
        STEP14D / "AVIRIS_Step14D_Probability_ANORTHOSITE_HOST.tif",

    "CHROMITITE_ASSOCIATED":
        STEP14D / "AVIRIS_Step14D_Probability_CHROMITITE_ASSOCIATED.tif",

    "GNEISSIC_METAMORPHIC":
        STEP14D / "AVIRIS_Step14D_Probability_GNEISSIC_METAMORPHIC.tif",

    "GRANITIC_FELSIC":
        STEP14D / "AVIRIS_Step14D_Probability_GRANITIC_FELSIC.tif",

    "MAFIC_ULTRAMAFIC":
        STEP14D / "AVIRIS_Step14D_Probability_MAFIC_ULTRAMAFIC.tif",
}

MAX_PROB = STEP14D / "AVIRIS_Step14D_Max_Probability.tif"
UNCERTAINTY = STEP14D / "AVIRIS_Step14D_Uncertainty.tif"
WINNER = STEP14D / "AVIRIS_Step14D_Broad_Group_Winner.tif"

# ============================================================
# THRESHOLDS
# ============================================================

HIGH_CONFIDENCE_THRESHOLD = 0.70
VERY_HIGH_THRESHOLD = 0.80

BLOCK_ROWS = 64

# ============================================================
# SCIENTIFIC BANDS
# ============================================================

GOOD_RANGES = [
    (21, 191),
    (220, 283),
    (324, 414),
]

GOOD_BANDS = []

for lo, hi in GOOD_RANGES:
    GOOD_BANDS.extend(range(lo, hi + 1))

GOOD_INDEX = np.array(
    [b - 1 for b in GOOD_BANDS],
    dtype=int
)

GOOD_COLS = [
    f"B{b:03d}"
    for b in GOOD_BANDS
]

# ============================================================
# 13 LITHOLOGY CLASSES
# ============================================================

SUBTYPE_CLASSES = [
    "CHROMITITE",
    "ANORTHOSITE_METAANORTHOSITE",
    "GABBRO_METAGABBRO",
    "PYROXENITE_MAFIC_CUMULATE",
    "ULTRAMAFIC_DUNITE",
    "HORNBLENDE_BIOTITE_GNEISS",
    "GRANITE_GRANITIC_ROCK",
    "GNEISS_GRANITIC_COMPLEX",
    "PEGMATITE",
    "MIGMATITE",
    "QUARTZITE",
    "DOLERITE",
    "DOLOMITE_CARBONATE",
]

SUBTYPE_CODES = {
    name: i + 1
    for i, name in enumerate(SUBTYPE_CLASSES)
}

SUBTYPE_TO_BROAD = {
    "CHROMITITE":
        "CHROMITITE_ASSOCIATED",

    "ANORTHOSITE_METAANORTHOSITE":
        "ANORTHOSITE_HOST",

    "GABBRO_METAGABBRO":
        "MAFIC_ULTRAMAFIC",

    "PYROXENITE_MAFIC_CUMULATE":
        "MAFIC_ULTRAMAFIC",

    "ULTRAMAFIC_DUNITE":
        "MAFIC_ULTRAMAFIC",

    "HORNBLENDE_BIOTITE_GNEISS":
        "GNEISSIC_METAMORPHIC",

    "GNEISS_GRANITIC_COMPLEX":
        "GNEISSIC_METAMORPHIC",

    "GRANITE_GRANITIC_ROCK":
        "GRANITIC_FELSIC",

    "PEGMATITE":
        "GRANITIC_FELSIC",

    "MIGMATITE":
        "GRANITIC_FELSIC",

    "QUARTZITE":
        "GRANITIC_FELSIC",

    "DOLERITE":
        "MAFIC_ULTRAMAFIC",

    "DOLOMITE_CARBONATE":
        "GRANITIC_FELSIC",
}

# ============================================================
# HELPERS
# ============================================================

def snv_vector(x):

    x = np.asarray(x, dtype=np.float32)

    good = np.isfinite(x)

    if good.sum() < 100:
        return np.full_like(x, np.nan)

    m = np.nanmean(x[good])
    s = np.nanstd(x[good])

    if not np.isfinite(s) or s < 1e-8:
        return np.full_like(x, np.nan)

    y = np.full_like(x, np.nan)

    y[good] = (
        x[good] - m
    ) / s

    return y


def snv_rows(X):

    X = np.asarray(X, dtype=np.float32)

    m = np.nanmean(
        X,
        axis=1,
        keepdims=True
    )

    s = np.nanstd(
        X,
        axis=1,
        keepdims=True
    )

    s[s < 1e-8] = np.nan

    return (
        X - m
    ) / s


def cosine_rows(X, ref):

    ref = np.asarray(ref, dtype=np.float32)

    xn = np.linalg.norm(
        X,
        axis=1
    )

    rn = np.linalg.norm(ref)

    dot = X @ ref

    den = xn * rn

    out = np.full(
        len(X),
        np.nan,
        dtype=np.float32
    )

    good = den > 1e-10

    out[good] = (
        dot[good] / den[good]
    )

    return out


# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 90)
print("STEP14E - HIGH-CONFIDENCE MASKS + 13-CLASS SUBTYPE OVERLAY")
print("=" * 90)

for p in [AVIRIS, FIELD_SPECTRA, MAX_PROB, UNCERTAINTY, WINNER]:
    if not p.exists():
        raise FileNotFoundError(p)

for p in PROB_FILES.values():
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# LOAD FIELD SPECTRA
# ============================================================

field = pd.read_csv(
    FIELD_SPECTRA
)

print("\nField spectra rows:", len(field))

# ============================================================
# CREATE SUBTYPE REFERENCE MEDIANS
# ============================================================

references = {}

reference_summary = []

for subtype in SUBTYPE_CLASSES:

    sub = field[
        field["lithology_family"] == subtype
    ].copy()

    if sub.empty:
        continue

    mat = (
        sub[GOOD_COLS]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(dtype=np.float32)
    )

    normalized = []

    for row in mat:

        r = snv_vector(row)

        if np.isfinite(r).sum() >= 300:
            normalized.append(r)

    if len(normalized) == 0:
        continue

    ref = np.nanmedian(
        np.vstack(normalized),
        axis=0
    )

    ref = snv_vector(ref)

    references[subtype] = ref

    reference_summary.append({
        "subtype": subtype,
        "field_reference_count": len(normalized),
        "broad_group": SUBTYPE_TO_BROAD[subtype]
    })

reference_df = pd.DataFrame(
    reference_summary
)

print("\nSubtype references available:")
print(
    reference_df.to_string(index=False)
)

# ============================================================
# OUTPUT PROFILES
# ============================================================

with rasterio.open(AVIRIS) as src:

    float_profile = src.profile.copy()

    float_profile.update(
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        BIGTIFF="YES"
    )

    byte_profile = src.profile.copy()

    byte_profile.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        BIGTIFF="YES"
    )

    # ========================================================
    # OUTPUT FILES
    # ========================================================

    high_mask_path = OUT / (
        "AVIRIS_Step14E_HighConfidence_BroadMask.tif"
    )

    very_high_mask_path = OUT / (
        "AVIRIS_Step14E_VeryHighConfidence_BroadMask.tif"
    )

    subtype_winner_path = OUT / (
        "AVIRIS_Step14E_13Class_ReferenceSubtype_Winner.tif"
    )

    subtype_similarity_path = OUT / (
        "AVIRIS_Step14E_Subtype_MaxSimilarity.tif"
    )

    subtype_margin_path = OUT / (
        "AVIRIS_Step14E_Subtype_SimilarityMargin.tif"
    )

    high_ds = rasterio.open(
        high_mask_path,
        "w",
        **byte_profile
    )

    very_high_ds = rasterio.open(
        very_high_mask_path,
        "w",
        **byte_profile
    )

    subtype_ds = rasterio.open(
        subtype_winner_path,
        "w",
        **byte_profile
    )

    similarity_ds = rasterio.open(
        subtype_similarity_path,
        "w",
        **float_profile
    )

    margin_ds = rasterio.open(
        subtype_margin_path,
        "w",
        **float_profile
    )

    # Open Step14D rasters
    prob_ds = {
        k: rasterio.open(v)
        for k, v in PROB_FILES.items()
    }

    maxprob_ds = rasterio.open(
        MAX_PROB
    )

    # ========================================================
    # BLOCK PROCESSING
    # ========================================================

    for row0 in range(
        0,
        src.height,
        BLOCK_ROWS
    ):

        h = min(
            BLOCK_ROWS,
            src.height - row0
        )

        window = Window(
            0,
            row0,
            src.width,
            h
        )

        maxp = maxprob_ds.read(
            1,
            window=window
        )

        broad_probs = {
            group: ds.read(
                1,
                window=window
            )
            for group, ds in prob_ds.items()
        }

        high_mask = (
            maxp >= HIGH_CONFIDENCE_THRESHOLD
        )

        very_high_mask = (
            maxp >= VERY_HIGH_THRESHOLD
        )

        # Output high-confidence broad mask
        high_out = np.zeros(
            (h, src.width),
            dtype=np.uint8
        )

        very_high_out = np.zeros(
            (h, src.width),
            dtype=np.uint8
        )

        # winner among broad classes
        broad_stack = np.stack(
            [
                broad_probs["ANORTHOSITE_HOST"],
                broad_probs["CHROMITITE_ASSOCIATED"],
                broad_probs["GNEISSIC_METAMORPHIC"],
                broad_probs["GRANITIC_FELSIC"],
                broad_probs["MAFIC_ULTRAMAFIC"],
            ],
            axis=0
        )

        broad_codes = np.argmax(
            broad_stack,
            axis=0
        ) + 1

        high_out[
            high_mask
        ] = broad_codes[
            high_mask
        ]

        very_high_out[
            very_high_mask
        ] = broad_codes[
            very_high_mask
        ]

        # ====================================================
        # SUBTYPE SIMILARITY
        # ====================================================

        cube = src.read(
            window=window
        ).astype(np.float32)

        cube[cube <= -9000] = np.nan

        flat = cube.reshape(
            src.count,
            h * src.width
        ).T

        sci = flat[:, GOOD_INDEX]

        finite = np.isfinite(
            sci
        ).sum(axis=1)

        valid = (
            finite >= 300
        )

        subtype_codes = np.zeros(
            h * src.width,
            dtype=np.uint8
        )

        max_sim = np.full(
            h * src.width,
            -9999.0,
            dtype=np.float32
        )

        margin = np.full(
            h * src.width,
            -9999.0,
            dtype=np.float32
        )

        idx = np.where(valid)[0]

        if len(idx):

            X = sci[idx]

            # row median fill
            med = np.nanmedian(
                X,
                axis=1
            )

            bad = ~np.isfinite(X)

            if bad.any():

                rr, cc = np.where(bad)

                X[rr, cc] = med[rr]

            X = snv_rows(X)

            # broad group per pixel
            broad_flat = broad_codes.reshape(-1)[idx]

            # mapping broad numeric code
            numeric_to_group = {
                1: "ANORTHOSITE_HOST",
                2: "CHROMITITE_ASSOCIATED",
                3: "GNEISSIC_METAMORPHIC",
                4: "GRANITIC_FELSIC",
                5: "MAFIC_ULTRAMAFIC",
            }

            for broad_code, broad_name in numeric_to_group.items():

                sub_idx_local = np.where(
                    broad_flat == broad_code
                )[0]

                if len(sub_idx_local) == 0:
                    continue

                candidate_subtypes = [
                    s for s in SUBTYPE_CLASSES
                    if (
                        SUBTYPE_TO_BROAD[s] == broad_name
                        and s in references
                    )
                ]

                if len(candidate_subtypes) == 0:
                    continue

                Xg = X[
                    sub_idx_local
                ]

                sims = []

                for subtype in candidate_subtypes:

                    sims.append(
                        cosine_rows(
                            Xg,
                            references[subtype]
                        )
                    )

                S = np.column_stack(
                    sims
                )

                order = np.argsort(
                    S,
                    axis=1
                )

                best_col = order[:, -1]

                best = S[
                    np.arange(len(S)),
                    best_col
                ]

                if S.shape[1] >= 2:

                    second = S[
                        np.arange(len(S)),
                        order[:, -2]
                    ]

                    marg = best - second

                else:

                    marg = np.full(
                        len(S),
                        np.nan,
                        dtype=np.float32
                    )

                global_idx = idx[
                    sub_idx_local
                ]

                chosen_codes = np.array(
                    [
                        SUBTYPE_CODES[
                            candidate_subtypes[c]
                        ]
                        for c in best_col
                    ],
                    dtype=np.uint8
                )

                subtype_codes[
                    global_idx
                ] = chosen_codes

                max_sim[
                    global_idx
                ] = best

                margin[
                    global_idx
                ] = marg

        # Only retain subtype winner in HIGH-confidence broad pixels
        high_flat = high_mask.reshape(-1)

        subtype_codes[
            ~high_flat
        ] = 0

        max_sim[
            ~high_flat
        ] = -9999.0

        margin[
            ~high_flat
        ] = -9999.0

        # ====================================================
        # WRITE
        # ====================================================

        high_ds.write(
            high_out,
            1,
            window=window
        )

        very_high_ds.write(
            very_high_out,
            1,
            window=window
        )

        subtype_ds.write(
            subtype_codes.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        similarity_ds.write(
            max_sim.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        margin_ds.write(
            margin.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        print(
            f"Processed rows {row0} - {row0+h-1}"
        )

    high_ds.close()
    very_high_ds.close()
    subtype_ds.close()
    similarity_ds.close()
    margin_ds.close()

    for ds in prob_ds.values():
        ds.close()

    maxprob_ds.close()

# ============================================================
# SUMMARY
# ============================================================

with rasterio.open(
    subtype_winner_path
) as ds:

    arr = ds.read(1)

subtype_rows = []

for subtype, code in SUBTYPE_CODES.items():

    count = int(
        np.sum(arr == code)
    )

    subtype_rows.append({
        "subtype": subtype,
        "code": code,
        "pixel_count": count,
        "area_km2": count * 16.0 / 1e6
    })

subtype_summary = pd.DataFrame(
    subtype_rows
)

subtype_summary.to_csv(
    OUT / (
        "AVIRIS_Step14E_13Class_Subtype_Area_Summary.csv"
    ),
    index=False
)

reference_df.to_csv(
    OUT / (
        "AVIRIS_Step14E_Subtype_Reference_Summary.csv"
    ),
    index=False
)

with pd.ExcelWriter(
    OUT / (
        "AVIRIS_Step14E_Subtype_Overlay_Summary.xlsx"
    ),
    engine="openpyxl"
) as writer:

    subtype_summary.to_excel(
        writer,
        sheet_name="Subtype_Areas",
        index=False
    )

    reference_df.to_excel(
        writer,
        sheet_name="Reference_Summary",
        index=False
    )

settings = {
    "step":
        "Step14E_HighConfidence_Subtype_Overlay",

    "high_confidence_threshold":
        HIGH_CONFIDENCE_THRESHOLD,

    "very_high_threshold":
        VERY_HIGH_THRESHOLD,

    "subtype_interpretation":
        (
            "13-class output is reference-guided subtype winner "
            "inside high-confidence Step14D broad-group pixels only."
        ),

    "important_warning":
        (
            "Subtype winner is not a validated geological boundary map. "
            "Use similarity and margin rasters to judge ambiguity."
        )
}

(
    OUT / "AVIRIS_Step14E_Settings.json"
).write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 90)
print("STEP14E COMPLETE")
print("=" * 90)

print("\n13-CLASS REFERENCE SUBTYPE AREA:")
print(
    subtype_summary.to_string(index=False)
)

print("\nOUTPUTS:")
print(high_mask_path)
print(very_high_mask_path)
print(subtype_winner_path)
print(subtype_similarity_path)
print(subtype_margin_path)

print("\nIMPORTANT:")
print(
    "Subtype winner is only a reference-guided interpretation within "
    "high-confidence broad-group pixels."
)

print(
    "Use MaxSimilarity + SimilarityMargin to identify ambiguous subtype assignments."
)

print(
    "CHROMITITE subtype is not direct chromite or PGE detection."
)

print("\nNEXT:")
print(
    "Step14F = ArcGIS final map preparation + legend + uncertainty filtering + "
    "field-check target extraction."
)

print("\nPASS")
