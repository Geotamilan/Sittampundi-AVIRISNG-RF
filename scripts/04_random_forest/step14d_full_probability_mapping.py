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
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

AVIRIS = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.dat"
)

STEP14C = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C_Hierarchical_ROI_Training_QC"
)

STEP14C2 = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C2_v2_Clean_RAW_ASD_Guided_Revalidation"
)

ROI_CSV = STEP14C / (
    "AVIRIS_Step14C_ROI_Pixel_Training_Data.csv"
)

ASD_MEDIAN_CSV = STEP14C2 / (
    "AVIRIS_Step14C2_v2_Clean_RAW_ASD_Family_Medians.csv"
)

ALIGNMENT_CSV = STEP14C2 / (
    "AVIRIS_Step14C2_v2_ASD_AVIRIS_Alignment.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step14D_Broad_Lithology_Probability_Uncertainty"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# GOOD BANDS
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

# ============================================================
# BROAD CLASSES
# ============================================================

BROAD_CLASSES = [
    "ANORTHOSITE_HOST",
    "CHROMITITE_ASSOCIATED",
    "GNEISSIC_METAMORPHIC",
    "GRANITIC_FELSIC",
    "MAFIC_ULTRAMAFIC",
]

CLASS_CODES = {
    "ANORTHOSITE_HOST": 1,
    "CHROMITITE_ASSOCIATED": 2,
    "GNEISSIC_METAMORPHIC": 3,
    "GRANITIC_FELSIC": 4,
    "MAFIC_ULTRAMAFIC": 5,
}

# ============================================================
# MASK SETTINGS
# ============================================================

NDVI_THRESHOLD = 0.45
MIN_MEAN_REFLECTANCE = 0.01
MAX_MEAN_REFLECTANCE = 1.20

RED_BAND = 54
NIR_BAND = 98

BLOCK_ROWS = 64

# Confidence thresholds
HIGH_CONF = 0.70
MODERATE_CONF = 0.55

# ============================================================
# HELPERS
# ============================================================

def snv_rows(X):

    X = np.asarray(X, dtype=np.float32)

    mean = np.nanmean(X, axis=1, keepdims=True)
    std = np.nanstd(X, axis=1, keepdims=True)

    std[std < 1e-8] = np.nan

    return (X - mean) / std


def cosine_similarity_rows(X, ref):

    ref = np.asarray(ref, dtype=np.float32)

    Xn = np.linalg.norm(X, axis=1)
    rn = np.linalg.norm(ref)

    dots = X @ ref

    denom = Xn * rn

    out = np.zeros(len(X), dtype=np.float32)

    good = denom > 1e-10

    out[good] = dots[good] / denom[good]

    return out


# ============================================================
# LOAD TRAINING DATA
# ============================================================

print("=" * 90)
print("STEP14D - FULL AVIRIS BROAD LITHOLOGY PROBABILITY + UNCERTAINTY")
print("=" * 90)

for p in [AVIRIS, ROI_CSV, ASD_MEDIAN_CSV, ALIGNMENT_CSV]:
    if not p.exists():
        raise FileNotFoundError(p)

roi = pd.read_csv(ROI_CSV)

roi = roi[
    roi["broad_group"].isin(BROAD_CLASSES)
].copy()

feature_cols = [
    c for c in roi.columns
    if re.match(r"^F\d{3}$", c)
]

if len(feature_cols) != 326:
    raise RuntimeError(
        f"Expected 326 ROI spectral features, found {len(feature_cols)}"
    )

X_train = (
    roi[feature_cols]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float32)
)

y_train = roi["broad_group"].astype(str).to_numpy()

valid_train = np.isfinite(X_train).all(axis=1)

X_train = X_train[valid_train]
y_train = y_train[valid_train]

print("\nTraining pixels:", len(X_train))

print("\nTraining broad-group counts:")
print(
    pd.Series(y_train)
    .value_counts()
    .to_string()
)

# ============================================================
# ASD GUIDANCE
# ============================================================

alignment = pd.read_csv(
    ALIGNMENT_CSV
)

asd_medians = pd.read_csv(
    ASD_MEDIAN_CSV
)

usable_guidance = alignment[
    alignment["guidance_status"].isin([
        "STRONG_ALIGNMENT",
        "MODERATE_ALIGNMENT",
        "WEAK_ALIGNMENT_USE_WITH_CAUTION"
    ])
]["lithology_family"].tolist()

print("\nASD guidance families:")
print(usable_guidance)

asd_refs = {}

for fam in usable_guidance:

    sub = asd_medians[
        asd_medians["lithology_family"] == fam
    ]

    if sub.empty:
        continue

    cols = [
        c for c in sub.columns
        if re.match(r"^F\d{3}$", c)
    ]

    ref = (
        sub.iloc[0][cols]
        .astype(float)
        .to_numpy(dtype=np.float32)
    )

    asd_refs[fam] = ref

# Add ASD features to training data
train_guidance = []

for fam, ref in asd_refs.items():

    sim = cosine_similarity_rows(
        X_train,
        ref
    )

    train_guidance.append(sim)

if train_guidance:

    ASD_TRAIN = np.column_stack(
        train_guidance
    )

    X_train_final = np.hstack([
        X_train,
        ASD_TRAIN
    ])

else:

    X_train_final = X_train

print("\nBase features :", X_train.shape[1])
print("ASD features  :", len(asd_refs))
print("Final features:", X_train_final.shape[1])

# ============================================================
# TRAIN FINAL MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=1000,
    max_features="sqrt",
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=20260930,
    n_jobs=-1
)

model.fit(
    X_train_final,
    y_train
)

print("\nFinal classifier trained.")

print("\nModel classes:")
print(model.classes_)

# ============================================================
# PREPARE OUTPUT RASTERS
# ============================================================

with rasterio.open(AVIRIS) as src:

    profile_float = src.profile.copy()

    profile_float.update(
        count=1,
        dtype="float32",
        nodata=-9999.0,
        compress="deflate",
        predictor=3,
        BIGTIFF="YES"
    )

    profile_byte = src.profile.copy()

    profile_byte.update(
        count=1,
        dtype="uint8",
        nodata=0,
        compress="deflate",
        BIGTIFF="YES"
    )

    probability_paths = {}

    probability_datasets = {}

    for cls in BROAD_CLASSES:

        p = OUT / f"AVIRIS_Step14D_Probability_{cls}.tif"

        probability_paths[cls] = p

        probability_datasets[cls] = rasterio.open(
            p,
            "w",
            **profile_float
        )

    max_prob_path = OUT / (
        "AVIRIS_Step14D_Max_Probability.tif"
    )

    uncertainty_path = OUT / (
        "AVIRIS_Step14D_Uncertainty.tif"
    )

    winner_path = OUT / (
        "AVIRIS_Step14D_Broad_Group_Winner.tif"
    )

    confidence_path = OUT / (
        "AVIRIS_Step14D_Confidence_Class.tif"
    )

    max_prob_ds = rasterio.open(
        max_prob_path,
        "w",
        **profile_float
    )

    uncertainty_ds = rasterio.open(
        uncertainty_path,
        "w",
        **profile_float
    )

    winner_ds = rasterio.open(
        winner_path,
        "w",
        **profile_byte
    )

    confidence_ds = rasterio.open(
        confidence_path,
        "w",
        **profile_byte
    )

    # ========================================================
    # FULL IMAGE BLOCK PROCESSING
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

        cube = src.read(
            window=window
        ).astype(np.float32)

        cube[cube <= -9000] = np.nan

        rows = h * src.width

        flat425 = (
            cube
            .reshape(
                src.count,
                rows
            )
            .T
        )

        sci = flat425[:, GOOD_INDEX]

        finite_count = np.isfinite(
            sci
        ).sum(axis=1)

        mean_refl = np.nanmean(
            sci,
            axis=1
        )

        red = flat425[:, RED_BAND - 1]
        nir = flat425[:, NIR_BAND - 1]

        ndvi = np.full(
            rows,
            np.nan,
            dtype=np.float32
        )

        den = nir + red

        ok_ndvi = (
            np.isfinite(red) &
            np.isfinite(nir) &
            (np.abs(den) > 1e-10)
        )

        ndvi[ok_ndvi] = (
            (nir[ok_ndvi] - red[ok_ndvi]) /
            den[ok_ndvi]
        )

        valid = (
            (finite_count >= 300) &
            np.isfinite(mean_refl) &
            (mean_refl >= MIN_MEAN_REFLECTANCE) &
            (mean_refl <= MAX_MEAN_REFLECTANCE) &
            (
                (~np.isfinite(ndvi)) |
                (ndvi <= NDVI_THRESHOLD)
            )
        )

        # Initialize outputs
        prob_block = {
            cls: np.full(
                rows,
                -9999.0,
                dtype=np.float32
            )
            for cls in BROAD_CLASSES
        }

        max_prob = np.full(
            rows,
            -9999.0,
            dtype=np.float32
        )

        uncertainty = np.full(
            rows,
            -9999.0,
            dtype=np.float32
        )

        winner = np.zeros(
            rows,
            dtype=np.uint8
        )

        confidence = np.zeros(
            rows,
            dtype=np.uint8
        )

        if valid.any():

            X = sci[valid]

            # Fill rare missing values row-wise
            row_median = np.nanmedian(
                X,
                axis=1,
                keepdims=True
            )

            bad = ~np.isfinite(X)

            if bad.any():
                X[bad] = np.take_along_axis(
                    np.repeat(
                        row_median,
                        X.shape[1],
                        axis=1
                    ),
                    np.indices(X.shape)[1],
                    axis=1
                )[bad]

            X = snv_rows(X)

            # Remove any remaining invalid
            finite_model = np.isfinite(
                X
            ).all(axis=1)

            valid_indices = np.where(
                valid
            )[0]

            model_indices = valid_indices[
                finite_model
            ]

            X_model = X[
                finite_model
            ]

            if len(X_model):

                guide_features = []

                for fam, ref in asd_refs.items():

                    sim = cosine_similarity_rows(
                        X_model,
                        ref
                    )

                    guide_features.append(
                        sim
                    )

                if guide_features:

                    G = np.column_stack(
                        guide_features
                    )

                    X_final = np.hstack([
                        X_model,
                        G
                    ])

                else:

                    X_final = X_model

                probs = model.predict_proba(
                    X_final
                )

                model_classes = list(
                    model.classes_
                )

                for cls in BROAD_CLASSES:

                    if cls in model_classes:

                        ci = model_classes.index(
                            cls
                        )

                        prob_block[cls][
                            model_indices
                        ] = probs[:, ci]

                pmax = np.max(
                    probs,
                    axis=1
                )

                win_idx = np.argmax(
                    probs,
                    axis=1
                )

                win_class = [
                    model_classes[i]
                    for i in win_idx
                ]

                max_prob[
                    model_indices
                ] = pmax

                uncertainty[
                    model_indices
                ] = 1.0 - pmax

                winner[
                    model_indices
                ] = np.array(
                    [
                        CLASS_CODES[c]
                        for c in win_class
                    ],
                    dtype=np.uint8
                )

                conf_vals = np.ones(
                    len(pmax),
                    dtype=np.uint8
                )

                conf_vals[
                    pmax >= MODERATE_CONF
                ] = 2

                conf_vals[
                    pmax >= HIGH_CONF
                ] = 3

                confidence[
                    model_indices
                ] = conf_vals

        # reshape and write
        for cls in BROAD_CLASSES:

            arr = prob_block[cls].reshape(
                h,
                src.width
            )

            probability_datasets[cls].write(
                arr,
                1,
                window=window
            )

        max_prob_ds.write(
            max_prob.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        uncertainty_ds.write(
            uncertainty.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        winner_ds.write(
            winner.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        confidence_ds.write(
            confidence.reshape(
                h,
                src.width
            ),
            1,
            window=window
        )

        print(
            f"Processed rows "
            f"{row0} - {row0+h-1}"
        )

    # close outputs
    for ds in probability_datasets.values():
        ds.close()

    max_prob_ds.close()
    uncertainty_ds.close()
    winner_ds.close()
    confidence_ds.close()

# ============================================================
# SUMMARY
# ============================================================

winner_counts = {
    cls: 0
    for cls in BROAD_CLASSES
}

confidence_counts = {
    "LOW": 0,
    "MODERATE": 0,
    "HIGH": 0
}

with rasterio.open(
    winner_path
) as ds:

    arr = ds.read(1)

    for cls, code in CLASS_CODES.items():

        winner_counts[cls] = int(
            np.sum(arr == code)
        )

with rasterio.open(
    confidence_path
) as ds:

    arr = ds.read(1)

    confidence_counts["LOW"] = int(
        np.sum(arr == 1)
    )

    confidence_counts["MODERATE"] = int(
        np.sum(arr == 2)
    )

    confidence_counts["HIGH"] = int(
        np.sum(arr == 3)
    )

summary_rows = []

for cls in BROAD_CLASSES:

    px = winner_counts[cls]

    summary_rows.append({
        "broad_group": cls,
        "winner_pixels": px,
        "area_km2": px * 16.0 / 1e6
    })

summary_df = pd.DataFrame(
    summary_rows
)

confidence_df = pd.DataFrame([
    {
        "confidence_class": k,
        "pixel_count": v,
        "area_km2": v * 16.0 / 1e6
    }
    for k, v in confidence_counts.items()
])

summary_df.to_csv(
    OUT / (
        "AVIRIS_Step14D_Broad_Group_Area_Summary.csv"
    ),
    index=False
)

confidence_df.to_csv(
    OUT / (
        "AVIRIS_Step14D_Confidence_Area_Summary.csv"
    ),
    index=False
)

with pd.ExcelWriter(
    OUT / (
        "AVIRIS_Step14D_Probability_Mapping_Summary.xlsx"
    ),
    engine="openpyxl"
) as writer:

    summary_df.to_excel(
        writer,
        sheet_name="Broad_Group_Areas",
        index=False
    )

    confidence_df.to_excel(
        writer,
        sheet_name="Confidence_Areas",
        index=False
    )

settings = {
    "step":
        "Step14D_Broad_Lithology_Probability_Uncertainty",

    "scientific_gate":
        "CAUTION_STEP14D_PROBABILITY_AND_UNCERTAINTY_ONLY",

    "broad_classes":
        BROAD_CLASSES,

    "class_codes":
        CLASS_CODES,

    "high_confidence_probability":
        HIGH_CONF,

    "moderate_confidence_probability":
        MODERATE_CONF,

    "vegetation_NDVI_threshold":
        NDVI_THRESHOLD,

    "ASD_guidance_families":
        list(asd_refs.keys()),

    "important_interpretation":
        (
            "Broad_Group_Winner is an interpretive spectral winner, "
            "not a validated geological boundary map. "
            "Probability and uncertainty rasters are the primary outputs."
        )
}

(
    OUT / "AVIRIS_Step14D_Settings.json"
).write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

print("\n" + "=" * 90)
print("STEP14D COMPLETE")
print("=" * 90)

print("\nBROAD GROUP WINNER AREA:")
print(
    summary_df.to_string(index=False)
)

print("\nCONFIDENCE AREA:")
print(
    confidence_df.to_string(index=False)
)

print("\nOUTPUT RASTERS:")

for cls in BROAD_CLASSES:
    print(
        f"Probability {cls}:"
    )
    print(
        probability_paths[cls]
    )

print("\nMax probability:")
print(max_prob_path)

print("\nUncertainty:")
print(uncertainty_path)

print("\nBroad-group winner:")
print(winner_path)

print("\nConfidence class:")
print(confidence_path)

print("\nIMPORTANT:")
print(
    "Use probability + uncertainty rasters as primary scientific outputs."
)

print(
    "Do NOT interpret Broad_Group_Winner as a final hard geological boundary map."
)

print(
    "Chromitite-associated probability is NOT direct chromite or PGE detection."
)

print("\nNEXT:")
print(
    "Step14E = ArcGIS QC + high-confidence lithology probability masks + "
    "13-class reference subtype overlays."
)

print("\nPASS")
