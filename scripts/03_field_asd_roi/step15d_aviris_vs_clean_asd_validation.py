# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP15A = ROOT / (
    "09_Mineral_Mapping/"
    "Step15A_13_Lithology_AVIRIS_Endmembers"
)

STEP14C2A = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C2A_ASD_WhiteReference_Audit"
)

AVIRIS_HDR = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.hdr"
)

AVIRIS_ENDMEMBERS = STEP15A / (
    "AVIRIS_Step15A_13_Endmembers_326ScientificBand_Wide.csv"
)

ENDMEMBER_LOCATIONS = STEP15A / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.csv"
)

CLEAN_ASD_AUDIT = STEP14C2A / (
    "AVIRIS_Step14C2A_Clean_NonWR_ASD_Candidates.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15D_AVIRIS_vs_Clean_SAC_ASD"
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

GOOD_COLS = [
    f"B{b:03d}"
    for b in GOOD_BANDS
]

# ============================================================
# LOCAL ASD FAMILIES
# ============================================================

ASD_FAMILIES = [
    "CHROMITITE",
    "ANORTHOSITE_METAANORTHOSITE",
    "GABBRO_METAGABBRO",
    "PYROXENITE_MAFIC_CUMULATE",
    "HORNBLENDE_BIOTITE_GNEISS",
    "GRANITE_GRANITIC_ROCK",
]

# ============================================================
# CONTINUUM WINDOWS
# ============================================================

WINDOWS = [
    (476.0, 1329.0),
    (1473.0, 1789.0),
    (1994.0, 2445.5),
]

# ============================================================
# HELPERS
# ============================================================

def parse_aviris_wavelengths(hdr):

    text = hdr.read_text(errors="ignore")

    m = re.search(
        r"wavelength\s*=\s*\{(.*?)\}",
        text,
        flags=re.I | re.S
    )

    if not m:
        raise RuntimeError(
            "Could not parse AVIRIS wavelengths."
        )

    vals = []

    for x in m.group(1).replace("\n", " ").split(","):
        try:
            vals.append(float(x.strip()))
        except Exception:
            pass

    return np.asarray(vals, dtype=float)


def parse_raw_asd(path):

    path = Path(path)

    if not path.exists():
        return None

    lines = path.read_text(
        errors="ignore"
    ).splitlines()

    wl = []
    refl = []

    for line in lines:

        parts = re.split(
            r"[\t, ]+",
            line.strip()
        )

        nums = []

        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass

        if len(nums) < 2:
            continue

        w = nums[0]
        r = nums[1]

        if (
            300 <= w <= 2600
            and np.isfinite(r)
        ):
            wl.append(w)
            refl.append(r)

    if len(wl) < 1000:
        return None

    df = pd.DataFrame({
        "wl": wl,
        "refl": refl
    })

    df = (
        df.groupby(
            "wl",
            as_index=False
        )
        .median()
        .sort_values("wl")
    )

    w = df["wl"].to_numpy(dtype=float)
    r = df["refl"].to_numpy(dtype=float)

    median_abs = np.nanmedian(
        np.abs(r)
    )

    if 2.0 < median_abs <= 100:
        r = r / 100.0

    return w, r


def smooth(x):

    x = np.asarray(x, dtype=float)

    if len(x) < 7:
        return x

    return savgol_filter(
        x,
        window_length=7,
        polyorder=2
    )


def minmax(x):

    x = np.asarray(x, dtype=float)

    out = np.full_like(
        x,
        np.nan
    )

    good = np.isfinite(x)

    if good.sum() < 10:
        return out

    lo = np.nanmin(x[good])
    hi = np.nanmax(x[good])

    if hi <= lo:
        return out

    out[good] = (
        x[good] - lo
    ) / (
        hi - lo
    )

    return out


def correlation(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    good = (
        np.isfinite(a) &
        np.isfinite(b)
    )

    if good.sum() < 50:
        return np.nan

    aa = a[good]
    bb = b[good]

    if (
        np.nanstd(aa) < 1e-10
        or
        np.nanstd(bb) < 1e-10
    ):
        return np.nan

    return float(
        np.corrcoef(
            aa,
            bb
        )[0, 1]
    )


def spectral_angle_deg(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    good = (
        np.isfinite(a) &
        np.isfinite(b)
    )

    if good.sum() < 250:
        return np.nan

    aa = a[good]
    bb = b[good]

    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)

    if na <= 0 or nb <= 0:
        return np.nan

    c = np.dot(
        aa,
        bb
    ) / (
        na * nb
    )

    c = np.clip(
        c,
        -1,
        1
    )

    return float(
        np.degrees(
            np.arccos(c)
        )
    )


def upper_hull_continuum(wl, refl):

    x = np.asarray(wl, dtype=float)
    y = np.asarray(refl, dtype=float)

    points = list(
        zip(x, y)
    )

    hull = []

    for p in points:

        while len(hull) >= 2:

            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = p

            cross = (
                (x2 - x1)
                * (y3 - y2)
                -
                (y2 - y1)
                * (x3 - x2)
            )

            if cross >= 0:
                hull.pop()
            else:
                break

        hull.append(p)

    hx = np.asarray(
        [p[0] for p in hull]
    )

    hy = np.asarray(
        [p[1] for p in hull]
    )

    return np.interp(
        x,
        hx,
        hy
    )


def continuum_removed_segments(
    wl,
    spectrum
):

    all_cr = []
    all_wl = []

    for lo, hi in WINDOWS:

        mask = (
            (wl >= lo)
            &
            (wl <= hi)
            &
            np.isfinite(spectrum)
        )

        w = wl[mask]
        r = spectrum[mask]

        if len(w) < 7:
            continue

        r = smooth(r)

        cont = upper_hull_continuum(
            w,
            r
        )

        good = (
            np.isfinite(cont)
            &
            (cont > 1e-12)
        )

        cr = np.full_like(
            r,
            np.nan
        )

        cr[good] = (
            r[good] /
            cont[good]
        )

        all_wl.extend(w.tolist())
        all_cr.extend(cr.tolist())

    return (
        np.asarray(all_wl),
        np.asarray(all_cr)
    )


def derivative_corr(
    a,
    b
):

    aa = smooth(
        minmax(a)
    )

    bb = smooth(
        minmax(b)
    )

    da = np.gradient(aa)
    db = np.gradient(bb)

    return correlation(
        da,
        db
    )


def corr_to_score(c):

    if not np.isfinite(c):
        return 0.0

    return float(
        np.clip(
            (c + 1.0) / 2.0,
            0,
            1
        )
    )


def sam_to_score(angle):

    if not np.isfinite(angle):
        return 0.0

    # Screening scale only:
    # 0 degrees -> 1
    # >=45 degrees -> 0
    return float(
        np.clip(
            1.0 - angle / 45.0,
            0,
            1
        )
    )


# ============================================================
# START
# ============================================================

print("=" * 90)
print("STEP15D - AVIRIS ENDMEMBERS vs CLEAN SAC ASD")
print("=" * 90)

for p in [
    AVIRIS_HDR,
    AVIRIS_ENDMEMBERS,
    CLEAN_ASD_AUDIT
]:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# WAVELENGTHS
# ============================================================

wl425 = parse_aviris_wavelengths(
    AVIRIS_HDR
)

if len(wl425) != 425:
    raise RuntimeError(
        f"Expected 425 wavelengths, found {len(wl425)}"
    )

wl326 = wl425[
    GOOD_INDEX
]

print("\nAVIRIS scientific bands:", len(wl326))

# ============================================================
# LOAD AVIRIS ENDMEMBERS
# ============================================================

aviris = pd.read_csv(
    AVIRIS_ENDMEMBERS
)

print("AVIRIS endmembers:", len(aviris))

X_aviris = (
    aviris[GOOD_COLS]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(dtype=float)
)

# ============================================================
# CLEAN ASD AUDIT
# ============================================================

audit = pd.read_csv(
    CLEAN_ASD_AUDIT
)

audit = audit[
    audit[
        "local_file"
    ]
    .astype(str)
    .str.lower()
    .str.endswith(
        ".asd.sco.txt"
    )
].copy()

audit = audit[
    audit[
        "local_family_v2"
    ].isin(
        ASD_FAMILIES
    )
].copy()

print("\nClean raw ASD files:", len(audit))

print("\nASD family counts:")
print(
    audit[
        "local_family_v2"
    ]
    .value_counts()
    .to_string()
)

# ============================================================
# PARSE + RESAMPLE ASD
# ============================================================

asd_rows = []
asd_matrix = []

for _, row in audit.iterrows():

    parsed = parse_raw_asd(
        row["local_path"]
    )

    if parsed is None:
        continue

    src_wl, src_refl = parsed

    spec = np.full(
        len(wl326),
        np.nan,
        dtype=float
    )

    valid = (
        (wl326 >= np.nanmin(src_wl))
        &
        (wl326 <= np.nanmax(src_wl))
    )

    spec[valid] = np.interp(
        wl326[valid],
        src_wl,
        src_refl
    )

    if np.isfinite(spec).sum() < 300:
        continue

    asd_rows.append({
        "local_file":
            row["local_file"],

        "lithology_family":
            row["local_family_v2"]
    })

    asd_matrix.append(
        spec
    )

asd_index = pd.DataFrame(
    asd_rows
)

ASD = np.vstack(
    asd_matrix
)

print("\nUsable clean ASD spectra:", len(asd_index))

# ============================================================
# FAMILY MEDIANS
# ============================================================

asd_family_medians = {}

family_rows = []

for family in ASD_FAMILIES:

    idx = np.where(
        asd_index[
            "lithology_family"
        ].to_numpy()
        ==
        family
    )[0]

    if len(idx) == 0:
        continue

    mat = ASD[
        idx,
        :
    ]

    median_spec = np.nanmedian(
        mat,
        axis=0
    )

    asd_family_medians[
        family
    ] = median_spec

    family_rows.append({
        "lithology_family":
            family,

        "clean_ASD_count":
            len(idx)
    })

family_summary = pd.DataFrame(
    family_rows
)

# ============================================================
# PAIRWISE AVIRIS vs ASD FAMILY COMPARISON
# ============================================================

comparison_rows = []

for i, av_row in aviris.iterrows():

    av_name = str(
        av_row["lithology_family"]
    )

    av_code = int(
        av_row["subtype_code"]
    )

    av_spec = X_aviris[i]

    av_shape = minmax(
        smooth(
            av_spec
        )
    )

    av_cr_wl, av_cr = (
        continuum_removed_segments(
            wl326,
            av_spec
        )
    )

    for family, asd_spec in (
        asd_family_medians.items()
    ):

        asd_shape = minmax(
            smooth(
                asd_spec
            )
        )

        raw_sam = spectral_angle_deg(
            av_spec,
            asd_spec
        )

        shape_corr = correlation(
            av_shape,
            asd_shape
        )

        deriv_corr = derivative_corr(
            av_spec,
            asd_spec
        )

        asd_cr_wl, asd_cr = (
            continuum_removed_segments(
                wl326,
                asd_spec
            )
        )

        # Same segmented wavelength grid
        if (
            len(av_cr) ==
            len(asd_cr)
            and
            len(av_cr) > 100
        ):
            cr_corr = correlation(
                1.0 - av_cr,
                1.0 - asd_cr
            )
        else:
            cr_corr = np.nan

        shape_score = corr_to_score(
            shape_corr
        )

        derivative_score = corr_to_score(
            deriv_corr
        )

        continuum_score = corr_to_score(
            cr_corr
        )

        sam_score = sam_to_score(
            raw_sam
        )

        composite = (
            0.30 * shape_score
            +
            0.25 * derivative_score
            +
            0.30 * continuum_score
            +
            0.15 * sam_score
        )

        comparison_rows.append({
            "subtype_code":
                av_code,

            "AVIRIS_lithology":
                av_name,

            "ASD_family":
                family,

            "raw_SAM_deg":
                raw_sam,

            "shape_correlation":
                shape_corr,

            "derivative_correlation":
                deriv_corr,

            "continuum_removed_correlation":
                cr_corr,

            "shape_score":
                shape_score,

            "derivative_score":
                derivative_score,

            "continuum_score":
                continuum_score,

            "SAM_score":
                sam_score,

            "cross_domain_composite_score":
                composite
        })

comparison = pd.DataFrame(
    comparison_rows
)

comparison[
    "rank_within_AVIRIS_endmember"
] = (
    comparison
    .groupby(
        "subtype_code"
    )[
        "cross_domain_composite_score"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

# ============================================================
# ENDMEMBER SUMMARY
# ============================================================

summary_rows = []

for code, sub in comparison.groupby(
    "subtype_code"
):

    sub = sub.sort_values(
        "cross_domain_composite_score",
        ascending=False
    )

    best = sub.iloc[0]

    av_name = best[
        "AVIRIS_lithology"
    ]

    expected_available = (
        av_name in
        asd_family_medians
    )

    expected = sub[
        sub[
            "ASD_family"
        ] == av_name
    ]

    if expected_available and not expected.empty:

        e = expected.iloc[0]

        expected_rank = int(
            e[
                "rank_within_AVIRIS_endmember"
            ]
        )

        expected_score = float(
            e[
                "cross_domain_composite_score"
            ]
        )

        expected_sam = float(
            e["raw_SAM_deg"]
        )

        expected_cr = float(
            e[
                "continuum_removed_correlation"
            ]
        )

        if (
            expected_rank == 1
            and expected_score >= 0.65
        ):
            status = (
                "LOCAL_ASD_SUPPORTS_CURRENT_LABEL"
            )

        elif (
            expected_rank <= 2
            and expected_score >= 0.55
        ):
            status = (
                "PARTIAL_LOCAL_ASD_SUPPORT"
            )

        else:
            status = (
                "WEAK_OR_CONFLICTING_LOCAL_ASD_SUPPORT"
            )

    else:

        expected_rank = np.nan
        expected_score = np.nan
        expected_sam = np.nan
        expected_cr = np.nan

        status = (
            "NO_SAME_FAMILY_LOCAL_ASD_REFERENCE"
        )

    summary_rows.append({
        "subtype_code":
            int(code),

        "AVIRIS_lithology":
            av_name,

        "best_matching_ASD_family":
            best[
                "ASD_family"
            ],

        "best_composite_score":
            float(
                best[
                    "cross_domain_composite_score"
                ]
            ),

        "best_raw_SAM_deg":
            float(
                best[
                    "raw_SAM_deg"
                ]
            ),

        "best_shape_correlation":
            float(
                best[
                    "shape_correlation"
                ]
            ),

        "best_derivative_correlation":
            float(
                best[
                    "derivative_correlation"
                ]
            ),

        "best_continuum_removed_correlation":
            float(
                best[
                    "continuum_removed_correlation"
                ]
            ),

        "same_family_ASD_available":
            expected_available,

        "expected_family_rank":
            expected_rank,

        "expected_family_score":
            expected_score,

        "expected_family_raw_SAM_deg":
            expected_sam,

        "expected_family_CR_correlation":
            expected_cr,

        "cross_domain_support_status":
            status
    })

summary = pd.DataFrame(
    summary_rows
).sort_values(
    "subtype_code"
)

# ============================================================
# ADD LOCATIONS
# ============================================================

if ENDMEMBER_LOCATIONS.exists():

    loc = pd.read_csv(
        ENDMEMBER_LOCATIONS
    )

    cols = [
        "subtype_code",
        "longitude",
        "latitude",
        "UTM_Easting_m",
        "UTM_Northing_m"
    ]

    cols = [
        c for c in cols
        if c in loc.columns
    ]

    summary = summary.merge(
        loc[cols],
        on="subtype_code",
        how="left"
    )

# ============================================================
# SAVE TABLES
# ============================================================

PAIR_CSV = OUT / (
    "AVIRIS_Step15D_13x6_AVIRIS_ASD_Comparison.csv"
)

SUMMARY_CSV = OUT / (
    "AVIRIS_Step15D_Endmember_CrossDomain_Summary.csv"
)

ASD_MEDIAN_CSV = OUT / (
    "AVIRIS_Step15D_Clean_ASD_Family_Medians_326Band.csv"
)

comparison.to_csv(
    PAIR_CSV,
    index=False
)

summary.to_csv(
    SUMMARY_CSV,
    index=False
)

# wide ASD family medians
median_rows = []

for family, spec in (
    asd_family_medians.items()
):

    rec = {
        "lithology_family":
            family
    }

    for j, b in enumerate(
        GOOD_BANDS
    ):
        rec[
            f"B{b:03d}"
        ] = spec[j]

    median_rows.append(rec)

asd_median_df = pd.DataFrame(
    median_rows
)

asd_median_df.to_csv(
    ASD_MEDIAN_CSV,
    index=False
)

# ============================================================
# MATRIX OUTPUTS
# ============================================================

score_matrix = comparison.pivot(
    index="AVIRIS_lithology",
    columns="ASD_family",
    values="cross_domain_composite_score"
)

sam_matrix = comparison.pivot(
    index="AVIRIS_lithology",
    columns="ASD_family",
    values="raw_SAM_deg"
)

cr_matrix = comparison.pivot(
    index="AVIRIS_lithology",
    columns="ASD_family",
    values="continuum_removed_correlation"
)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15D_AVIRIS_vs_Clean_ASD_Validation.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Endmember_Summary",
        index=False
    )

    comparison.to_excel(
        writer,
        sheet_name="All_13x6_Comparisons",
        index=False
    )

    score_matrix.to_excel(
        writer,
        sheet_name="Composite_Score_Matrix"
    )

    sam_matrix.to_excel(
        writer,
        sheet_name="SAM_Matrix"
    )

    cr_matrix.to_excel(
        writer,
        sheet_name="Continuum_Corr_Matrix"
    )

    family_summary.to_excel(
        writer,
        sheet_name="ASD_Family_Counts",
        index=False
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step15D_AVIRIS_vs_Clean_SAC_ASD",

    "clean_ASD_input":
        str(CLEAN_ASD_AUDIT),

    "white_reference_used":
        False,

    "clean_raw_ASD_families":
        list(
            asd_family_medians.keys()
        ),

    "weights": {
        "spectral_shape":
            0.30,

        "first_derivative":
            0.25,

        "continuum_removed_shape":
            0.30,

        "raw_SAM":
            0.15
    },

    "important_note":
        (
            "Composite scores are cross-domain consistency screening "
            "metrics, not calibrated probabilities or mineral IDs."
        ),

    "warning":
        (
            "ASD powder/lump laboratory spectra and airborne AVIRIS-NG "
            "spectra have different acquisition domains. A poor match does "
            "not automatically disprove the lithology, and a good match "
            "does not independently confirm it."
        )
}

(
    OUT / "AVIRIS_Step15D_Settings.json"
).write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 90)
print("STEP15D COMPLETE")
print("=" * 90)

print("\nClean raw ASD spectra:",
      len(asd_index))

print("Clean ASD families:",
      len(asd_family_medians))

print("\nAVIRIS vs LOCAL ASD SUMMARY:")

show_cols = [
    "subtype_code",
    "AVIRIS_lithology",
    "best_matching_ASD_family",
    "best_composite_score",
    "best_raw_SAM_deg",
    "best_continuum_removed_correlation",
    "same_family_ASD_available",
    "expected_family_rank",
    "expected_family_score",
    "cross_domain_support_status",
]

print(
    summary[
        show_cols
    ].to_string(
        index=False
    )
)

print("\nComposite score matrix:")
print(
    score_matrix.round(
        4
    ).to_string()
)

print("\nOutput folder:")
print(OUT)

print("\nExcel:")
print(XLSX)

print("\nIMPORTANT:")
print(
    "This is cross-domain spectral consistency screening."
)

print(
    "It is NOT an independent lithology confirmation and "
    "not direct chromite/PGE identification."
)

print("\nNEXT:")
print(
    "Step15E = 13-endmember internal separability / SAM matrix "
    "+ duplicate/redundancy QC."
)

print("\nPASS")
