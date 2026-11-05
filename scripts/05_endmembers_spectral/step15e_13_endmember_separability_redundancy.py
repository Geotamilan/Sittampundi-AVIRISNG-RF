# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP15A = ROOT / (
    "09_Mineral_Mapping/"
    "Step15A_13_Lithology_AVIRIS_Endmembers"
)

INPUT_CSV = STEP15A / (
    "AVIRIS_Step15A_13_Endmembers_326ScientificBand_Wide.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15E_13_Endmember_Separability_Redundancy_QC"
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
    GOOD_BANDS.extend(
        range(lo, hi + 1)
    )

GOOD_COLS = [
    f"B{b:03d}"
    for b in GOOD_BANDS
]

# Scientific wavelength ranges corresponding to retained bands.
# Actual wavelengths are not necessary for raw SAM/correlation.
#
# Continuum removal will be done by retained-band index ranges
# corresponding approximately to:
# 476-1328 nm
# 1473-1789 nm
# 1994-2445 nm
# ============================================================

SEGMENTS = [
    (0, 171),      # bands 21-191
    (171, 235),    # bands 220-283
    (235, 326),    # bands 324-414
]

# ============================================================
# QC THRESHOLDS
#
# These are heuristic separability screening thresholds,
# not universal geological classification thresholds.
# ============================================================

SAM_VERY_LOW = 5.0
SAM_LOW = 10.0
SAM_MODERATE = 15.0

CORR_EXTREME = 0.995
CORR_HIGH = 0.980

REDUNDANCY_HIGH = 0.85
REDUNDANCY_MODERATE = 0.70

# ============================================================
# HELPERS
# ============================================================

def smooth(x):

    x = np.asarray(
        x,
        dtype=float
    )

    if len(x) < 7:
        return x.copy()

    return savgol_filter(
        x,
        window_length=7,
        polyorder=2
    )


def minmax(x):

    x = np.asarray(
        x,
        dtype=float
    )

    out = np.full_like(
        x,
        np.nan,
        dtype=float
    )

    good = np.isfinite(x)

    if good.sum() < 10:
        return out

    lo = np.nanmin(
        x[good]
    )

    hi = np.nanmax(
        x[good]
    )

    if hi <= lo:
        return out

    out[good] = (
        x[good] - lo
    ) / (
        hi - lo
    )

    return out


def correlation(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    good = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if good.sum() < 50:
        return np.nan

    aa = a[good]
    bb = b[good]

    if (
        np.nanstd(aa) < 1e-12
        or
        np.nanstd(bb) < 1e-12
    ):
        return np.nan

    return float(
        np.corrcoef(
            aa,
            bb
        )[0, 1]
    )


def spectral_angle_deg(a, b):

    a = np.asarray(
        a,
        dtype=float
    )

    b = np.asarray(
        b,
        dtype=float
    )

    good = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if good.sum() < 250:
        return np.nan

    aa = a[good]
    bb = b[good]

    na = np.linalg.norm(
        aa
    )

    nb = np.linalg.norm(
        bb
    )

    if na <= 0 or nb <= 0:
        return np.nan

    cosine = np.dot(
        aa,
        bb
    ) / (
        na * nb
    )

    cosine = np.clip(
        cosine,
        -1,
        1
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def upper_hull_continuum_index(y):

    y = np.asarray(
        y,
        dtype=float
    )

    x = np.arange(
        len(y),
        dtype=float
    )

    points = list(
        zip(
            x,
            y
        )
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


def continuum_removed_vector(spec):

    spec = np.asarray(
        spec,
        dtype=float
    )

    result = []

    for start, end in SEGMENTS:

        seg = spec[
            start:end
        ]

        seg = smooth(
            seg
        )

        continuum = upper_hull_continuum_index(
            seg
        )

        cr = np.full_like(
            seg,
            np.nan,
            dtype=float
        )

        good = (
            np.isfinite(seg)
            &
            np.isfinite(continuum)
            &
            (continuum > 1e-12)
        )

        cr[good] = (
            seg[good]
            /
            continuum[good]
        )

        result.extend(
            cr.tolist()
        )

    return np.asarray(
        result,
        dtype=float
    )


def derivative_vector(spec):

    s = minmax(
        smooth(
            spec
        )
    )

    return np.gradient(
        s
    )


def sam_similarity_score(angle):

    if not np.isfinite(angle):
        return np.nan

    # 0 deg -> 1
    # >= 20 deg -> 0
    return float(
        np.clip(
            1.0 - angle / 20.0,
            0,
            1
        )
    )


def corr_score(c):

    if not np.isfinite(c):
        return np.nan

    return float(
        np.clip(
            (c + 1.0) / 2.0,
            0,
            1
        )
    )


def sam_qc_label(angle):

    if not np.isfinite(angle):
        return "NO_DATA"

    if angle < SAM_VERY_LOW:
        return "VERY_LOW_SEPARATION"

    if angle < SAM_LOW:
        return "LOW_SEPARATION"

    if angle < SAM_MODERATE:
        return "MODERATE_SEPARATION"

    return "BETTER_SEPARATION"


def redundancy_label(score):

    if not np.isfinite(score):
        return "NO_DATA"

    if score >= REDUNDANCY_HIGH:
        return "HIGH_REDUNDANCY"

    if score >= REDUNDANCY_MODERATE:
        return "MODERATE_REDUNDANCY"

    return "LOWER_REDUNDANCY"


# ============================================================
# START
# ============================================================

print("=" * 90)
print("STEP15E - 13 ENDMEMBER INTERNAL SEPARABILITY / REDUNDANCY QC")
print("=" * 90)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        INPUT_CSV
    )

df = pd.read_csv(
    INPUT_CSV
)

if len(df) != 13:

    print(
        f"WARNING: expected 13 endmembers, found {len(df)}"
    )

missing = [
    c
    for c in GOOD_COLS
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing spectral columns: {missing[:10]}"
    )

names = (
    df[
        "lithology_family"
    ]
    .astype(str)
    .tolist()
)

codes = (
    df[
        "subtype_code"
    ]
    .astype(int)
    .tolist()
)

X = (
    df[
        GOOD_COLS
    ]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=float
    )
)

n = len(names)

print("\nEndmembers:", n)
print("Bands:", X.shape[1])

# ============================================================
# PRECOMPUTE TRANSFORMS
# ============================================================

X_smooth = np.vstack(
    [
        smooth(row)
        for row in X
    ]
)

X_shape = np.vstack(
    [
        minmax(row)
        for row in X_smooth
    ]
)

X_deriv = np.vstack(
    [
        derivative_vector(row)
        for row in X
    ]
)

X_cr = np.vstack(
    [
        continuum_removed_vector(row)
        for row in X
    ]
)

# ============================================================
# MATRICES
# ============================================================

sam = np.zeros(
    (n, n),
    dtype=float
)

raw_corr = np.zeros(
    (n, n),
    dtype=float
)

shape_corr = np.zeros(
    (n, n),
    dtype=float
)

deriv_corr = np.zeros(
    (n, n),
    dtype=float
)

cr_corr = np.zeros(
    (n, n),
    dtype=float
)

redundancy = np.zeros(
    (n, n),
    dtype=float
)

pair_rows = []

for i in range(n):

    for j in range(n):

        angle = spectral_angle_deg(
            X[i],
            X[j]
        )

        rc = correlation(
            X[i],
            X[j]
        )

        sc = correlation(
            X_shape[i],
            X_shape[j]
        )

        dc = correlation(
            X_deriv[i],
            X_deriv[j]
        )

        cc = correlation(
            1.0 - X_cr[i],
            1.0 - X_cr[j]
        )

        sam[i, j] = angle
        raw_corr[i, j] = rc
        shape_corr[i, j] = sc
        deriv_corr[i, j] = dc
        cr_corr[i, j] = cc

        if i == j:

            redundancy[i, j] = 1.0

        else:

            score = (
                0.30
                * sam_similarity_score(
                    angle
                )
                +
                0.20
                * corr_score(
                    sc
                )
                +
                0.20
                * corr_score(
                    dc
                )
                +
                0.30
                * corr_score(
                    cc
                )
            )

            redundancy[i, j] = score

            if i < j:

                pair_rows.append({
                    "code_A":
                        codes[i],

                    "lithology_A":
                        names[i],

                    "code_B":
                        codes[j],

                    "lithology_B":
                        names[j],

                    "raw_SAM_deg":
                        angle,

                    "raw_correlation":
                        rc,

                    "shape_correlation":
                        sc,

                    "derivative_correlation":
                        dc,

                    "continuum_removed_correlation":
                        cc,

                    "redundancy_score":
                        score,

                    "SAM_separation_QC":
                        sam_qc_label(
                            angle
                        ),

                    "redundancy_QC":
                        redundancy_label(
                            score
                        )
                })

pairs = pd.DataFrame(
    pair_rows
)

# ============================================================
# MATRIX DATAFRAMES
# ============================================================

labels = [
    f"{code:02d}_{name}"
    for code, name in zip(
        codes,
        names
    )
]

sam_df = pd.DataFrame(
    sam,
    index=labels,
    columns=labels
)

raw_corr_df = pd.DataFrame(
    raw_corr,
    index=labels,
    columns=labels
)

shape_corr_df = pd.DataFrame(
    shape_corr,
    index=labels,
    columns=labels
)

deriv_corr_df = pd.DataFrame(
    deriv_corr,
    index=labels,
    columns=labels
)

cr_corr_df = pd.DataFrame(
    cr_corr,
    index=labels,
    columns=labels
)

redundancy_df = pd.DataFrame(
    redundancy,
    index=labels,
    columns=labels
)

# ============================================================
# NEAREST / MOST REDUNDANT NEIGHBOUR
# ============================================================

nearest_rows = []

for i in range(n):

    candidate_indices = [
        j
        for j in range(n)
        if j != i
    ]

    # minimum SAM neighbour
    sam_values = [
        sam[i, j]
        for j in candidate_indices
    ]

    nearest_sam_j = candidate_indices[
        int(
            np.nanargmin(
                sam_values
            )
        )
    ]

    # highest redundancy neighbour
    red_values = [
        redundancy[i, j]
        for j in candidate_indices
    ]

    nearest_red_j = candidate_indices[
        int(
            np.nanargmax(
                red_values
            )
        )
    ]

    nearest_rows.append({
        "subtype_code":
            codes[i],

        "lithology_family":
            names[i],

        "nearest_by_SAM":
            names[
                nearest_sam_j
            ],

        "nearest_SAM_deg":
            float(
                sam[
                    i,
                    nearest_sam_j
                ]
            ),

        "SAM_QC":
            sam_qc_label(
                sam[
                    i,
                    nearest_sam_j
                ]
            ),

        "most_redundant_with":
            names[
                nearest_red_j
            ],

        "maximum_redundancy_score":
            float(
                redundancy[
                    i,
                    nearest_red_j
                ]
            ),

        "redundancy_QC":
            redundancy_label(
                redundancy[
                    i,
                    nearest_red_j
                ]
            ),

        "shape_correlation_with_nearest":
            float(
                shape_corr[
                    i,
                    nearest_red_j
                ]
            ),

        "derivative_correlation_with_nearest":
            float(
                deriv_corr[
                    i,
                    nearest_red_j
                ]
            ),

        "CR_correlation_with_nearest":
            float(
                cr_corr[
                    i,
                    nearest_red_j
                ]
            )
    })

nearest = pd.DataFrame(
    nearest_rows
)

# ============================================================
# CLASS-LEVEL SEPARABILITY SUMMARY
# ============================================================

class_rows = []

for i in range(n):

    other = [
        j
        for j in range(n)
        if j != i
    ]

    min_sam = float(
        np.nanmin(
            sam[
                i,
                other
            ]
        )
    )

    median_sam = float(
        np.nanmedian(
            sam[
                i,
                other
            ]
        )
    )

    max_red = float(
        np.nanmax(
            redundancy[
                i,
                other
            ]
        )
    )

    high_red_count = int(
        np.sum(
            redundancy[
                i,
                other
            ]
            >= REDUNDANCY_HIGH
        )
    )

    low_sam_count = int(
        np.sum(
            sam[
                i,
                other
            ]
            < SAM_LOW
        )
    )

    if (
        min_sam >= SAM_MODERATE
        and
        max_red < REDUNDANCY_MODERATE
    ):

        readiness = (
            "GOOD_INTERNAL_SEPARABILITY"
        )

    elif (
        min_sam >= SAM_LOW
        and
        max_red < REDUNDANCY_HIGH
    ):

        readiness = (
            "MODERATE_INTERNAL_SEPARABILITY"
        )

    else:

        readiness = (
            "POOR_OR_REDUNDANT_ENDMEMBER"
        )

    class_rows.append({
        "subtype_code":
            codes[i],

        "lithology_family":
            names[i],

        "minimum_pairwise_SAM_deg":
            min_sam,

        "median_pairwise_SAM_deg":
            median_sam,

        "maximum_redundancy_score":
            max_red,

        "pairs_with_SAM_below_10deg":
            low_sam_count,

        "pairs_with_high_redundancy":
            high_red_count,

        "internal_separability_status":
            readiness
    })

class_summary = pd.DataFrame(
    class_rows
)

# ============================================================
# GLOBAL QC
# ============================================================

offdiag_sam = sam[
    ~np.eye(
        n,
        dtype=bool
    )
]

offdiag_red = redundancy[
    ~np.eye(
        n,
        dtype=bool
    )
]

global_min_sam = float(
    np.nanmin(
        offdiag_sam
    )
)

global_median_sam = float(
    np.nanmedian(
        offdiag_sam
    )
)

global_max_red = float(
    np.nanmax(
        offdiag_red
    )
)

high_redundant_pairs = pairs[
    pairs[
        "redundancy_score"
    ]
    >= REDUNDANCY_HIGH
].copy()

very_low_sam_pairs = pairs[
    pairs[
        "raw_SAM_deg"
    ]
    < SAM_VERY_LOW
].copy()

low_sam_pairs = pairs[
    pairs[
        "raw_SAM_deg"
    ]
    < SAM_LOW
].copy()

if (
    len(high_redundant_pairs) == 0
    and
    len(low_sam_pairs) == 0
):

    GLOBAL_GATE = (
        "PASS_13_ENDMEMBERS_ARE_REASONABLY_SEPARABLE"
    )

elif (
    len(high_redundant_pairs) <= 3
    and
    len(low_sam_pairs) <= 5
):

    GLOBAL_GATE = (
        "CAUTION_SOME_ENDMEMBER_REDUNDANCY"
    )

else:

    GLOBAL_GATE = (
        "FAIL_FOR_HARD_13_CLASS_MAPPING_HIGH_REDUNDANCY"
    )

# ============================================================
# SAVE CSV
# ============================================================

pairs.to_csv(
    OUT / (
        "AVIRIS_Step15E_All_78_Endmember_Pairs.csv"
    ),
    index=False
)

nearest.to_csv(
    OUT / (
        "AVIRIS_Step15E_Nearest_Endmember_QC.csv"
    ),
    index=False
)

class_summary.to_csv(
    OUT / (
        "AVIRIS_Step15E_Class_Separability_Summary.csv"
    ),
    index=False
)

sam_df.to_csv(
    OUT / (
        "AVIRIS_Step15E_SAM_Matrix_deg.csv"
    )
)

redundancy_df.to_csv(
    OUT / (
        "AVIRIS_Step15E_Redundancy_Matrix.csv"
    )
)

# ============================================================
# SAM MATRIX FIGURE
# ============================================================

fig = plt.figure(
    figsize=(14, 12)
)

ax = fig.add_subplot(111)

im = ax.imshow(
    sam
)

ax.set_xticks(
    np.arange(n)
)

ax.set_yticks(
    np.arange(n)
)

ax.set_xticklabels(
    [
        str(c)
        for c in codes
    ]
)

ax.set_yticklabels(
    [
        str(c)
        for c in codes
    ]
)

ax.set_xlabel(
    "Lithology Code"
)

ax.set_ylabel(
    "Lithology Code"
)

ax.set_title(
    "AVIRIS-NG 13-Endmember Pairwise Spectral Angle (degrees)"
)

fig.colorbar(
    im,
    ax=ax,
    label="SAM (degrees)"
)

fig.tight_layout()

SAM_PNG = OUT / (
    "AVIRIS_Step15E_SAM_Matrix.png"
)

fig.savefig(
    SAM_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# REDUNDANCY MATRIX FIGURE
# ============================================================

fig = plt.figure(
    figsize=(14, 12)
)

ax = fig.add_subplot(111)

im = ax.imshow(
    redundancy
)

ax.set_xticks(
    np.arange(n)
)

ax.set_yticks(
    np.arange(n)
)

ax.set_xticklabels(
    [
        str(c)
        for c in codes
    ]
)

ax.set_yticklabels(
    [
        str(c)
        for c in codes
    ]
)

ax.set_xlabel(
    "Lithology Code"
)

ax.set_ylabel(
    "Lithology Code"
)

ax.set_title(
    "AVIRIS-NG 13-Endmember Composite Spectral Redundancy"
)

fig.colorbar(
    im,
    ax=ax,
    label="Redundancy Score"
)

fig.tight_layout()

RED_PNG = OUT / (
    "AVIRIS_Step15E_Redundancy_Matrix.png"
)

fig.savefig(
    RED_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# HIERARCHICAL CLUSTERING
#
# Distance based on redundancy:
# distance = 1 - redundancy
# ============================================================

distance = (
    1.0 - redundancy
)

distance = (
    distance
    +
    distance.T
) / 2.0

np.fill_diagonal(
    distance,
    0.0
)

distance = np.clip(
    distance,
    0,
    1
)

condensed = squareform(
    distance,
    checks=False
)

Z = linkage(
    condensed,
    method="average"
)

fig = plt.figure(
    figsize=(15, 8)
)

ax = fig.add_subplot(111)

dendrogram(
    Z,
    labels=[
        f"{codes[i]} {names[i]}"
        for i in range(n)
    ],
    leaf_rotation=90,
    ax=ax
)

ax.set_title(
    "Hierarchical Clustering of 13 AVIRIS-NG Image Endmembers"
)

ax.set_ylabel(
    "Composite Spectral Distance"
)

fig.tight_layout()

DENDRO_PNG = OUT / (
    "AVIRIS_Step15E_Endmember_Dendrogram.png"
)

fig.savefig(
    DENDRO_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15E_13_Endmember_Separability_QC.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    class_summary.to_excel(
        writer,
        sheet_name="Class_Summary",
        index=False
    )

    nearest.to_excel(
        writer,
        sheet_name="Nearest_Neighbour",
        index=False
    )

    pairs.to_excel(
        writer,
        sheet_name="All_78_Pairs",
        index=False
    )

    high_redundant_pairs.to_excel(
        writer,
        sheet_name="High_Redundancy",
        index=False
    )

    low_sam_pairs.to_excel(
        writer,
        sheet_name="SAM_Below_10deg",
        index=False
    )

    sam_df.to_excel(
        writer,
        sheet_name="SAM_Matrix"
    )

    raw_corr_df.to_excel(
        writer,
        sheet_name="Raw_Correlation"
    )

    shape_corr_df.to_excel(
        writer,
        sheet_name="Shape_Correlation"
    )

    deriv_corr_df.to_excel(
        writer,
        sheet_name="Derivative_Correlation"
    )

    cr_corr_df.to_excel(
        writer,
        sheet_name="Continuum_Correlation"
    )

    redundancy_df.to_excel(
        writer,
        sheet_name="Redundancy_Matrix"
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step15E_13_Endmember_Separability_Redundancy",

    "number_of_endmembers":
        n,

    "bands":
        X.shape[1],

    "SAM_thresholds_deg": {
        "very_low_separation":
            SAM_VERY_LOW,

        "low_separation":
            SAM_LOW,

        "moderate_separation":
            SAM_MODERATE
    },

    "redundancy_thresholds": {
        "high":
            REDUNDANCY_HIGH,

        "moderate":
            REDUNDANCY_MODERATE
    },

    "redundancy_weights": {
        "SAM_similarity":
            0.30,

        "spectral_shape":
            0.20,

        "first_derivative":
            0.20,

        "continuum_removed_shape":
            0.30
    },

    "global_gate":
        GLOBAL_GATE,

    "important_note":
        (
            "Thresholds are heuristic spectral-QC criteria. "
            "They do not prove or disprove geological identity."
        )
}

(
    OUT / "AVIRIS_Step15E_Settings.json"
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
print("STEP15E COMPLETE")
print("=" * 90)

print("\nGLOBAL SEPARABILITY:")
print(
    "Minimum pairwise SAM:",
    round(
        global_min_sam,
        4
    ),
    "deg"
)

print(
    "Median pairwise SAM:",
    round(
        global_median_sam,
        4
    ),
    "deg"
)

print(
    "Maximum redundancy score:",
    round(
        global_max_red,
        4
    )
)

print(
    "Pairs with SAM < 5 deg:",
    len(
        very_low_sam_pairs
    )
)

print(
    "Pairs with SAM < 10 deg:",
    len(
        low_sam_pairs
    )
)

print(
    "Pairs with redundancy >= 0.85:",
    len(
        high_redundant_pairs
    )
)

print("\nGLOBAL GATE:")
print(
    GLOBAL_GATE
)

print("\nCLASS SEPARABILITY SUMMARY:")

print(
    class_summary.to_string(
        index=False
    )
)

print("\nNEAREST / MOST REDUNDANT END-MEMBER:")

print(
    nearest[
        [
            "subtype_code",
            "lithology_family",
            "nearest_by_SAM",
            "nearest_SAM_deg",
            "SAM_QC",
            "most_redundant_with",
            "maximum_redundancy_score",
            "redundancy_QC"
        ]
    ].to_string(
        index=False
    )
)

print("\nTOP 20 MOST REDUNDANT PAIRS:")

print(
    pairs.sort_values(
        "redundancy_score",
        ascending=False
    )[
        [
            "lithology_A",
            "lithology_B",
            "raw_SAM_deg",
            "shape_correlation",
            "derivative_correlation",
            "continuum_removed_correlation",
            "redundancy_score",
            "redundancy_QC"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)

print("\nSAM matrix image:")
print(
    SAM_PNG
)

print("\nRedundancy matrix image:")
print(
    RED_PNG
)

print("\nDendrogram:")
print(
    DENDRO_PNG
)

print("\nExcel:")
print(
    XLSX
)

print("\nIMPORTANT:")
print(
    "If GLOBAL GATE reports FAIL_FOR_HARD_13_CLASS_MAPPING_HIGH_REDUNDANCY, "
    "do not use the 13 image endmembers for a hard lithology classification."
)

print(
    "In that case we will merge or retain only spectrally defensible endmember groups."
)

print("\nNEXT:")
print(
    "Step15F = scientifically defensible endmember set / merged classes "
    "based on Step15D + Step15E evidence."
)

print("\nPASS")
