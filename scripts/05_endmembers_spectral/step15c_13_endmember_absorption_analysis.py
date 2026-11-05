# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP15B = ROOT / (
    "09_Mineral_Mapping/"
    "Step15B_13_Endmember_Spectral_Library"
)

LONG_CSV = STEP15B / (
    "AVIRIS_Step15B_13_Endmember_Spectra_Long.csv"
)

LOCATION_CSV = STEP15B / (
    "AVIRIS_Step15B_13_Endmember_Location_Summary.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15C_13_Endmember_Absorption_Features"
)

OUT.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT / "Individual_Continuum_Removal_Plots"
PLOT_DIR.mkdir(exist_ok=True)

# ============================================================
# ANALYSIS SETTINGS
# ============================================================

SAVGOL_WINDOW = 7
SAVGOL_ORDER = 2

MIN_DEPTH_PERCENT = 1.0
MIN_PROMINENCE_PERCENT = 0.5

EDGE_EXCLUSION_NM = 12.0

# Spectral windows
WINDOWS = {
    "VNIR_NIR": (476.0, 1329.0),
    "SWIR1": (1473.0, 1789.0),
    "SWIR2": (1994.0, 2445.5),
}

# Useful diagnostic sub-windows for interpretation
DIAGNOSTIC_WINDOWS = {
    "VIS_FE_OXIDE": (500, 800),
    "MAFIC_1UM": (850, 1300),
    "OH_ALOH": (2100, 2250),
    "MG_FE_OH": (2250, 2400),
}

# ============================================================
# HELPERS
# ============================================================

def upper_hull_continuum(wl, refl):
    """
    Compute upper convex hull continuum for one spectral segment.
    Returns continuum evaluated at all wavelengths.
    """

    x = np.asarray(wl, dtype=float)
    y = np.asarray(refl, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)

    x = x[good]
    y = y[good]

    if len(x) < 3:
        return None, None, None

    pts = list(zip(x, y))

    hull = []

    for p in pts:
        while len(hull) >= 2:
            x1, y1 = hull[-2]
            x2, y2 = hull[-1]
            x3, y3 = p

            cross = (
                (x2 - x1) * (y3 - y2)
                - (y2 - y1) * (x3 - x2)
            )

            # keep upper hull
            if cross >= 0:
                hull.pop()
            else:
                break

        hull.append(p)

    hull_x = np.array([p[0] for p in hull], dtype=float)
    hull_y = np.array([p[1] for p in hull], dtype=float)

    continuum = np.interp(
        x,
        hull_x,
        hull_y
    )

    return x, y, continuum


def continuum_remove(wl, refl):
    result = upper_hull_continuum(wl, refl)

    if result[0] is None:
        return None

    x, y, continuum = result

    cr = np.full_like(y, np.nan)

    good = (
        np.isfinite(y)
        & np.isfinite(continuum)
        & (continuum > 1e-12)
    )

    cr[good] = y[good] / continuum[good]

    return x, y, continuum, cr


def interpolate_crossing(x1, y1, x2, y2, target):
    if not all(np.isfinite([x1, y1, x2, y2, target])):
        return np.nan

    if y2 == y1:
        return (x1 + x2) / 2.0

    return x1 + (
        (target - y1)
        * (x2 - x1)
        / (y2 - y1)
    )


def estimate_fwhm(wl, cr, peak_idx, depth):
    """
    FWHM in absorption-depth space.
    cr baseline = 1.
    center depth = 1 - cr.
    half-depth continuum-removal level = 1 - depth/2.
    """

    half_level = 1.0 - depth / 2.0

    left_cross = np.nan
    right_cross = np.nan

    # left
    for i in range(peak_idx, 0, -1):
        y1 = cr[i - 1]
        y2 = cr[i]

        if (
            np.isfinite(y1)
            and np.isfinite(y2)
            and (
                (y1 - half_level)
                * (y2 - half_level)
                <= 0
            )
        ):
            left_cross = interpolate_crossing(
                wl[i - 1],
                y1,
                wl[i],
                y2,
                half_level
            )
            break

    # right
    for i in range(peak_idx, len(cr) - 1):
        y1 = cr[i]
        y2 = cr[i + 1]

        if (
            np.isfinite(y1)
            and np.isfinite(y2)
            and (
                (y1 - half_level)
                * (y2 - half_level)
                <= 0
            )
        ):
            right_cross = interpolate_crossing(
                wl[i],
                y1,
                wl[i + 1],
                y2,
                half_level
            )
            break

    if (
        np.isfinite(left_cross)
        and np.isfinite(right_cross)
        and right_cross > left_cross
    ):
        return (
            float(right_cross - left_cross),
            float(left_cross),
            float(right_cross)
        )

    return np.nan, left_cross, right_cross


def absorption_area(wl, cr, left_nm, right_nm):
    if not np.isfinite(left_nm) or not np.isfinite(right_nm):
        return np.nan

    mask = (
        (wl >= left_nm)
        & (wl <= right_nm)
        & np.isfinite(cr)
    )

    if mask.sum() < 2:
        return np.nan

    depth_curve = 1.0 - cr[mask]

    return float(
        np.trapezoid(
            depth_curve,
            wl[mask]
        )
    )


def diagnostic_region(center):
    for name, (lo, hi) in DIAGNOSTIC_WINDOWS.items():
        if lo <= center <= hi:
            return name

    return "OTHER"


# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 90)
print("STEP15C - 13 AVIRIS-NG ENDMEMBER ABSORPTION FEATURE ANALYSIS")
print("=" * 90)

for p in [LONG_CSV, LOCATION_CSV]:
    if not p.exists():
        raise FileNotFoundError(p)

long_df = pd.read_csv(LONG_CSV)
locations = pd.read_csv(LOCATION_CSV)

required = [
    "subtype_code",
    "lithology_family",
    "wavelength_nm",
    "reflectance",
]

for c in required:
    if c not in long_df.columns:
        raise RuntimeError(
            f"Required column missing: {c}"
        )

print("\nRows loaded:", len(long_df))
print(
    "Endmembers:",
    long_df["lithology_family"].nunique()
)

# ============================================================
# FEATURE EXTRACTION
# ============================================================

feature_rows = []
continuum_rows = []
summary_rows = []

for (code, lith), sub in long_df.groupby(
    ["subtype_code", "lithology_family"]
):

    sub = sub.sort_values(
        "wavelength_nm"
    ).copy()

    wl_all = sub[
        "wavelength_nm"
    ].to_numpy(dtype=float)

    refl_all = sub[
        "reflectance"
    ].to_numpy(dtype=float)

    print("\n" + "-" * 90)
    print(
        f"{int(code):02d} - {lith}"
    )

    lith_features = []

    for window_name, (lo, hi) in WINDOWS.items():

        mask = (
            (wl_all >= lo)
            & (wl_all <= hi)
            & np.isfinite(refl_all)
        )

        wl = wl_all[mask]
        refl = refl_all[mask]

        if len(wl) < 7:
            continue

        # Smooth raw reflectance
        win = min(
            SAVGOL_WINDOW,
            len(refl)
            if len(refl) % 2 == 1
            else len(refl) - 1
        )

        if win < 5:
            smooth = refl.copy()
        else:
            smooth = savgol_filter(
                refl,
                window_length=win,
                polyorder=min(
                    SAVGOL_ORDER,
                    win - 1
                )
            )

        cr_result = continuum_remove(
            wl,
            smooth
        )

        if cr_result is None:
            continue

        x, y, continuum, cr = cr_result

        depth_curve = (
            1.0 - cr
        ) * 100.0

        # Find absorption minima using peaks in depth
        peaks, props = find_peaks(
            depth_curve,
            height=MIN_DEPTH_PERCENT,
            prominence=MIN_PROMINENCE_PERCENT,
            distance=2
        )

        for i in range(len(x)):

            continuum_rows.append({
                "subtype_code":
                    int(code),

                "lithology_family":
                    lith,

                "spectral_window":
                    window_name,

                "wavelength_nm":
                    x[i],

                "smoothed_reflectance":
                    y[i],

                "continuum":
                    continuum[i],

                "continuum_removed":
                    cr[i],

                "absorption_depth_percent":
                    depth_curve[i]
            })

        for k, pidx in enumerate(peaks):

            center = float(
                x[pidx]
            )

            # Edge rejection
            if (
                center - lo < EDGE_EXCLUSION_NM
                or
                hi - center < EDGE_EXCLUSION_NM
            ):
                continue

            depth_fraction = float(
                1.0 - cr[pidx]
            )

            depth_percent = (
                depth_fraction * 100.0
            )

            prominence = float(
                props[
                    "prominences"
                ][k]
            )

            fwhm, left_nm, right_nm = estimate_fwhm(
                x,
                cr,
                pidx,
                depth_fraction
            )

            area = absorption_area(
                x,
                cr,
                left_nm,
                right_nm
            )

            rec = {
                "subtype_code":
                    int(code),

                "lithology_family":
                    lith,

                "spectral_window":
                    window_name,

                "diagnostic_region":
                    diagnostic_region(
                        center
                    ),

                "absorption_center_nm":
                    center,

                "absorption_depth_percent":
                    depth_percent,

                "prominence_percent":
                    prominence,

                "FWHM_nm":
                    fwhm,

                "half_depth_left_nm":
                    left_nm,

                "half_depth_right_nm":
                    right_nm,

                "absorption_area":
                    area,
            }

            feature_rows.append(
                rec
            )

            lith_features.append(
                rec
            )

    # ========================================================
    # PER-LITHOLOGY SUMMARY
    # ========================================================

    if lith_features:

        lf = pd.DataFrame(
            lith_features
        )

        lf = lf.sort_values(
            [
                "absorption_depth_percent",
                "prominence_percent"
            ],
            ascending=False
        )

        strongest = lf.iloc[0]

        top3 = lf.head(3)

        top3_string = "; ".join(
            [
                (
                    f"{r.absorption_center_nm:.1f} nm "
                    f"({r.absorption_depth_percent:.2f}%)"
                )
                for _, r in top3.iterrows()
            ]
        )

        summary_rows.append({
            "subtype_code":
                int(code),

            "lithology_family":
                lith,

            "total_detected_features":
                len(lf),

            "strongest_center_nm":
                strongest[
                    "absorption_center_nm"
                ],

            "strongest_depth_percent":
                strongest[
                    "absorption_depth_percent"
                ],

            "strongest_FWHM_nm":
                strongest[
                    "FWHM_nm"
                ],

            "strongest_area":
                strongest[
                    "absorption_area"
                ],

            "strongest_prominence_percent":
                strongest[
                    "prominence_percent"
                ],

            "strongest_window":
                strongest[
                    "spectral_window"
                ],

            "strongest_diagnostic_region":
                strongest[
                    "diagnostic_region"
                ],

            "top3_absorptions":
                top3_string,
        })

        print(
            "Detected features:",
            len(lf)
        )

        print(
            "Strongest:",
            f"{strongest['absorption_center_nm']:.2f} nm",
            f"depth={strongest['absorption_depth_percent']:.2f}%"
        )

    else:

        summary_rows.append({
            "subtype_code":
                int(code),

            "lithology_family":
                lith,

            "total_detected_features":
                0,

            "strongest_center_nm":
                np.nan,

            "strongest_depth_percent":
                np.nan,

            "strongest_FWHM_nm":
                np.nan,

            "strongest_area":
                np.nan,

            "strongest_prominence_percent":
                np.nan,

            "strongest_window":
                "",

            "strongest_diagnostic_region":
                "",

            "top3_absorptions":
                "",
        })

        print(
            "Detected features: 0"
        )

# ============================================================
# DATAFRAMES
# ============================================================

features = pd.DataFrame(
    feature_rows
)

continuum_df = pd.DataFrame(
    continuum_rows
)

summary = pd.DataFrame(
    summary_rows
)

# ============================================================
# MERGE LOCATIONS
# ============================================================

location_cols = [
    "subtype_code",
    "lithology_family",
    "longitude",
    "latitude",
    "UTM_Easting_m",
    "UTM_Northing_m",
    "broad_probability",
    "uncertainty",
    "subtype_similarity",
]

location_cols = [
    c for c in location_cols
    if c in locations.columns
]

summary = summary.merge(
    locations[
        location_cols
    ],
    on=[
        "subtype_code",
        "lithology_family"
    ],
    how="left"
)

# ============================================================
# FEATURE QC FLAGS
# ============================================================

if not features.empty:

    features[
        "depth_qc"
    ] = np.where(
        features[
            "absorption_depth_percent"
        ] >= 3.0,
        "STRONGER_FEATURE",
        "WEAK_FEATURE"
    )

    features[
        "fwhm_qc"
    ] = np.where(
        features[
            "FWHM_nm"
        ].between(
            5,
            300,
            inclusive="both"
        ),
        "PLAUSIBLE_WIDTH",
        "REVIEW_WIDTH"
    )

# ============================================================
# SAVE TABLES
# ============================================================

FEATURE_CSV = OUT / (
    "AVIRIS_Step15C_All_Absorption_Features.csv"
)

SUMMARY_CSV = OUT / (
    "AVIRIS_Step15C_Endmember_Absorption_Summary.csv"
)

CONTINUUM_CSV = OUT / (
    "AVIRIS_Step15C_Continuum_Removed_Spectra.csv"
)

features.to_csv(
    FEATURE_CSV,
    index=False
)

summary.to_csv(
    SUMMARY_CSV,
    index=False
)

continuum_df.to_csv(
    CONTINUUM_CSV,
    index=False
)

# ============================================================
# INDIVIDUAL CONTINUUM-REMOVAL PLOTS
# ============================================================

for (code, lith), sub in continuum_df.groupby(
    ["subtype_code", "lithology_family"]
):

    fig = plt.figure(
        figsize=(12, 7)
    )

    ax = fig.add_subplot(111)

    for window_name, ss in sub.groupby(
        "spectral_window"
    ):

        ax.plot(
            ss["wavelength_nm"],
            ss["continuum_removed"],
            linewidth=1.4,
            label=window_name
        )

    lith_feat = features[
        (
            features[
                "subtype_code"
            ] == code
        )
        &
        (
            features[
                "lithology_family"
            ] == lith
        )
    ]

    if not lith_feat.empty:

        for _, r in lith_feat.iterrows():

            ax.axvline(
                r[
                    "absorption_center_nm"
                ],
                alpha=0.25
            )

    ax.axhline(
        1.0,
        linewidth=0.8,
        alpha=0.4
    )

    ax.set_xlabel(
        "Wavelength (nm)"
    )

    ax.set_ylabel(
        "Continuum-Removed Reflectance"
    )

    ax.set_title(
        f"{int(code):02d} - "
        f"{lith.replace('_', ' ').title()}"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    out_png = PLOT_DIR / (
        f"{int(code):02d}_{lith}_Continuum_Removal.png"
    )

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# COMBINED STRONGEST-FEATURE PLOT
# ============================================================

fig = plt.figure(
    figsize=(14, 8)
)

ax = fig.add_subplot(111)

for (code, lith), sub in continuum_df.groupby(
    ["subtype_code", "lithology_family"]
):

    ax.plot(
        sub["wavelength_nm"],
        sub["continuum_removed"],
        linewidth=1.0,
        alpha=0.8,
        label=(
            f"{int(code)} "
            f"{lith.replace('_', ' ').title()}"
        )
    )

ax.set_xlabel(
    "Wavelength (nm)"
)

ax.set_ylabel(
    "Continuum-Removed Reflectance"
)

ax.set_title(
    "Continuum-Removed AVIRIS-NG Spectra of 13 Image-Derived Endmembers"
)

ax.legend(
    fontsize=7,
    ncol=2
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

COMBINED_PNG = OUT / (
    "AVIRIS_Step15C_13_Endmembers_Continuum_Removed.png"
)

fig.savefig(
    COMBINED_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15C_13_Endmember_Absorption_Features.xlsx"
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

    features.to_excel(
        writer,
        sheet_name="All_Features",
        index=False
    )

    continuum_df.to_excel(
        writer,
        sheet_name="Continuum_Removed",
        index=False
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step15C_13_Endmember_Absorption_Features",

    "input":
        str(LONG_CSV),

    "savgol_window":
        SAVGOL_WINDOW,

    "savgol_order":
        SAVGOL_ORDER,

    "minimum_depth_percent":
        MIN_DEPTH_PERCENT,

    "minimum_prominence_percent":
        MIN_PROMINENCE_PERCENT,

    "edge_exclusion_nm":
        EDGE_EXCLUSION_NM,

    "spectral_windows":
        WINDOWS,

    "diagnostic_windows":
        DIAGNOSTIC_WINDOWS,

    "important_warning":
        (
            "Detected absorption features describe the selected "
            "image-derived AVIRIS-NG endmember spectra. "
            "They are not mineral identifications by themselves."
        )
}

(
    OUT / "AVIRIS_Step15C_Settings.json"
).write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("STEP15C COMPLETE")
print("=" * 90)

print("\nENDMEMBER ABSORPTION SUMMARY:")

show_cols = [
    "subtype_code",
    "lithology_family",
    "total_detected_features",
    "strongest_center_nm",
    "strongest_depth_percent",
    "strongest_FWHM_nm",
    "strongest_area",
    "strongest_prominence_percent",
    "strongest_window",
    "strongest_diagnostic_region",
    "top3_absorptions",
]

print(
    summary[
        show_cols
    ].to_string(
        index=False
    )
)

print("\nTotal absorption features:",
      len(features))

print("\nFeature table:")
print(FEATURE_CSV)

print("\nSummary table:")
print(SUMMARY_CSV)

print("\nContinuum-removed spectra:")
print(CONTINUUM_CSV)

print("\nCombined continuum-removal plot:")
print(COMBINED_PNG)

print("\nIndividual plots:")
print(PLOT_DIR)

print("\nExcel workbook:")
print(XLSX)

print("\nIMPORTANT:")
print(
    "Absorption centers, depths, FWHM, area and prominence "
    "are spectral descriptors only."
)

print(
    "Do not assign mineral names solely from these features."
)

print(
    "Next scientific step: compare these 13 AVIRIS endmembers "
    "against clean SAC ASD family spectra."
)

print("\nPASS")
