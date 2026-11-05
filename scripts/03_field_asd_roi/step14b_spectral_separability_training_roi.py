# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import box
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

AVIRIS = ROOT / \
    "03_Mosaic/Final_425Band_Mosaic/AVIRIS_NG_mosaic_master_425band_4m.dat"

STEP14A = ROOT / \
    "09_Mineral_Mapping/Step14A_Lithology_Training_Data_Preparation"

FIELD_CSV = STEP14A / \
    "AVIRIS_Step14A_Field_Samples_Classified.csv"

SPECTRA_CSV = STEP14A / \
    "AVIRIS_Step14A_Field_Spectra_3x3Median_Wide.csv"

WAVELENGTH_CSV = STEP14A / \
    "AVIRIS_Step14A_Band_Wavelength_Table.csv"

OUT = ROOT / \
    "09_Mineral_Mapping/Step14B_Spectral_Separability_Training_ROI"

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SCIENTIFIC GOOD BANDS
# Original 425-band numbering
# ============================================================

GOOD_RANGES_1BASED = [
    (21, 191),
    (220, 283),
    (324, 414),
]

MIN_VALID_BANDS = 250

# These are QC heuristics, not formal mineral-ID thresholds.
PAIR_SAM_GOOD_DEG = 8.0
PAIR_SAM_BORDERLINE_DEG = 5.0
RATIO_GOOD = 2.0
RATIO_BORDERLINE = 1.25

PIXEL_SIZE_M = 4.0
ROI_HALF_SIZE_M = 6.0       # 12 m × 12 m, approximately 3×3 pixels

# ============================================================
# HELPERS
# ============================================================

def good_band_numbers():
    nums = []
    for lo, hi in GOOD_RANGES_1BASED:
        nums.extend(range(lo, hi + 1))
    return nums


def spectral_angle_deg(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = np.isfinite(a) & np.isfinite(b)

    if ok.sum() < MIN_VALID_BANDS:
        return np.nan

    aa = a[ok]
    bb = b[ok]

    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)

    if na <= 0 or nb <= 0:
        return np.nan

    cosang = np.dot(aa, bb) / (na * nb)
    cosang = np.clip(cosang, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosang)))


def unit_normalize(spec):
    x = np.asarray(spec, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)

    ok = np.isfinite(x)

    if ok.sum() == 0:
        return out

    norm = np.linalg.norm(x[ok])

    if norm <= 0:
        return out

    out[ok] = x[ok] / norm
    return out


def robust_class_median(matrix):
    return np.nanmedian(matrix, axis=0)


def within_class_sam(matrix, median_spec):
    vals = []

    for row in matrix:
        ang = spectral_angle_deg(row, median_spec)

        if np.isfinite(ang):
            vals.append(ang)

    if not vals:
        return np.nan, np.nan, np.nan

    vals = np.asarray(vals)

    return (
        float(np.median(vals)),
        float(np.percentile(vals, 75)),
        float(np.max(vals)),
    )


def pair_status(sam_deg, separation_ratio):
    if not np.isfinite(sam_deg):
        return "INSUFFICIENT_DATA"

    if sam_deg >= PAIR_SAM_GOOD_DEG and separation_ratio >= RATIO_GOOD:
        return "GOOD_SEPARATION"

    if sam_deg >= PAIR_SAM_BORDERLINE_DEG and separation_ratio >= RATIO_BORDERLINE:
        return "BORDERLINE_SEPARATION"

    return "LOW_SEPARATION_REVIEW_OR_HIERARCHICAL_CLASSIFIER"


def sample_strength(n):
    if n >= 5:
        return "STRONG_SAMPLE_SUPPORT"
    elif n >= 3:
        return "LIMITED_SAMPLE_SUPPORT"
    elif n >= 2:
        return "LOW_SAMPLE_SUPPORT"
    elif n == 1:
        return "VERY_LOW_SINGLE_REFERENCE"
    else:
        return "NO_SAMPLE"


# ============================================================
# LOAD INPUTS
# ============================================================

print("=" * 80)
print("STEP14B - SAC SPECTRAL SEPARABILITY + TRAINING ROI PREPARATION")
print("=" * 80)

for p in [AVIRIS, FIELD_CSV, SPECTRA_CSV, WAVELENGTH_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Missing required input:\n{p}")

field = pd.read_csv(FIELD_CSV)
spectra = pd.read_csv(SPECTRA_CSV)
wave = pd.read_csv(WAVELENGTH_CSV)

print("\nField samples :", len(field))
print("Spectra rows  :", len(spectra))
print("Bands in HDR table :", len(wave))

if len(field) != len(spectra):
    print("\nWARNING: field/spectra row counts differ.")

# ============================================================
# BAND SELECTION
# ============================================================

good_nums = good_band_numbers()

band_cols = []

for b in good_nums:
    c = f"B{b:03d}"
    if c in spectra.columns:
        band_cols.append(c)

print("\nScientific good bands selected:", len(band_cols))

if len(band_cols) < 300:
    raise RuntimeError(
        f"Expected approximately 326 good bands, found only {len(band_cols)}"
    )

# merge training role if needed
if "step14a_training_role" not in spectra.columns:
    merge_cols = [
        "field_index",
        "lithology_family",
        "step14a_training_role"
    ]

    merge_cols = [c for c in merge_cols if c in field.columns]

    spectra = spectra.merge(
        field[merge_cols],
        on=[c for c in ["field_index", "lithology_family"]
            if c in field.columns and c in spectra.columns],
        how="left"
    )

# ============================================================
# CLASS MATRICES AND MEDIAN SPECTRA
# ============================================================

classes = sorted(
    spectra["lithology_family"].dropna().astype(str).unique().tolist()
)

print("\nLithology classes found:", len(classes))

class_matrices = {}
class_medians = {}
class_stats = []

for fam in classes:

    sub = spectra[spectra["lithology_family"] == fam].copy()

    mat = sub[band_cols].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=float)

    class_matrices[fam] = mat

    med = robust_class_median(mat)
    class_medians[fam] = med

    med_sam, p75_sam, max_sam = within_class_sam(mat, med)

    class_stats.append({
        "lithology_family": fam,
        "sample_count": len(sub),
        "sample_strength": sample_strength(len(sub)),
        "median_within_class_SAM_deg": med_sam,
        "p75_within_class_SAM_deg": p75_sam,
        "maximum_within_class_SAM_deg": max_sam,
    })

class_stats = pd.DataFrame(class_stats)

# ============================================================
# PAIRWISE CLASS SEPARABILITY
# ============================================================

pair_rows = []

for i in range(len(classes)):
    for j in range(i + 1, len(classes)):

        a = classes[i]
        b = classes[j]

        sam = spectral_angle_deg(
            class_medians[a],
            class_medians[b]
        )

        wa = class_stats.loc[
            class_stats["lithology_family"] == a,
            "median_within_class_SAM_deg"
        ].iloc[0]

        wb = class_stats.loc[
            class_stats["lithology_family"] == b,
            "median_within_class_SAM_deg"
        ].iloc[0]

        pooled_within = np.nanmean([wa, wb])

        if np.isfinite(pooled_within) and pooled_within > 1e-9:
            ratio = sam / pooled_within
        else:
            ratio = np.nan

        pair_rows.append({
            "class_A": a,
            "class_B": b,
            "median_to_median_SAM_deg": sam,
            "pooled_median_within_class_SAM_deg": pooled_within,
            "between_to_within_ratio": ratio,
            "separability_status": pair_status(sam, ratio)
        })

pairwise = pd.DataFrame(pair_rows)

# ============================================================
# NEAREST COMPETING CLASS FOR EACH LITHOLOGY
# ============================================================

nearest_rows = []

for fam in classes:

    sub = pairwise[
        (pairwise["class_A"] == fam) |
        (pairwise["class_B"] == fam)
    ].copy()

    if sub.empty:
        continue

    sub = sub.sort_values("median_to_median_SAM_deg")

    r = sub.iloc[0]

    other = r["class_B"] if r["class_A"] == fam else r["class_A"]

    nearest_rows.append({
        "lithology_family": fam,
        "nearest_competing_class": other,
        "nearest_class_SAM_deg": r["median_to_median_SAM_deg"],
        "nearest_class_between_to_within_ratio": r["between_to_within_ratio"],
        "nearest_class_separability_status": r["separability_status"]
    })

nearest = pd.DataFrame(nearest_rows)

class_stats = class_stats.merge(
    nearest,
    on="lithology_family",
    how="left"
)

# ============================================================
# LEAVE-ONE-SAMPLE-OUT NEAREST MEDIAN SAM CLASSIFICATION
#
# Important:
# Each field sample is left out from its own class centroid.
# This avoids testing a sample against a centroid that contains itself.
# ============================================================

pred_rows = []

for idx, row in spectra.iterrows():

    true_class = str(row["lithology_family"])

    sample = row[band_cols].apply(
        pd.to_numeric,
        errors="coerce"
    ).to_numpy(dtype=float)

    candidate_angles = {}

    for fam in classes:

        fam_sub = spectra[
            spectra["lithology_family"] == fam
        ].copy()

        if fam == true_class:
            fam_sub = fam_sub.drop(index=idx, errors="ignore")

        if len(fam_sub) == 0:
            continue

        fam_mat = fam_sub[band_cols].apply(
            pd.to_numeric,
            errors="coerce"
        ).to_numpy(dtype=float)

        fam_med = np.nanmedian(fam_mat, axis=0)

        ang = spectral_angle_deg(sample, fam_med)

        if np.isfinite(ang):
            candidate_angles[fam] = ang

    if candidate_angles:

        ordered = sorted(
            candidate_angles.items(),
            key=lambda x: x[1]
        )

        pred_class = ordered[0][0]
        best_sam = ordered[0][1]

        if len(ordered) >= 2:
            second_class = ordered[1][0]
            second_sam = ordered[1][1]
            margin = second_sam - best_sam
        else:
            second_class = ""
            second_sam = np.nan
            margin = np.nan

    else:
        pred_class = "NO_PREDICTION"
        best_sam = np.nan
        second_class = ""
        second_sam = np.nan
        margin = np.nan

    pred_rows.append({
        "field_index": row.get("field_index", idx),
        "sample_label": row.get("sample_label", f"SAMPLE_{idx+1}"),
        "true_lithology": true_class,
        "predicted_lithology": pred_class,
        "correct_prediction": pred_class == true_class,
        "best_SAM_deg": best_sam,
        "second_best_class": second_class,
        "second_best_SAM_deg": second_sam,
        "SAM_margin_deg": margin,
    })

pred = pd.DataFrame(pred_rows)

valid_pred = pred[
    pred["predicted_lithology"] != "NO_PREDICTION"
].copy()

if not valid_pred.empty:

    labels = sorted(
        set(valid_pred["true_lithology"]) |
        set(valid_pred["predicted_lithology"])
    )

    cm = confusion_matrix(
        valid_pred["true_lithology"],
        valid_pred["predicted_lithology"],
        labels=labels
    )

    accuracy = accuracy_score(
        valid_pred["true_lithology"],
        valid_pred["predicted_lithology"]
    )

    try:
        balanced_acc = balanced_accuracy_score(
            valid_pred["true_lithology"],
            valid_pred["predicted_lithology"]
        )
    except Exception:
        balanced_acc = np.nan

    cm_df = pd.DataFrame(
        cm,
        index=[f"TRUE_{x}" for x in labels],
        columns=[f"PRED_{x}" for x in labels]
    )

else:

    accuracy = np.nan
    balanced_acc = np.nan
    cm_df = pd.DataFrame()

# ============================================================
# PER-CLASS LOOCV PERFORMANCE
# ============================================================

perf_rows = []

for fam in classes:

    sub = pred[
        pred["true_lithology"] == fam
    ].copy()

    n = len(sub)

    if n > 0:
        correct = int(sub["correct_prediction"].sum())
        recall = correct / n
        med_margin = sub["SAM_margin_deg"].median()
    else:
        correct = 0
        recall = np.nan
        med_margin = np.nan

    perf_rows.append({
        "lithology_family": fam,
        "sample_count": n,
        "LOOCV_correct": correct,
        "LOOCV_recall": recall,
        "median_LOOCV_SAM_margin_deg": med_margin
    })

perf = pd.DataFrame(perf_rows)

class_stats = class_stats.merge(
    perf,
    on=["lithology_family", "sample_count"],
    how="left"
)

# ============================================================
# FINAL STEP14B CLASS STRATEGY
# ============================================================

def strategy(row):

    n = int(row["sample_count"])

    sep = str(
        row.get(
            "nearest_class_separability_status",
            ""
        )
    )

    recall = row.get("LOOCV_recall", np.nan)

    if n >= 5 and sep == "GOOD_SEPARATION":
        return "PRIMARY_FULL_LITHOLOGY_CLASS"

    if n >= 3 and sep in [
        "GOOD_SEPARATION",
        "BORDERLINE_SEPARATION"
    ]:
        return "KEEP_SEPARATE_WITH_CAUTION"

    if n >= 2:
        return "KEEP_AS_REFERENCE_HIERARCHICAL_OR_AUGMENT"

    return "REFERENCE_ONLY_AUGMENT_REQUIRED"

class_stats["step14b_recommended_strategy"] = class_stats.apply(
    strategy,
    axis=1
)

# Explicitly preserve all original lithology names
class_stats["automatic_merge_applied"] = "NO"

# ============================================================
# CREATE 3x3-PIXEL-APPROX TRAINING ROI POLYGONS
# ============================================================

with rasterio.open(AVIRIS) as src:
    raster_crs = src.crs
    transform = src.transform

roi_records = []
roi_geoms = []

for _, row in field.iterrows():

    if "field_E" not in row or "field_N" not in row:
        continue

    x = pd.to_numeric(row["field_E"], errors="coerce")
    y = pd.to_numeric(row["field_N"], errors="coerce")

    if not np.isfinite(x) or not np.isfinite(y):
        continue

    geom = box(
        x - ROI_HALF_SIZE_M,
        y - ROI_HALF_SIZE_M,
        x + ROI_HALF_SIZE_M,
        y + ROI_HALF_SIZE_M,
    )

    roi_records.append({
        "field_index": row.get("field_index", np.nan),
        "sample_label": row.get("sample_label_for_step14a",
                                row.get("sample_label", "")),
        "lithology_family": row.get("lithology_family", ""),
        "step14a_training_role": row.get("step14a_training_role", ""),
        "roi_width_m": ROI_HALF_SIZE_M * 2,
        "roi_height_m": ROI_HALF_SIZE_M * 2,
        "nominal_pixel_footprint": "approximately_3x3_pixels",
        "important_note":
            "Training ROI around field point; inspect against AVIRIS and remove mixed/anthropogenic pixels before Step14C."
    })

    roi_geoms.append(geom)

roi_gdf = gpd.GeoDataFrame(
    roi_records,
    geometry=roi_geoms,
    crs=raster_crs
)

# ============================================================
# CLASS MEDIAN SPECTRA OUTPUT
# ============================================================

median_rows = []

for fam in classes:

    rec = {
        "lithology_family": fam,
        "sample_count": len(class_matrices[fam])
    }

    med = class_medians[fam]

    for i, c in enumerate(band_cols):
        rec[c] = med[i]

    median_rows.append(rec)

median_spectra = pd.DataFrame(median_rows)

# ============================================================
# SAVE OUTPUTS
# ============================================================

PAIR_CSV = OUT / \
    "AVIRIS_Step14B_Pairwise_Lithology_Separability.csv"

CLASS_CSV = OUT / \
    "AVIRIS_Step14B_Lithology_Class_QC_and_Strategy.csv"

LOOCV_CSV = OUT / \
    "AVIRIS_Step14B_LOOCV_SAM_Predictions.csv"

CM_CSV = OUT / \
    "AVIRIS_Step14B_LOOCV_Confusion_Matrix.csv"

MEDIAN_CSV = OUT / \
    "AVIRIS_Step14B_Class_Median_Spectra.csv"

GPKG = OUT / \
    "AVIRIS_Step14B_Lithology_Training_ROIs.gpkg"

XLSX = OUT / \
    "AVIRIS_Step14B_Spectral_Separability_and_ROI_Summary.xlsx"

QC_TXT = OUT / \
    "AVIRIS_Step14B_QC_Report.txt"

SETTINGS_JSON = OUT / \
    "AVIRIS_Step14B_Settings.json"

pairwise.to_csv(PAIR_CSV, index=False)
class_stats.to_csv(CLASS_CSV, index=False)
pred.to_csv(LOOCV_CSV, index=False)
cm_df.to_csv(CM_CSV)
median_spectra.to_csv(MEDIAN_CSV, index=False)

if GPKG.exists():
    GPKG.unlink()

roi_gdf.to_file(
    GPKG,
    layer="Step14B_Field_Training_ROIs_3x3",
    driver="GPKG"
)

with pd.ExcelWriter(XLSX, engine="openpyxl") as writer:

    class_stats.to_excel(
        writer,
        sheet_name="Class_QC_Strategy",
        index=False
    )

    pairwise.to_excel(
        writer,
        sheet_name="Pairwise_Separability",
        index=False
    )

    pred.to_excel(
        writer,
        sheet_name="LOOCV_Predictions",
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name="LOOCV_Confusion_Matrix"
    )

# ============================================================
# QC REPORT
# ============================================================

settings = {
    "step": "Step14B_Spectral_Separability_Training_ROI",
    "scientific_good_band_ranges_1based": GOOD_RANGES_1BASED,
    "good_band_count_used": len(band_cols),
    "roi_size_m": 12,
    "roi_nominal_pixels": "3x3",
    "pairwise_SAM_good_deg": PAIR_SAM_GOOD_DEG,
    "pairwise_SAM_borderline_deg": PAIR_SAM_BORDERLINE_DEG,
    "between_within_ratio_good": RATIO_GOOD,
    "between_within_ratio_borderline": RATIO_BORDERLINE,
    "automatic_lithology_merge": False,
    "warning":
        "Step14B assesses separability and prepares training ROIs. "
        "It is not a final lithology classification map and not direct PGE detection."
}

SETTINGS_JSON.write_text(
    json.dumps(settings, indent=2),
    encoding="utf-8"
)

with open(QC_TXT, "w", encoding="utf-8") as f:

    f.write("STEP14B SPECTRAL SEPARABILITY + TRAINING ROI QC\n")
    f.write("=" * 80 + "\n\n")

    f.write(
        "Purpose: assess field-lithology spectral separability and prepare "
        "training ROIs for Step14C full-image lithology classification.\n\n"
    )

    f.write(
        "All original lithology classes are preserved. "
        "No automatic lithology merging was applied.\n\n"
    )

    f.write(
        f"Scientific good bands used: {len(band_cols)}\n"
    )

    f.write(
        f"LOOCV overall accuracy: {accuracy}\n"
    )

    f.write(
        f"LOOCV balanced accuracy: {balanced_acc}\n\n"
    )

    f.write("CLASS QC AND STRATEGY\n")
    f.write("-" * 80 + "\n")
    f.write(class_stats.to_string(index=False))
    f.write("\n\n")

    f.write("LOW-SEPARATION PAIRS\n")
    f.write("-" * 80 + "\n")

    low = pairwise[
        pairwise["separability_status"] ==
        "LOW_SEPARATION_REVIEW_OR_HIERARCHICAL_CLASSIFIER"
    ]

    if low.empty:
        f.write("None\n")
    else:
        f.write(
            low.sort_values(
                "median_to_median_SAM_deg"
            ).to_string(index=False)
        )

    f.write("\n\n")

    f.write(
        "Important scientific note:\n"
        "Small classes with 1-2 field samples cannot be treated as statistically "
        "robust supervised classes. They are retained as lithological reference "
        "classes and should be augmented using additional field/ASD/geology evidence "
        "or handled in a hierarchical classifier.\n"
    )

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP14B COMPLETE")
print("=" * 80)

print("\nScientific good bands used :", len(band_cols))
print("Lithology classes retained :", len(classes))
print("Field training ROIs        :", len(roi_gdf))

print("\nLOOCV overall accuracy      :", round(accuracy, 4)
      if np.isfinite(accuracy) else "NA")

print("LOOCV balanced accuracy     :", round(balanced_acc, 4)
      if np.isfinite(balanced_acc) else "NA")

print("\nCLASS QC / STRATEGY:")

show_cols = [
    "lithology_family",
    "sample_count",
    "median_within_class_SAM_deg",
    "nearest_competing_class",
    "nearest_class_SAM_deg",
    "nearest_class_separability_status",
    "LOOCV_recall",
    "step14b_recommended_strategy"
]

print(
    class_stats[show_cols]
    .sort_values("sample_count", ascending=False)
    .to_string(index=False)
)

print("\nLOW-SEPARATION PAIRS:")

low = pairwise[
    pairwise["separability_status"] ==
    "LOW_SEPARATION_REVIEW_OR_HIERARCHICAL_CLASSIFIER"
].sort_values("median_to_median_SAM_deg")

if low.empty:
    print("None")
else:
    print(
        low[
            [
                "class_A",
                "class_B",
                "median_to_median_SAM_deg",
                "between_to_within_ratio"
            ]
        ].head(20).to_string(index=False)
    )

print("\nArcGIS GeoPackage:")
print(GPKG)

print("\nLayer:")
print("Step14B_Field_Training_ROIs_3x3")

print("\nExcel summary:")
print(XLSX)

print("\nOutput folder:")
print(OUT)

print("\nIMPORTANT:")
print("No lithology class was automatically merged.")
print("All original SAC lithology classes remain available for Step14C.")

print("\nNEXT:")
print(
    "Step14C = hierarchical / multi-class full AVIRIS lithology classification "
    "with vegetation + anthropogenic masking"
)

print("\nPASS")

