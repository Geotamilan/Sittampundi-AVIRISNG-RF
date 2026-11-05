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

from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

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

FIELD_CSV = STEP14A / "AVIRIS_Step14A_Field_Samples_Classified.csv"

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C_Hierarchical_ROI_Training_QC"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SCIENTIFIC BANDS
# ============================================================

GOOD_RANGES_1BASED = [
    (21, 191),
    (220, 283),
    (324, 414),
]

GOOD_BANDS_1BASED = []

for lo, hi in GOOD_RANGES_1BASED:
    GOOD_BANDS_1BASED.extend(range(lo, hi + 1))

GOOD_INDEX_0BASED = np.array(
    [b - 1 for b in GOOD_BANDS_1BASED],
    dtype=int
)

# ============================================================
# ROI / MASK SETTINGS
# ============================================================

ROI_RADIUS_PIXELS = 2      # 5x5 around each field location
MIN_REFLECTANCE = 0.01
MAX_REFLECTANCE = 1.20

NDVI_VEGETATION_THRESHOLD = 0.45

# approximate 1-based bands
BLUE_BAND = 20
GREEN_BAND = 36
RED_BAND = 54
NIR_BAND = 98

# ============================================================
# HIERARCHICAL GROUPS
# ============================================================

BROAD_GROUP_MAP = {

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

    "GRANITE_GRANITIC_ROCK":
        "GRANITIC_FELSIC",

    "PEGMATITE":
        "GRANITIC_FELSIC",

    "MIGMATITE":
        "GRANITIC_FELSIC",

    "HORNBLENDE_BIOTITE_GNEISS":
        "GNEISSIC_METAMORPHIC",

    "GNEISS_GRANITIC_COMPLEX":
        "GNEISSIC_METAMORPHIC",

    # sparse classes retained as reference-only
    "QUARTZITE":
        "REFERENCE_OTHER",

    "DOLERITE":
        "REFERENCE_OTHER",

    "DOLOMITE_CARBONATE":
        "REFERENCE_OTHER",
}

# ============================================================
# HELPERS
# ============================================================

def numeric(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def normalize_spectrum(x):
    x = np.asarray(x, dtype=np.float32)

    good = np.isfinite(x)

    if good.sum() == 0:
        return np.full_like(x, np.nan)

    y = x.copy()

    # continuum-independent shape normalization
    mean = np.nanmean(y[good])
    std = np.nanstd(y[good])

    if not np.isfinite(std) or std < 1e-8:
        return np.full_like(x, np.nan)

    y[good] = (y[good] - mean) / std

    return y


def pixel_valid(spec425):

    if len(spec425) < 425:
        return False, np.nan, "INVALID_BAND_COUNT"

    science = spec425[GOOD_INDEX_0BASED]

    finite = np.isfinite(science)

    if finite.sum() < 300:
        return False, np.nan, "TOO_FEW_VALID_BANDS"

    mean_refl = np.nanmean(science)

    if mean_refl < MIN_REFLECTANCE:
        return False, np.nan, "TOO_DARK"

    if mean_refl > MAX_REFLECTANCE:
        return False, np.nan, "EXTREME_REFLECTANCE"

    red = spec425[RED_BAND - 1]
    nir = spec425[NIR_BAND - 1]

    if (
        np.isfinite(red)
        and np.isfinite(nir)
        and abs(nir + red) > 1e-12
    ):
        ndvi = (nir - red) / (nir + red)
    else:
        ndvi = np.nan

    if np.isfinite(ndvi) and ndvi > NDVI_VEGETATION_THRESHOLD:
        return False, ndvi, "VEGETATION"

    return True, ndvi, "VALID_GEOLOGICAL_CANDIDATE"


# ============================================================
# LOAD FIELD DATA
# ============================================================

print("=" * 80)
print("STEP14C - HIERARCHICAL ROI TRAINING + GROUPED QC")
print("=" * 80)

if not AVIRIS.exists():
    raise FileNotFoundError(AVIRIS)

if not FIELD_CSV.exists():
    raise FileNotFoundError(FIELD_CSV)

field = pd.read_csv(FIELD_CSV)

print("\nField samples:", len(field))

if "lithology_family" not in field.columns:
    raise ValueError("lithology_family column missing.")

if "field_E" not in field.columns or "field_N" not in field.columns:
    raise ValueError("field_E / field_N columns missing.")

field["broad_group"] = field["lithology_family"].map(BROAD_GROUP_MAP)

field["broad_group"] = field["broad_group"].fillna(
    "REFERENCE_OTHER"
)

print("\nField lithology counts:")
print(field["lithology_family"].value_counts().to_string())

print("\nBroad group counts:")
print(field["broad_group"].value_counts().to_string())

# ============================================================
# EXTRACT 5x5 PIXEL TRAINING DATA
# ============================================================

training_rows = []

with rasterio.open(AVIRIS) as src:

    print("\nAVIRIS:")
    print(" size:", src.width, "x", src.height)
    print(" bands:", src.count)
    print(" CRS:", src.crs)

    if src.count != 425:
        raise RuntimeError(
            f"Expected 425 bands, found {src.count}"
        )

    for site_i, row in field.iterrows():

        x = numeric(row["field_E"])
        y = numeric(row["field_N"])

        if not np.isfinite(x) or not np.isfinite(y):
            continue

        rr, cc = src.index(x, y)

        r0 = rr - ROI_RADIUS_PIXELS
        c0 = cc - ROI_RADIUS_PIXELS

        size = ROI_RADIUS_PIXELS * 2 + 1

        cube = src.read(
            window=Window(c0, r0, size, size),
            boundless=True,
            fill_value=-9999
        ).astype(np.float32)

        cube[cube <= -9000] = np.nan

        for dr in range(size):
            for dc in range(size):

                spec425 = cube[:, dr, dc]

                valid, ndvi, mask_status = pixel_valid(spec425)

                if not valid:
                    continue

                sci = spec425[GOOD_INDEX_0BASED]

                norm = normalize_spectrum(sci)

                if np.isfinite(norm).sum() < 300:
                    continue

                rec = {
                    "field_site_id": int(site_i),
                    "sample_label": str(
                        row.get(
                            "sample_label_for_step14a",
                            row.get("sample_label", site_i)
                        )
                    ),
                    "lithology_family": row["lithology_family"],
                    "broad_group": row["broad_group"],
                    "pixel_row": int(r0 + dr),
                    "pixel_col": int(c0 + dc),
                    "ndvi": ndvi,
                    "mask_status": mask_status,
                }

                for j, value in enumerate(norm):
                    rec[f"F{j+1:03d}"] = value

                training_rows.append(rec)

        if (site_i + 1) % 10 == 0:
            print(" field sites processed:", site_i + 1)

train = pd.DataFrame(training_rows)

print("\nExtracted clean ROI pixels:", len(train))

if train.empty:
    raise RuntimeError("No valid training pixels extracted.")

# ============================================================
# PIXEL COUNTS
# ============================================================

pixel_counts = (
    train.groupby(
        ["lithology_family", "broad_group"]
    )
    .agg(
        field_sites=("field_site_id", "nunique"),
        training_pixels=("field_site_id", "size")
    )
    .reset_index()
)

print("\nROI training pixel counts:")
print(pixel_counts.to_string(index=False))

# ============================================================
# FEATURE MATRIX
# ============================================================

feature_cols = [
    c for c in train.columns
    if re.match(r"^F\d{3}$", c)
]

X = train[feature_cols].to_numpy(
    dtype=np.float32
)

# ============================================================
# GROUPED CROSS VALIDATION
#
# Critical:
# Leave one FIELD SITE out.
# Pixels from the held-out site never occur in training.
# ============================================================

usable = train[
    train["broad_group"] != "REFERENCE_OTHER"
].copy()

X_u = usable[feature_cols].to_numpy(
    dtype=np.float32
)

y_u = usable["broad_group"].astype(str).to_numpy()

groups = usable["field_site_id"].to_numpy()

logo = LeaveOneGroupOut()

pred_all = np.empty(
    len(usable),
    dtype=object
)

prob_max = np.full(
    len(usable),
    np.nan,
    dtype=float
)

fold_count = 0

for tr_idx, te_idx in logo.split(
    X_u,
    y_u,
    groups
):

    y_train = y_u[tr_idx]

    # A fold can fail if one class disappears completely.
    if len(np.unique(y_train)) < 2:
        continue

    clf = RandomForestClassifier(
        n_estimators=700,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=20260909,
        n_jobs=-1
    )

    clf.fit(
        X_u[tr_idx],
        y_train
    )

    pred = clf.predict(
        X_u[te_idx]
    )

    pred_all[te_idx] = pred

    probs = clf.predict_proba(
        X_u[te_idx]
    )

    prob_max[te_idx] = np.max(
        probs,
        axis=1
    )

    fold_count += 1

valid_eval = pd.notna(pred_all)

y_true = y_u[valid_eval]
y_pred = pred_all[valid_eval]

acc = accuracy_score(
    y_true,
    y_pred
)

bal_acc = balanced_accuracy_score(
    y_true,
    y_pred
)

labels = sorted(
    np.unique(
        np.concatenate(
            [y_true, y_pred]
        )
    )
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"TRUE_{x}" for x in labels],
    columns=[f"PRED_{x}" for x in labels]
)

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).T.reset_index().rename(
    columns={"index": "class"}
)

usable["grouped_cv_prediction"] = pred_all
usable["grouped_cv_max_probability"] = prob_max
usable["grouped_cv_correct"] = (
    usable["broad_group"] ==
    usable["grouped_cv_prediction"]
)

# ============================================================
# FIELD-SITE LEVEL PERFORMANCE
#
# Majority vote among pixels at each held-out field site
# ============================================================

site_rows = []

for site_id, sub in usable.groupby(
    "field_site_id"
):

    truth = sub["broad_group"].iloc[0]

    preds = sub[
        "grouped_cv_prediction"
    ].dropna()

    if preds.empty:
        majority = "NO_PREDICTION"
        fraction = np.nan
    else:
        vc = preds.value_counts()
        majority = vc.index[0]
        fraction = vc.iloc[0] / len(preds)

    site_rows.append({
        "field_site_id": site_id,
        "sample_label": sub["sample_label"].iloc[0],
        "lithology_family": sub["lithology_family"].iloc[0],
        "true_broad_group": truth,
        "predicted_broad_group": majority,
        "majority_vote_fraction": fraction,
        "site_correct": majority == truth,
        "valid_training_pixels": len(sub)
    })

site_qc = pd.DataFrame(site_rows)

site_acc = accuracy_score(
    site_qc["true_broad_group"],
    site_qc["predicted_broad_group"]
)

site_bal_acc = balanced_accuracy_score(
    site_qc["true_broad_group"],
    site_qc["predicted_broad_group"]
)

# ============================================================
# BROAD GROUP READINESS
# ============================================================

group_summary = (
    usable.groupby("broad_group")
    .agg(
        field_sites=("field_site_id", "nunique"),
        training_pixels=("field_site_id", "size"),
        pixel_cv_accuracy=("grouped_cv_correct", "mean")
    )
    .reset_index()
)

site_group_acc = (
    site_qc.groupby("true_broad_group")
    .agg(
        field_sites_tested=("field_site_id", "count"),
        field_site_accuracy=("site_correct", "mean")
    )
    .reset_index()
    .rename(
        columns={
            "true_broad_group":
                "broad_group"
        }
    )
)

group_summary = group_summary.merge(
    site_group_acc,
    on="broad_group",
    how="left"
)

def readiness(row):

    n = row["field_sites"]
    a = row["field_site_accuracy"]

    if n >= 5 and a >= 0.70:
        return "READY_FOR_FULL_IMAGE_BROAD_GROUP_MAPPING"

    if n >= 3 and a >= 0.50:
        return "MAPPING_WITH_CAUTION"

    return "REFERENCE_ONLY_OR_NEEDS_MORE_TRAINING"


group_summary[
    "step14c_mapping_readiness"
] = group_summary.apply(
    readiness,
    axis=1
)

# ============================================================
# SAVE
# ============================================================

TRAIN_CSV = OUT / (
    "AVIRIS_Step14C_ROI_Pixel_Training_Data.csv"
)

PIXEL_COUNTS_CSV = OUT / (
    "AVIRIS_Step14C_ROI_Pixel_Counts.csv"
)

PIXEL_CV_CSV = OUT / (
    "AVIRIS_Step14C_GroupedCV_Pixel_Predictions.csv"
)

SITE_QC_CSV = OUT / (
    "AVIRIS_Step14C_FieldSite_GroupedCV_QC.csv"
)

GROUP_SUMMARY_CSV = OUT / (
    "AVIRIS_Step14C_Broad_Group_Readiness.csv"
)

CM_CSV = OUT / (
    "AVIRIS_Step14C_GroupedCV_Confusion_Matrix.csv"
)

REPORT_CSV = OUT / (
    "AVIRIS_Step14C_GroupedCV_Classification_Report.csv"
)

XLSX = OUT / (
    "AVIRIS_Step14C_Hierarchical_Training_QC.xlsx"
)

SETTINGS = OUT / (
    "AVIRIS_Step14C_Settings.json"
)

QC_TXT = OUT / (
    "AVIRIS_Step14C_QC_Report.txt"
)

train.to_csv(
    TRAIN_CSV,
    index=False
)

pixel_counts.to_csv(
    PIXEL_COUNTS_CSV,
    index=False
)

usable.to_csv(
    PIXEL_CV_CSV,
    index=False
)

site_qc.to_csv(
    SITE_QC_CSV,
    index=False
)

group_summary.to_csv(
    GROUP_SUMMARY_CSV,
    index=False
)

cm_df.to_csv(
    CM_CSV
)

report_df.to_csv(
    REPORT_CSV,
    index=False
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    pixel_counts.to_excel(
        writer,
        sheet_name="ROI_Pixel_Counts",
        index=False
    )

    group_summary.to_excel(
        writer,
        sheet_name="Broad_Group_Readiness",
        index=False
    )

    site_qc.to_excel(
        writer,
        sheet_name="Field_Site_CV",
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name="Confusion_Matrix"
    )

    report_df.to_excel(
        writer,
        sheet_name="Classification_Report",
        index=False
    )

settings = {
    "step":
        "Step14C_Hierarchical_ROI_Training_QC",

    "roi_pixels":
        "5x5 around each field point",

    "field_site_grouped_cv":
        True,

    "scientific_good_bands":
        len(GOOD_BANDS_1BASED),

    "vegetation_ndvi_threshold":
        NDVI_VEGETATION_THRESHOLD,

    "broad_group_map":
        BROAD_GROUP_MAP,

    "important_note":
        "No final lithology map is produced in Step14C. "
        "This is a reliability gate before full-image mapping."
}

SETTINGS.write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

with open(
    QC_TXT,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP14C HIERARCHICAL ROI TRAINING QC\n"
    )
    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Clean ROI pixels: {len(train)}\n"
    )

    f.write(
        f"Grouped CV folds: {fold_count}\n"
    )

    f.write(
        f"Pixel-level overall accuracy: {acc:.6f}\n"
    )

    f.write(
        f"Pixel-level balanced accuracy: {bal_acc:.6f}\n"
    )

    f.write(
        f"Field-site overall accuracy: {site_acc:.6f}\n"
    )

    f.write(
        f"Field-site balanced accuracy: {site_bal_acc:.6f}\n\n"
    )

    f.write(
        "Broad group readiness:\n"
    )

    f.write(
        group_summary.to_string(
            index=False
        )
    )

    f.write(
        "\n\nIMPORTANT:\n"
        "Field-site grouped validation is used so neighboring pixels "
        "from the same field location do not leak into training/testing.\n"
    )

    f.write(
        "This step does not prove chromitite, chromite or PGE occurrence.\n"
    )

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("STEP14C COMPLETE")
print("=" * 80)

print("\nClean ROI pixels              :", len(train))
print("Field sites                   :", field["field_E"].notna().sum())
print("Grouped CV folds              :", fold_count)

print("\nPIXEL-LEVEL GROUPED CV")
print("Overall accuracy              :", round(acc, 4))
print("Balanced accuracy             :", round(bal_acc, 4))

print("\nFIELD-SITE LEVEL GROUPED CV")
print("Overall accuracy              :", round(site_acc, 4))
print("Balanced accuracy             :", round(site_bal_acc, 4))

print("\nBROAD GROUP READINESS:")
print(
    group_summary.to_string(
        index=False
    )
)

print("\nFIELD-SITE CONFUSIONS:")
bad = site_qc[
    ~site_qc["site_correct"]
]

if bad.empty:
    print("None")
else:
    print(
        bad[
            [
                "sample_label",
                "lithology_family",
                "true_broad_group",
                "predicted_broad_group",
                "majority_vote_fraction"
            ]
        ].to_string(
            index=False
        )
    )

print("\nOutput folder:")
print(OUT)

print("\nExcel:")
print(XLSX)

print("\nIMPORTANT:")
print(
    "13 original lithology classes are preserved as reference classes."
)

print(
    "Only broad groups with adequate field-site validation will be "
    "allowed into Step14D full-image mapping."
)

print("\nNEXT:")
print(
    "Step14D = full AVIRIS broad-lithology probability mapping + "
    "13-class reference-guided subtype assignment."
)

print("\nPASS")
