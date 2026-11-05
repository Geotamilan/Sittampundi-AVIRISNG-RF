# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP14A = ROOT / (
    "09_Mineral_Mapping/"
    "Step14A_Lithology_Training_Data_Preparation"
)

STEP14C = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C_Hierarchical_ROI_Training_QC"
)

STEP14C2A = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C2A_ASD_WhiteReference_Audit"
)

AVIRIS_HDR = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.hdr"
)

FIELD_SPECTRA = STEP14A / (
    "AVIRIS_Step14A_Field_Spectra_3x3Median_Wide.csv"
)

ROI_PIXELS = STEP14C / (
    "AVIRIS_Step14C_ROI_Pixel_Training_Data.csv"
)

CLEAN_AUDIT = STEP14C2A / (
    "AVIRIS_Step14C2A_Clean_NonWR_ASD_Candidates.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C2_v2_Clean_RAW_ASD_Guided_Revalidation"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SCIENTIFIC AVIRIS BANDS
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
# LITHOLOGY DEFINITIONS
# ============================================================

ALL_LITHOLOGIES = [
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

ASD_EXPECTED_FAMILIES = [
    "CHROMITITE",
    "ANORTHOSITE_METAANORTHOSITE",
    "GABBRO_METAGABBRO",
    "PYROXENITE_MAFIC_CUMULATE",
    "HORNBLENDE_BIOTITE_GNEISS",
    "GRANITE_GRANITIC_ROCK",
]

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

def parse_aviris_wavelengths(hdr_path):

    text = hdr_path.read_text(errors="ignore")

    m = re.search(
        r"wavelength\s*=\s*\{(.*?)\}",
        text,
        flags=re.I | re.S
    )

    if not m:
        raise RuntimeError(
            "Could not parse wavelengths from AVIRIS HDR."
        )

    values = []

    for item in m.group(1).replace("\n", " ").split(","):

        try:
            values.append(float(item.strip()))
        except Exception:
            pass

    return np.asarray(values, dtype=float)


def parse_raw_asd_txt(path):

    path = Path(path)

    if not path.exists():
        return None

    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return None

    wl = []
    refl = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = re.split(r"[\t, ]+", line)

        nums = []

        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass

        if len(nums) < 2:
            continue

        wavelength = nums[0]
        value = nums[1]

        if not (300 <= wavelength <= 2600):
            continue

        if not np.isfinite(value):
            continue

        wl.append(wavelength)
        refl.append(value)

    if len(wl) < 1000:
        return None

    wl = np.asarray(wl, dtype=float)
    refl = np.asarray(refl, dtype=float)

    order = np.argsort(wl)

    wl = wl[order]
    refl = refl[order]

    tmp = pd.DataFrame({
        "wl": wl,
        "refl": refl
    })

    tmp = (
        tmp.groupby("wl", as_index=False)
        ["refl"]
        .median()
    )

    wl = tmp["wl"].to_numpy(dtype=float)
    refl = tmp["refl"].to_numpy(dtype=float)

    # Handle percent reflectance only if necessary
    med = np.nanmedian(np.abs(refl))

    if 2.0 < med <= 100.0:
        refl = refl / 100.0

    return wl, refl


def snv(x):

    x = np.asarray(x, dtype=float)

    result = np.full_like(
        x,
        np.nan,
        dtype=float
    )

    good = np.isfinite(x)

    if good.sum() < 100:
        return result

    mean = np.nanmean(x[good])
    std = np.nanstd(x[good])

    if not np.isfinite(std) or std < 1e-10:
        return result

    result[good] = (
        x[good] - mean
    ) / std

    return result


def cosine_similarity(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ok = (
        np.isfinite(a) &
        np.isfinite(b)
    )

    if ok.sum() < 250:
        return np.nan

    aa = a[ok]
    bb = b[ok]

    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)

    if na <= 0 or nb <= 0:
        return np.nan

    return float(
        np.dot(aa, bb) /
        (na * nb)
    )


def spectral_angle_deg(a, b):

    c = cosine_similarity(a, b)

    if not np.isfinite(c):
        return np.nan

    c = np.clip(c, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(c))
    )


def grouped_rf_cv(X, y, groups, seed):

    logo = LeaveOneGroupOut()

    predictions = np.full(
        len(y),
        None,
        dtype=object
    )

    probabilities = np.full(
        len(y),
        np.nan,
        dtype=float
    )

    fold_count = 0

    for train_idx, test_idx in logo.split(
        X,
        y,
        groups
    ):

        y_train = y[train_idx]

        if len(np.unique(y_train)) < 2:
            continue

        model = RandomForestClassifier(
            n_estimators=900,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1
        )

        model.fit(
            X[train_idx],
            y_train
        )

        predictions[test_idx] = model.predict(
            X[test_idx]
        )

        p = model.predict_proba(
            X[test_idx]
        )

        probabilities[test_idx] = np.max(
            p,
            axis=1
        )

        fold_count += 1

    valid = pd.notna(predictions)

    yt = y[valid]
    yp = predictions[valid]

    acc = accuracy_score(
        yt,
        yp
    )

    bal = balanced_accuracy_score(
        yt,
        yp
    )

    labels = sorted(
        set(yt) | set(yp)
    )

    cm = confusion_matrix(
        yt,
        yp,
        labels=labels
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "TRUE_" + str(x)
            for x in labels
        ],
        columns=[
            "PRED_" + str(x)
            for x in labels
        ]
    )

    return {
        "prediction": predictions,
        "probability": probabilities,
        "accuracy": acc,
        "balanced_accuracy": bal,
        "folds": fold_count,
        "confusion_matrix": cm_df
    }


def site_majority_vote(
    table,
    predictions,
    truth_column
):

    work = table.copy()

    work["cv_prediction"] = predictions

    rows = []

    for site_id, sub in work.groupby(
        "field_site_id"
    ):

        truth = str(
            sub[truth_column].iloc[0]
        )

        p = sub[
            "cv_prediction"
        ].dropna()

        if p.empty:

            pred = "NO_PREDICTION"
            vote_fraction = np.nan

        else:

            counts = p.value_counts()

            pred = str(counts.index[0])

            vote_fraction = float(
                counts.iloc[0] / len(p)
            )

        rows.append({
            "field_site_id": site_id,
            "sample_label":
                sub["sample_label"].iloc[0],
            "lithology_family":
                sub["lithology_family"].iloc[0],
            "true_class":
                truth,
            "predicted_class":
                pred,
            "majority_vote_fraction":
                vote_fraction,
            "correct":
                pred == truth
        })

    result = pd.DataFrame(rows)

    acc = accuracy_score(
        result["true_class"],
        result["predicted_class"]
    )

    bal = balanced_accuracy_score(
        result["true_class"],
        result["predicted_class"]
    )

    return result, acc, bal


# ============================================================
# START
# ============================================================

print("=" * 90)
print("STEP14C2-v2 - CLEAN RAW ASD-GUIDED REVALIDATION")
print("=" * 90)

for path in [
    AVIRIS_HDR,
    FIELD_SPECTRA,
    ROI_PIXELS,
    CLEAN_AUDIT
]:
    if not path.exists():
        raise FileNotFoundError(path)

# ============================================================
# AVIRIS WAVELENGTHS
# ============================================================

wl425 = parse_aviris_wavelengths(
    AVIRIS_HDR
)

if len(wl425) != 425:
    raise RuntimeError(
        f"Expected 425 wavelengths; found {len(wl425)}"
    )

wl326 = wl425[GOOD_INDEX]

print("\nScientific AVIRIS bands:", len(wl326))

# ============================================================
# LOAD CLEAN ASD AUDIT
# ============================================================

audit = pd.read_csv(
    CLEAN_AUDIT
)

print("\nClean non-WR ASD candidates:", len(audit))

# Primary raw ASD files only.
# Transformed CR/DV ENVI products are NOT used as raw cross-domain references.

audit["primary_raw_asd"] = (
    audit["local_file"]
    .astype(str)
    .str.lower()
    .str.endswith(".asd.sco.txt")
)

raw_audit = audit[
    audit["primary_raw_asd"]
].copy()

excluded_products = audit[
    ~audit["primary_raw_asd"]
].copy()

print("Primary raw .asd.sco.txt files:", len(raw_audit))
print("Excluded transformed/non-primary products:", len(excluded_products))

if not excluded_products.empty:

    print("\nExcluded non-primary files:")

    print(
        excluded_products[
            [
                "local_family_v2",
                "local_file"
            ]
        ].to_string(index=False)
    )

# ============================================================
# PARSE CLEAN RAW ASD SPECTRA
# ============================================================

asd_rows = []
asd_matrix = []

for _, row in raw_audit.iterrows():

    path = Path(row["local_path"])

    parsed = parse_raw_asd_txt(path)

    if parsed is None:
        continue

    source_wl, source_refl = parsed

    interpolated = np.full(
        len(wl326),
        np.nan,
        dtype=float
    )

    valid = (
        (wl326 >= np.nanmin(source_wl)) &
        (wl326 <= np.nanmax(source_wl))
    )

    interpolated[valid] = np.interp(
        wl326[valid],
        source_wl,
        source_refl
    )

    normalized = snv(
        interpolated
    )

    if np.isfinite(normalized).sum() < 300:
        continue

    asd_rows.append({
        "asd_id":
            len(asd_rows),

        "local_file":
            row["local_file"],

        "local_path":
            row["local_path"],

        "lithology_family":
            row["local_family_v2"],

        "sample_preparation":
            row["sample_preparation_v2"],

        "source_points":
            len(source_wl),

        "source_wavelength_min_nm":
            float(np.nanmin(source_wl)),

        "source_wavelength_max_nm":
            float(np.nanmax(source_wl)),

        "valid_AVIRIS_bands":
            int(np.isfinite(normalized).sum())
    })

    asd_matrix.append(
        normalized
    )

asd_index = pd.DataFrame(
    asd_rows
)

if len(asd_index) == 0:
    raise RuntimeError(
        "No clean raw ASD spectra parsed."
    )

ASD = np.vstack(
    asd_matrix
)

print("\nUsable CLEAN RAW ASD spectra:", len(asd_index))

print("\nClean RAW ASD family counts:")
print(
    asd_index[
        "lithology_family"
    ]
    .value_counts()
    .to_string()
)

# ============================================================
# ASD FAMILY MEDIANS
# ============================================================

asd_family_medians = {}

asd_median_rows = []

for family in ASD_EXPECTED_FAMILIES:

    idx = np.where(
        asd_index[
            "lithology_family"
        ].to_numpy() == family
    )[0]

    if len(idx) == 0:
        continue

    family_spectra = ASD[idx, :]

    med = np.nanmedian(
        family_spectra,
        axis=0
    )

    med = snv(med)

    asd_family_medians[
        family
    ] = med

    rec = {
        "lithology_family":
            family,

        "asd_raw_spectrum_count":
            len(idx)
    }

    for i, value in enumerate(med):
        rec[f"F{i+1:03d}"] = value

    asd_median_rows.append(rec)

asd_medians_df = pd.DataFrame(
    asd_median_rows
)

# ============================================================
# LOAD FIELD AVIRIS SPECTRA
# ============================================================

field_spec = pd.read_csv(
    FIELD_SPECTRA
)

good_cols = [
    f"B{b:03d}"
    for b in GOOD_BANDS
]

field_family_medians = {}

for family in ASD_EXPECTED_FAMILIES:

    sub = field_spec[
        field_spec[
            "lithology_family"
        ] == family
    ].copy()

    if sub.empty:
        continue

    matrix = (
        sub[good_cols]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .to_numpy(dtype=float)
    )

    # Normalize each airborne sample separately
    normalized_samples = []

    for row in matrix:

        n = snv(row)

        if np.isfinite(n).sum() >= 300:
            normalized_samples.append(n)

    if len(normalized_samples) == 0:
        continue

    med = np.nanmedian(
        np.vstack(normalized_samples),
        axis=0
    )

    field_family_medians[
        family
    ] = snv(med)

# ============================================================
# ASD ↔ FIELD AVIRIS ALIGNMENT
# ============================================================

alignment_rows = []

for family in ASD_EXPECTED_FAMILIES:

    asd_ref = asd_family_medians.get(
        family
    )

    airborne_ref = field_family_medians.get(
        family
    )

    n_asd = int(
        (
            asd_index[
                "lithology_family"
            ] == family
        ).sum()
    )

    if asd_ref is None or airborne_ref is None:

        alignment_rows.append({
            "lithology_family":
                family,

            "clean_raw_asd_count":
                n_asd,

            "shape_correlation":
                np.nan,

            "shape_angle_deg":
                np.nan,

            "guidance_status":
                "NO_COMMON_REFERENCE"
        })

        continue

    corr = cosine_similarity(
        asd_ref,
        airborne_ref
    )

    angle = spectral_angle_deg(
        asd_ref,
        airborne_ref
    )

    if corr >= 0.70:
        status = "STRONG_ALIGNMENT"

    elif corr >= 0.45:
        status = "MODERATE_ALIGNMENT"

    elif corr >= 0.20:
        status = "WEAK_ALIGNMENT_USE_WITH_CAUTION"

    else:
        status = "POOR_ALIGNMENT_EXCLUDE_FROM_GUIDANCE"

    alignment_rows.append({
        "lithology_family":
            family,

        "clean_raw_asd_count":
            n_asd,

        "shape_correlation":
            corr,

        "shape_angle_deg":
            angle,

        "guidance_status":
            status
    })

alignment = pd.DataFrame(
    alignment_rows
)

print("\nASD ↔ FIELD AVIRIS ALIGNMENT:")
print(
    alignment.to_string(index=False)
)

# ============================================================
# ALLOWED ASD GUIDANCE FAMILIES
# ============================================================

usable_guidance = []

for _, row in alignment.iterrows():

    family = row[
        "lithology_family"
    ]

    n = int(
        row["clean_raw_asd_count"]
    )

    status = row[
        "guidance_status"
    ]

    if (
        n >= 2 and
        status in [
            "STRONG_ALIGNMENT",
            "MODERATE_ALIGNMENT",
            "WEAK_ALIGNMENT_USE_WITH_CAUTION"
        ]
    ):
        usable_guidance.append(
            family
        )

print("\nASD families admitted as guidance:")
print(usable_guidance)

# ============================================================
# LOAD STEP14C ROI PIXELS
# ============================================================

roi = pd.read_csv(
    ROI_PIXELS
)

feature_cols = [
    c for c in roi.columns
    if re.match(r"^F\d{3}$", c)
]

if len(feature_cols) != 326:
    raise RuntimeError(
        f"Expected 326 ROI spectral features; found {len(feature_cols)}"
    )

X_base = (
    roi[feature_cols]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(dtype=np.float32)
)

# Remove rows with missing spectral features
valid_rows = np.isfinite(
    X_base
).all(axis=1)

print("\nROI pixels before finite-data QC:", len(roi))
print("ROI pixels retained:", int(valid_rows.sum()))
print("ROI pixels removed :", int((~valid_rows).sum()))

roi = roi.loc[
    valid_rows
].reset_index(drop=True)

X_base = X_base[
    valid_rows
]

# ============================================================
# ADD ASD SIMILARITY FEATURES
# ============================================================

asd_feature_arrays = []
asd_feature_names = []

for family in usable_guidance:

    reference = asd_family_medians[
        family
    ]

    similarities = []

    for spectrum in X_base:

        similarities.append(
            cosine_similarity(
                spectrum,
                reference
            )
        )

    similarities = np.asarray(
        similarities,
        dtype=np.float32
    )

    # Replace rare invalid values conservatively
    similarities[
        ~np.isfinite(similarities)
    ] = 0.0

    name = (
        "ASD_SIM_" +
        family
    )

    roi[name] = similarities

    asd_feature_arrays.append(
        similarities
    )

    asd_feature_names.append(
        name
    )

if asd_feature_arrays:

    ASD_GUIDE = np.column_stack(
        asd_feature_arrays
    )

    X_guided = np.hstack(
        [
            X_base,
            ASD_GUIDE
        ]
    )

else:

    X_guided = X_base.copy()

print("\nBaseline spectral features:", X_base.shape[1])
print("ASD guidance features     :", len(asd_feature_names))
print("Guided feature count      :", X_guided.shape[1])

# ============================================================
# BROAD GROUP VALIDATION
# ============================================================

broad_mask = (
    roi[
        "broad_group"
    ] != "REFERENCE_OTHER"
)

broad = roi.loc[
    broad_mask
].reset_index(drop=True)

broad_indices = np.where(
    broad_mask.to_numpy()
)[0]

Xb_base = X_base[
    broad_indices
]

Xb_guided = X_guided[
    broad_indices
]

yb = (
    broad[
        "broad_group"
    ]
    .astype(str)
    .to_numpy()
)

gb = broad[
    "field_site_id"
].to_numpy()

baseline_broad = grouped_rf_cv(
    Xb_base,
    yb,
    gb,
    20260920
)

guided_broad = grouped_rf_cv(
    Xb_guided,
    yb,
    gb,
    20260921
)

site_base_broad, site_base_acc, site_base_bal = (
    site_majority_vote(
        broad,
        baseline_broad[
            "prediction"
        ],
        "broad_group"
    )
)

site_guided_broad, site_guided_acc, site_guided_bal = (
    site_majority_vote(
        broad,
        guided_broad[
            "prediction"
        ],
        "broad_group"
    )
)

# ============================================================
# PER-BROAD-GROUP PERFORMANCE
# ============================================================

broad_performance_rows = []

for group in sorted(
    broad["broad_group"].unique()
):

    base_sub = site_base_broad[
        site_base_broad[
            "true_class"
        ] == group
    ]

    guide_sub = site_guided_broad[
        site_guided_broad[
            "true_class"
        ] == group
    ]

    broad_performance_rows.append({
        "broad_group":
            group,

        "field_sites":
            len(guide_sub),

        "baseline_site_accuracy":
            float(
                base_sub["correct"].mean()
            ) if len(base_sub) else np.nan,

        "ASD_guided_site_accuracy":
            float(
                guide_sub["correct"].mean()
            ) if len(guide_sub) else np.nan,
    })

broad_performance = pd.DataFrame(
    broad_performance_rows
)

broad_performance[
    "accuracy_change"
] = (
    broad_performance[
        "ASD_guided_site_accuracy"
    ] -
    broad_performance[
        "baseline_site_accuracy"
    ]
)

# ============================================================
# INDIVIDUAL LITHOLOGY VALIDATION
# ============================================================

site_counts = (
    roi.groupby(
        "lithology_family"
    )["field_site_id"]
    .nunique()
)

eligible_classes = site_counts[
    site_counts >= 2
].index.tolist()

ind_mask = roi[
    "lithology_family"
].isin(
    eligible_classes
)

individual = roi.loc[
    ind_mask
].reset_index(drop=True)

ind_indices = np.where(
    ind_mask.to_numpy()
)[0]

Xi_base = X_base[
    ind_indices
]

Xi_guided = X_guided[
    ind_indices
]

yi = (
    individual[
        "lithology_family"
    ]
    .astype(str)
    .to_numpy()
)

gi = individual[
    "field_site_id"
].to_numpy()

baseline_ind = grouped_rf_cv(
    Xi_base,
    yi,
    gi,
    20260922
)

guided_ind = grouped_rf_cv(
    Xi_guided,
    yi,
    gi,
    20260923
)

site_base_ind, site_base_ind_acc, site_base_ind_bal = (
    site_majority_vote(
        individual,
        baseline_ind[
            "prediction"
        ],
        "lithology_family"
    )
)

site_guided_ind, site_guided_ind_acc, site_guided_ind_bal = (
    site_majority_vote(
        individual,
        guided_ind[
            "prediction"
        ],
        "lithology_family"
    )
)

# ============================================================
# INDIVIDUAL LITHOLOGY READINESS
# ============================================================

readiness_rows = []

for family in ALL_LITHOLOGIES:

    n_sites = int(
        site_counts.get(
            family,
            0
        )
    )

    asd_count = int(
        (
            asd_index[
                "lithology_family"
            ] == family
        ).sum()
    )

    al = alignment[
        alignment[
            "lithology_family"
        ] == family
    ]

    if len(al):
        alignment_status = (
            al.iloc[0][
                "guidance_status"
            ]
        )
    else:
        alignment_status = (
            "NO_LOCAL_ASD_REFERENCE"
        )

    guided_sub = site_guided_ind[
        site_guided_ind[
            "lithology_family"
        ] == family
    ]

    if len(guided_sub):

        guided_accuracy = float(
            guided_sub[
                "correct"
            ].mean()
        )

    else:

        guided_accuracy = np.nan

    if (
        n_sites >= 5 and
        np.isfinite(guided_accuracy) and
        guided_accuracy >= 0.70
    ):
        readiness_status = (
            "READY_FOR_REFERENCE_GUIDED_SUBTYPE_PROBABILITY"
        )

    elif (
        n_sites >= 3 and
        np.isfinite(guided_accuracy) and
        guided_accuracy >= 0.50
    ):
        readiness_status = (
            "SUBTYPE_PROBABILITY_WITH_CAUTION"
        )

    elif n_sites >= 2:
        readiness_status = (
            "REFERENCE_ONLY_NOT_READY_FOR_HARD_CLASS"
        )

    else:
        readiness_status = (
            "REFERENCE_ONLY_SINGLE_OR_NO_VALID_SITE"
        )

    readiness_rows.append({
        "lithology_family":
            family,

        "field_sites":
            n_sites,

        "clean_raw_ASD_spectra":
            asd_count,

        "ASD_alignment_status":
            alignment_status,

        "ASD_guided_field_site_accuracy":
            guided_accuracy,

        "step14c2_v2_readiness":
            readiness_status
    })

readiness = pd.DataFrame(
    readiness_rows
)

# ============================================================
# MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame([
    {
        "model":
            "BROAD_BASELINE",

        "pixel_accuracy":
            baseline_broad[
                "accuracy"
            ],

        "pixel_balanced_accuracy":
            baseline_broad[
                "balanced_accuracy"
            ],

        "field_site_accuracy":
            site_base_acc,

        "field_site_balanced_accuracy":
            site_base_bal
    },

    {
        "model":
            "BROAD_CLEAN_RAW_ASD_GUIDED",

        "pixel_accuracy":
            guided_broad[
                "accuracy"
            ],

        "pixel_balanced_accuracy":
            guided_broad[
                "balanced_accuracy"
            ],

        "field_site_accuracy":
            site_guided_acc,

        "field_site_balanced_accuracy":
            site_guided_bal
    },

    {
        "model":
            "INDIVIDUAL_BASELINE",

        "pixel_accuracy":
            baseline_ind[
                "accuracy"
            ],

        "pixel_balanced_accuracy":
            baseline_ind[
                "balanced_accuracy"
            ],

        "field_site_accuracy":
            site_base_ind_acc,

        "field_site_balanced_accuracy":
            site_base_ind_bal
    },

    {
        "model":
            "INDIVIDUAL_CLEAN_RAW_ASD_GUIDED",

        "pixel_accuracy":
            guided_ind[
                "accuracy"
            ],

        "pixel_balanced_accuracy":
            guided_ind[
                "balanced_accuracy"
            ],

        "field_site_accuracy":
            site_guided_ind_acc,

        "field_site_balanced_accuracy":
            site_guided_ind_bal
    }
])

# ============================================================
# SCIENTIFIC GATE
# ============================================================

supported_group_acc = (
    broad_performance[
        "ASD_guided_site_accuracy"
    ]
    .dropna()
)

minimum_group_accuracy = (
    float(
        supported_group_acc.min()
    )
    if len(supported_group_acc)
    else np.nan
)

if (
    site_guided_bal >= 0.70 and
    minimum_group_accuracy >= 0.50
):

    final_gate = (
        "PASS_STEP14D_BROAD_PROBABILITY_MAPPING"
    )

elif site_guided_bal >= 0.55:

    final_gate = (
        "CAUTION_STEP14D_PROBABILITY_AND_UNCERTAINTY_ONLY"
    )

else:

    final_gate = (
        "STOP_IMPROVE_TRAINING_BEFORE_FULL_IMAGE_MAPPING"
    )

# ============================================================
# SAVE OUTPUTS
# ============================================================

ASD_INDEX_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Clean_RAW_ASD_Index.csv"
)

ASD_MEDIAN_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Clean_RAW_ASD_Family_Medians.csv"
)

ALIGNMENT_CSV = OUT / (
    "AVIRIS_Step14C2_v2_ASD_AVIRIS_Alignment.csv"
)

COMPARISON_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Model_Comparison.csv"
)

BROAD_PERFORMANCE_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Broad_Group_Performance.csv"
)

READINESS_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Lithology_Readiness.csv"
)

BROAD_SITE_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Broad_FieldSite_QC.csv"
)

IND_SITE_CSV = OUT / (
    "AVIRIS_Step14C2_v2_Individual_FieldSite_QC.csv"
)

XLSX = OUT / (
    "AVIRIS_Step14C2_v2_Clean_RAW_ASD_Revalidation.xlsx"
)

QC_TXT = OUT / (
    "AVIRIS_Step14C2_v2_QC_Report.txt"
)

SETTINGS_JSON = OUT / (
    "AVIRIS_Step14C2_v2_Settings.json"
)

asd_index.to_csv(
    ASD_INDEX_CSV,
    index=False
)

asd_medians_df.to_csv(
    ASD_MEDIAN_CSV,
    index=False
)

alignment.to_csv(
    ALIGNMENT_CSV,
    index=False
)

comparison.to_csv(
    COMPARISON_CSV,
    index=False
)

broad_performance.to_csv(
    BROAD_PERFORMANCE_CSV,
    index=False
)

readiness.to_csv(
    READINESS_CSV,
    index=False
)

site_guided_broad.to_csv(
    BROAD_SITE_CSV,
    index=False
)

site_guided_ind.to_csv(
    IND_SITE_CSV,
    index=False
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    comparison.to_excel(
        writer,
        sheet_name="Model_Comparison",
        index=False
    )

    broad_performance.to_excel(
        writer,
        sheet_name="Broad_Group_Performance",
        index=False
    )

    readiness.to_excel(
        writer,
        sheet_name="Lithology_Readiness",
        index=False
    )

    alignment.to_excel(
        writer,
        sheet_name="ASD_AVIRIS_Alignment",
        index=False
    )

    asd_index.to_excel(
        writer,
        sheet_name="Clean_RAW_ASD_Index",
        index=False
    )

    site_guided_broad.to_excel(
        writer,
        sheet_name="Broad_FieldSite_QC",
        index=False
    )

    site_guided_ind.to_excel(
        writer,
        sheet_name="Individual_FieldSite_QC",
        index=False
    )

settings = {
    "step":
        "Step14C2_v2_Clean_RAW_ASD_Guided_Revalidation",

    "white_reference_files_used":
        False,

    "transformed_CR_DV_products_used_as_raw_training":
        False,

    "primary_ASD_definition":
        "*.asd.sco.txt clean non-WR files only",

    "clean_raw_ASD_spectra":
        int(len(asd_index)),

    "ASD_guidance_families":
        usable_guidance,

    "AVIRIS_scientific_bands":
        326,

    "field_site_grouped_cross_validation":
        True,

    "final_scientific_gate":
        final_gate,

    "warning":
        (
            "ASD spectra are independent spectral-reference guidance only. "
            "They are not treated as interchangeable airborne AVIRIS pixels. "
            "No direct PGE detection is claimed."
        )
}

SETTINGS_JSON.write_text(
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
        "STEP14C2-v2 CLEAN RAW ASD-GUIDED REVALIDATION\n"
    )

    f.write(
        "=" * 90 + "\n\n"
    )

    f.write(
        f"Clean raw ASD spectra: {len(asd_index)}\n"
    )

    f.write(
        f"ASD guidance families: {len(usable_guidance)}\n"
    )

    f.write(
        "White-reference spectra used: NO\n"
    )

    f.write(
        "CR/DV transformed products used as raw training: NO\n\n"
    )

    f.write("MODEL COMPARISON\n")
    f.write(
        comparison.to_string(index=False)
    )

    f.write("\n\nBROAD GROUP PERFORMANCE\n")
    f.write(
        broad_performance.to_string(index=False)
    )

    f.write("\n\nASD / AVIRIS ALIGNMENT\n")
    f.write(
        alignment.to_string(index=False)
    )

    f.write("\n\nLITHOLOGY READINESS\n")
    f.write(
        readiness.to_string(index=False)
    )

    f.write("\n\nFINAL SCIENTIFIC GATE\n")
    f.write(final_gate + "\n")

    f.write(
        "\nNo direct chromite or PGE detection is claimed.\n"
    )

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 90)
print("STEP14C2-v2 COMPLETE")
print("=" * 90)

print("\nClean non-WR candidates        :", len(audit))
print("Primary raw ASD files          :", len(raw_audit))
print("Usable clean RAW ASD spectra   :", len(asd_index))
print("ASD families represented       :", asd_index["lithology_family"].nunique())
print("ASD guidance families admitted :", len(usable_guidance))

print("\nMODEL COMPARISON:")
print(
    comparison.to_string(index=False)
)

print("\nBROAD GROUP PERFORMANCE:")
print(
    broad_performance.to_string(index=False)
)

print("\nLITHOLOGY READINESS:")
print(
    readiness.to_string(index=False)
)

print("\nFINAL SCIENTIFIC GATE:")
print(final_gate)

print("\nOutput folder:")
print(OUT)

print("\nExcel summary:")
print(XLSX)

print("\nIMPORTANT:")
print("White-reference spectra were excluded.")
print("CR/DV transformed products were excluded from raw ASD guidance.")
print("All 13 SAC lithology names remain preserved.")
print("No direct chromite/PGE detection is claimed.")

print("\nPASS")
