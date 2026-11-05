# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
    f1_score,
)

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP14C2 = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C2_v2_Clean_RAW_ASD_Guided_Revalidation"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step16A_Official_RF_Accuracy_Assessment"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# OFFICIAL 5 BROAD CLASSES
# ============================================================

CLASSES = [
    "ANORTHOSITE_HOST",
    "CHROMITITE_ASSOCIATED",
    "GNEISSIC_METAMORPHIC",
    "GRANITIC_FELSIC",
    "MAFIC_ULTRAMAFIC",
]

SHORT_LABELS = [
    "Anorthosite\nHost",
    "Chromitite-\nAssociated",
    "Gneissic\nMetamorphic",
    "Granitic\nFelsic",
    "Mafic-\nUltramafic",
]

# Previous QC values expected from Step14C2-v2.
# Used ONLY as a consistency check.
EXPECTED_SITE_COUNT = 43
EXPECTED_OA = 0.604651
EXPECTED_BA = 0.574048

BOOTSTRAP_ITERATIONS = 10000
RANDOM_SEED = 42

# ============================================================
# NORMALIZE CLASS NAMES
# ============================================================

def normalize_class(value):

    if pd.isna(value):
        return np.nan

    s = str(value).strip().upper()

    s = re.sub(
        r"[^A-Z0-9]+",
        "_",
        s
    ).strip("_")

    aliases = {
        "ANORTHOSITE":
            "ANORTHOSITE_HOST",

        "ANORTHOSITE_HOST":
            "ANORTHOSITE_HOST",

        "ANORTHOSITE_METAANORTHOSITE":
            "ANORTHOSITE_HOST",

        "CHROMITITE":
            "CHROMITITE_ASSOCIATED",

        "CHROMITITE_ASSOC":
            "CHROMITITE_ASSOCIATED",

        "CHROMITITE_ASSOCIATED":
            "CHROMITITE_ASSOCIATED",

        "GNEISSIC":
            "GNEISSIC_METAMORPHIC",

        "GNEISSIC_METAMORPHIC":
            "GNEISSIC_METAMORPHIC",

        "GRANITIC":
            "GRANITIC_FELSIC",

        "GRANITIC_FELSIC":
            "GRANITIC_FELSIC",

        "MAFIC":
            "MAFIC_ULTRAMAFIC",

        "MAFIC_ULTRAMAFIC":
            "MAFIC_ULTRAMAFIC",
    }

    return aliases.get(
        s,
        s
    )


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_true_column(columns):

    cols = list(columns)

    priority = [
        "true_broad_group",
        "actual_broad_group",
        "true_group",
        "actual_group",
        "true_label",
        "actual_label",
        "y_true",
        "broad_group_true",
        "observed_broad_group",
        "reference_broad_group",
    ]

    lower = {
        c.lower(): c
        for c in cols
    }

    for p in priority:
        if p in lower:
            return lower[p]

    candidates = []

    for c in cols:

        x = c.lower()

        if (
            (
                "true" in x
                or "actual" in x
                or "observed" in x
                or "reference" in x
            )
            and
            (
                "group" in x
                or "label" in x
                or "class" in x
            )
        ):
            candidates.append(c)

    return (
        candidates[0]
        if candidates
        else None
    )


def find_pred_column(columns):

    cols = list(columns)

    priority = [
        "predicted_broad_group",
        "pred_broad_group",
        "predicted_group",
        "pred_group",
        "predicted_label",
        "pred_label",
        "y_pred",
        "broad_group_predicted",
    ]

    lower = {
        c.lower(): c
        for c in cols
    }

    for p in priority:
        if p in lower:
            return lower[p]

    candidates = []

    for c in cols:

        x = c.lower()

        if (
            (
                "pred" in x
                or "prediction" in x
            )
            and
            (
                "group" in x
                or "label" in x
                or "class" in x
            )
        ):
            candidates.append(c)

    return (
        candidates[0]
        if candidates
        else None
    )


def find_site_column(columns):

    cols = list(columns)

    priority = [
        "field_site_id",
        "field_site",
        "site_id",
        "site",
        "source_id",
        "sample_id",
        "location_id",
        "point_id",
        "roi_id",
    ]

    lower = {
        c.lower(): c
        for c in cols
    }

    for p in priority:
        if p in lower:
            return lower[p]

    for c in cols:

        x = c.lower()

        if (
            "site" in x
            or "location_id" in x
            or "source_id" in x
        ):
            return c

    return None


# ============================================================
# FIND EXISTING FINAL VALIDATION PREDICTION TABLE
# ============================================================

print("=" * 95)
print("STEP16A - OFFICIAL RANDOM FOREST ACCURACY ASSESSMENT")
print("=" * 95)

if not STEP14C2.exists():
    raise FileNotFoundError(
        STEP14C2
    )

csv_files = list(
    STEP14C2.rglob("*.csv")
)

print(
    "\nCSV files found in Step14C2-v2:",
    len(csv_files)
)

candidate_tables = []

for path in csv_files:

    try:

        preview = pd.read_csv(
            path,
            nrows=10
        )

    except Exception:

        continue

    true_col = find_true_column(
        preview.columns
    )

    pred_col = find_pred_column(
        preview.columns
    )

    if (
        true_col is None
        or pred_col is None
    ):
        continue

    filename = path.name.lower()

    score = 0

    if "broad" in filename:
        score += 20

    if "guided" in filename:
        score += 20

    if "clean" in filename:
        score += 10

    if "site" in filename:
        score += 20

    if "prediction" in filename or "pred" in filename:
        score += 10

    if "pixel" in filename:
        score -= 10

    site_col = find_site_column(
        preview.columns
    )

    if site_col is not None:
        score += 10

    candidate_tables.append({
        "path": path,
        "score": score,
        "true_col": true_col,
        "pred_col": pred_col,
        "site_col": site_col,
    })

if not candidate_tables:

    print("\nAvailable CSV filenames:")

    for p in csv_files:
        print(" ", p.name)

    raise RuntimeError(
        "\nCould not automatically find a CSV containing both "
        "true and predicted broad-group labels.\n"
        "Do NOT substitute pixel-level random validation.\n"
        "Send the filenames printed above for a safe adjustment."
    )

candidate_tables = sorted(
    candidate_tables,
    key=lambda x: x["score"],
    reverse=True
)

chosen = candidate_tables[0]

print("\nCandidate validation tables:")

for item in candidate_tables[:10]:

    print(
        f"score={item['score']:3d} | "
        f"{item['path'].name} | "
        f"true={item['true_col']} | "
        f"pred={item['pred_col']} | "
        f"site={item['site_col']}"
    )

print("\nSELECTED VALIDATION TABLE:")
print(chosen["path"])

# ============================================================
# LOAD SELECTED TABLE
# ============================================================

df = pd.read_csv(
    chosen["path"]
)

true_col = chosen[
    "true_col"
]

pred_col = chosen[
    "pred_col"
]

site_col = chosen[
    "site_col"
]

df["_true"] = df[
    true_col
].map(
    normalize_class
)

df["_pred"] = df[
    pred_col
].map(
    normalize_class
)

# Only official 5 broad classes
valid = (
    df["_true"].isin(CLASSES)
    &
    df["_pred"].isin(CLASSES)
)

excluded = int(
    (~valid).sum()
)

df = df[
    valid
].copy()

print("\nRows retained in official five classes:", len(df))
print("Rows excluded:", excluded)

# ============================================================
# CONVERT TO FIELD-SITE LEVEL IF NECESSARY
# ============================================================

if site_col is not None:

    df["_site"] = (
        df[
            site_col
        ]
        .astype(str)
        .str.strip()
    )

    unique_sites = int(
        df["_site"].nunique()
    )

    print(
        "Unique field sites:",
        unique_sites
    )

    if len(df) == unique_sites:

        site_df = df[
            [
                "_site",
                "_true",
                "_pred"
            ]
        ].copy()

    else:

        print(
            "\nMultiple rows per field site detected."
        )

        print(
            "Aggregating predictions to one prediction per field site."
        )

        # ----------------------------------------------------
        # Check for class probability columns first
        # ----------------------------------------------------

        probability_columns = {}

        for c in df.columns:

            cl = c.lower()

            if (
                "prob" not in cl
                and "proba" not in cl
            ):
                continue

            normalized_name = normalize_class(
                re.sub(
                    r"(?i)probability|proba|prob|score",
                    "",
                    c
                )
            )

            for klass in CLASSES:

                if (
                    klass in normalize_class(c)
                    or
                    normalized_name == klass
                ):
                    probability_columns[
                        klass
                    ] = c

        site_rows = []

        for site, g in df.groupby(
            "_site"
        ):

            true_values = (
                g["_true"]
                .dropna()
                .value_counts()
            )

            if len(true_values) == 0:
                continue

            true_label = (
                true_values.index[0]
            )

            if len(probability_columns) >= 5:

                mean_probs = {}

                for klass in CLASSES:

                    c = probability_columns[
                        klass
                    ]

                    mean_probs[
                        klass
                    ] = pd.to_numeric(
                        g[c],
                        errors="coerce"
                    ).mean()

                predicted_label = max(
                    mean_probs,
                    key=mean_probs.get
                )

                aggregation_method = (
                    "MEAN_SITE_PROBABILITY"
                )

            else:

                votes = (
                    g["_pred"]
                    .value_counts()
                )

                predicted_label = (
                    votes.index[0]
                )

                aggregation_method = (
                    "MAJORITY_PIXEL_VOTE"
                )

            site_rows.append({
                "_site":
                    site,

                "_true":
                    true_label,

                "_pred":
                    predicted_label,

                "_aggregation":
                    aggregation_method,

                "_n_pixels":
                    len(g),
            })

        site_df = pd.DataFrame(
            site_rows
        )

else:

    # No site column.
    # Safe only if the table already appears to be site-level.
    if len(df) <= 60:

        site_df = df[
            [
                "_true",
                "_pred"
            ]
        ].copy()

        site_df[
            "_site"
        ] = [
            f"SITE_{i+1:03d}"
            for i in range(len(site_df))
        ]

        site_df[
            "_aggregation"
        ] = (
            "TABLE_ALREADY_SITE_LEVEL"
        )

    else:

        raise RuntimeError(
            "\nSelected prediction table has more than 60 rows "
            "but no identifiable site ID.\n"
            "Cannot safely calculate official site-level accuracy."
        )

# ============================================================
# FINAL CLEAN SITE TABLE
# ============================================================

site_df = site_df[
    site_df["_true"].isin(CLASSES)
    &
    site_df["_pred"].isin(CLASSES)
].copy()

site_df = site_df.reset_index(
    drop=True
)

y_true = site_df[
    "_true"
].to_numpy()

y_pred = site_df[
    "_pred"
].to_numpy()

N = len(site_df)

print("\nOFFICIAL VALIDATION SITES:", N)

print("\nREFERENCE SITE COUNTS:")

print(
    site_df[
        "_true"
    ]
    .value_counts()
    .reindex(
        CLASSES,
        fill_value=0
    )
    .to_string()
)

# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=CLASSES
)

cm_df = pd.DataFrame(
    cm,
    index=CLASSES,
    columns=CLASSES
)

# Row-normalized matrix = Producer perspective
row_sums = cm.sum(
    axis=1,
    keepdims=True
)

cm_row_pct = np.divide(
    cm,
    row_sums,
    out=np.zeros_like(
        cm,
        dtype=float
    ),
    where=row_sums != 0
) * 100.0

cm_row_pct_df = pd.DataFrame(
    cm_row_pct,
    index=CLASSES,
    columns=CLASSES
)

# ============================================================
# GLOBAL METRICS
# ============================================================

oa = accuracy_score(
    y_true,
    y_pred
)

ba = balanced_accuracy_score(
    y_true,
    y_pred
)

kappa = cohen_kappa_score(
    y_true,
    y_pred,
    labels=CLASSES
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=CLASSES,
    average="weighted",
    zero_division=0
)

# ============================================================
# CLASS METRICS
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASSES,
        zero_division=0
    )
)

predicted_support = cm.sum(
    axis=0
)

class_rows = []

for i, klass in enumerate(
    CLASSES
):

    tp = int(
        cm[i, i]
    )

    fn = int(
        cm[i, :].sum()
        - tp
    )

    fp = int(
        cm[:, i].sum()
        - tp
    )

    tn = int(
        cm.sum()
        - tp
        - fn
        - fp
    )

    class_rows.append({
        "class":
            klass,

        "reference_sites":
            int(
                support[i]
            ),

        "predicted_sites":
            int(
                predicted_support[i]
            ),

        "TP":
            tp,

        "FN":
            fn,

        "FP":
            fp,

        "TN":
            tn,

        # Remote sensing terminology
        "Producers_Accuracy":
            float(
                recall[i]
            ),

        "Producers_Accuracy_percent":
            float(
                recall[i] * 100
            ),

        "Users_Accuracy":
            float(
                precision[i]
            ),

        "Users_Accuracy_percent":
            float(
                precision[i] * 100
            ),

        # Machine-learning terminology
        "Recall":
            float(
                recall[i]
            ),

        "Precision":
            float(
                precision[i]
            ),

        "F1_score":
            float(
                f1[i]
            ),
    })

class_metrics = pd.DataFrame(
    class_rows
)

# ============================================================
# BOOTSTRAP 95% CONFIDENCE INTERVALS
#
# Resample FIELD SITES, not pixels.
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

boot_oa = []
boot_ba = []
boot_kappa = []

for _ in range(
    BOOTSTRAP_ITERATIONS
):

    idx = rng.integers(
        0,
        N,
        N
    )

    yt = y_true[
        idx
    ]

    yp = y_pred[
        idx
    ]

    boot_oa.append(
        accuracy_score(
            yt,
            yp
        )
    )

    # Balanced accuracy only if >=2 true classes
    if len(np.unique(yt)) >= 2:

        boot_ba.append(
            balanced_accuracy_score(
                yt,
                yp
            )
        )

        try:

            k = cohen_kappa_score(
                yt,
                yp,
                labels=CLASSES
            )

            if np.isfinite(k):
                boot_kappa.append(
                    k
                )

        except Exception:
            pass

def ci95(values):

    values = np.asarray(
        values,
        dtype=float
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        )
    )

oa_lo, oa_hi = ci95(
    boot_oa
)

ba_lo, ba_hi = ci95(
    boot_ba
)

kap_lo, kap_hi = ci95(
    boot_kappa
)

# ============================================================
# OFFICIAL SUMMARY
# ============================================================

summary = pd.DataFrame([
    {
        "validation_level":
            "FIELD_SITE_GROUPED_CROSS_VALIDATION",

        "number_of_validation_sites":
            N,

        "number_of_classes":
            len(CLASSES),

        "Overall_Accuracy":
            oa,

        "Overall_Accuracy_percent":
            oa * 100,

        "OA_95CI_lower":
            oa_lo,

        "OA_95CI_upper":
            oa_hi,

        "Balanced_Accuracy":
            ba,

        "Balanced_Accuracy_percent":
            ba * 100,

        "BA_95CI_lower":
            ba_lo,

        "BA_95CI_upper":
            ba_hi,

        "Cohens_Kappa":
            kappa,

        "Kappa_95CI_lower":
            kap_lo,

        "Kappa_95CI_upper":
            kap_hi,

        "Macro_F1":
            macro_f1,

        "Weighted_F1":
            weighted_f1,

        "primary_accuracy_for_paper":
            "FIELD_SITE_OVERALL_ACCURACY",

        "validation_status":
            "INTERNAL_GROUPED_CROSS_VALIDATION_NOT_EXTERNAL_INDEPENDENT_VALIDATION"
    }
])

# ============================================================
# CONSISTENCY CHECK WITH PREVIOUS STEP14C2-v2
# ============================================================

site_count_match = (
    N == EXPECTED_SITE_COUNT
)

oa_match = (
    abs(
        oa - EXPECTED_OA
    )
    <= 0.002
)

ba_match = (
    abs(
        ba - EXPECTED_BA
    )
    <= 0.002
)

if (
    site_count_match
    and
    oa_match
    and
    ba_match
):

    consistency_gate = (
        "PASS_MATCHES_PREVIOUS_STEP14C2_V2_VALIDATION"
    )

else:

    consistency_gate = (
        "REVIEW_DIFFERENCE_FROM_PREVIOUS_STEP14C2_V2"
    )

# ============================================================
# FREEZE TABLE
# ============================================================

freeze = pd.DataFrame([
    {
        "metric":
            "Validation sites",
        "official_value":
            N,
        "unit":
            "sites"
    },
    {
        "metric":
            "Overall Accuracy",
        "official_value":
            oa * 100,
        "unit":
            "%"
    },
    {
        "metric":
            "Balanced Accuracy",
        "official_value":
            ba * 100,
        "unit":
            "%"
    },
    {
        "metric":
            "Cohen Kappa",
        "official_value":
            kappa,
        "unit":
            "coefficient"
    },
    {
        "metric":
            "Macro F1",
        "official_value":
            macro_f1,
        "unit":
            "score"
    },
    {
        "metric":
            "Weighted F1",
        "official_value":
            weighted_f1,
        "unit":
            "score"
    },
])

# ============================================================
# CONFUSION MATRIX FIGURE — COUNTS
# ============================================================

fig = plt.figure(
    figsize=(10, 9)
)

ax = fig.add_subplot(111)

im = ax.imshow(
    cm
)

ax.set_xticks(
    np.arange(
        len(CLASSES)
    )
)

ax.set_yticks(
    np.arange(
        len(CLASSES)
    )
)

ax.set_xticklabels(
    SHORT_LABELS,
    rotation=35,
    ha="right"
)

ax.set_yticklabels(
    SHORT_LABELS
)

ax.set_xlabel(
    "Predicted broad lithological group"
)

ax.set_ylabel(
    "Reference field-site broad lithological group"
)

ax.set_title(
    "Random Forest Field-Site Confusion Matrix"
)

threshold = (
    cm.max() / 2.0
    if cm.max() > 0
    else 0
)

for i in range(
    cm.shape[0]
):

    for j in range(
        cm.shape[1]
    ):

        ax.text(
            j,
            i,
            str(
                cm[i, j]
            ),
            ha="center",
            va="center",
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            ),
            fontsize=11
        )

fig.colorbar(
    im,
    ax=ax,
    label="Number of field sites"
)

fig.tight_layout()

CM_COUNTS_PNG = OUT / (
    "AVIRIS_Step16A_RF_Confusion_Matrix_Counts.png"
)

fig.savefig(
    CM_COUNTS_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# NORMALIZED CONFUSION MATRIX FIGURE
# ============================================================

fig = plt.figure(
    figsize=(10, 9)
)

ax = fig.add_subplot(111)

im = ax.imshow(
    cm_row_pct,
    vmin=0,
    vmax=100
)

ax.set_xticks(
    np.arange(
        len(CLASSES)
    )
)

ax.set_yticks(
    np.arange(
        len(CLASSES)
    )
)

ax.set_xticklabels(
    SHORT_LABELS,
    rotation=35,
    ha="right"
)

ax.set_yticklabels(
    SHORT_LABELS
)

ax.set_xlabel(
    "Predicted broad lithological group"
)

ax.set_ylabel(
    "Reference field-site broad lithological group"
)

ax.set_title(
    "Random Forest Field-Site Confusion Matrix — Row Normalized"
)

for i in range(
    cm_row_pct.shape[0]
):

    for j in range(
        cm_row_pct.shape[1]
    ):

        value = cm_row_pct[
            i,
            j
        ]

        ax.text(
            j,
            i,
            f"{value:.1f}%",
            ha="center",
            va="center",
            color=(
                "white"
                if value >= 50
                else "black"
            ),
            fontsize=10
        )

fig.colorbar(
    im,
    ax=ax,
    label="Reference-class percentage (%)"
)

fig.tight_layout()

CM_PERCENT_PNG = OUT / (
    "AVIRIS_Step16A_RF_Confusion_Matrix_RowNormalized.png"
)

fig.savefig(
    CM_PERCENT_PNG,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# SAVE CSV
# ============================================================

SUMMARY_CSV = OUT / (
    "AVIRIS_Step16A_Official_RF_Accuracy_Summary.csv"
)

CLASS_CSV = OUT / (
    "AVIRIS_Step16A_Classwise_Accuracy.csv"
)

CM_CSV = OUT / (
    "AVIRIS_Step16A_Confusion_Matrix_Counts.csv"
)

CM_PCT_CSV = OUT / (
    "AVIRIS_Step16A_Confusion_Matrix_RowNormalized_Percent.csv"
)

SITE_CSV = OUT / (
    "AVIRIS_Step16A_FieldSite_Validation_Predictions.csv"
)

FREEZE_CSV = OUT / (
    "AVIRIS_Step16A_OFFICIAL_ACCURACY_FREEZE.csv"
)

summary.to_csv(
    SUMMARY_CSV,
    index=False
)

class_metrics.to_csv(
    CLASS_CSV,
    index=False
)

cm_df.to_csv(
    CM_CSV
)

cm_row_pct_df.to_csv(
    CM_PCT_CSV
)

site_df.to_csv(
    SITE_CSV,
    index=False
)

freeze.to_csv(
    FREEZE_CSV,
    index=False
)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step16A_Official_RF_Accuracy_Assessment.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="Official_Accuracy",
        index=False
    )

    freeze.to_excel(
        writer,
        sheet_name="Accuracy_FREEZE",
        index=False
    )

    class_metrics.to_excel(
        writer,
        sheet_name="Classwise_Accuracy",
        index=False
    )

    cm_df.to_excel(
        writer,
        sheet_name="Confusion_Counts"
    )

    cm_row_pct_df.to_excel(
        writer,
        sheet_name="Confusion_Row_Percent"
    )

    site_df.to_excel(
        writer,
        sheet_name="FieldSite_Predictions",
        index=False
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step16A_Official_RF_Accuracy_Assessment",

    "classification_level":
        "5 broad lithological groups",

    "primary_validation_unit":
        "field site",

    "validation_design":
        (
            "Grouped/leave-field-site-out cross-validation. "
            "Neighbouring ROI pixels from the same field site "
            "are not treated as independent validation samples."
        ),

    "source_prediction_table":
        str(
            chosen["path"]
        ),

    "official_metrics": {
        "overall_accuracy":
            float(oa),

        "balanced_accuracy":
            float(ba),

        "cohen_kappa":
            float(kappa),

        "macro_F1":
            float(macro_f1),

        "weighted_F1":
            float(weighted_f1),
    },

    "consistency_gate":
        consistency_gate,

    "important_scientific_note":
        (
            "These values are internal field-site grouped "
            "cross-validation accuracy estimates. They must not "
            "be described as independent external validation."
        ),

    "mapping_note":
        (
            "Accuracy applies to the five broad Random Forest "
            "lithological groups, not to the 13 reference-guided "
            "subtypes and not to SG1/SG2 spectral clustering."
        )
}

(
    OUT / "AVIRIS_Step16A_Settings.json"
).write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# TERMINAL REPORT
# ============================================================

print("\n" + "=" * 95)
print("STEP16A COMPLETE")
print("=" * 95)

print("\nOFFICIAL VALIDATION DESIGN:")
print(
    "Field-site grouped cross-validation"
)

print(
    "Broad lithological classes:",
    len(CLASSES)
)

print(
    "Validation field sites:",
    N
)

print("\nOFFICIAL RANDOM FOREST ACCURACY:")

print(
    f"Overall Accuracy       : {oa:.6f} "
    f"({oa*100:.2f}%)"
)

print(
    f"Balanced Accuracy      : {ba:.6f} "
    f"({ba*100:.2f}%)"
)

print(
    f"Cohen's Kappa          : {kappa:.6f}"
)

print(
    f"Macro F1               : {macro_f1:.6f}"
)

print(
    f"Weighted F1            : {weighted_f1:.6f}"
)

print(
    f"OA 95% bootstrap CI    : "
    f"{oa_lo*100:.2f}% - {oa_hi*100:.2f}%"
)

print(
    f"BA 95% bootstrap CI    : "
    f"{ba_lo*100:.2f}% - {ba_hi*100:.2f}%"
)

print(
    f"Kappa 95% bootstrap CI : "
    f"{kap_lo:.4f} - {kap_hi:.4f}"
)

print("\nCLASS-WISE ACCURACY:")

print(
    class_metrics[
        [
            "class",
            "reference_sites",
            "predicted_sites",
            "Producers_Accuracy_percent",
            "Users_Accuracy_percent",
            "F1_score"
        ]
    ].to_string(
        index=False
    )
)

print("\nCONFUSION MATRIX — COUNTS:")
print(
    cm_df.to_string()
)

print("\nCONFUSION MATRIX — ROW NORMALIZED (%):")
print(
    cm_row_pct_df.round(
        2
    ).to_string()
)

print("\nCONSISTENCY CHECK:")
print(
    consistency_gate
)

if consistency_gate.startswith(
    "REVIEW"
):

    print(
        "\nWARNING:"
    )

    print(
        "The result differs from the previously recorded "
        "Step14C2-v2 site-level OA ≈60.47% / BA ≈57.40%."
    )

    print(
        "Do not freeze the result until the selected prediction "
        "table is confirmed."
    )

else:

    print(
        "\nOFFICIAL ACCURACY FREEZE STATUS:"
    )

    print(
        "READY_TO_FREEZE_FOR_CURRENT_WORKFLOW"
    )

print("\nOutputs:")

print(
    "Official summary:",
    SUMMARY_CSV
)

print(
    "Class accuracy:",
    CLASS_CSV
)

print(
    "Confusion matrix:",
    CM_CSV
)

print(
    "Official freeze:",
    FREEZE_CSV
)

print(
    "Excel:",
    XLSX
)

print(
    "Confusion-matrix figure:",
    CM_COUNTS_PNG
)

print(
    "Normalized matrix figure:",
    CM_PERCENT_PNG
)

print("\nIMPORTANT:")

print(
    "Producer's Accuracy = Recall."
)

print(
    "User's Accuracy = Precision."
)

print(
    "Primary publication accuracy = FIELD-SITE Overall Accuracy."
)

print(
    "Do not report pixel-level random-split accuracy as the "
    "official geological mapping accuracy."
)

print(
    "This accuracy belongs to the five broad RF classes only."
)

print("\nPASS")
