# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# ROOT
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

MINERAL = ROOT / "09_Mineral_Mapping"

STEP15A = MINERAL / "Step15A_13_Lithology_AVIRIS_Endmembers"
STEP15C = MINERAL / "Step15C_13_Endmember_Absorption_Features"
STEP15D = MINERAL / "Step15D_AVIRIS_vs_Clean_SAC_ASD"
STEP15E = MINERAL / "Step15E_13_Endmember_Separability_Redundancy_QC"
STEP15F = MINERAL / "Step15F_Defensible_Spectral_Group_Endmembers"
STEP15G = MINERAL / "Step15G_Reduced_Spectral_Group_Map"
STEP16A = MINERAL / "Step16A_Official_RF_Accuracy_Assessment"

OUT = MINERAL / "Step16B_Final_Evidence_Synthesis"
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# INPUT FILES
# ============================================================

LOCATIONS_CSV = STEP15A / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.csv"
)

ABSORPTION_CSV = STEP15C / (
    "AVIRIS_Step15C_Endmember_Absorption_Summary.csv"
)

CROSSDOMAIN_CSV = STEP15D / (
    "AVIRIS_Step15D_Endmember_CrossDomain_Summary.csv"
)

SEPARABILITY_CSV = STEP15E / (
    "AVIRIS_Step15E_Class_Separability_Summary.csv"
)

NEAREST_CSV = STEP15E / (
    "AVIRIS_Step15E_Nearest_Endmember_QC.csv"
)

MEMBERSHIP_CSV = STEP15F / (
    "AVIRIS_Step15F_Original13_to_SpectralGroups.csv"
)

GROUP_SUMMARY_CSV = STEP15F / (
    "AVIRIS_Step15F_Spectral_Group_Summary.csv"
)

AREA_CSV = STEP15G / (
    "AVIRIS_Step15G_Spectral_Group_Area_Summary.csv"
)

CONTRIBUTION_CSV = STEP15G / (
    "AVIRIS_Step15G_Original_Lithology_Contribution.csv"
)

RF_CLASS_CSV = STEP16A / (
    "AVIRIS_Step16A_Classwise_Accuracy.csv"
)

RF_SUMMARY_CSV = STEP16A / (
    "AVIRIS_Step16A_Official_RF_Accuracy_Summary.csv"
)

# ============================================================
# CHECK
# ============================================================

required = [
    LOCATIONS_CSV,
    ABSORPTION_CSV,
    CROSSDOMAIN_CSV,
    SEPARABILITY_CSV,
    NEAREST_CSV,
    MEMBERSHIP_CSV,
    GROUP_SUMMARY_CSV,
    AREA_CSV,
    CONTRIBUTION_CSV,
    RF_CLASS_CSV,
    RF_SUMMARY_CSV,
]

print("=" * 100)
print("STEP16B - FINAL EVIDENCE SYNTHESIS")
print("=" * 100)

for p in required:

    if not p.exists():

        raise FileNotFoundError(
            f"Required input not found:\n{p}"
        )

print("\nAll required inputs found.")

# ============================================================
# 13 LITHOLOGY DEFINITIONS
# ============================================================

LITHOLOGIES = {
    1: "CHROMITITE",
    2: "ANORTHOSITE_METAANORTHOSITE",
    3: "GABBRO_METAGABBRO",
    4: "PYROXENITE_MAFIC_CUMULATE",
    5: "ULTRAMAFIC_DUNITE",
    6: "HORNBLENDE_BIOTITE_GNEISS",
    7: "GRANITE_GRANITIC_ROCK",
    8: "GNEISS_GRANITIC_COMPLEX",
    9: "PEGMATITE",
    10: "MIGMATITE",
    11: "QUARTZITE",
    12: "DOLERITE",
    13: "DOLOMITE_CARBONATE",
}

# ============================================================
# BROAD RF PARENT
#
# Only these five groups received official Step16A RF accuracy.
#
# Quartzite / Dolerite / Dolomite were reference-other classes
# in the training hierarchy and therefore must NOT inherit
# five-class RF accuracy.
# ============================================================

RF_PARENT = {
    1: "CHROMITITE_ASSOCIATED",

    2: "ANORTHOSITE_HOST",

    3: "MAFIC_ULTRAMAFIC",
    4: "MAFIC_ULTRAMAFIC",
    5: "MAFIC_ULTRAMAFIC",

    6: "GNEISSIC_METAMORPHIC",
    8: "GNEISSIC_METAMORPHIC",

    7: "GRANITIC_FELSIC",
    9: "GRANITIC_FELSIC",
    10: "GRANITIC_FELSIC",

    11: "REFERENCE_OTHER_OUTSIDE_RF5",
    12: "REFERENCE_OTHER_OUTSIDE_RF5",
    13: "REFERENCE_OTHER_OUTSIDE_RF5",
}

# ============================================================
# LOAD TABLES
# ============================================================

locations = pd.read_csv(
    LOCATIONS_CSV
)

absorption = pd.read_csv(
    ABSORPTION_CSV
)

cross = pd.read_csv(
    CROSSDOMAIN_CSV
)

separability = pd.read_csv(
    SEPARABILITY_CSV
)

nearest = pd.read_csv(
    NEAREST_CSV
)

membership = pd.read_csv(
    MEMBERSHIP_CSV
)

group_summary = pd.read_csv(
    GROUP_SUMMARY_CSV
)

group_area = pd.read_csv(
    AREA_CSV
)

contribution = pd.read_csv(
    CONTRIBUTION_CSV
)

rf_class = pd.read_csv(
    RF_CLASS_CSV
)

rf_summary = pd.read_csv(
    RF_SUMMARY_CSV
)

# ============================================================
# HELPERS
# ============================================================

def first_existing_column(df, candidates):

    lower = {
        c.lower(): c
        for c in df.columns
    }

    for c in candidates:

        if c.lower() in lower:
            return lower[
                c.lower()
            ]

    return None


def safe_value(row, column, default=np.nan):

    if column is None:
        return default

    try:
        return row[column]
    except Exception:
        return default


def find_by_code_or_name(
    df,
    code,
    name,
    code_candidates,
    name_candidates
):

    code_col = first_existing_column(
        df,
        code_candidates
    )

    name_col = first_existing_column(
        df,
        name_candidates
    )

    if code_col is not None:

        vals = pd.to_numeric(
            df[code_col],
            errors="coerce"
        )

        sub = df[
            vals == code
        ]

        if len(sub):
            return sub.iloc[0]

    if name_col is not None:

        sub = df[
            df[name_col]
            .astype(str)
            .str.upper()
            ==
            str(name).upper()
        ]

        if len(sub):
            return sub.iloc[0]

    return None


# ============================================================
# OFFICIAL RF SUMMARY
# ============================================================

rf_official = rf_summary.iloc[0]

oa_col = first_existing_column(
    rf_summary,
    ["Overall_Accuracy"]
)

ba_col = first_existing_column(
    rf_summary,
    ["Balanced_Accuracy"]
)

kappa_col = first_existing_column(
    rf_summary,
    ["Cohens_Kappa"]
)

macro_col = first_existing_column(
    rf_summary,
    ["Macro_F1"]
)

weighted_col = first_existing_column(
    rf_summary,
    ["Weighted_F1"]
)

nsite_col = first_existing_column(
    rf_summary,
    ["number_of_validation_sites"]
)

OA = float(
    rf_official[oa_col]
)

BA = float(
    rf_official[ba_col]
)

KAPPA = float(
    rf_official[kappa_col]
)

MACRO_F1 = float(
    rf_official[macro_col]
)

WEIGHTED_F1 = float(
    rf_official[weighted_col]
)

N_SITES = int(
    rf_official[nsite_col]
)

# ============================================================
# RF CLASS LOOKUP
# ============================================================

rf_lookup = {}

for _, row in rf_class.iterrows():

    klass = str(
        row["class"]
    )

    rf_lookup[
        klass
    ] = {
        "reference_sites":
            int(
                row[
                    "reference_sites"
                ]
            ),

        "producer_accuracy":
            float(
                row[
                    "Producers_Accuracy"
                ]
            ),

        "producer_accuracy_percent":
            float(
                row[
                    "Producers_Accuracy_percent"
                ]
            ),

        "user_accuracy":
            float(
                row[
                    "Users_Accuracy"
                ]
            ),

        "user_accuracy_percent":
            float(
                row[
                    "Users_Accuracy_percent"
                ]
            ),

        "F1":
            float(
                row[
                    "F1_score"
                ]
            )
    }

# ============================================================
# BUILD 13-CLASS MASTER EVIDENCE TABLE
# ============================================================

rows = []

for code in range(1, 14):

    name = LITHOLOGIES[
        code
    ]

    broad_parent = RF_PARENT[
        code
    ]

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    loc = find_by_code_or_name(
        locations,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "lithology_family",
            "lithology_reference"
        ]
    )

    # --------------------------------------------------------
    # Cross-domain ASD
    # --------------------------------------------------------

    cr = find_by_code_or_name(
        cross,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "AVIRIS_lithology"
        ]
    )

    if cr is not None:

        asd_available = bool(
            cr[
                "same_family_ASD_available"
            ]
        )

        expected_rank = (
            cr[
                "expected_family_rank"
            ]
        )

        expected_score = (
            cr[
                "expected_family_score"
            ]
        )

        best_asd = (
            cr[
                "best_matching_ASD_family"
            ]
        )

        best_asd_score = (
            cr[
                "best_composite_score"
            ]
        )

        cross_status = (
            cr[
                "cross_domain_support_status"
            ]
        )

    else:

        asd_available = False
        expected_rank = np.nan
        expected_score = np.nan
        best_asd = np.nan
        best_asd_score = np.nan
        cross_status = "NO_STEP15D_RECORD"

    # --------------------------------------------------------
    # Step15E separability
    # --------------------------------------------------------

    sep = find_by_code_or_name(
        separability,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "lithology_family"
        ]
    )

    near = find_by_code_or_name(
        nearest,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "lithology_family"
        ]
    )

    if sep is not None:

        min_sam = float(
            sep[
                "minimum_pairwise_SAM_deg"
            ]
        )

        median_sam = float(
            sep[
                "median_pairwise_SAM_deg"
            ]
        )

        max_redundancy = float(
            sep[
                "maximum_redundancy_score"
            ]
        )

        sep_status = (
            sep[
                "internal_separability_status"
            ]
        )

    else:

        min_sam = np.nan
        median_sam = np.nan
        max_redundancy = np.nan
        sep_status = "NO_STEP15E_RECORD"

    if near is not None:

        nearest_sam_name = (
            near[
                "nearest_by_SAM"
            ]
        )

        nearest_sam_deg = float(
            near[
                "nearest_SAM_deg"
            ]
        )

        most_redundant_with = (
            near[
                "most_redundant_with"
            ]
        )

    else:

        nearest_sam_name = np.nan
        nearest_sam_deg = np.nan
        most_redundant_with = np.nan

    # --------------------------------------------------------
    # Step15F spectral group
    # --------------------------------------------------------

    mem = find_by_code_or_name(
        membership,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "lithology_family"
        ]
    )

    if mem is not None:

        spectral_group = (
            mem[
                "spectral_group"
            ]
        )

        is_group_medoid = bool(
            mem[
                "is_group_medoid"
            ]
        )

        redundancy_medoid = float(
            mem[
                "redundancy_with_group_medoid"
            ]
        )

    else:

        spectral_group = np.nan
        is_group_medoid = False
        redundancy_medoid = np.nan

    # --------------------------------------------------------
    # Step15G contribution
    # --------------------------------------------------------

    cont = find_by_code_or_name(
        contribution,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "lithology_reference"
        ]
    )

    if cont is not None:

        candidate_pixels = int(
            cont[
                "pixel_count"
            ]
        )

        candidate_area = float(
            cont[
                "area_km2"
            ]
        )

    else:

        candidate_pixels = 0
        candidate_area = 0.0

    # --------------------------------------------------------
    # Step15C absorption
    # --------------------------------------------------------

    ab = find_by_code_or_name(
        absorption,
        code,
        name,
        [
            "subtype_code"
        ],
        [
            "lithology_family",
            "AVIRIS_lithology",
            "lithology"
        ]
    )

    strongest_feature_nm = np.nan
    strongest_depth = np.nan
    number_features = np.nan

    if ab is not None:

        fcol = first_existing_column(
            absorption,
            [
                "strongest_feature_nm",
                "strongest_absorption_nm",
                "strongest_center_nm",
                "strongest_wavelength_nm"
            ]
        )

        dcol = first_existing_column(
            absorption,
            [
                "strongest_depth_percent",
                "strongest_absorption_depth_percent",
                "strongest_depth_pct"
            ]
        )

        ncol = first_existing_column(
            absorption,
            [
                "number_of_features",
                "n_features",
                "feature_count"
            ]
        )

        strongest_feature_nm = safe_value(
            ab,
            fcol
        )

        strongest_depth = safe_value(
            ab,
            dcol
        )

        number_features = safe_value(
            ab,
            ncol
        )

    # --------------------------------------------------------
    # RF parent accuracy
    # --------------------------------------------------------

    if broad_parent in rf_lookup:

        rfm = rf_lookup[
            broad_parent
        ]

        rf_applicable = True

        rf_pa = rfm[
            "producer_accuracy_percent"
        ]

        rf_ua = rfm[
            "user_accuracy_percent"
        ]

        rf_f1 = rfm[
            "F1"
        ]

        rf_refsites = rfm[
            "reference_sites"
        ]

    else:

        rf_applicable = False
        rf_pa = np.nan
        rf_ua = np.nan
        rf_f1 = np.nan
        rf_refsites = np.nan

    # ========================================================
    # EVIDENCE TIER
    # ========================================================

    if (
        cross_status
        ==
        "LOCAL_ASD_SUPPORTS_CURRENT_LABEL"
        and
        rf_applicable
        and
        rf_pa >= 80
        and
        rf_ua >= 80
    ):

        evidence_tier = (
            "TIER_A_STRONGEST_AVAILABLE_ASSOCIATION_SUPPORT"
        )

        paper_interpretation = (
            "Strongest available integrated evidence. "
            "Interpret as a high-priority lithology-associated "
            "spectral target only; requires geological/field "
            "confirmation."
        )

    elif (
        cross_status
        ==
        "PARTIAL_LOCAL_ASD_SUPPORT"
    ):

        evidence_tier = (
            "TIER_B_PARTIAL_LOCAL_SPECTRAL_SUPPORT"
        )

        paper_interpretation = (
            "Reference-guided candidate with partial local ASD "
            "spectral consistency. Interpret cautiously."
        )

    elif (
        cross_status
        ==
        "WEAK_OR_CONFLICTING_LOCAL_ASD_SUPPORT"
    ):

        evidence_tier = (
            "TIER_C_WEAK_OR_CONFLICTING_SUPPORT"
        )

        paper_interpretation = (
            "Reference-guided image candidate only. Local ASD "
            "evidence does not uniquely support the current "
            "lithological label."
        )

    elif (
        cross_status
        ==
        "NO_SAME_FAMILY_LOCAL_ASD_REFERENCE"
    ):

        evidence_tier = (
            "TIER_D_REFERENCE_ONLY_NO_SAME_FAMILY_ASD"
        )

        paper_interpretation = (
            "Image/reference-guided candidate with no same-family "
            "clean local ASD validation. Use for field/reference "
            "interpretation only."
        )

    else:

        evidence_tier = (
            "TIER_D_REFERENCE_ONLY"
        )

        paper_interpretation = (
            "Reference information only."
        )

    # Every subtype failed hard 13-class separability.
    separability_warning = (
        "Do not interpret as an independently separable hard "
        "lithological image class because Step15E demonstrated "
        "substantial endmember redundancy."
    )

    # ========================================================
    # CLAIM STATUS
    # ========================================================

    if code == 1:

        recommended_term = (
            "chromitite-associated spectral target / "
            "chromitite-associated broad RF class"
        )

        prohibited_claim = (
            "Do not claim direct chromite occurrence, confirmed "
            "chromitite body, ore boundary, or PGE detection."
        )

    elif code == 4:

        recommended_term = (
            "pyroxenite/mafic-cumulate reference-guided candidate"
        )

        prohibited_claim = (
            "Do not claim confirmed pyroxenite geological boundary."
        )

    else:

        recommended_term = (
            "reference-guided lithological candidate"
        )

        prohibited_claim = (
            "Do not present as a validated hard lithological boundary."
        )

    # --------------------------------------------------------
    # Location fields
    # --------------------------------------------------------

    def locval(candidates):

        if loc is None:
            return np.nan

        col = first_existing_column(
            locations,
            candidates
        )

        return safe_value(
            loc,
            col
        )

    rows.append({
        "subtype_code":
            code,

        "lithology_reference":
            name,

        "RF_broad_parent":
            broad_parent,

        "RF_accuracy_applicable":
            rf_applicable,

        "RF_reference_sites":
            rf_refsites,

        "RF_Producers_Accuracy_percent":
            rf_pa,

        "RF_Users_Accuracy_percent":
            rf_ua,

        "RF_F1":
            rf_f1,

        "local_same_family_ASD_available":
            asd_available,

        "ASD_expected_family_rank":
            expected_rank,

        "ASD_expected_family_score":
            expected_score,

        "best_matching_local_ASD_family":
            best_asd,

        "best_local_ASD_composite_score":
            best_asd_score,

        "cross_domain_support_status":
            cross_status,

        "minimum_pairwise_SAM_deg":
            min_sam,

        "median_pairwise_SAM_deg":
            median_sam,

        "maximum_redundancy_score":
            max_redundancy,

        "internal_separability_status":
            sep_status,

        "nearest_endmember_by_SAM":
            nearest_sam_name,

        "nearest_endmember_SAM_deg":
            nearest_sam_deg,

        "most_redundant_with":
            most_redundant_with,

        "spectral_group":
            spectral_group,

        "is_spectral_group_medoid":
            is_group_medoid,

        "redundancy_with_group_medoid":
            redundancy_medoid,

        "Step15G_candidate_pixels":
            candidate_pixels,

        "Step15G_candidate_area_km2":
            candidate_area,

        "strongest_absorption_feature_nm":
            strongest_feature_nm,

        "strongest_absorption_depth_percent":
            strongest_depth,

        "number_absorption_features":
            number_features,

        "endmember_longitude":
            locval(
                [
                    "longitude"
                ]
            ),

        "endmember_latitude":
            locval(
                [
                    "latitude"
                ]
            ),

        "endmember_UTM_Easting_m":
            locval(
                [
                    "UTM_Easting_m"
                ]
            ),

        "endmember_UTM_Northing_m":
            locval(
                [
                    "UTM_Northing_m"
                ]
            ),

        "final_evidence_tier":
            evidence_tier,

        "recommended_paper_term":
            recommended_term,

        "paper_interpretation":
            paper_interpretation,

        "separability_warning":
            separability_warning,

        "prohibited_claim":
            prohibited_claim
    })

master = pd.DataFrame(
    rows
)

# ============================================================
# PROJECT-LEVEL EVIDENCE LEDGER
# ============================================================

ledger_rows = [
    {
        "evidence_component":
            "Random Forest broad lithological classification",

        "result":
            (
                f"Field-site OA={OA*100:.2f}%; "
                f"Balanced Accuracy={BA*100:.2f}%; "
                f"Kappa={KAPPA:.3f}; "
                f"Macro-F1={MACRO_F1:.3f}; "
                f"n={N_SITES} sites"
            ),

        "strength":
            "MODERATE",

        "can_claim":
            (
                "Five-class broad lithological probability "
                "classification with moderate internal "
                "field-site grouped cross-validation performance."
            ),

        "cannot_claim":
            (
                "Do not describe as high-accuracy or independent "
                "external validation."
            )
    },

    {
        "evidence_component":
            "Chromitite-associated RF broad class",

        "result":
            (
                "Producer's Accuracy=91.67%; "
                "User's Accuracy=100%; "
                "F1=0.9565"
            ),

        "strength":
            "STRONGEST_CLASS_WITHIN_CURRENT_RF_MODEL",

        "can_claim":
            (
                "Chromitite-associated class showed the strongest "
                "discrimination within the current five-class model."
            ),

        "cannot_claim":
            (
                "Do not equate classification performance with "
                "direct chromite or PGE detection."
            )
    },

    {
        "evidence_component":
            "13 reference-guided AVIRIS image endmembers",

        "result":
            (
                "All 13 retained as image/reference spectra, "
                "but internal separability failed for hard "
                "13-class mapping."
            ),

        "strength":
            "REFERENCE_GUIDED_ONLY",

        "can_claim":
            (
                "Thirteen representative image spectra/endmember "
                "locations were extracted for comparative "
                "spectral interpretation."
            ),

        "cannot_claim":
            (
                "Do not claim that 13 lithological classes are "
                "spectrally independently resolved."
            )
    },

    {
        "evidence_component":
            "Step15E endmember separability",

        "result":
            (
                "Minimum SAM=1.73 deg; median SAM=5.90 deg; "
                "46 pairs had redundancy >=0.85."
            ),

        "strength":
            "STRONG_EVIDENCE_OF_REDUNDANCY",

        "can_claim":
            (
                "Substantial spectral redundancy was observed "
                "among the 13 image-derived endmembers."
            ),

        "cannot_claim":
            (
                "Do not use original 13 endmembers for defensible "
                "hard 13-class spectral classification."
            )
    },

    {
        "evidence_component":
            "Step15F hierarchical spectral grouping",

        "result":
            (
                "Two data-driven spectral groups selected; "
                "silhouette score=0.7495."
            ),

        "strength":
            "STRONG_DATA_DRIVEN_GROUPING",

        "can_claim":
            (
                "The image endmembers are more defensibly "
                "represented as two spectral-behaviour groups."
            ),

        "cannot_claim":
            (
                "SG1 and SG2 are not geological units and must "
                "not be renamed using their medoid lithologies."
            )
    },

    {
        "evidence_component":
            "Step15G reduced spectral-group mapping",

        "result":
            (
                "SG1=9.138688 km2; "
                "SG2=14.885568 km2; "
                "total=24.024256 km2."
            ),

        "strength":
            "INTERPRETIVE_SPECTRAL_MAP",

        "can_claim":
            (
                "Spatial distribution of two data-driven "
                "spectral groups within the high-confidence "
                "mapping domain."
            ),

        "cannot_claim":
            (
                "Do not describe SG1 and SG2 as mapped rock units."
            )
    },

    {
        "evidence_component":
            "Local ASD cross-domain screening",

        "result":
            (
                "Chromitite image endmember received same-family "
                "local ASD support; pyroxenite received partial "
                "support; several other expected lithologies were "
                "weak/conflicting or lacked same-family ASD."
            ),

        "strength":
            "MIXED",

        "can_claim":
            (
                "Local laboratory spectra provide supporting "
                "cross-domain consistency evidence for selected "
                "reference spectra."
            ),

        "cannot_claim":
            (
                "Do not describe laboratory-airborne comparison "
                "as independent lithological validation."
            )
    },

    {
        "evidence_component":
            "Mineral/PGE interpretation",

        "result":
            (
                "No direct AVIRIS spectral evidence establishes "
                "chromite or PGE mineral occurrence."
            ),

        "strength":
            "NOT_SUPPORTED_FOR_DIRECT_IDENTIFICATION",

        "can_claim":
            (
                "Prioritize chromitite-associated spectral targets "
                "for geological follow-up."
            ),

        "cannot_claim":
            (
                "No direct chromite mapping and no PGE detection claim."
            )
    }
]

ledger = pd.DataFrame(
    ledger_rows
)

# ============================================================
# FINAL CLAIM MATRIX
# ============================================================

claims = pd.DataFrame([
    {
        "claim_id": "C01",
        "paper_statement":
            (
                "A supervised Random Forest model was used to "
                "generate broad lithological probability classes."
            ),
        "status": "SUPPORTED",
        "use_in_paper": "YES"
    },

    {
        "claim_id": "C02",
        "paper_statement":
            (
                "The five-class broad RF model achieved an "
                "overall field-site grouped accuracy of 60.47%."
            ),
        "status": "SUPPORTED",
        "use_in_paper": "YES"
    },

    {
        "claim_id": "C03",
        "paper_statement":
            (
                "The chromitite-associated broad class achieved "
                "91.67% producer's accuracy and 100% user's accuracy."
            ),
        "status": "SUPPORTED",
        "use_in_paper": "YES_WITH_CAUTION"
    },

    {
        "claim_id": "C04",
        "paper_statement":
            (
                "Thirteen reference-guided image spectra were "
                "identified as candidate lithological endmembers."
            ),
        "status": "SUPPORTED",
        "use_in_paper": "YES"
    },

    {
        "claim_id": "C05",
        "paper_statement":
            (
                "The 13 candidate endmembers represent 13 "
                "spectrally separable validated lithological units."
            ),
        "status": "NOT_SUPPORTED",
        "use_in_paper": "NO"
    },

    {
        "claim_id": "C06",
        "paper_statement":
            (
                "Two data-driven spectral groups provide a more "
                "defensible representation of image spectral "
                "variability than a hard 13-class map."
            ),
        "status": "SUPPORTED",
        "use_in_paper": "YES"
    },

    {
        "claim_id": "C07",
        "paper_statement":
            (
                "SG1 is a dolerite geological unit."
            ),
        "status": "NOT_SUPPORTED",
        "use_in_paper": "NO"
    },

    {
        "claim_id": "C08",
        "paper_statement":
            (
                "SG2 is a pyroxenite geological unit."
            ),
        "status": "NOT_SUPPORTED",
        "use_in_paper": "NO"
    },

    {
        "claim_id": "C09",
        "paper_statement":
            (
                "AVIRIS-NG directly detected chromite."
            ),
        "status": "NOT_SUPPORTED",
        "use_in_paper": "NO"
    },

    {
        "claim_id": "C10",
        "paper_statement":
            (
                "AVIRIS-NG directly detected PGE mineralization."
            ),
        "status": "NOT_SUPPORTED",
        "use_in_paper": "NO"
    },

    {
        "claim_id": "C11",
        "paper_statement":
            (
                "The workflow identifies chromitite-associated "
                "spectral targets suitable for field follow-up."
            ),
        "status": "SUPPORTED_WITH_CAUTION",
        "use_in_paper": "YES"
    },

    {
        "claim_id": "C12",
        "paper_statement":
            (
                "Field-site grouped validation was used to reduce "
                "spatial leakage between training and validation."
            ),
        "status": "SUPPORTED",
        "use_in_paper": "YES"
    },
])

# ============================================================
# TIER SUMMARY
# ============================================================

tier_summary = (
    master[
        "final_evidence_tier"
    ]
    .value_counts()
    .rename_axis(
        "evidence_tier"
    )
    .reset_index(
        name="number_of_lithology_references"
    )
)

# ============================================================
# SG SUMMARY
# ============================================================

sg_summary = group_area.copy()

sg_summary[
    "total_mapped_area_km2"
] = sg_summary[
    "mapped_area_km2"
].sum()

sg_summary[
    "mapped_area_percent"
] = (
    sg_summary[
        "mapped_area_km2"
    ]
    /
    sg_summary[
        "total_mapped_area_km2"
    ]
    * 100.0
)

# ============================================================
# PUBLICATION NUMBERS FREEZE
# ============================================================

publication_freeze = pd.DataFrame([
    {
        "parameter":
            "RF validation sites",
        "value":
            N_SITES,
        "unit":
            "field sites"
    },

    {
        "parameter":
            "RF Overall Accuracy",
        "value":
            OA * 100,
        "unit":
            "%"
    },

    {
        "parameter":
            "RF Balanced Accuracy",
        "value":
            BA * 100,
        "unit":
            "%"
    },

    {
        "parameter":
            "RF Cohen Kappa",
        "value":
            KAPPA,
        "unit":
            "coefficient"
    },

    {
        "parameter":
            "RF Macro F1",
        "value":
            MACRO_F1,
        "unit":
            "score"
    },

    {
        "parameter":
            "RF Weighted F1",
        "value":
            WEIGHTED_F1,
        "unit":
            "score"
    },

    {
        "parameter":
            "Initial reference-guided image spectra",
        "value":
            13,
        "unit":
            "endmember candidates"
    },

    {
        "parameter":
            "Final defensible spectral groups",
        "value":
            2,
        "unit":
            "spectral groups"
    },

    {
        "parameter":
            "SG1 mapped area",
        "value":
            9.138688,
        "unit":
            "km2"
    },

    {
        "parameter":
            "SG2 mapped area",
        "value":
            14.885568,
        "unit":
            "km2"
    },

    {
        "parameter":
            "Total Step15G mapped area",
        "value":
            24.024256,
        "unit":
            "km2"
    }
])

# ============================================================
# WRITE CSV
# ============================================================

MASTER_CSV = OUT / (
    "AVIRIS_Step16B_13_Lithology_Final_Evidence_Master.csv"
)

LEDGER_CSV = OUT / (
    "AVIRIS_Step16B_Project_Evidence_Ledger.csv"
)

CLAIMS_CSV = OUT / (
    "AVIRIS_Step16B_Paper_Claim_Matrix.csv"
)

FREEZE_CSV = OUT / (
    "AVIRIS_Step16B_Publication_Numbers_FREEZE.csv"
)

master.to_csv(
    MASTER_CSV,
    index=False
)

ledger.to_csv(
    LEDGER_CSV,
    index=False
)

claims.to_csv(
    CLAIMS_CSV,
    index=False
)

publication_freeze.to_csv(
    FREEZE_CSV,
    index=False
)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step16B_Final_Evidence_Synthesis.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    master.to_excel(
        writer,
        sheet_name="13_Lithology_Evidence",
        index=False
    )

    ledger.to_excel(
        writer,
        sheet_name="Project_Evidence_Ledger",
        index=False
    )

    claims.to_excel(
        writer,
        sheet_name="Paper_Claim_Matrix",
        index=False
    )

    publication_freeze.to_excel(
        writer,
        sheet_name="Publication_FREEZE",
        index=False
    )

    tier_summary.to_excel(
        writer,
        sheet_name="Evidence_Tiers",
        index=False
    )

    sg_summary.to_excel(
        writer,
        sheet_name="SG1_SG2_Summary",
        index=False
    )

    rf_class.to_excel(
        writer,
        sheet_name="RF_Class_Accuracy",
        index=False
    )

    rf_summary.to_excel(
        writer,
        sheet_name="RF_Official_Accuracy",
        index=False
    )

# ============================================================
# FINAL TEXT REPORT
# ============================================================

REPORT = OUT / (
    "AVIRIS_Step16B_Final_Scientific_Interpretation.txt"
)

report_text = f"""
STEP16B FINAL SCIENTIFIC EVIDENCE SYNTHESIS
===========================================

1. MACHINE-LEARNING COMPONENT
-----------------------------
The supervised classification component of the workflow is a
Random Forest broad-lithological probability model.

Official field-site grouped validation:
Validation sites       : {N_SITES}
Overall Accuracy       : {OA*100:.2f}%
Balanced Accuracy      : {BA*100:.2f}%
Cohen's Kappa          : {KAPPA:.4f}
Macro F1               : {MACRO_F1:.4f}
Weighted F1            : {WEIGHTED_F1:.4f}

These metrics apply ONLY to the five broad RF classes.

2. 13 REFERENCE-GUIDED IMAGE SPECTRA
------------------------------------
Thirteen image-derived/reference-guided spectra were retained as
candidate lithological representatives and endmember locations.

They must NOT be interpreted as 13 independently validated
lithological image classes.

3. INTERNAL SPECTRAL SEPARABILITY
---------------------------------
Step15E demonstrated strong spectral redundancy among the 13
candidate image endmembers.

Therefore:
- Hard 13-class lithological mapping is not supported.
- Original lithology labels remain reference/field-validation
  information.

4. DATA-DRIVEN SPECTRAL GROUPING
--------------------------------
Hierarchical spectral grouping reduced the 13 image spectra to two
defensible spectral-behaviour groups:

SG1
SG2

These are spectral groups, NOT geological formations or rock units.

5. CHROMITITE-ASSOCIATED INTERPRETATION
---------------------------------------
The chromitite-associated broad RF class was the strongest-performing
class within the current field-site grouped validation.

Local clean ASD comparison also supported the current chromitite
reference spectrum more strongly than most other lithological labels.

However, the AVIRIS image spectrum itself was spectrally redundant
with several other image-derived endmembers.

Therefore the defensible wording is:

"chromitite-associated spectral target"

and NOT:

"confirmed chromitite"
"chromite map"
"PGE map"
"direct PGE detection"

6. FINAL PAPER POSITION
-----------------------
The scientifically defensible contribution is an integrated
hyperspectral-machine-learning workflow combining:

AVIRIS-NG preprocessing
+ spectral dimensionality analysis
+ field-based ROI training
+ Random Forest broad-group probability mapping
+ uncertainty assessment
+ reference-guided subtype interpretation
+ image endmember extraction
+ continuum-removal / absorption analysis
+ local ASD cross-domain screening
+ spectral separability and redundancy testing
+ hierarchical spectral grouping
+ field-target prioritisation.

7. FINAL INTERPRETATION
-----------------------
The study should be presented as an uncertainty-aware,
reference-guided lithological and chromitite-associated spectral
targeting workflow.

It should NOT be presented as a high-accuracy hard 13-class
lithological map or a direct mineral/PGE detection system.
"""

REPORT.write_text(
    report_text.strip() + "\n",
    encoding="utf-8"
)

# ============================================================
# SETTINGS
# ============================================================

SETTINGS = {
    "step":
        "Step16B_Final_Evidence_Synthesis",

    "official_RF_accuracy_percent":
        OA * 100,

    "official_RF_balanced_accuracy_percent":
        BA * 100,

    "official_RF_kappa":
        KAPPA,

    "initial_reference_guided_endmembers":
        13,

    "defensible_spectral_groups":
        2,

    "primary_final_mapping_interpretation":
        (
            "Uncertainty-aware broad RF lithological probability "
            "mapping plus data-driven spectral-group interpretation."
        ),

    "chromitite_wording":
        (
            "chromitite-associated spectral target"
        ),

    "prohibited_direct_claims": [
        "confirmed chromitite body from AVIRIS alone",
        "direct chromite map",
        "direct PGE detection",
        "validated hard 13-class lithology map",
        "SG1 equals Dolerite",
        "SG2 equals Pyroxenite"
    ]
}

(
    OUT / "AVIRIS_Step16B_Settings.json"
).write_text(
    json.dumps(
        SETTINGS,
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("STEP16B COMPLETE")
print("=" * 100)

print("\nOFFICIAL RF PERFORMANCE:")
print(
    f"Overall Accuracy  : {OA*100:.2f}%"
)
print(
    f"Balanced Accuracy : {BA*100:.2f}%"
)
print(
    f"Kappa             : {KAPPA:.4f}"
)
print(
    f"Macro F1          : {MACRO_F1:.4f}"
)
print(
    f"Weighted F1       : {WEIGHTED_F1:.4f}"
)

print("\nFINAL 13-LITHOLOGY EVIDENCE STATUS:")

print(
    master[
        [
            "subtype_code",
            "lithology_reference",
            "RF_broad_parent",
            "RF_Producers_Accuracy_percent",
            "RF_Users_Accuracy_percent",
            "cross_domain_support_status",
            "minimum_pairwise_SAM_deg",
            "maximum_redundancy_score",
            "spectral_group",
            "Step15G_candidate_area_km2",
            "final_evidence_tier"
        ]
    ].to_string(
        index=False
    )
)

print("\nEVIDENCE TIER COUNTS:")

print(
    tier_summary.to_string(
        index=False
    )
)

print("\nFINAL PAPER CLAIM MATRIX:")

print(
    claims[
        [
            "claim_id",
            "status",
            "use_in_paper",
            "paper_statement"
        ]
    ].to_string(
        index=False
    )
)

print("\nFINAL SCIENTIFIC GATE:")
print(
    "ACCEPT_INTEGRATED_UNCERTAINTY_AWARE_WORKFLOW"
)

print("\nMAIN PAPER POSITION:")
print(
    "Random-Forest broad lithological probability mapping + "
    "reference-guided hyperspectral interpretation + "
    "spectral redundancy/uncertainty assessment."
)

print("\nDO NOT CLAIM:")
print(
    "Hard validated 13-class lithology map."
)
print(
    "Direct chromite detection."
)
print(
    "Direct PGE detection."
)
print(
    "SG1=Dolerite or SG2=Pyroxenite."
)

print("\nOUTPUT FOLDER:")
print(
    OUT
)

print("\nMASTER EVIDENCE TABLE:")
print(
    MASTER_CSV
)

print("\nPAPER CLAIM MATRIX:")
print(
    CLAIMS_CSV
)

print("\nPUBLICATION NUMBERS FREEZE:")
print(
    FREEZE_CSV
)

print("\nEXCEL:")
print(
    XLSX
)

print("\nSCIENTIFIC INTERPRETATION REPORT:")
print(
    REPORT
)

print("\nNEXT:")
print(
    "Step17 = final paper-ready figures, methodology workflow, "
    "Results, Discussion, limitations, tables and manuscript package."
)

print("\nPASS")
