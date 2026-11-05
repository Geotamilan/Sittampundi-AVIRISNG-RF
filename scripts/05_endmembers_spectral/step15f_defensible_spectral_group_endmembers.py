# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP15A = ROOT / (
    "09_Mineral_Mapping/"
    "Step15A_13_Lithology_AVIRIS_Endmembers"
)

STEP15B = ROOT / (
    "09_Mineral_Mapping/"
    "Step15B_13_Endmember_Spectral_Library"
)

STEP15D = ROOT / (
    "09_Mineral_Mapping/"
    "Step15D_AVIRIS_vs_Clean_SAC_ASD"
)

STEP15E = ROOT / (
    "09_Mineral_Mapping/"
    "Step15E_13_Endmember_Separability_Redundancy_QC"
)

SPECTRA_CSV = STEP15A / (
    "AVIRIS_Step15A_13_Endmembers_326ScientificBand_Wide.csv"
)

LOCATION_CSV = STEP15A / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.csv"
)

REDUNDANCY_CSV = STEP15E / (
    "AVIRIS_Step15E_Redundancy_Matrix.csv"
)

PAIR_CSV = STEP15E / (
    "AVIRIS_Step15E_All_78_Endmember_Pairs.csv"
)

STEP15D_SUMMARY = STEP15D / (
    "AVIRIS_Step15D_Endmember_CrossDomain_Summary.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15F_Defensible_Spectral_Group_Endmembers"
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

GOOD_COLS = [
    f"B{b:03d}"
    for b in GOOD_BANDS
]

# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 90)
print("STEP15F - DEFENSIBLE REDUCED SPECTRAL-GROUP ENDMEMBER SET")
print("=" * 90)

for p in [
    SPECTRA_CSV,
    LOCATION_CSV,
    REDUNDANCY_CSV,
    PAIR_CSV,
    STEP15D_SUMMARY,
]:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# LOAD ORIGINAL 13 ENDMEMBERS
# ============================================================

spectra = pd.read_csv(SPECTRA_CSV)
locations = pd.read_csv(LOCATION_CSV)
crossdomain = pd.read_csv(STEP15D_SUMMARY)

names = spectra["lithology_family"].astype(str).tolist()
codes = spectra["subtype_code"].astype(int).tolist()

X = (
    spectra[GOOD_COLS]
    .apply(pd.to_numeric, errors="coerce")
    .to_numpy(dtype=np.float32)
)

n = len(names)

print("\nOriginal candidate endmembers:", n)

# ============================================================
# LOAD REDUNDANCY MATRIX
# ============================================================

red_df = pd.read_csv(
    REDUNDANCY_CSV,
    index_col=0
)

redundancy = red_df.to_numpy(dtype=float)

if redundancy.shape != (n, n):
    raise RuntimeError(
        f"Unexpected redundancy matrix shape: {redundancy.shape}"
    )

# spectral distance
distance = 1.0 - redundancy

distance = (
    distance + distance.T
) / 2.0

np.fill_diagonal(
    distance,
    0.0
)

distance = np.clip(
    distance,
    0.0,
    1.0
)

# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

condensed = squareform(
    distance,
    checks=False
)

Z = linkage(
    condensed,
    method="average"
)

# ============================================================
# TEST k = 2 .. 6
# ============================================================

evaluation_rows = []

best_k = None
best_score = -999

for k in range(2, 7):

    labels = fcluster(
        Z,
        t=k,
        criterion="maxclust"
    )

    unique = np.unique(labels)

    if len(unique) < 2:
        continue

    try:

        score = silhouette_score(
            distance,
            labels,
            metric="precomputed"
        )

    except Exception:

        score = np.nan

    counts = pd.Series(
        labels
    ).value_counts()

    min_group_size = int(
        counts.min()
    )

    max_group_size = int(
        counts.max()
    )

    evaluation_rows.append({
        "requested_k": k,
        "actual_groups": len(unique),
        "silhouette_score": score,
        "minimum_group_size": min_group_size,
        "maximum_group_size": max_group_size,
    })

    if (
        np.isfinite(score)
        and score > best_score
    ):
        best_score = float(score)
        best_k = k

evaluation = pd.DataFrame(
    evaluation_rows
)

if best_k is None:
    raise RuntimeError(
        "Could not determine defensible spectral-group count."
    )

print("\nCLUSTER COUNT EVALUATION:")
print(
    evaluation.to_string(index=False)
)

print(
    "\nSelected spectral groups:",
    best_k
)

print(
    "Best silhouette:",
    round(best_score, 4)
)

# ============================================================
# FINAL GROUP ASSIGNMENT
# ============================================================

cluster_labels = fcluster(
    Z,
    t=best_k,
    criterion="maxclust"
)

# reorder clusters deterministically using
# smallest original subtype code in each cluster

old_groups = sorted(
    np.unique(cluster_labels)
)

group_order = sorted(
    old_groups,
    key=lambda g: min(
        codes[i]
        for i in range(n)
        if cluster_labels[i] == g
    )
)

group_remap = {
    old: i + 1
    for i, old in enumerate(group_order)
}

final_groups = np.array(
    [
        group_remap[g]
        for g in cluster_labels
    ],
    dtype=int
)

# ============================================================
# MEDOID PER GROUP
# ============================================================

membership_rows = []
group_rows = []

medoid_indices = []

for group_id in sorted(
    np.unique(final_groups)
):

    idx = np.where(
        final_groups == group_id
    )[0]

    local_distance = distance[
        np.ix_(idx, idx)
    ]

    mean_distance = np.nanmean(
        local_distance,
        axis=1
    )

    medoid_local = int(
        np.nanargmin(mean_distance)
    )

    medoid_idx = int(
        idx[medoid_local]
    )

    medoid_indices.append(
        medoid_idx
    )

    constituent_names = [
        names[i]
        for i in idx
    ]

    constituent_codes = [
        codes[i]
        for i in idx
    ]

    # within-group pair statistics
    if len(idx) > 1:

        vals = []

        for a in range(len(idx)):
            for b in range(
                a + 1,
                len(idx)
            ):
                vals.append(
                    redundancy[
                        idx[a],
                        idx[b]
                    ]
                )

        mean_redundancy = float(
            np.nanmean(vals)
        )

        min_redundancy = float(
            np.nanmin(vals)
        )

    else:

        mean_redundancy = np.nan
        min_redundancy = np.nan

    medoid_name = names[
        medoid_idx
    ]

    # Step15D ASD support
    cd = crossdomain[
        crossdomain[
            "AVIRIS_lithology"
        ] == medoid_name
    ]

    if len(cd):

        cd = cd.iloc[0]

        asd_status = cd[
            "cross_domain_support_status"
        ]

        best_asd = cd[
            "best_matching_ASD_family"
        ]

        best_score_asd = cd[
            "best_composite_score"
        ]

    else:

        asd_status = "NO_STEP15D_RECORD"
        best_asd = np.nan
        best_score_asd = np.nan

    group_rows.append({
        "spectral_group":
            f"SG{group_id}",

        "group_id":
            group_id,

        "number_of_original_endmembers":
            len(idx),

        "constituent_codes":
            ",".join(
                map(
                    str,
                    constituent_codes
                )
            ),

        "constituent_lithologies":
            " | ".join(
                constituent_names
            ),

        "medoid_subtype_code":
            codes[medoid_idx],

        "medoid_lithology_reference":
            medoid_name,

        "mean_within_group_redundancy":
            mean_redundancy,

        "minimum_within_group_redundancy":
            min_redundancy,

        "medoid_best_ASD_family":
            best_asd,

        "medoid_ASD_composite_score":
            best_score_asd,

        "medoid_cross_domain_status":
            asd_status,

        "recommended_map_name":
            f"Spectral Group SG{group_id}",

        "interpretation":
            (
                "DATA_DRIVEN_SPECTRAL_GROUP_"
                "NOT_VALIDATED_LITHOLOGY"
            )
    })

    for i in idx:

        membership_rows.append({
            "spectral_group":
                f"SG{group_id}",

            "group_id":
                group_id,

            "subtype_code":
                codes[i],

            "lithology_family":
                names[i],

            "is_group_medoid":
                bool(
                    i == medoid_idx
                ),

            "distance_to_group_medoid":
                float(
                    distance[
                        i,
                        medoid_idx
                    ]
                ),

            "redundancy_with_group_medoid":
                float(
                    redundancy[
                        i,
                        medoid_idx
                    ]
                )
        })

membership = pd.DataFrame(
    membership_rows
)

group_summary = pd.DataFrame(
    group_rows
)

# ============================================================
# MEDOID LOCATION TABLE
# ============================================================

medoid_rows = []

for _, group in group_summary.iterrows():

    code = int(
        group[
            "medoid_subtype_code"
        ]
    )

    loc = locations[
        locations[
            "subtype_code"
        ] == code
    ]

    if len(loc):

        loc = loc.iloc[0]

        medoid_rows.append({
            "spectral_group":
                group[
                    "spectral_group"
                ],

            "medoid_subtype_code":
                code,

            "medoid_lithology_reference":
                group[
                    "medoid_lithology_reference"
                ],

            "image_row_1based":
                loc.get(
                    "image_row_1based",
                    np.nan
                ),

            "image_col_1based":
                loc.get(
                    "image_col_1based",
                    np.nan
                ),

            "UTM_Easting_m":
                loc.get(
                    "UTM_Easting_m",
                    np.nan
                ),

            "UTM_Northing_m":
                loc.get(
                    "UTM_Northing_m",
                    np.nan
                ),

            "longitude":
                loc.get(
                    "longitude",
                    np.nan
                ),

            "latitude":
                loc.get(
                    "latitude",
                    np.nan
                ),

            "broad_probability":
                loc.get(
                    "broad_probability",
                    np.nan
                ),

            "uncertainty":
                loc.get(
                    "uncertainty",
                    np.nan
                ),

            "subtype_similarity":
                loc.get(
                    "subtype_similarity",
                    np.nan
                ),
        })

medoid_locations = pd.DataFrame(
    medoid_rows
)

# ============================================================
# BUILD REDUCED ENDMEMBER MATRIX
# ============================================================

reduced_spectra = X[
    medoid_indices,
    :
]

reduced_names = [
    f"SG{i+1}_{names[idx]}"
    for i, idx in enumerate(
        medoid_indices
    )
]

# ============================================================
# WAVELENGTHS FROM STEP15B LONG TABLE
# ============================================================

LONG15B = STEP15B / (
    "AVIRIS_Step15B_13_Endmember_Spectra_Long.csv"
)

if not LONG15B.exists():
    raise FileNotFoundError(
        LONG15B
    )

long15b = pd.read_csv(
    LONG15B
)

first_code = int(
    long15b[
        "subtype_code"
    ].iloc[0]
)

wl = (
    long15b[
        long15b[
            "subtype_code"
        ] == first_code
    ]
    .sort_values(
        "scientific_band_index_1based"
    )[
        "wavelength_nm"
    ]
    .to_numpy(
        dtype=float
    )
)

if len(wl) != 326:
    raise RuntimeError(
        f"Expected 326 wavelengths; found {len(wl)}"
    )

# ============================================================
# SAVE REDUCED ENVI SPECTRAL LIBRARY
# ============================================================

SLI = OUT / (
    "AVIRIS_Step15F_Defensible_Spectral_Group_Medoids_326Band.sli"
)

HDR = OUT / (
    "AVIRIS_Step15F_Defensible_Spectral_Group_Medoids_326Band.hdr"
)

reduced_spectra.astype(
    "<f4"
).tofile(
    SLI
)

spectra_names_text = ",\n".join(
    [
        f"  {name}"
        for name in reduced_names
    ]
)

wl_text = ",\n".join(
    [
        f"  {v:.6f}"
        for v in wl
    ]
)

hdr_text = f"""ENVI
description = {{
Reduced data-driven AVIRIS-NG spectral-group medoid library.
Generated after Step15E redundancy QC.
These spectral groups are NOT validated geological lithology classes.
}}
samples = 326
lines = {len(reduced_names)}
bands = 1
header offset = 0
file type = ENVI Spectral Library
data type = 4
interleave = bsq
sensor type = AVIRIS-NG
byte order = 0
wavelength units = Nanometers
spectra names = {{
{spectra_names_text}
}}
wavelength = {{
{wl_text}
}}
"""

HDR.write_text(
    hdr_text,
    encoding="utf-8"
)

# ============================================================
# REDUCED WIDE CSV
# ============================================================

reduced_rows = []

for group_number, idx in enumerate(
    medoid_indices,
    start=1
):

    rec = {
        "spectral_group":
            f"SG{group_number}",

        "source_subtype_code":
            codes[idx],

        "source_lithology_reference":
            names[idx]
    }

    for j, col in enumerate(
        GOOD_COLS
    ):
        rec[col] = X[
            idx,
            j
        ]

    reduced_rows.append(
        rec
    )

reduced_df = pd.DataFrame(
    reduced_rows
)

# ============================================================
# BETWEEN-GROUP SEPARABILITY
# ============================================================

between_rows = []

for a in range(
    len(medoid_indices)
):

    for b in range(
        a + 1,
        len(medoid_indices)
    ):

        ia = medoid_indices[a]
        ib = medoid_indices[b]

        between_rows.append({
            "group_A":
                f"SG{a+1}",

            "group_B":
                f"SG{b+1}",

            "medoid_A":
                names[ia],

            "medoid_B":
                names[ib],

            "redundancy_score":
                float(
                    redundancy[
                        ia,
                        ib
                    ]
                ),

            "spectral_distance":
                float(
                    distance[
                        ia,
                        ib
                    ]
                )
        })

between = pd.DataFrame(
    between_rows
)

# ============================================================
# PLOTS
# ============================================================

# Dendrogram
fig = plt.figure(
    figsize=(16, 8)
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

ax.set_ylabel(
    "Composite Spectral Distance"
)

ax.set_title(
    f"Step15F Spectral Grouping of 13 AVIRIS-NG Endmembers "
    f"(selected groups = {best_k})"
)

fig.tight_layout()

DENDRO = OUT / (
    "AVIRIS_Step15F_Selected_Spectral_Groups_Dendrogram.png"
)

fig.savefig(
    DENDRO,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# Reduced medoid spectra
fig = plt.figure(
    figsize=(15, 9)
)

ax = fig.add_subplot(111)

for i in range(
    len(medoid_indices)
):

    ax.plot(
        wl,
        reduced_spectra[i],
        linewidth=1.5,
        label=reduced_names[i]
    )

ax.set_xlabel(
    "Wavelength (nm)"
)

ax.set_ylabel(
    "AVIRIS-NG Reflectance"
)

ax.set_title(
    "Scientifically Reduced AVIRIS-NG Spectral-Group Medoids"
)

ax.legend(
    fontsize=8
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

MEDOID_PLOT = OUT / (
    "AVIRIS_Step15F_Reduced_Spectral_Group_Medoids.png"
)

fig.savefig(
    MEDOID_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# SAVE CSV
# ============================================================

evaluation.to_csv(
    OUT / (
        "AVIRIS_Step15F_Cluster_Count_Evaluation.csv"
    ),
    index=False
)

membership.to_csv(
    OUT / (
        "AVIRIS_Step15F_Original13_to_SpectralGroups.csv"
    ),
    index=False
)

group_summary.to_csv(
    OUT / (
        "AVIRIS_Step15F_Spectral_Group_Summary.csv"
    ),
    index=False
)

medoid_locations.to_csv(
    OUT / (
        "AVIRIS_Step15F_Spectral_Group_Medoid_Locations.csv"
    ),
    index=False
)

reduced_df.to_csv(
    OUT / (
        "AVIRIS_Step15F_Reduced_Endmember_Spectra_326Band.csv"
    ),
    index=False
)

between.to_csv(
    OUT / (
        "AVIRIS_Step15F_Between_Group_Separability.csv"
    ),
    index=False
)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15F_Defensible_Spectral_Group_Endmembers.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    evaluation.to_excel(
        writer,
        sheet_name="Cluster_Evaluation",
        index=False
    )

    group_summary.to_excel(
        writer,
        sheet_name="Spectral_Group_Summary",
        index=False
    )

    membership.to_excel(
        writer,
        sheet_name="Original13_Membership",
        index=False
    )

    medoid_locations.to_excel(
        writer,
        sheet_name="Medoid_Locations",
        index=False
    )

    between.to_excel(
        writer,
        sheet_name="Between_Group_QC",
        index=False
    )

    reduced_df.to_excel(
        writer,
        sheet_name="Reduced_326Band_Library",
        index=False
    )

# ============================================================
# SCIENTIFIC GATE
# ============================================================

if best_score >= 0.50:
    spectral_gate = (
        "ACCEPT_REDUCED_DATA_DRIVEN_SPECTRAL_GROUPS"
    )

elif best_score >= 0.30:
    spectral_gate = (
        "CAUTION_REDUCED_SPECTRAL_GROUPS_ONLY"
    )

else:
    spectral_gate = (
        "POOR_GLOBAL_SEPARABILITY_USE_REFERENCE_ONLY"
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step15F_Defensible_Spectral_Group_Endmembers",

    "original_endmembers":
        n,

    "selected_spectral_groups":
        int(best_k),

    "best_silhouette_score":
        float(best_score),

    "clustering_method":
        "Average-linkage hierarchical clustering",

    "distance_definition":
        "1 - Step15E composite redundancy score",

    "representative_definition":
        "Within-group medoid",

    "spectral_gate":
        spectral_gate,

    "important_warning":
        (
            "Spectral groups are data-driven spectral families. "
            "They must NOT automatically be renamed as geological "
            "lithology classes."
        ),

    "original_13_status":
        (
            "Retain as reference-guided candidate endmembers and "
            "field-validation locations only."
        )
}

(
    OUT / "AVIRIS_Step15F_Settings.json"
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

print("\n" + "=" * 90)
print("STEP15F COMPLETE")
print("=" * 90)

print("\nORIGINAL ENDMEMBERS:")
print(n)

print("\nSELECTED DEFENSIBLE SPECTRAL GROUPS:")
print(best_k)

print("\nBEST SILHOUETTE SCORE:")
print(
    round(
        best_score,
        6
    )
)

print("\nSPECTRAL GATE:")
print(
    spectral_gate
)

print("\nSPECTRAL GROUP SUMMARY:")
print(
    group_summary[
        [
            "spectral_group",
            "number_of_original_endmembers",
            "constituent_lithologies",
            "medoid_subtype_code",
            "medoid_lithology_reference",
            "mean_within_group_redundancy",
            "medoid_best_ASD_family",
            "medoid_cross_domain_status",
        ]
    ].to_string(
        index=False
    )
)

print("\nORIGINAL 13 -> SPECTRAL GROUP MEMBERSHIP:")
print(
    membership[
        [
            "spectral_group",
            "subtype_code",
            "lithology_family",
            "is_group_medoid",
            "redundancy_with_group_medoid"
        ]
    ].to_string(
        index=False
    )
)

print("\nMEDOID LOCATIONS:")
print(
    medoid_locations.to_string(
        index=False
    )
)

print("\nReduced ENVI spectral library:")
print(
    SLI
)

print("\nReduced ENVI header:")
print(
    HDR
)

print("\nExcel:")
print(
    XLSX
)

print("\nDendrogram:")
print(
    DENDRO
)

print("\nReduced spectral plot:")
print(
    MEDOID_PLOT
)

print("\nIMPORTANT:")
print(
    "These are SPECTRAL GROUPS, not automatically geological units."
)

print(
    "Keep all original 13 endmember locations for field/reference purposes."
)

print(
    "Do not perform hard 13-class lithology mapping from the original "
    "13 image-derived endmembers."
)

print("\nNEXT:")
print(
    "Step15G = map only the defensible reduced spectral groups, "
    "with uncertainty and original lithology-reference overlays."
)

print("\nPASS")
