# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import Point

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP14D = ROOT / (
    "09_Mineral_Mapping/"
    "Step14D_Broad_Lithology_Probability_Uncertainty"
)

STEP14E = ROOT / (
    "09_Mineral_Mapping/"
    "Step14E_HighConfidence_Subtype_Overlay"
)

STEP15A = ROOT / (
    "09_Mineral_Mapping/"
    "Step15A_13_Lithology_AVIRIS_Endmembers"
)

STEP15F = ROOT / (
    "09_Mineral_Mapping/"
    "Step15F_Defensible_Spectral_Group_Endmembers"
)

SUBTYPE_RASTER = STEP14E / (
    "AVIRIS_Step14E_13Class_ReferenceSubtype_Winner.tif"
)

MAX_PROB = STEP14D / (
    "AVIRIS_Step14D_Max_Probability.tif"
)

UNCERTAINTY = STEP14D / (
    "AVIRIS_Step14D_Uncertainty.tif"
)

MEMBERSHIP_CSV = STEP15F / (
    "AVIRIS_Step15F_Original13_to_SpectralGroups.csv"
)

GROUP_SUMMARY_CSV = STEP15F / (
    "AVIRIS_Step15F_Spectral_Group_Summary.csv"
)

MEDOID_LOCATIONS_CSV = STEP15F / (
    "AVIRIS_Step15F_Spectral_Group_Medoid_Locations.csv"
)

ORIGINAL_LOCATIONS_CSV = STEP15A / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15G_Reduced_Spectral_Group_Map"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# SETTINGS
# ============================================================

PIXEL_AREA_M2 = 16.0

# Step14E already represents high-confidence broad-group pixels.
# We additionally generate a stricter display product.
VERY_HIGH_PROBABILITY = 0.80
VERY_LOW_UNCERTAINTY = 0.20

# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 90)
print("STEP15G - REDUCED DEFENSIBLE SPECTRAL-GROUP MAP")
print("=" * 90)

for p in [
    SUBTYPE_RASTER,
    MAX_PROB,
    UNCERTAINTY,
    MEMBERSHIP_CSV,
    GROUP_SUMMARY_CSV,
    MEDOID_LOCATIONS_CSV,
    ORIGINAL_LOCATIONS_CSV,
]:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# LOAD MEMBERSHIP
# ============================================================

membership = pd.read_csv(
    MEMBERSHIP_CSV
)

group_summary = pd.read_csv(
    GROUP_SUMMARY_CSV
)

print("\nORIGINAL 13 -> REDUCED SPECTRAL GROUPS:")

print(
    membership[
        [
            "spectral_group",
            "group_id",
            "subtype_code",
            "lithology_family",
            "is_group_medoid",
            "redundancy_with_group_medoid"
        ]
    ].sort_values(
        [
            "group_id",
            "subtype_code"
        ]
    ).to_string(
        index=False
    )
)

group_ids = sorted(
    membership[
        "group_id"
    ].unique()
)

print(
    "\nSpectral groups:",
    group_ids
)

# ============================================================
# LOAD RASTERS
# ============================================================

with rasterio.open(
    SUBTYPE_RASTER
) as ds:

    subtype = ds.read(
        1
    )

    profile = ds.profile.copy()

    transform = ds.transform
    crs = ds.crs
    width = ds.width
    height = ds.height

with rasterio.open(
    MAX_PROB
) as ds:

    maxprob = ds.read(
        1
    ).astype(
        np.float32
    )

with rasterio.open(
    UNCERTAINTY
) as ds:

    uncertainty = ds.read(
        1
    ).astype(
        np.float32
    )

maxprob[
    maxprob <= -9000
] = np.nan

uncertainty[
    uncertainty <= -9000
] = np.nan

print("\nRaster size:", width, "x", height)
print("CRS:", crs)

# ============================================================
# COLLAPSE 13 REFERENCE SUBTYPES -> SG1 / SG2
# ============================================================

group_map = np.zeros(
    subtype.shape,
    dtype=np.uint8
)

for _, row in membership.iterrows():

    subtype_code = int(
        row[
            "subtype_code"
        ]
    )

    group_id = int(
        row[
            "group_id"
        ]
    )

    group_map[
        subtype == subtype_code
    ] = group_id

# ============================================================
# STRICT / VERY-HIGH-CONFIDENCE DISPLAY MAP
# ============================================================

very_high_mask = (
    (group_map > 0)
    &
    np.isfinite(maxprob)
    &
    np.isfinite(uncertainty)
    &
    (maxprob >= VERY_HIGH_PROBABILITY)
    &
    (uncertainty <= VERY_LOW_UNCERTAINTY)
)

very_high_group_map = np.zeros_like(
    group_map,
    dtype=np.uint8
)

very_high_group_map[
    very_high_mask
] = group_map[
    very_high_mask
]

# ============================================================
# MASKED SOURCE CONFIDENCE / UNCERTAINTY
# ============================================================

masked_probability = np.full(
    subtype.shape,
    -9999.0,
    dtype=np.float32
)

masked_uncertainty = np.full(
    subtype.shape,
    -9999.0,
    dtype=np.float32
)

mapped = (
    group_map > 0
)

masked_probability[
    mapped
] = maxprob[
    mapped
]

masked_uncertainty[
    mapped
] = uncertainty[
    mapped
]

# ============================================================
# OUTPUT PROFILES
# ============================================================

byte_profile = profile.copy()

byte_profile.update(
    driver="GTiff",
    count=1,
    dtype="uint8",
    nodata=0,
    compress="deflate",
    BIGTIFF="YES"
)

byte_profile.pop(
    "interleave",
    None
)

float_profile = profile.copy()

float_profile.update(
    driver="GTiff",
    count=1,
    dtype="float32",
    nodata=-9999.0,
    compress="deflate",
    predictor=3,
    BIGTIFF="YES"
)

float_profile.pop(
    "interleave",
    None
)

# ============================================================
# SAVE RASTERS
# ============================================================

GROUP_MAP_TIF = OUT / (
    "AVIRIS_Step15G_Reduced_Spectral_Group_Map.tif"
)

VERY_HIGH_TIF = OUT / (
    "AVIRIS_Step15G_VeryHighConfidence_Spectral_Group_Map.tif"
)

PROB_TIF = OUT / (
    "AVIRIS_Step15G_Source_Broad_Probability.tif"
)

UNC_TIF = OUT / (
    "AVIRIS_Step15G_Source_Uncertainty.tif"
)

with rasterio.open(
    GROUP_MAP_TIF,
    "w",
    **byte_profile
) as ds:

    ds.write(
        group_map,
        1
    )

with rasterio.open(
    VERY_HIGH_TIF,
    "w",
    **byte_profile
) as ds:

    ds.write(
        very_high_group_map,
        1
    )

with rasterio.open(
    PROB_TIF,
    "w",
    **float_profile
) as ds:

    ds.write(
        masked_probability,
        1
    )

with rasterio.open(
    UNC_TIF,
    "w",
    **float_profile
) as ds:

    ds.write(
        masked_uncertainty,
        1
    )

# ============================================================
# GROUP AREA SUMMARY
# ============================================================

area_rows = []

for group_id in group_ids:

    normal_count = int(
        np.sum(
            group_map == group_id
        )
    )

    strict_count = int(
        np.sum(
            very_high_group_map == group_id
        )
    )

    gs = group_summary[
        group_summary[
            "group_id"
        ] == group_id
    ]

    if len(gs):

        gs = gs.iloc[0]

        constituents = gs[
            "constituent_lithologies"
        ]

        medoid = gs[
            "medoid_lithology_reference"
        ]

    else:

        constituents = ""
        medoid = ""

    area_rows.append({
        "spectral_group":
            f"SG{group_id}",

        "group_id":
            group_id,

        "mapped_pixels":
            normal_count,

        "mapped_area_km2":
            normal_count
            * PIXEL_AREA_M2
            / 1e6,

        "very_high_pixels":
            strict_count,

        "very_high_area_km2":
            strict_count
            * PIXEL_AREA_M2
            / 1e6,

        "medoid_reference":
            medoid,

        "constituent_lithology_references":
            constituents,

        "recommended_map_label":
            f"Spectral Group SG{group_id}",

        "scientific_status":
            "DATA_DRIVEN_SPECTRAL_GROUP_NOT_GEOLOGICAL_UNIT"
    })

area_df = pd.DataFrame(
    area_rows
)

# ============================================================
# ORIGINAL SUBTYPE CONTRIBUTION SUMMARY
# ============================================================

contribution_rows = []

for _, row in membership.iterrows():

    code = int(
        row[
            "subtype_code"
        ]
    )

    gid = int(
        row[
            "group_id"
        ]
    )

    count = int(
        np.sum(
            subtype == code
        )
    )

    contribution_rows.append({
        "spectral_group":
            f"SG{gid}",

        "group_id":
            gid,

        "subtype_code":
            code,

        "lithology_reference":
            row[
                "lithology_family"
            ],

        "pixel_count":
            count,

        "area_km2":
            count
            * PIXEL_AREA_M2
            / 1e6,

        "is_group_medoid":
            bool(
                row[
                    "is_group_medoid"
                ]
            ),

        "redundancy_with_group_medoid":
            row[
                "redundancy_with_group_medoid"
            ],
    })

contribution_df = pd.DataFrame(
    contribution_rows
)

# ============================================================
# CONFIDENCE STATISTICS BY SPECTRAL GROUP
# ============================================================

confidence_rows = []

for gid in group_ids:

    mask = (
        group_map == gid
    )

    confidence_rows.append({
        "spectral_group":
            f"SG{gid}",

        "pixel_count":
            int(
                mask.sum()
            ),

        "mean_source_probability":
            float(
                np.nanmean(
                    maxprob[
                        mask
                    ]
                )
            ),

        "median_source_probability":
            float(
                np.nanmedian(
                    maxprob[
                        mask
                    ]
                )
            ),

        "mean_source_uncertainty":
            float(
                np.nanmean(
                    uncertainty[
                        mask
                    ]
                )
            ),

        "median_source_uncertainty":
            float(
                np.nanmedian(
                    uncertainty[
                        mask
                    ]
                )
            )
    })

confidence_df = pd.DataFrame(
    confidence_rows
)

# ============================================================
# CREATE ORIGINAL 13 REFERENCE POINTS
# ============================================================

locations = pd.read_csv(
    ORIGINAL_LOCATIONS_CSV
)

locations = locations.merge(
    membership[
        [
            "spectral_group",
            "group_id",
            "subtype_code",
            "is_group_medoid",
            "redundancy_with_group_medoid"
        ]
    ],
    on="subtype_code",
    how="left"
)

locations = locations.rename(
    columns={
        "lithology_family":
            "lithology_reference"
    }
)

ref_geometry = [
    Point(
        float(x),
        float(y)
    )
    for x, y in zip(
        locations[
            "UTM_Easting_m"
        ],
        locations[
            "UTM_Northing_m"
        ]
    )
]

reference_gdf = gpd.GeoDataFrame(
    locations,
    geometry=ref_geometry,
    crs=crs
)

# ============================================================
# CREATE SG MEDOID POINTS
# ============================================================

medoid_locations = pd.read_csv(
    MEDOID_LOCATIONS_CSV
)

medoid_geometry = [
    Point(
        float(x),
        float(y)
    )
    for x, y in zip(
        medoid_locations[
            "UTM_Easting_m"
        ],
        medoid_locations[
            "UTM_Northing_m"
        ]
    )
]

medoid_gdf = gpd.GeoDataFrame(
    medoid_locations,
    geometry=medoid_geometry,
    crs=crs
)

# ============================================================
# SAVE GEOPACKAGE
# ============================================================

GPKG = OUT / (
    "AVIRIS_Step15G_Spectral_Group_Reference_Points.gpkg"
)

if GPKG.exists():
    GPKG.unlink()

reference_gdf.to_file(
    GPKG,
    layer="Step15G_Original13_Reference_Points",
    driver="GPKG"
)

medoid_gdf.to_file(
    GPKG,
    layer="Step15G_Spectral_Group_Medoids",
    driver="GPKG"
)

# ============================================================
# LEGEND TABLE
# ============================================================

legend_df = pd.DataFrame([
    {
        "raster_value": 1,
        "spectral_group": "SG1",
        "map_label": "Spectral Group SG1",
        "medoid_reference": (
            area_df.loc[
                area_df[
                    "group_id"
                ] == 1,
                "medoid_reference"
            ].iloc[0]
            if 1 in group_ids
            else ""
        ),
        "interpretation":
            "Data-driven spectral group; not a geological unit"
    },
    {
        "raster_value": 2,
        "spectral_group": "SG2",
        "map_label": "Spectral Group SG2",
        "medoid_reference": (
            area_df.loc[
                area_df[
                    "group_id"
                ] == 2,
                "medoid_reference"
            ].iloc[0]
            if 2 in group_ids
            else ""
        ),
        "interpretation":
            "Data-driven spectral group; not a geological unit"
    }
])

# ============================================================
# SAVE CSV
# ============================================================

AREA_CSV = OUT / (
    "AVIRIS_Step15G_Spectral_Group_Area_Summary.csv"
)

CONTRIB_CSV = OUT / (
    "AVIRIS_Step15G_Original_Lithology_Contribution.csv"
)

CONF_CSV = OUT / (
    "AVIRIS_Step15G_Group_Source_Confidence_Summary.csv"
)

LOOKUP_CSV = OUT / (
    "AVIRIS_Step15G_ArcGIS_Legend_Lookup.csv"
)

area_df.to_csv(
    AREA_CSV,
    index=False
)

contribution_df.to_csv(
    CONTRIB_CSV,
    index=False
)

confidence_df.to_csv(
    CONF_CSV,
    index=False
)

legend_df.to_csv(
    LOOKUP_CSV,
    index=False
)

# ============================================================
# EXCEL
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15G_Reduced_Spectral_Group_Map_Summary.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    area_df.to_excel(
        writer,
        sheet_name="Group_Areas",
        index=False
    )

    contribution_df.to_excel(
        writer,
        sheet_name="Original13_Contribution",
        index=False
    )

    confidence_df.to_excel(
        writer,
        sheet_name="Source_Confidence",
        index=False
    )

    membership.to_excel(
        writer,
        sheet_name="Membership",
        index=False
    )

    group_summary.to_excel(
        writer,
        sheet_name="Step15F_Group_Summary",
        index=False
    )

    locations.drop(
        columns=[],
        errors="ignore"
    ).to_excel(
        writer,
        sheet_name="Reference_Locations",
        index=False
    )

    medoid_locations.to_excel(
        writer,
        sheet_name="Medoid_Locations",
        index=False
    )

    legend_df.to_excel(
        writer,
        sheet_name="ArcGIS_Legend",
        index=False
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step15G_Reduced_Spectral_Group_Map",

    "mapping_method":
        (
            "Reclassification of the Step14E high-confidence "
            "13-class reference-subtype raster according to "
            "Step15F data-driven spectral-group membership."
        ),

    "number_of_spectral_groups":
        len(group_ids),

    "very_high_probability_threshold":
        VERY_HIGH_PROBABILITY,

    "very_low_uncertainty_threshold":
        VERY_LOW_UNCERTAINTY,

    "important_scientific_note":
        (
            "SG1 and SG2 are spectral-behaviour groups. "
            "They are not validated geological units."
        ),

    "medoid_note":
        (
            "Dolerite and Pyroxenite/Mafic Cumulate are medoid "
            "reference spectra only; their names must not be used "
            "as geological labels for SG1 or SG2."
        ),

    "original_13_note":
        (
            "Original lithology names and locations are retained "
            "as reference-guided field-validation information."
        )
}

(
    OUT / "AVIRIS_Step15G_Settings.json"
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
print("STEP15G COMPLETE")
print("=" * 90)

print("\nREDUCED SPECTRAL-GROUP AREA:")

print(
    area_df[
        [
            "spectral_group",
            "mapped_pixels",
            "mapped_area_km2",
            "very_high_pixels",
            "very_high_area_km2",
            "medoid_reference"
        ]
    ].to_string(
        index=False
    )
)

print("\nSOURCE CONFIDENCE BY GROUP:")

print(
    confidence_df.to_string(
        index=False
    )
)

print("\nORIGINAL 13 CONTRIBUTION:")

print(
    contribution_df
    .sort_values(
        [
            "group_id",
            "subtype_code"
        ]
    )[
        [
            "spectral_group",
            "subtype_code",
            "lithology_reference",
            "pixel_count",
            "area_km2",
            "is_group_medoid"
        ]
    ]
    .to_string(
        index=False
    )
)

print("\nMAIN SPECTRAL-GROUP MAP:")
print(
    GROUP_MAP_TIF
)

print("\nVERY-HIGH-CONFIDENCE MAP:")
print(
    VERY_HIGH_TIF
)

print("\nSOURCE PROBABILITY:")
print(
    PROB_TIF
)

print("\nSOURCE UNCERTAINTY:")
print(
    UNC_TIF
)

print("\nArcGIS GeoPackage:")
print(
    GPKG
)

print("\nLayers:")
print(
    "Step15G_Original13_Reference_Points"
)
print(
    "Step15G_Spectral_Group_Medoids"
)

print("\nExcel:")
print(
    XLSX
)

print("\nSCIENTIFIC INTERPRETATION:")
print(
    "SG1 and SG2 are data-driven spectral groups only."
)

print(
    "Do NOT rename SG1 as Dolerite."
)

print(
    "Do NOT rename SG2 as Pyroxenite."
)

print(
    "Original 13 lithology names remain reference / "
    "field-validation labels."
)

print("\nNEXT:")
print(
    "Step15H = ArcGIS final spectral-group display + "
    "original 13 endmember reference locations + "
    "paper-ready figure preparation."
)

print("\nPASS")
