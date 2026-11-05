# -*- coding: utf-8 -*-
import os

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio

from shapely.geometry import Point
from pyproj import Transformer

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

AVIRIS = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.dat"
)

AVIRIS_HDR = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.hdr"
)

STEP14D = ROOT / (
    "09_Mineral_Mapping/"
    "Step14D_Broad_Lithology_Probability_Uncertainty"
)

STEP14E = ROOT / (
    "09_Mineral_Mapping/"
    "Step14E_HighConfidence_Subtype_Overlay"
)

SUBTYPE_RASTER = STEP14E / (
    "AVIRIS_Step14E_13Class_ReferenceSubtype_Winner.tif"
)

SIMILARITY_RASTER = STEP14E / (
    "AVIRIS_Step14E_Subtype_MaxSimilarity.tif"
)

MARGIN_RASTER = STEP14E / (
    "AVIRIS_Step14E_Subtype_SimilarityMargin.tif"
)

MAX_PROB_RASTER = STEP14D / (
    "AVIRIS_Step14D_Max_Probability.tif"
)

UNCERTAINTY_RASTER = STEP14D / (
    "AVIRIS_Step14D_Uncertainty.tif"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15A_13_Lithology_AVIRIS_Endmembers"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# LITHOLOGY CLASSES
# ============================================================

CLASSES = {
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
# SCIENTIFIC GOOD BANDS
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
# HELPERS
# ============================================================

def parse_wavelengths(hdr):

    text = hdr.read_text(
        errors="ignore"
    )

    m = re.search(
        r"wavelength\s*=\s*\{(.*?)\}",
        text,
        flags=re.I | re.S
    )

    if not m:
        raise RuntimeError(
            "Could not parse wavelength list from ENVI header."
        )

    vals = []

    for x in m.group(1).replace(
        "\n", " "
    ).split(","):

        try:
            vals.append(
                float(x.strip())
            )
        except Exception:
            pass

    return np.asarray(
        vals,
        dtype=float
    )


def scale01(arr, mask):

    vals = arr[mask]

    vals = vals[
        np.isfinite(vals)
    ]

    if len(vals) == 0:
        return np.zeros_like(
            arr,
            dtype=np.float32
        )

    lo = np.nanpercentile(
        vals,
        5
    )

    hi = np.nanpercentile(
        vals,
        95
    )

    if (
        not np.isfinite(lo) or
        not np.isfinite(hi) or
        hi <= lo
    ):

        out = np.zeros_like(
            arr,
            dtype=np.float32
        )

        out[mask] = 0.5

        return out

    out = np.zeros_like(
        arr,
        dtype=np.float32
    )

    out[mask] = np.clip(
        (arr[mask] - lo) /
        (hi - lo),
        0,
        1
    )

    return out


# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 90)
print("STEP15A - EXTRACT 13 AVIRIS-NG LITHOLOGY ENDMEMBERS")
print("=" * 90)

for p in [
    AVIRIS,
    AVIRIS_HDR,
    SUBTYPE_RASTER,
    SIMILARITY_RASTER,
    MARGIN_RASTER,
    MAX_PROB_RASTER,
    UNCERTAINTY_RASTER
]:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# WAVELENGTHS
# ============================================================

wl425 = parse_wavelengths(
    AVIRIS_HDR
)

if len(wl425) != 425:

    raise RuntimeError(
        f"Expected 425 wavelengths; found {len(wl425)}"
    )

wl326 = wl425[
    GOOD_INDEX
]

print("\nFull AVIRIS bands       :", len(wl425))
print("Scientific good bands  :", len(wl326))

# ============================================================
# LOAD CLASSIFICATION / QC RASTERS
# ============================================================

with rasterio.open(
    SUBTYPE_RASTER
) as ds:

    subtype = ds.read(1)

    transform = ds.transform
    crs = ds.crs

    height = ds.height
    width = ds.width

with rasterio.open(
    SIMILARITY_RASTER
) as ds:

    similarity = ds.read(1).astype(
        np.float32
    )

with rasterio.open(
    MARGIN_RASTER
) as ds:

    margin = ds.read(1).astype(
        np.float32
    )

with rasterio.open(
    MAX_PROB_RASTER
) as ds:

    maxprob = ds.read(1).astype(
        np.float32
    )

with rasterio.open(
    UNCERTAINTY_RASTER
) as ds:

    uncertainty = ds.read(1).astype(
        np.float32
    )

similarity[
    similarity <= -9000
] = np.nan

margin[
    margin <= -9000
] = np.nan

maxprob[
    maxprob <= -9000
] = np.nan

uncertainty[
    uncertainty <= -9000
] = np.nan

print("\nRaster size:", width, "x", height)
print("Raster CRS :", crs)

# ============================================================
# SELECT BEST PIXEL PER LITHOLOGY
#
# score =
# 40% broad probability
# 40% subtype similarity
# 20% subtype margin where available
#
# For single-subtype groups where margin is NaN:
# 50% probability + 50% similarity
# ============================================================

selected = []

for code, name in CLASSES.items():

    mask = (
        (subtype == code) &
        np.isfinite(maxprob) &
        np.isfinite(similarity)
    )

    n = int(
        mask.sum()
    )

    print("\n" + "-" * 90)
    print(code, name)
    print("Candidate pixels:", n)

    if n == 0:

        selected.append({
            "subtype_code": code,
            "lithology_family": name,
            "candidate_pixels": 0,
            "status": "NO_CANDIDATE_PIXEL"
        })

        continue

    prob_scaled = scale01(
        maxprob,
        mask
    )

    sim_scaled = scale01(
        similarity,
        mask
    )

    margin_available = (
        mask &
        np.isfinite(margin)
    )

    # If most candidate pixels have a valid subtype margin,
    # include margin in the ranking.
    margin_fraction = (
        margin_available.sum() /
        mask.sum()
    )

    if margin_fraction >= 0.50:

        margin_scaled = scale01(
            margin,
            margin_available
        )

        score = (
            0.40 * prob_scaled +
            0.40 * sim_scaled +
            0.20 * margin_scaled
        )

        ranking_logic = (
            "40% probability + 40% similarity + 20% margin"
        )

    else:

        score = (
            0.50 * prob_scaled +
            0.50 * sim_scaled
        )

        ranking_logic = (
            "50% probability + 50% similarity; margin unavailable"
        )

    score[
        ~mask
    ] = -9999.0

    flat_index = int(
        np.nanargmax(score)
    )

    row, col = np.unravel_index(
        flat_index,
        score.shape
    )

    x, y = rasterio.transform.xy(
        transform,
        row,
        col,
        offset="center"
    )

    selected.append({
        "subtype_code":
            code,

        "lithology_family":
            name,

        "candidate_pixels":
            n,

        "image_row_0based":
            int(row),

        "image_col_0based":
            int(col),

        "image_row_1based":
            int(row + 1),

        "image_col_1based":
            int(col + 1),

        "UTM_Easting_m":
            float(x),

        "UTM_Northing_m":
            float(y),

        "broad_probability":
            float(
                maxprob[row, col]
            ),

        "uncertainty":
            float(
                uncertainty[row, col]
            ),

        "subtype_similarity":
            float(
                similarity[row, col]
            ),

        "subtype_margin":
            (
                float(
                    margin[row, col]
                )
                if np.isfinite(
                    margin[row, col]
                )
                else np.nan
            ),

        "endmember_selection_score":
            float(
                score[row, col]
            ),

        "ranking_logic":
            ranking_logic,

        "status":
            "SELECTED_AVIRIS_ENDMEMBER"
    })

# ============================================================
# LOCATION TABLE
# ============================================================

locations = pd.DataFrame(
    selected
)

# ============================================================
# CONVERT TO WGS84
# ============================================================

transformer = Transformer.from_crs(
    crs,
    "EPSG:4326",
    always_xy=True
)

longitudes = []
latitudes = []

for _, r in locations.iterrows():

    if r["status"] != "SELECTED_AVIRIS_ENDMEMBER":

        longitudes.append(
            np.nan
        )

        latitudes.append(
            np.nan
        )

        continue

    lon, lat = transformer.transform(
        r["UTM_Easting_m"],
        r["UTM_Northing_m"]
    )

    longitudes.append(
        lon
    )

    latitudes.append(
        lat
    )

locations[
    "longitude"
] = longitudes

locations[
    "latitude"
] = latitudes

# ============================================================
# EXTRACT 425-BAND AVIRIS SPECTRA
# ============================================================

spectra425_rows = []
spectra326_rows = []
long425_rows = []
long326_rows = []

with rasterio.open(
    AVIRIS
) as src:

    for _, r in locations.iterrows():

        if r["status"] != "SELECTED_AVIRIS_ENDMEMBER":
            continue

        row = int(
            r["image_row_0based"]
        )

        col = int(
            r["image_col_0based"]
        )

        spectrum425 = src.read(
            window=rasterio.windows.Window(
                col,
                row,
                1,
                1
            )
        ).reshape(
            src.count
        ).astype(
            np.float64
        )

        spectrum425[
            spectrum425 <= -9000
        ] = np.nan

        spectrum326 = spectrum425[
            GOOD_INDEX
        ]

        # ---------------------------
        # Wide 425-band
        # ---------------------------

        rec425 = {
            "subtype_code":
                int(
                    r["subtype_code"]
                ),

            "lithology_family":
                r["lithology_family"],

            "longitude":
                r["longitude"],

            "latitude":
                r["latitude"],

            "UTM_Easting_m":
                r["UTM_Easting_m"],

            "UTM_Northing_m":
                r["UTM_Northing_m"],
        }

        for i in range(425):

            rec425[
                f"B{i+1:03d}"
            ] = spectrum425[i]

        spectra425_rows.append(
            rec425
        )

        # ---------------------------
        # Wide 326-band
        # ---------------------------

        rec326 = {
            "subtype_code":
                int(
                    r["subtype_code"]
                ),

            "lithology_family":
                r["lithology_family"],

            "longitude":
                r["longitude"],

            "latitude":
                r["latitude"],

            "UTM_Easting_m":
                r["UTM_Easting_m"],

            "UTM_Northing_m":
                r["UTM_Northing_m"],
        }

        for j, original_band in enumerate(
            GOOD_BANDS
        ):

            rec326[
                f"B{original_band:03d}"
            ] = spectrum326[j]

        spectra326_rows.append(
            rec326
        )

        # ---------------------------
        # Long 425-band
        # ---------------------------

        for i in range(425):

            long425_rows.append({
                "subtype_code":
                    int(
                        r["subtype_code"]
                    ),

                "lithology_family":
                    r["lithology_family"],

                "band_1based":
                    i + 1,

                "wavelength_nm":
                    wl425[i],

                "reflectance":
                    spectrum425[i],

                "longitude":
                    r["longitude"],

                "latitude":
                    r["latitude"]
            })

        # ---------------------------
        # Long 326 scientific bands
        # ---------------------------

        for j, original_band in enumerate(
            GOOD_BANDS
        ):

            long326_rows.append({
                "subtype_code":
                    int(
                        r["subtype_code"]
                    ),

                "lithology_family":
                    r["lithology_family"],

                "scientific_band_index_1based":
                    j + 1,

                "original_425_band_1based":
                    original_band,

                "wavelength_nm":
                    wl326[j],

                "reflectance":
                    spectrum326[j],

                "longitude":
                    r["longitude"],

                "latitude":
                    r["latitude"]
            })

spectra425 = pd.DataFrame(
    spectra425_rows
)

spectra326 = pd.DataFrame(
    spectra326_rows
)

long425 = pd.DataFrame(
    long425_rows
)

long326 = pd.DataFrame(
    long326_rows
)

# ============================================================
# CREATE GEOPACKAGE POINTS
# ============================================================

valid_locations = locations[
    locations[
        "status"
    ] == "SELECTED_AVIRIS_ENDMEMBER"
].copy()

geometry = [
    Point(
        x,
        y
    )
    for x, y in zip(
        valid_locations[
            "UTM_Easting_m"
        ],
        valid_locations[
            "UTM_Northing_m"
        ]
    )
]

gdf = gpd.GeoDataFrame(
    valid_locations,
    geometry=geometry,
    crs=crs
)

GPKG = OUT / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.gpkg"
)

if GPKG.exists():
    GPKG.unlink()

gdf.to_file(
    GPKG,
    layer="Step15A_13_Lithology_Endmembers",
    driver="GPKG"
)

# ============================================================
# SAVE CSV
# ============================================================

LOCATION_CSV = OUT / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.csv"
)

SPEC425_CSV = OUT / (
    "AVIRIS_Step15A_13_Endmembers_425Band_Wide.csv"
)

SPEC326_CSV = OUT / (
    "AVIRIS_Step15A_13_Endmembers_326ScientificBand_Wide.csv"
)

LONG425_CSV = OUT / (
    "AVIRIS_Step15A_13_Endmembers_425Band_Long.csv"
)

LONG326_CSV = OUT / (
    "AVIRIS_Step15A_13_Endmembers_326ScientificBand_Long.csv"
)

locations.to_csv(
    LOCATION_CSV,
    index=False
)

spectra425.to_csv(
    SPEC425_CSV,
    index=False
)

spectra326.to_csv(
    SPEC326_CSV,
    index=False
)

long425.to_csv(
    LONG425_CSV,
    index=False
)

long326.to_csv(
    LONG326_CSV,
    index=False
)

# ============================================================
# WAVELENGTH TABLE
# ============================================================

wavelength_table = pd.DataFrame({
    "band_1based":
        np.arange(
            1,
            426
        ),

    "wavelength_nm":
        wl425,

    "scientific_good_band":
        [
            (i in GOOD_BANDS)
            for i in range(
                1,
                426
            )
        ]
})

wavelength_table.to_csv(
    OUT / (
        "AVIRIS_Step15A_425Band_Wavelength_Table.csv"
    ),
    index=False
)

# ============================================================
# EXCEL OUTPUT
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15A_13_Lithology_Endmembers.xlsx"
)

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    locations.to_excel(
        writer,
        sheet_name="Endmember_Locations",
        index=False
    )

    spectra326.to_excel(
        writer,
        sheet_name="326Band_Endmembers",
        index=False
    )

    long326.to_excel(
        writer,
        sheet_name="326Band_Long",
        index=False
    )

    wavelength_table.to_excel(
        writer,
        sheet_name="Wavelengths",
        index=False
    )

# ============================================================
# SETTINGS / INTERPRETATION
# ============================================================

settings = {
    "step":
        "Step15A_13_Lithology_AVIRIS_Endmembers",

    "source_subtype_raster":
        str(SUBTYPE_RASTER),

    "source_AVIRIS":
        str(AVIRIS),

    "endmember_selection":
        (
            "Best AVIRIS pixel within each Step14E subtype candidate class, "
            "ranked by broad-group probability, subtype similarity, "
            "and subtype margin where available."
        ),

    "full_spectral_bands":
        425,

    "scientific_good_bands":
        326,

    "important_warning":
        (
            "These are representative image-derived spectral endmembers "
            "for the current reference-guided lithology interpretation. "
            "They are not independent field-confirmed lithological truth. "
            "Chromitite endmember is not direct chromite/PGE detection."
        )
}

(
    OUT / "AVIRIS_Step15A_Settings.json"
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
print("STEP15A COMPLETE")
print("=" * 90)

print("\n13 LITHOLOGY AVIRIS-NG ENDMEMBER LOCATIONS:")

show_cols = [
    "subtype_code",
    "lithology_family",
    "image_row_1based",
    "image_col_1based",
    "UTM_Easting_m",
    "UTM_Northing_m",
    "longitude",
    "latitude",
    "broad_probability",
    "uncertainty",
    "subtype_similarity",
    "subtype_margin",
    "endmember_selection_score",
    "status"
]

print(
    locations[
        show_cols
    ].to_string(
        index=False
    )
)

print("\nSelected endmembers:",
      int(
          (
              locations["status"] ==
              "SELECTED_AVIRIS_ENDMEMBER"
          ).sum()
      ))

print("\nGeoPackage:")
print(GPKG)

print("\nArcGIS layer:")
print("Step15A_13_Lithology_Endmembers")

print("\nExcel:")
print(XLSX)

print("\n425-band spectra:")
print(SPEC425_CSV)

print("\n326 scientific-band spectra:")
print(SPEC326_CSV)

print("\nIMPORTANT:")
print(
    "Each location is the selected representative AVIRIS-NG pixel "
    "for that reference-guided lithology class."
)

print(
    "These image-derived endmembers should be validated against "
    "field geology and laboratory spectra before being described "
    "as definitive lithological endmembers."
)

print("\nPASS")
