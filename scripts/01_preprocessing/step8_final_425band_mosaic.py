from pathlib import Path
from contextlib import ExitStack
import csv
import re
import shutil
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.enums import ColorInterp
from scipy.ndimage import distance_transform_edt
import spectral as spy
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path.cwd()

INDIR = (
    ROOT
    / "03_Mosaic"
    / "Normalized_425Band_4m"
)

MASK_FILE = (
    ROOT
    / "03_Mosaic"
    / "Post_Normalization_QC"
    / "AVIRIS_FINAL_326band_scientific_mask.csv"
)

STEP7C_SUMMARY = (
    ROOT
    / "03_Mosaic"
    / "Post_Normalization_QC"
    / "AVIRIS_Step7C_pair_summary.csv"
)

COEFF_FILE = (
    ROOT
    / "03_Mosaic"
    / "Radiometric_Normalization_Model_v3_FINAL"
    / "AVIRIS_425band_FINAL_normalization_coefficients_TO_MASTER_v3.csv"
)

OUTDIR = (
    ROOT
    / "03_Mosaic"
    / "Final_425Band_Mosaic"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL OUTPUTS
# ============================================================

MOSAIC_DAT = (
    OUTDIR
    / "AVIRIS_NG_mosaic_master_425band_4m.dat"
)

MOSAIC_HDR = (
    OUTDIR
    / "AVIRIS_NG_mosaic_master_425band_4m.hdr"
)

COVERAGE_TIF = (
    OUTDIR
    / "AVIRIS_NG_mosaic_coverage_count.tif"
)

SCENE_INDEX_TIF = (
    OUTDIR
    / "AVIRIS_NG_mosaic_scene_index.tif"
)

SCENE_LEGEND_CSV = (
    OUTDIR
    / "AVIRIS_NG_mosaic_scene_index_legend.csv"
)

QC_CSV = (
    OUTDIR
    / "AVIRIS_NG_mosaic_QC.csv"
)

RGB_TIF = (
    OUTDIR
    / "AVIRIS_NG_mosaic_RGB_QC.tif"
)

RGB_PNG = (
    OUTDIR
    / "AVIRIS_NG_mosaic_RGB_QC.png"
)


# ============================================================
# SETTINGS
# ============================================================

MASTER_SCENE = "ang20160110t053932"

SCENES = [
    "ang20160110t053216",
    "ang20160110t053932",
    "ang20160110t054703",
    "ang20160110t055421",
]

SCENE_INDEX = {
    "ang20160110t053216": 1,
    "ang20160110t053932": 2,
    "ang20160110t054703": 3,
    "ang20160110t055421": 4,
}

EXPECTED_BANDS = 425

TARGET_RESOLUTION = 4.0

RGB_BANDS = [
    54,     # ~641.9 nm
    36,     # ~551.7 nm
    20,     # ~471.6 nm
]

# Linear distance weighting.
#
# 1.0 = normal distance feathering.
#
# >1 would make the higher-quality interior of each scene
# dominate more strongly.
WEIGHT_POWER = 1.0

FALLBACK_NODATA = -9999.0

FORCE_OVERWRITE = False


# ============================================================
# INPUT FILE HELPERS
# ============================================================

def scene_dat(scene):

    return (
        INDIR
        / f"{scene}_normalized_to_053932_4m.dat"
    )


def scene_hdr(scene):

    return (
        INDIR
        / f"{scene}_normalized_to_053932_4m.hdr"
    )


# ============================================================
# ENVI HEADER HELPERS
# ============================================================

def set_envi_field(text, key, value):

    escaped = re.escape(
        key
    )

    pattern = re.compile(
        rf"(?ims)^[ \t]*{escaped}[ \t]*="
        rf"[ \t]*(\{{.*?\}}|[^\r\n]*)"
    )

    replacement = (
        f"{key} = {value}"
    )

    if pattern.search(text):

        text = pattern.sub(
            replacement,
            text,
            count=1
        )

    else:

        if not text.endswith("\n"):
            text += "\n"

        text += (
            replacement
            + "\n"
        )

    return text


def format_envi_list(values):

    values = list(
        values
    )

    lines = []

    # 12 entries per text line keeps the header readable.
    for start in range(
        0,
        len(values),
        12
    ):

        part = values[
            start:start + 12
        ]

        lines.append(
            ", ".join(
                str(v)
                for v in part
            )
        )

    return (
        "{\n"
        + ",\n".join(lines)
        + "\n}"
    )


# ============================================================
# STRETCH FUNCTION FOR RGB QC
# ============================================================

def stretch_byte(
    array,
    valid
):

    output = np.zeros(
        array.shape,
        dtype=np.uint8
    )

    values = array[
        valid
    ]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:

        return output

    low, high = np.percentile(
        values,
        [2, 98]
    )

    if high <= low:

        return output

    scaled = (
        array - low
    ) / (
        high - low
    )

    scaled = np.clip(
        scaled,
        0,
        1
    )

    output = (
        scaled * 255
    ).astype(
        np.uint8
    )

    output[
        ~valid
    ] = 0

    return output


# ============================================================
# START
# ============================================================

print("=" * 96)

print(
    "AVIRIS-NG STEP 8 — "
    "FINAL 425-BAND MOSAIC"
)

print("=" * 96)

print(
    "\nINPUT:"
)

print(
    "Step 7B normalized 425-band cubes."
)

print(
    "\nMETHOD:"
)

print(
    "Distance-to-valid-edge weighted feathering."
)

print(
    "\nIMPORTANT:"
)

print(
    "No input raster will be modified."
)

print(
    "No spatial resampling will occur."
)

print(
    "No new radiometric normalization will occur."
)

print(
    "No additive correction will occur."
)

print(
    "No reflectance clipping will occur."
)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

if (
    MOSAIC_DAT.exists()
    and
    not FORCE_OVERWRITE
):

    raise RuntimeError(
        "\nFinal mosaic already exists:\n"
        f"{MOSAIC_DAT}\n\n"
        "Set FORCE_OVERWRITE=True only if you "
        "deliberately want to rebuild it."
    )


if FORCE_OVERWRITE:

    for path in [

        MOSAIC_DAT,
        MOSAIC_HDR,
        Path(
            str(MOSAIC_DAT)
            + ".aux.xml"
        ),
        COVERAGE_TIF,
        SCENE_INDEX_TIF,
        RGB_TIF,
        RGB_PNG,
        QC_CSV,
        SCENE_LEGEND_CSV,

    ]:

        if path.exists():

            path.unlink()


# ============================================================
# LOAD FINAL SCIENTIFIC MASK
# ============================================================

if not MASK_FILE.exists():

    raise FileNotFoundError(
        MASK_FILE
    )


mask_df = pd.read_csv(
    MASK_FILE
)


if len(mask_df) != EXPECTED_BANDS:

    raise RuntimeError(
        "Final scientific mask "
        "does not contain 425 bands."
    )


if (
    "final_recommended_for_scientific_analysis"
    not in mask_df.columns
):

    raise RuntimeError(
        "Final scientific mask column missing."
    )


final_science_mask = (
    mask_df[
        "final_recommended_for_scientific_analysis"
    ]
    .astype(bool)
    .to_numpy()
)


bbl = np.where(
    final_science_mask,
    1,
    0
).astype(int)


science_band_count = int(
    np.sum(
        bbl == 1
    )
)


excluded_band_count = int(
    np.sum(
        bbl == 0
    )
)


print(
    "\nFinal scientific bands:",
    science_band_count
)

print(
    "Excluded bands:",
    excluded_band_count
)


if science_band_count != 326:

    raise RuntimeError(
        "\nExpected final scientific mask "
        "to contain 326 bands."
    )


# ============================================================
# SOURCE VALIDATION
# ============================================================

print("\n")
print("=" * 96)
print("SOURCE DATA VALIDATION")
print("=" * 96)


scene_info = {}


for scene in SCENES:

    dat = scene_dat(
        scene
    )

    hdr = scene_hdr(
        scene
    )


    if not dat.exists():

        raise FileNotFoundError(
            dat
        )


    if not hdr.exists():

        raise FileNotFoundError(
            hdr
        )


    with rasterio.open(
        dat
    ) as src:


        if src.count != EXPECTED_BANDS:

            raise RuntimeError(
                f"{scene}: expected 425 bands, "
                f"found {src.count}."
            )


        if str(src.crs) != "EPSG:32643":

            raise RuntimeError(
                f"{scene}: unexpected CRS "
                f"{src.crs}"
            )


        # North-up only.
        if not (
            np.isclose(
                src.transform.a,
                TARGET_RESOLUTION
            )
            and
            np.isclose(
                src.transform.e,
                -TARGET_RESOLUTION
            )
            and
            np.isclose(
                src.transform.b,
                0.0
            )
            and
            np.isclose(
                src.transform.d,
                0.0
            )
        ):

            raise RuntimeError(
                f"{scene}: raster is not on the "
                "expected north-up 4 m grid."
            )


        scene_info[
            scene
        ] = {

            "dat":
                dat,

            "hdr":
                hdr,

            "width":
                src.width,

            "height":
                src.height,

            "crs":
                src.crs,

            "transform":
                src.transform,

            "bounds":
                src.bounds,

            "nodata":
                src.nodata,
        }


        print(
            f"\n{scene}"
        )

        print(
            f"  Dimensions : "
            f"{src.width} x "
            f"{src.height}"
        )

        print(
            f"  Bands      : "
            f"{src.count}"
        )

        print(
            f"  Resolution : "
            f"{src.res}"
        )

        print(
            f"  CRS        : "
            f"{src.crs}"
        )

        print(
            f"  Nodata     : "
            f"{src.nodata}"
        )

        print(
            f"  Transform  :"
        )

        print(
            src.transform
        )


# ============================================================
# COMMON CRS CHECK
# ============================================================

crs_values = {
    str(
        scene_info[
            scene
        ]["crs"]
    )
    for scene in SCENES
}


if len(
    crs_values
) != 1:

    raise RuntimeError(
        "Input CRS values differ."
    )


OUTPUT_CRS = (
    scene_info[
        MASTER_SCENE
    ]["crs"]
)


MASTER_NODATA = (
    scene_info[
        MASTER_SCENE
    ]["nodata"]
)


if MASTER_NODATA is None:

    OUTPUT_NODATA = (
        FALLBACK_NODATA
    )

else:

    OUTPUT_NODATA = float(
        MASTER_NODATA
    )


# ============================================================
# VERIFY COMMON PIXEL LATTICE
# ============================================================

master_transform = (
    scene_info[
        MASTER_SCENE
    ]["transform"]
)


for scene in SCENES:

    transform = (
        scene_info[
            scene
        ]["transform"]
    )


    dx = (
        transform.c
        -
        master_transform.c
    ) / TARGET_RESOLUTION


    dy = (
        master_transform.f
        -
        transform.f
    ) / TARGET_RESOLUTION


    if not np.isclose(
        dx,
        round(dx),
        atol=1e-6
    ):

        raise RuntimeError(
            f"{scene}: X grid phase mismatch."
        )


    if not np.isclose(
        dy,
        round(dy),
        atol=1e-6
    ):

        raise RuntimeError(
            f"{scene}: Y grid phase mismatch."
        )


print(
    "\nCommon 4 m pixel lattice: PASS"
)


# ============================================================
# MOSAIC GRID
# ============================================================

GLOBAL_LEFT = min(
    scene_info[
        scene
    ]["transform"].c
    for scene in SCENES
)


GLOBAL_TOP = max(
    scene_info[
        scene
    ]["transform"].f
    for scene in SCENES
)


# Calculate exact scene offsets.
for scene in SCENES:

    info = scene_info[
        scene
    ]


    col_float = (
        info[
            "transform"
        ].c
        -
        GLOBAL_LEFT
    ) / TARGET_RESOLUTION


    row_float = (
        GLOBAL_TOP
        -
        info[
            "transform"
        ].f
    ) / TARGET_RESOLUTION


    if not np.isclose(
        col_float,
        round(
            col_float
        ),
        atol=1e-6
    ):

        raise RuntimeError(
            f"{scene}: non-integer "
            "mosaic column offset."
        )


    if not np.isclose(
        row_float,
        round(
            row_float
        ),
        atol=1e-6
    ):

        raise RuntimeError(
            f"{scene}: non-integer "
            "mosaic row offset."
        )


    info[
        "col_off"
    ] = int(
        round(
            col_float
        )
    )


    info[
        "row_off"
    ] = int(
        round(
            row_float
        )
    )


MOSAIC_WIDTH = max(

    scene_info[
        scene
    ]["col_off"]
    +
    scene_info[
        scene
    ]["width"]

    for scene in SCENES
)


MOSAIC_HEIGHT = max(

    scene_info[
        scene
    ]["row_off"]
    +
    scene_info[
        scene
    ]["height"]

    for scene in SCENES
)


MOSAIC_TRANSFORM = Affine(

    TARGET_RESOLUTION,
    0.0,
    GLOBAL_LEFT,

    0.0,
    -TARGET_RESOLUTION,
    GLOBAL_TOP
)


GLOBAL_RIGHT = (
    GLOBAL_LEFT
    +
    MOSAIC_WIDTH
    *
    TARGET_RESOLUTION
)


GLOBAL_BOTTOM = (
    GLOBAL_TOP
    -
    MOSAIC_HEIGHT
    *
    TARGET_RESOLUTION
)


print("\n")
print("=" * 96)
print("FINAL MOSAIC GRID")
print("=" * 96)


print(
    f"\nWidth       : "
    f"{MOSAIC_WIDTH}"
)

print(
    f"Height      : "
    f"{MOSAIC_HEIGHT}"
)

print(
    f"Bands       : "
    f"{EXPECTED_BANDS}"
)

print(
    f"Resolution  : "
    f"{TARGET_RESOLUTION} m"
)

print(
    f"CRS         : "
    f"{OUTPUT_CRS}"
)

print(
    f"NoData      : "
    f"{OUTPUT_NODATA}"
)

print(
    "\nBounds:"
)

print(
    f"  Left   : {GLOBAL_LEFT}"
)

print(
    f"  Bottom : {GLOBAL_BOTTOM}"
)

print(
    f"  Right  : {GLOBAL_RIGHT}"
)

print(
    f"  Top    : {GLOBAL_TOP}"
)

print(
    "\nTransform:"
)

print(
    MOSAIC_TRANSFORM
)


# ============================================================
# STORAGE REQUIREMENT
# ============================================================

estimated_bytes = (

    MOSAIC_WIDTH
    *
    MOSAIC_HEIGHT
    *
    EXPECTED_BANDS
    *
    4
)


estimated_gib = (
    estimated_bytes
    /
    (1024 ** 3)
)


free_bytes = (
    shutil.disk_usage(
        OUTDIR
    ).free
)


free_gib = (
    free_bytes
    /
    (1024 ** 3)
)


print(
    f"\nEstimated mosaic size : "
    f"{estimated_gib:.2f} GiB"
)

print(
    f"Free disk space       : "
    f"{free_gib:.2f} GiB"
)


if free_bytes < (
    estimated_bytes
    *
    1.20
):

    raise RuntimeError(
        "Insufficient free disk space "
        "for safe Step 8 processing."
    )


# ============================================================
# BUILD SPATIAL VALIDITY + FEATHER WEIGHTS
#
# Band 54 (~641.9 nm) is used only to determine the
# spatial footprint.
#
# The same spatial weight is then used for ALL wavelengths.
# ============================================================

print("\n")
print("=" * 96)
print("BUILDING SPATIAL FEATHER WEIGHTS")
print("=" * 96)


coverage_count = np.zeros(
    (
        MOSAIC_HEIGHT,
        MOSAIC_WIDTH
    ),
    dtype=np.uint8
)


dominant_scene_index = np.zeros(
    (
        MOSAIC_HEIGHT,
        MOSAIC_WIDTH
    ),
    dtype=np.uint8
)


dominant_weight = np.zeros(
    (
        MOSAIC_HEIGHT,
        MOSAIC_WIDTH
    ),
    dtype=np.float32
)


for scene in SCENES:

    info = scene_info[
        scene
    ]


    print(
        f"\n{scene}"
    )


    with rasterio.open(
        info["dat"]
    ) as src:


        footprint_band = src.read(
            54
        ).astype(
            np.float32
        )


        valid = np.isfinite(
            footprint_band
        )


        if src.nodata is not None:

            valid &= (
                footprint_band
                !=
                src.nodata
            )


    # --------------------------------------------------------
    # Pad with False so even an entirely-valid rectangular
    # scene has a mathematically correct edge distance.
    # --------------------------------------------------------

    padded = np.pad(
        valid,
        pad_width=1,
        mode="constant",
        constant_values=False
    )


    distance = (
        distance_transform_edt(
            padded
        )[
            1:-1,
            1:-1
        ]
        .astype(
            np.float32
        )
    )


    distance[
        ~valid
    ] = 0.0


    if WEIGHT_POWER != 1.0:

        weight = (
            distance
            **
            WEIGHT_POWER
        ).astype(
            np.float32
        )

    else:

        weight = distance


    info[
        "base_valid"
    ] = valid


    info[
        "weight"
    ] = weight


    r0 = info[
        "row_off"
    ]

    c0 = info[
        "col_off"
    ]

    r1 = (
        r0
        +
        info[
            "height"
        ]
    )

    c1 = (
        c0
        +
        info[
            "width"
        ]
    )


    target_coverage = (
        coverage_count[
            r0:r1,
            c0:c1
        ]
    )


    target_coverage[
        valid
    ] += 1


    local_dominant_weight = (
        dominant_weight[
            r0:r1,
            c0:c1
        ]
    )


    local_dominant_index = (
        dominant_scene_index[
            r0:r1,
            c0:c1
        ]
    )


    update = (
        valid
        &
        (
            weight
            >
            local_dominant_weight
        )
    )


    local_dominant_weight[
        update
    ] = weight[
        update
    ]


    local_dominant_index[
        update
    ] = (
        SCENE_INDEX[
            scene
        ]
    )


    print(
        f"  Valid footprint pixels : "
        f"{np.sum(valid):,}"
    )


    print(
        f"  Maximum edge distance  : "
        f"{np.max(distance):.2f} pixels "
        f"= "
        f"{np.max(distance) * TARGET_RESOLUTION:.2f} m"
    )


# ============================================================
# COVERAGE STATISTICS
# ============================================================

mosaic_valid_pixels = int(
    np.sum(
        coverage_count > 0
    )
)


overlap_pixels = int(
    np.sum(
        coverage_count >= 2
    )
)


print("\n")
print("=" * 96)
print("SPATIAL COVERAGE")
print("=" * 96)


for count in range(
    0,
    int(
        np.max(
            coverage_count
        )
    ) + 1
):

    pixels = int(
        np.sum(
            coverage_count
            ==
            count
        )
    )

    area_km2 = (
        pixels
        *
        TARGET_RESOLUTION
        *
        TARGET_RESOLUTION
        /
        1_000_000.0
    )


    print(
        f"\nCoverage = {count}"
    )

    print(
        f"  Pixels : "
        f"{pixels:,}"
    )

    print(
        f"  Area   : "
        f"{area_km2:.3f} km²"
    )


print(
    f"\nTotal valid mosaic pixels : "
    f"{mosaic_valid_pixels:,}"
)

print(
    f"Overlap pixels (>=2)      : "
    f"{overlap_pixels:,}"
)


# ============================================================
# WRITE COVERAGE COUNT
# ============================================================

coverage_profile = {

    "driver":
        "GTiff",

    "width":
        MOSAIC_WIDTH,

    "height":
        MOSAIC_HEIGHT,

    "count":
        1,

    "dtype":
        "uint8",

    "crs":
        OUTPUT_CRS,

    "transform":
        MOSAIC_TRANSFORM,

    "nodata":
        0,

    "compress":
        "deflate",

    "tiled":
        True,
}


with rasterio.open(
    COVERAGE_TIF,
    "w",
    **coverage_profile
) as dst:

    dst.write(
        coverage_count,
        1
    )

    dst.set_band_description(
        1,
        "Number of valid normalized AVIRIS-NG scenes"
    )


# ============================================================
# WRITE DOMINANT SCENE INDEX
# ============================================================

with rasterio.open(
    SCENE_INDEX_TIF,
    "w",
    **coverage_profile
) as dst:

    dst.write(
        dominant_scene_index,
        1
    )

    dst.set_band_description(
        1,
        "Dominant contributor under distance-weighted feathering"
    )


# ============================================================
# WRITE SCENE INDEX LEGEND
# ============================================================

legend_rows = []


for scene in SCENES:

    legend_rows.append({

        "scene_index":
            SCENE_INDEX[
                scene
            ],

        "scene":
            scene,

        "role":
            (
                "radiometric master"
                if
                scene
                ==
                MASTER_SCENE
                else
                "normalized to master"
            ),
    })


pd.DataFrame(
    legend_rows
).to_csv(
    SCENE_LEGEND_CSV,
    index=False
)


# ============================================================
# CREATE FINAL 425-BAND MOSAIC
# ============================================================

print("\n")
print("=" * 96)
print("CREATING FINAL 425-BAND MOSAIC")
print("=" * 96)


profile = {

    "driver":
        "ENVI",

    "width":
        MOSAIC_WIDTH,

    "height":
        MOSAIC_HEIGHT,

    "count":
        EXPECTED_BANDS,

    "dtype":
        "float32",

    "crs":
        OUTPUT_CRS,

    "transform":
        MOSAIC_TRANSFORM,

    "nodata":
        OUTPUT_NODATA,

    "interleave":
        "bil",
}


start_time = time.time()


with ExitStack() as stack:

    sources = {

        scene:
            stack.enter_context(
                rasterio.open(
                    scene_info[
                        scene
                    ]["dat"]
                )
            )

        for scene in SCENES
    }


    dst = stack.enter_context(

        rasterio.open(
            MOSAIC_DAT,
            "w",
            **profile
        )
    )


    for band in range(
        1,
        EXPECTED_BANDS + 1
    ):


        numerator = np.zeros(
            (
                MOSAIC_HEIGHT,
                MOSAIC_WIDTH
            ),
            dtype=np.float64
        )


        denominator = np.zeros(
            (
                MOSAIC_HEIGHT,
                MOSAIC_WIDTH
            ),
            dtype=np.float64
        )


        for scene in SCENES:

            info = scene_info[
                scene
            ]

            src = sources[
                scene
            ]


            data = src.read(
                band
            ).astype(
                np.float32
            )


            valid = (
                info[
                    "base_valid"
                ]
                &
                np.isfinite(
                    data
                )
            )


            if src.nodata is not None:

                valid &= (
                    data
                    !=
                    src.nodata
                )


            if not np.any(
                valid
            ):

                continue


            weight = info[
                "weight"
            ]


            r0 = info[
                "row_off"
            ]

            c0 = info[
                "col_off"
            ]

            r1 = (
                r0
                +
                info[
                    "height"
                ]
            )

            c1 = (
                c0
                +
                info[
                    "width"
                ]
            )


            local_num = (
                numerator[
                    r0:r1,
                    c0:c1
                ]
            )


            local_den = (
                denominator[
                    r0:r1,
                    c0:c1
                ]
            )


            # Boolean indexing means only valid source pixels
            # contribute to numerator and denominator.
            local_num[
                valid
            ] += (
                data[
                    valid
                ].astype(
                    np.float64
                )
                *
                weight[
                    valid
                ].astype(
                    np.float64
                )
            )


            local_den[
                valid
            ] += (
                weight[
                    valid
                ].astype(
                    np.float64
                )
            )


        output = np.full(
            (
                MOSAIC_HEIGHT,
                MOSAIC_WIDTH
            ),
            OUTPUT_NODATA,
            dtype=np.float32
        )


        valid_output = (
            denominator
            >
            0
        )


        output[
            valid_output
        ] = (
            numerator[
                valid_output
            ]
            /
            denominator[
                valid_output
            ]
        ).astype(
            np.float32
        )


        dst.write(
            output,
            band
        )


        if (
            band == 1
            or
            band % 25 == 0
            or
            band == EXPECTED_BANDS
        ):

            elapsed = (
                time.time()
                -
                start_time
            )


            print(
                f"  Band "
                f"{band:3d}/425 complete "
                f"({elapsed/60:.1f} min)"
            )


# ============================================================
# VERIFY ENVI HEADER
# ============================================================

if not MOSAIC_HDR.exists():

    alternate_hdr = Path(
        str(
            MOSAIC_DAT
        )
        +
        ".hdr"
    )

    if alternate_hdr.exists():

        MOSAIC_HDR = (
            alternate_hdr
        )

    else:

        raise RuntimeError(
            "Final ENVI mosaic header "
            "was not generated."
        )


# ============================================================
# READ SPECTRAL METADATA FROM MASTER NORMALIZED HEADER
# ============================================================

master_header = scene_hdr(
    MASTER_SCENE
)


master_spectral = spy.open_image(
    str(
        master_header
    )
)


master_metadata = (
    master_spectral.metadata
)


# ============================================================
# PATCH FINAL MOSAIC HEADER
#
# Keep the spatial metadata written by Rasterio.
# Add only spectral/provenance metadata from the master.
# ============================================================

header_text = (
    MOSAIC_HDR.read_text(
        encoding="utf-8",
        errors="ignore"
    )
)


description = (
    "{AVIRIS-NG L2 final 425-band mosaic; "
    "registered 4 m normalized flight lines; "
    "distance-to-valid-edge weighted overlap feathering; "
    "radiometric master = "
    f"{MASTER_SCENE}; "
    "final recommended scientific bands = 326}"
)


header_text = set_envi_field(
    header_text,
    "description",
    description
)


# ------------------------------------------------------------
# Spectral lists
# ------------------------------------------------------------

for key in [
    "wavelength",
    "fwhm",
]:

    if key in master_metadata:

        values = (
            master_metadata[
                key
            ]
        )

        if len(
            values
        ) == EXPECTED_BANDS:

            header_text = set_envi_field(

                header_text,

                key,

                format_envi_list(
                    values
                )
            )


# ------------------------------------------------------------
# Optional source band names
# ------------------------------------------------------------

if (
    "band names"
    in
    master_metadata
):

    values = (
        master_metadata[
            "band names"
        ]
    )

    if (
        isinstance(
            values,
            list
        )
        and
        len(
            values
        )
        ==
        EXPECTED_BANDS
    ):

        header_text = set_envi_field(

            header_text,

            "band names",

            format_envi_list(
                values
            )
        )


# ------------------------------------------------------------
# Scalar spectral metadata
# ------------------------------------------------------------

for key in [

    "wavelength units",
    "sensor type",
    "reflectance scale factor",

]:

    if key in master_metadata:

        header_text = set_envi_field(

            header_text,

            key,

            str(
                master_metadata[
                    key
                ]
            )
        )


# ------------------------------------------------------------
# Final BBL
# ------------------------------------------------------------

header_text = set_envi_field(

    header_text,

    "bbl",

    format_envi_list(
        [
            int(x)
            for x
            in bbl
        ]
    )
)


# ------------------------------------------------------------
# Display defaults
# ------------------------------------------------------------

header_text = set_envi_field(

    header_text,

    "default bands",

    "{54, 36, 20}"
)


# ------------------------------------------------------------
# NoData
# ------------------------------------------------------------

header_text = set_envi_field(

    header_text,

    "data ignore value",

    str(
        OUTPUT_NODATA
    )
)


# ------------------------------------------------------------
# Mosaic provenance
# ------------------------------------------------------------

header_text = set_envi_field(

    header_text,

    "mosaic source scenes",

    "{"
    +
    ", ".join(
        SCENES
    )
    +
    "}"
)


header_text = set_envi_field(

    header_text,

    "mosaic radiometric master",

    MASTER_SCENE
)


header_text = set_envi_field(

    header_text,

    "mosaic blending method",

    "{distance-to-valid-edge weighted feathering}"
)


header_text = set_envi_field(

    header_text,

    "mosaic feather weight power",

    str(
        WEIGHT_POWER
    )
)


header_text = set_envi_field(

    header_text,

    "mosaic spatial resampling",

    "{none in Step 8}"
)


header_text = set_envi_field(

    header_text,

    "mosaic radiometric correction",

    "{none in Step 8; Step 7B normalized inputs used}"
)


header_text = set_envi_field(

    header_text,

    "scientific analysis bands",

    str(
        science_band_count
    )
)


header_text = set_envi_field(

    header_text,

    "excluded bands",

    str(
        excluded_band_count
    )
)


MOSAIC_HDR.write_text(
    header_text,
    encoding="utf-8"
)


# ============================================================
# FINAL RASTER VALIDATION
# ============================================================

print("\n")
print("=" * 96)
print("FINAL MOSAIC VALIDATION")
print("=" * 96)


with rasterio.open(
    MOSAIC_DAT
) as check:


    if (
        check.count
        != EXPECTED_BANDS
    ):

        raise RuntimeError(
            "Final mosaic does not "
            "contain 425 bands."
        )


    if (
        check.width
        != MOSAIC_WIDTH
        or
        check.height
        != MOSAIC_HEIGHT
    ):

        raise RuntimeError(
            "Final mosaic dimensions changed."
        )


    if (
        check.crs
        != OUTPUT_CRS
    ):

        raise RuntimeError(
            "Final mosaic CRS mismatch."
        )


    if not (
        np.isclose(
            check.res[0],
            TARGET_RESOLUTION
        )
        and
        np.isclose(
            check.res[1],
            TARGET_RESOLUTION
        )
    ):

        raise RuntimeError(
            "Final mosaic pixel size mismatch."
        )


# ============================================================
# SPECTRAL HEADER VALIDATION
# ============================================================

mosaic_spectral = spy.open_image(
    str(
        MOSAIC_HDR
    )
)


mosaic_metadata = (
    mosaic_spectral.metadata
)


if (
    "wavelength"
    not in
    mosaic_metadata
):

    raise RuntimeError(
        "Final mosaic wavelength metadata missing."
    )


output_wavelengths = np.asarray(
    mosaic_metadata[
        "wavelength"
    ],
    dtype=float
)


if len(
    output_wavelengths
) != EXPECTED_BANDS:

    raise RuntimeError(
        "Final mosaic wavelength count "
        "is not 425."
    )


if "bbl" not in mosaic_metadata:

    raise RuntimeError(
        "Final mosaic BBL missing."
    )


output_bbl = np.asarray(
    mosaic_metadata[
        "bbl"
    ],
    dtype=int
)


if len(
    output_bbl
) != EXPECTED_BANDS:

    raise RuntimeError(
        "Final mosaic BBL count "
        "is not 425."
    )


if int(
    np.sum(
        output_bbl
    )
) != 326:

    raise RuntimeError(
        "Final mosaic BBL does not "
        "contain 326 recommended bands."
    )


print(
    "\n425 bands           : PASS"
)

print(
    "425 wavelengths     : PASS"
)

print(
    "326-band BBL        : PASS"
)

print(
    "EPSG:32643          : PASS"
)

print(
    "4 m grid            : PASS"
)

print(
    "No Step-8 resampling: PASS"
)


# ============================================================
# CREATE FINAL RGB QUICKLOOK
# ============================================================

print("\n")
print("=" * 96)
print("CREATING FINAL RGB QC")
print("=" * 96)


with rasterio.open(
    MOSAIC_DAT
) as src:


    rgb_float = src.read(
        RGB_BANDS
    ).astype(
        np.float32
    )


    rgb_byte = np.zeros(
        rgb_float.shape,
        dtype=np.uint8
    )


    for channel in range(
        3
    ):


        valid = np.isfinite(
            rgb_float[
                channel
            ]
        )


        if src.nodata is not None:

            valid &= (
                rgb_float[
                    channel
                ]
                !=
                src.nodata
            )


        rgb_byte[
            channel
        ] = stretch_byte(

            rgb_float[
                channel
            ],

            valid
        )


    rgb_profile = {

        "driver":
            "GTiff",

        "width":
            src.width,

        "height":
            src.height,

        "count":
            3,

        "dtype":
            "uint8",

        "crs":
            src.crs,

        "transform":
            src.transform,

        "nodata":
            0,

        "compress":
            "deflate",

        "tiled":
            True,
    }


    with rasterio.open(
        RGB_TIF,
        "w",
        **rgb_profile
    ) as dst:


        dst.write(
            rgb_byte
        )


        dst.colorinterp = (

            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue
        )


    rgb_display = np.moveaxis(
        rgb_byte,
        0,
        -1
    )


    plt.figure(
        figsize=(15, 8)
    )


    plt.imshow(
        rgb_display
    )


    plt.title(
        "AVIRIS-NG Final 425-Band Mosaic\n"
        "RGB 641.9 / 551.7 / 471.6 nm"
    )


    plt.axis(
        "off"
    )


    plt.tight_layout()


    plt.savefig(
        RGB_PNG,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


print(
    RGB_PNG
)


# ============================================================
# FILE SIZE CHECK
# ============================================================

actual_bytes = (
    MOSAIC_DAT.stat().st_size
)


expected_bytes = (
    MOSAIC_WIDTH
    *
    MOSAIC_HEIGHT
    *
    EXPECTED_BANDS
    *
    4
)


if actual_bytes != expected_bytes:

    raise RuntimeError(
        "\nUnexpected ENVI binary size.\n"
        f"Expected: {expected_bytes:,}\n"
        f"Actual  : {actual_bytes:,}"
    )


actual_gib = (
    actual_bytes
    /
    (1024 ** 3)
)


# ============================================================
# FINAL QC CSV
# ============================================================

coverage_values = {}


for count in range(
    1,
    int(
        np.max(
            coverage_count
        )
    ) + 1
):

    coverage_values[
        f"pixels_with_{count}_scene_coverage"
    ] = int(
        np.sum(
            coverage_count
            ==
            count
        )
    )


qc_record = {

    "output":
        str(
            MOSAIC_DAT
        ),

    "width":
        MOSAIC_WIDTH,

    "height":
        MOSAIC_HEIGHT,

    "bands":
        EXPECTED_BANDS,

    "scientific_bands_bbl1":
        science_band_count,

    "excluded_bands_bbl0":
        excluded_band_count,

    "pixel_size_m":
        TARGET_RESOLUTION,

    "crs":
        str(
            OUTPUT_CRS
        ),

    "nodata":
        OUTPUT_NODATA,

    "interleave":
        "BIL",

    "dtype":
        "Float32",

    "mosaic_size_GiB":
        actual_gib,

    "valid_mosaic_pixels":
        mosaic_valid_pixels,

    "overlap_pixels_ge_2_scenes":
        overlap_pixels,

    "valid_area_km2":
        (
            mosaic_valid_pixels
            *
            TARGET_RESOLUTION
            *
            TARGET_RESOLUTION
            /
            1_000_000.0
        ),

    "overlap_area_km2":
        (
            overlap_pixels
            *
            TARGET_RESOLUTION
            *
            TARGET_RESOLUTION
            /
            1_000_000.0
        ),

    "blend_method":
        "distance-to-valid-edge weighted feathering",

    "weight_power":
        WEIGHT_POWER,

    "step8_spatial_resampling":
        "none",

    "step8_radiometric_correction":
        "none",

    "radiometric_master":
        MASTER_SCENE,

    "source_scene_count":
        len(
            SCENES
        ),

    "wavelength_min_nm":
        float(
            output_wavelengths.min()
        ),

    "wavelength_max_nm":
        float(
            output_wavelengths.max()
        ),

    "status":
        "STEP8_MOSAIC_CREATED_VALIDATION_PASS",
}


qc_record.update(
    coverage_values
)


pd.DataFrame(
    [
        qc_record
    ]
).to_csv(
    QC_CSV,
    index=False
)


# ============================================================
# ARCHIVE PROVENANCE FILES
# ============================================================

for source in [

    MASK_FILE,
    STEP7C_SUMMARY,
    COEFF_FILE,

]:

    if source.exists():

        destination = (
            OUTDIR
            /
            source.name
        )


        if (
            source.resolve()
            !=
            destination.resolve()
        ):

            shutil.copy2(
                source,
                destination
            )


# ============================================================
# FINAL REPORT
# ============================================================

elapsed_total = (
    time.time()
    -
    start_time
)


print("\n")
print("=" * 96)

print(
    "STEP 8 FINAL 425-BAND MOSAIC COMPLETE"
)

print("=" * 96)


print(
    f"\nProcessing time : "
    f"{elapsed_total/60:.1f} minutes"
)


print(
    f"\nDimensions      : "
    f"{MOSAIC_WIDTH} x "
    f"{MOSAIC_HEIGHT}"
)


print(
    f"Bands           : "
    f"{EXPECTED_BANDS}"
)


print(
    f"Scientific BBL=1: "
    f"{science_band_count}"
)


print(
    f"Excluded BBL=0 : "
    f"{excluded_band_count}"
)


print(
    f"Resolution      : "
    f"{TARGET_RESOLUTION} m"
)


print(
    f"CRS             : "
    f"{OUTPUT_CRS}"
)


print(
    f"File size       : "
    f"{actual_gib:.2f} GiB"
)


print(
    "\nFinal ENVI mosaic:"
)

print(
    MOSAIC_DAT
)


print(
    "\nHeader:"
)

print(
    MOSAIC_HDR
)


print(
    "\nCoverage count:"
)

print(
    COVERAGE_TIF
)


print(
    "\nDominant scene index:"
)

print(
    SCENE_INDEX_TIF
)


print(
    "\nRGB QC:"
)

print(
    RGB_PNG
)


print(
    "\nMosaic QC:"
)

print(
    QC_CSV
)


print("\n")
print("=" * 96)
print("IMPORTANT")
print("=" * 96)


print(
    "\n1. Original L2 cubes remain untouched."
)


print(
    "2. Registered cubes remain untouched."
)


print(
    "3. Step 7B normalized cubes remain untouched."
)


print(
    "4. All 425 spectral bands are retained."
)


print(
    "5. Final ENVI BBL contains "
    "326 recommended scientific bands."
)


print(
    "6. Distance weighting is spatial only "
    "and identical for all wavelengths."
)


print(
    "7. No spatial resampling occurred in Step 8."
)


print(
    "8. No additional radiometric correction occurred."
)


print(
    "9. No additive offset or clipping was applied."
)


print(
    "\nDO NOT start PCA/MNF yet."
)


print(
    "First visually and quantitatively inspect "
    "the final mosaic in Step 8B QC."
)
