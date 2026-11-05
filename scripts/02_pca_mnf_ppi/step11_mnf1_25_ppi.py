from pathlib import Path
import json
import re
import shutil
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

MNF_DIR = (
    ROOT
    / "05_MNF"
    / "Step10C_MNF1_40"
)

SOURCE_DAT = (
    MNF_DIR
    / "AVIRIS_NG_MNF1_40_4m.dat"
)

SOURCE_HDR = (
    MNF_DIR
    / "AVIRIS_NG_MNF1_40_4m.hdr"
)

STEP10C_SUMMARY = (
    MNF_DIR
    / "AVIRIS_Step10C_MNF1_40_summary.csv"
)

MNF_STATS = (
    MNF_DIR
    / "AVIRIS_MNF1_40_component_statistics.csv"
)

MNF_EIGEN_CSV = (
    ROOT
    / "05_MNF"
    / "Step10B_MNF_Statistics"
    / "AVIRIS_MNF_eigenvalues_SNR.csv"
)

COVERAGE_TIF = (
    ROOT
    / "03_Mosaic"
    / "Final_425Band_Mosaic"
    / "AVIRIS_NG_mosaic_coverage_count.tif"
)

OUTDIR = (
    ROOT
    / "06_PPI_Endmembers"
    / "Step11_MNF1_25_PPI"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUTS
# ============================================================

PPI_INPUT_DAT = (
    OUTDIR
    / "AVIRIS_NG_MNF1_25_PPI_input_4m.dat"
)

PPI_INPUT_HDR = (
    OUTDIR
    / "AVIRIS_NG_MNF1_25_PPI_input_4m.hdr"
)

MASK_TIF = (
    OUTDIR
    / "AVIRIS_MNF1_25_PPI_valid_edge_mask.tif"
)

EDGE_DISTANCE_TIF = (
    OUTDIR
    / "AVIRIS_PPI_distance_from_invalid_edge_pixels.tif"
)

PPI_COUNTS_TIF = (
    OUTDIR
    / "AVIRIS_MNF1_25_PPI_counts.tif"
)

ALL_CANDIDATES_CSV = (
    OUTDIR
    / "AVIRIS_PPI_all_candidates.csv"
)

THINNED_CANDIDATES_CSV = (
    OUTDIR
    / "AVIRIS_PPI_top300_spatially_thinned_candidates.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step11_PPI_summary.csv"
)

SETTINGS_JSON = (
    OUTDIR
    / "AVIRIS_Step11_PPI_settings.json"
)

MASK_PNG = (
    OUTDIR
    / "AVIRIS_PPI_valid_mask_quicklook.png"
)

PPI_PNG = (
    OUTDIR
    / "AVIRIS_PPI_count_quicklook.png"
)

TOP_PNG = (
    OUTDIR
    / "AVIRIS_PPI_top300_candidate_locations.png"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SOURCE_COMPONENTS = 40

PPI_COMPONENTS = 25

# ------------------------------------------------------------
# Edge protection
#
# 8 pixels × 4 m = 32 m inward buffer.
# ------------------------------------------------------------

EDGE_EROSION_PIXELS = 8


# ------------------------------------------------------------
# FULL-SCENE COARSE PPI
#
# Every masked pixel is evaluated.
#
# For each coarse random direction retain the strongest
# positive and negative K pixels.
# ------------------------------------------------------------

COARSE_SKEWERS = 128

COARSE_EXTREME_K = 40

COARSE_BLOCK_ROWS = 24


# ------------------------------------------------------------
# REFINED PPI
#
# Run many more skewers only on the candidate pool.
# ------------------------------------------------------------

FINE_SKEWERS = 10000

FINE_SKEWER_BATCH = 256

FINE_EXTREME_K = 3


# ------------------------------------------------------------
# Spatial thinning for later endmember inspection
# ------------------------------------------------------------

THIN_RADIUS_PIXELS = 3

THIN_MAX_CANDIDATES = 300


RANDOM_SEED_COARSE = 20260903

RANDOM_SEED_FINE = 20260904

FORCE_OVERWRITE = False

PPI_NODATA = 65535


# ============================================================
# HEADER HELPERS
# ============================================================

def set_envi_field(text, key, value):

    escaped = re.escape(key)

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

        text += replacement + "\n"

    return text


def format_envi_list(
    values,
    values_per_line=10
):

    values = list(values)

    lines = []

    for start in range(
        0,
        len(values),
        values_per_line
    ):

        part = values[
            start:start + values_per_line
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
# RANDOM UNIT SKEWERS
# ============================================================

def random_unit_skewers(
    dimensions,
    number,
    rng
):

    vectors = rng.standard_normal(
        (
            dimensions,
            number
        )
    ).astype(
        np.float32
    )

    norms = np.linalg.norm(
        vectors,
        axis=0
    )

    vectors /= norms[
        None,
        :
    ]

    return vectors


# ============================================================
# START
# ============================================================

print("=" * 100)

print(
    "AVIRIS-NG STEP 11 — "
    "MNF1–25 PPI + CONSERVATIVE EDGE MASK"
)

print("=" * 100)

print("\nIMPORTANT:")

print(
    "The 326-band scientific cube will NOT be modified."
)

print(
    "The MNF1–40 cube will NOT be modified."
)

print(
    "MNF1–25 are copied into a separate PPI input."
)

print(
    "The valid footprint is eroded by "
    f"{EDGE_EROSION_PIXELS} pixels "
    f"= {EDGE_EROSION_PIXELS*4} m."
)

print(
    "No thematic water/vegetation mask is applied yet."
)

print(
    "Water, vegetation, soil and rock endmembers are "
    "therefore allowed to compete naturally in this first PPI."
)


# ============================================================
# INPUT CHECK
# ============================================================

required = [
    SOURCE_DAT,
    SOURCE_HDR,
    STEP10C_SUMMARY,
    MNF_STATS,
    MNF_EIGEN_CSV,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# STEP 10C STATUS
# ============================================================

step10c = pd.read_csv(
    STEP10C_SUMMARY
)

status = str(
    step10c.iloc[0][
        "step10c_status"
    ]
).strip().upper()


if status != "PASS":

    raise RuntimeError(
        f"Step 10C status is {status}. "
        "PPI is not authorized."
    )


print(
    "\nStep 10C status: PASS"
)


# ============================================================
# LOAD MNF EIGEN/SNR TABLE
# ============================================================

mnf_eigen = pd.read_csv(
    MNF_EIGEN_CSV
)


if len(mnf_eigen) < PPI_COMPONENTS:

    raise RuntimeError(
        "MNF eigenvalue table is incomplete."
    )


ppi_eigenvalues = (
    mnf_eigen[
        "MNF_eigenvalue"
    ]
    .to_numpy(
        dtype=float
    )[
        :PPI_COMPONENTS
    ]
)


ppi_snr = (
    mnf_eigen[
        "estimated_SNR"
    ]
    .to_numpy(
        dtype=float
    )[
        :PPI_COMPONENTS
    ]
)


print("\nPPI COMPONENT RANGE")

print("-" * 100)

print(
    "MNF1–MNF25"
)

print(
    f"MNF1 estimated SNR  : "
    f"{ppi_snr[0]:.3f}"
)

print(
    f"MNF19 estimated SNR : "
    f"{ppi_snr[18]:.3f}"
)

print(
    f"MNF25 estimated SNR : "
    f"{ppi_snr[24]:.3f}"
)


# ============================================================
# SOURCE VALIDATION
# ============================================================

with rasterio.open(
    SOURCE_DAT
) as src:

    if src.count != EXPECTED_SOURCE_COMPONENTS:

        raise RuntimeError(
            f"Expected 40 MNF components, "
            f"found {src.count}."
        )

    if str(src.crs) != "EPSG:32643":

        raise RuntimeError(
            f"Unexpected CRS: {src.crs}"
        )

    if not (
        np.isclose(
            src.res[0],
            4.0
        )
        and
        np.isclose(
            src.res[1],
            4.0
        )
    ):

        raise RuntimeError(
            "MNF cube is not 4 m."
        )

    width = src.width
    height = src.height
    transform = src.transform
    crs = src.crs
    nodata = src.nodata


print("\nSOURCE")

print("-" * 100)

print(
    f"Dimensions : "
    f"{width} x {height}"
)

print(
    f"Components : "
    f"{EXPECTED_SOURCE_COMPONENTS}"
)

print(
    f"CRS        : "
    f"{crs}"
)

print(
    f"Resolution : "
    f"4 m"
)

print(
    f"NoData     : "
    f"{nodata}"
)


# ============================================================
# OUTPUT PROTECTION
# ============================================================

existing_outputs = [
    PPI_INPUT_DAT,
    PPI_INPUT_HDR,
    MASK_TIF,
    EDGE_DISTANCE_TIF,
    PPI_COUNTS_TIF,
    ALL_CANDIDATES_CSV,
    THINNED_CANDIDATES_CSV,
    SUMMARY_CSV,
]


if not FORCE_OVERWRITE:

    for path in existing_outputs:

        if path.exists():

            raise RuntimeError(
                "\nExisting Step 11 output found:\n"
                f"{path}\n\n"
                "Nothing was overwritten."
            )


else:

    for path in existing_outputs + [
        MASK_PNG,
        PPI_PNG,
        TOP_PNG,
        SETTINGS_JSON,
        Path(
            str(PPI_INPUT_DAT)
            + ".aux.xml"
        ),
    ]:

        if path.exists():

            path.unlink()


# ============================================================
# DISK SPACE
# ============================================================

ppi_input_bytes = (
    width
    *
    height
    *
    PPI_COMPONENTS
    *
    4
)


ppi_input_gib = (
    ppi_input_bytes
    /
    (1024 ** 3)
)


free_gib = (
    shutil.disk_usage(
        OUTDIR
    ).free
    /
    (1024 ** 3)
)


print("\n")
print("=" * 100)
print("DISK SPACE")
print("=" * 100)

print(
    f"\nMNF1–25 PPI cube : "
    f"{ppi_input_gib:.2f} GiB"
)

print(
    f"Free disk space  : "
    f"{free_gib:.2f} GiB"
)


if free_gib < (
    ppi_input_gib
    *
    1.35
):

    raise RuntimeError(
        "Insufficient disk space."
    )


# ============================================================
# CREATE MNF1–25 PPI INPUT
#
# Direct band copy only.
# ============================================================

print("\n")
print("=" * 100)

print(
    "CREATING MNF1–25 PPI INPUT"
)

print("=" * 100)


ppi_profile = {

    "driver":
        "ENVI",

    "width":
        width,

    "height":
        height,

    "count":
        PPI_COMPONENTS,

    "dtype":
        "float32",

    "crs":
        crs,

    "transform":
        transform,

    "nodata":
        nodata,

    "interleave":
        "bil",
}


copy_start = time.time()


with rasterio.open(
    SOURCE_DAT
) as src, \
     rasterio.open(
         PPI_INPUT_DAT,
         "w",
         **ppi_profile
     ) as dst:

    for band in range(
        1,
        PPI_COMPONENTS + 1
    ):

        dst.write(
            src.read(
                band
            ),
            band
        )

        if (
            band == 1
            or
            band % 5 == 0
            or
            band == PPI_COMPONENTS
        ):

            print(
                f"  MNF{band:02d}/25 copied"
            )


print(
    f"\nBand copy completed in "
    f"{(time.time()-copy_start)/60:.2f} min"
)


# ============================================================
# FIND GENERATED ENVI HEADER
# ============================================================

if not PPI_INPUT_HDR.exists():

    alt = Path(
        str(PPI_INPUT_DAT)
        +
        ".hdr"
    )

    if alt.exists():

        PPI_INPUT_HDR = alt

    else:

        raise RuntimeError(
            "PPI-input ENVI header missing."
        )


# ============================================================
# PATCH PPI INPUT HEADER
# ============================================================

header_text = (
    PPI_INPUT_HDR.read_text(
        encoding="utf-8",
        errors="ignore"
    )
)


header_text = set_envi_field(
    header_text,
    "description",
    (
        "{AVIRIS-NG primary PPI input; "
        "MNF1-MNF25 selected after Step 10D "
        "visual MNF quality assessment}"
    )
)


header_text = set_envi_field(
    header_text,
    "band names",
    format_envi_list(
        [
            f"MNF{x}"
            for x in range(
                1,
                PPI_COMPONENTS + 1
            )
        ]
    )
)


header_text = set_envi_field(
    header_text,
    "bbl",
    format_envi_list(
        [1]
        *
        PPI_COMPONENTS
    )
)


header_text = set_envi_field(
    header_text,
    "default bands",
    "{1, 2, 3}"
)


header_text = set_envi_field(
    header_text,
    "ppi source",
    "{AVIRIS_NG_MNF1_40_4m.dat}"
)


header_text = set_envi_field(
    header_text,
    "ppi selected mnf components",
    "{1-25}"
)


header_text = set_envi_field(
    header_text,
    "mnf eigenvalues",
    format_envi_list(
        [
            f"{x:.10g}"
            for x in ppi_eigenvalues
        ]
    )
)


header_text = set_envi_field(
    header_text,
    "mnf estimated snr",
    format_envi_list(
        [
            f"{x:.10g}"
            for x in ppi_snr
        ]
    )
)


header_text = set_envi_field(
    header_text,
    "ppi spatial resampling",
    "{none}"
)


PPI_INPUT_HDR.write_text(
    header_text,
    encoding="utf-8"
)


# ============================================================
# BUILD BASE VALID FOOTPRINT
# ============================================================

print("\n")
print("=" * 100)

print(
    "CREATING CONSERVATIVE PPI VALID MASK"
)

print("=" * 100)


with rasterio.open(
    PPI_INPUT_DAT
) as src:

    band1 = src.read(
        1
    )

    base_valid = np.isfinite(
        band1
    )

    if src.nodata is not None:

        base_valid &= (
            band1
            !=
            src.nodata
        )


# Also intersect the Step 8 coverage map when available.
if COVERAGE_TIF.exists():

    with rasterio.open(
        COVERAGE_TIF
    ) as cov:

        if (
            cov.width == width
            and
            cov.height == height
            and
            cov.transform == transform
        ):

            base_valid &= (
                cov.read(
                    1
                )
                >
                0
            )


base_valid_count = int(
    np.sum(
        base_valid
    )
)


# ============================================================
# DISTANCE FROM INVALID / EXTERNAL EDGE
# ============================================================

edge_distance = (
    distance_transform_edt(
        base_valid
    )
    .astype(
        np.float32
    )
)


ppi_mask = (
    base_valid
    &
    (
        edge_distance
        >
        EDGE_EROSION_PIXELS
    )
)


ppi_mask_count = int(
    np.sum(
        ppi_mask
    )
)


removed_edge_pixels = (
    base_valid_count
    -
    ppi_mask_count
)


print(
    f"\nOriginal valid footprint : "
    f"{base_valid_count:,} pixels"
)

print(
    f"PPI valid mask           : "
    f"{ppi_mask_count:,} pixels"
)

print(
    f"Edge pixels excluded     : "
    f"{removed_edge_pixels:,}"
)

print(
    f"Edge erosion             : "
    f"{EDGE_EROSION_PIXELS} pixels "
    f"= {EDGE_EROSION_PIXELS*4} m"
)


# ============================================================
# WRITE MASK
# ============================================================

mask_profile = {

    "driver":
        "GTiff",

    "width":
        width,

    "height":
        height,

    "count":
        1,

    "dtype":
        "uint8",

    "crs":
        crs,

    "transform":
        transform,

    "nodata":
        0,

    "compress":
        "deflate",

    "tiled":
        True,
}


with rasterio.open(
    MASK_TIF,
    "w",
    **mask_profile
) as dst:

    dst.write(
        ppi_mask.astype(
            np.uint8
        ),
        1
    )

    dst.set_band_description(
        1,
        "PPI valid pixels after 32 m edge erosion"
    )


distance_profile = dict(
    mask_profile
)

distance_profile[
    "dtype"
] = "float32"

distance_profile[
    "nodata"
] = 0.0


with rasterio.open(
    EDGE_DISTANCE_TIF,
    "w",
    **distance_profile
) as dst:

    dst.write(
        edge_distance,
        1
    )


# ============================================================
# MASK QUICKLOOK
# ============================================================

plt.figure(
    figsize=(15, 7)
)

plt.imshow(
    ppi_mask,
    cmap="gray"
)

plt.title(
    "AVIRIS-NG Step 11 PPI Valid Mask\n"
    "8-pixel / 32-m inward edge protection"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    MASK_PNG,
    dpi=180,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# COARSE FULL-SCENE PPI
# ============================================================

print("\n")
print("=" * 100)

print(
    "STAGE A — FULL-SCENE COARSE PPI"
)

print("=" * 100)


print(
    f"\nDimensions       : "
    f"{PPI_COMPONENTS}"
)

print(
    f"Random skewers   : "
    f"{COARSE_SKEWERS}"
)

print(
    f"Positive extrema : "
    f"{COARSE_EXTREME_K} / skewer"
)

print(
    f"Negative extrema : "
    f"{COARSE_EXTREME_K} / skewer"
)

print(
    f"Masked pixels    : "
    f"{ppi_mask_count:,}"
)


rng_coarse = np.random.default_rng(
    RANDOM_SEED_COARSE
)


coarse_skewers = random_unit_skewers(

    PPI_COMPONENTS,

    COARSE_SKEWERS,

    rng_coarse
)


top_scores = np.full(

    (
        COARSE_EXTREME_K,
        COARSE_SKEWERS
    ),

    -np.inf,

    dtype=np.float32
)


bottom_scores = np.full(

    (
        COARSE_EXTREME_K,
        COARSE_SKEWERS
    ),

    np.inf,

    dtype=np.float32
)


top_ids = np.full(

    (
        COARSE_EXTREME_K,
        COARSE_SKEWERS
    ),

    -1,

    dtype=np.int64
)


bottom_ids = np.full(

    (
        COARSE_EXTREME_K,
        COARSE_SKEWERS
    ),

    -1,

    dtype=np.int64
)


coarse_start = time.time()


total_blocks = int(
    np.ceil(
        height
        /
        COARSE_BLOCK_ROWS
    )
)


with rasterio.open(
    PPI_INPUT_DAT
) as src:

    for block_number, row_off in enumerate(

        range(
            0,
            height,
            COARSE_BLOCK_ROWS
        ),

        start=1
    ):

        rows = min(
            COARSE_BLOCK_ROWS,
            height - row_off
        )


        local_mask = (
            ppi_mask[
                row_off:
                row_off + rows,
                :
            ]
        )


        if not np.any(
            local_mask
        ):

            continue


        window = Window(
            0,
            row_off,
            width,
            rows
        )


        cube = src.read(
            window=window
        ).astype(
            np.float32,
            copy=False
        )


        valid = (
            local_mask
            &
            np.all(
                np.isfinite(
                    cube
                ),
                axis=0
            )
        )


        if src.nodata is not None:

            valid &= np.all(
                cube
                !=
                src.nodata,
                axis=0
            )


        if not np.any(
            valid
        ):

            continue


        # N x 25
        X = (
            cube[
                :,
                valid
            ]
            .T
        )


        local_rows, local_cols = np.where(
            valid
        )


        global_rows = (
            local_rows
            +
            row_off
        )


        linear_ids = (

            global_rows.astype(
                np.int64
            )
            *
            width

            +

            local_cols.astype(
                np.int64
            )
        )


        # N × 128 projection matrix.
        projections = (
            X
            @
            coarse_skewers
        )


        n_local = (
            projections.shape[0]
        )


        k_local = min(
            COARSE_EXTREME_K,
            n_local
        )


        # ----------------------------------------------------
        # Local positive extremes
        # ----------------------------------------------------

        top_local_index = np.argpartition(

            projections,

            n_local - k_local,

            axis=0

        )[
            n_local - k_local:
            ,
            :
        ]


        top_local_scores = np.take_along_axis(

            projections,

            top_local_index,

            axis=0
        )


        top_local_ids = (
            linear_ids[
                top_local_index
            ]
        )


        # ----------------------------------------------------
        # Local negative extremes
        # ----------------------------------------------------

        bottom_local_index = np.argpartition(

            projections,

            k_local - 1,

            axis=0

        )[
            :k_local,
            :
        ]


        bottom_local_scores = np.take_along_axis(

            projections,

            bottom_local_index,

            axis=0
        )


        bottom_local_ids = (
            linear_ids[
                bottom_local_index
            ]
        )


        # ----------------------------------------------------
        # Merge local top candidates with global top K
        # ----------------------------------------------------

        combined_scores = np.vstack(
            [
                top_scores,
                top_local_scores
            ]
        )


        combined_ids = np.vstack(
            [
                top_ids,
                top_local_ids
            ]
        )


        total_rows = (
            combined_scores.shape[0]
        )


        keep = np.argpartition(

            combined_scores,

            total_rows
            -
            COARSE_EXTREME_K,

            axis=0

        )[
            total_rows
            -
            COARSE_EXTREME_K:
            ,
            :
        ]


        top_scores = np.take_along_axis(
            combined_scores,
            keep,
            axis=0
        )


        top_ids = np.take_along_axis(
            combined_ids,
            keep,
            axis=0
        )


        # ----------------------------------------------------
        # Merge bottom candidates
        # ----------------------------------------------------

        combined_scores = np.vstack(
            [
                bottom_scores,
                bottom_local_scores
            ]
        )


        combined_ids = np.vstack(
            [
                bottom_ids,
                bottom_local_ids
            ]
        )


        keep = np.argpartition(

            combined_scores,

            COARSE_EXTREME_K - 1,

            axis=0

        )[
            :COARSE_EXTREME_K,
            :
        ]


        bottom_scores = np.take_along_axis(
            combined_scores,
            keep,
            axis=0
        )


        bottom_ids = np.take_along_axis(
            combined_ids,
            keep,
            axis=0
        )


        del cube
        del X
        del projections


        if (
            block_number == 1
            or
            block_number % 10 == 0
            or
            block_number == total_blocks
        ):

            elapsed = (
                time.time()
                -
                coarse_start
            )


            print(
                f"Block "
                f"{block_number:3d}/"
                f"{total_blocks} "
                f"| elapsed "
                f"{elapsed/60:.1f} min"
            )


# ============================================================
# BUILD UNIQUE CANDIDATE POOL
# ============================================================

all_coarse_ids = np.concatenate(

    [
        top_ids.ravel(),
        bottom_ids.ravel()
    ]
)


all_coarse_ids = (
    all_coarse_ids[
        all_coarse_ids >= 0
    ]
)


candidate_ids, coarse_votes = np.unique(

    all_coarse_ids,

    return_counts=True
)


candidate_count = int(
    len(
        candidate_ids
    )
)


print("\n")
print("=" * 100)

print(
    "COARSE PPI COMPLETE"
)

print("=" * 100)


print(
    f"\nUnique candidate pixels: "
    f"{candidate_count:,}"
)


if candidate_count < 100:

    raise RuntimeError(
        "PPI coarse stage produced too few candidates."
    )


# ============================================================
# CANDIDATE ROW / COL / COORDINATES
# ============================================================

candidate_rows = (
    candidate_ids
    //
    width
).astype(
    int
)


candidate_cols = (
    candidate_ids
    %
    width
).astype(
    int
)


xs, ys = rasterio.transform.xy(

    transform,

    candidate_rows,

    candidate_cols,

    offset="center"
)


coords = list(
    zip(
        xs,
        ys
    )
)


# ============================================================
# READ 25-D CANDIDATE SPECTRA
# ============================================================

print(
    "\nReading candidate MNF spectra..."
)


with rasterio.open(
    PPI_INPUT_DAT
) as src:

    samples = list(
        src.sample(

            coords,

            indexes=list(
                range(
                    1,
                    PPI_COMPONENTS + 1
                )
            )
        )
    )


candidate_matrix = np.asarray(
    samples,
    dtype=np.float32
)


if candidate_matrix.shape != (
    candidate_count,
    PPI_COMPONENTS
):

    raise RuntimeError(
        "Candidate spectral matrix shape is incorrect."
    )


if not np.all(
    np.isfinite(
        candidate_matrix
    )
):

    raise RuntimeError(
        "Non-finite PPI candidate values found."
    )


if nodata is not None:

    if np.any(
        candidate_matrix
        ==
        nodata
    ):

        raise RuntimeError(
            "NoData candidate reached refined PPI."
        )


# ============================================================
# REFINED PPI
# ============================================================

print("\n")
print("=" * 100)

print(
    "STAGE B — REFINED 10,000-SKEWER PPI"
)

print("=" * 100)


print(
    f"\nCandidate pool      : "
    f"{candidate_count:,}"
)

print(
    f"Random skewers      : "
    f"{FINE_SKEWERS:,}"
)

print(
    f"Top votes/skewer    : "
    f"{FINE_EXTREME_K}"
)

print(
    f"Bottom votes/skewer : "
    f"{FINE_EXTREME_K}"
)


fine_counts = np.zeros(
    candidate_count,
    dtype=np.uint32
)


rng_fine = np.random.default_rng(
    RANDOM_SEED_FINE
)


fine_start = time.time()


processed_skewers = 0


while processed_skewers < FINE_SKEWERS:

    batch = min(

        FINE_SKEWER_BATCH,

        FINE_SKEWERS
        -
        processed_skewers
    )


    skewers = random_unit_skewers(

        PPI_COMPONENTS,

        batch,

        rng_fine
    )


    projections = (
        candidate_matrix
        @
        skewers
    )


    n_candidates = (
        projections.shape[0]
    )


    k = min(
        FINE_EXTREME_K,
        n_candidates
    )


    # Positive extremes.
    top_idx = np.argpartition(

        projections,

        n_candidates - k,

        axis=0

    )[
        n_candidates-k:
        ,
        :
    ]


    # Negative extremes.
    bottom_idx = np.argpartition(

        projections,

        k - 1,

        axis=0

    )[
        :k,
        :
    ]


    np.add.at(
        fine_counts,
        top_idx.ravel(),
        1
    )


    np.add.at(
        fine_counts,
        bottom_idx.ravel(),
        1
    )


    processed_skewers += batch


    if (
        processed_skewers
        %
        1024
        ==
        0

        or

        processed_skewers
        ==
        FINE_SKEWERS
    ):

        print(
            f"  {processed_skewers:,}/"
            f"{FINE_SKEWERS:,} "
            f"skewers complete"
        )


print(
    f"\nRefined PPI completed in "
    f"{(time.time()-fine_start)/60:.2f} min"
)


# ============================================================
# BUILD CANDIDATE TABLE
# ============================================================

candidate_df = pd.DataFrame({

    "linear_pixel_id":
        candidate_ids,

    "row":
        candidate_rows,

    "col":
        candidate_cols,

    "easting_m":
        np.asarray(
            xs,
            dtype=float
        ),

    "northing_m":
        np.asarray(
            ys,
            dtype=float
        ),

    "coarse_PPI_votes":
        coarse_votes.astype(
            int
        ),

    "refined_PPI_count":
        fine_counts.astype(
            int
        ),
})


candidate_df = candidate_df.sort_values(

    [
        "refined_PPI_count",
        "coarse_PPI_votes"
    ],

    ascending=[
        False,
        False
    ]
).reset_index(
    drop=True
)


candidate_df.insert(

    0,

    "PPI_rank",

    np.arange(
        1,
        len(candidate_df) + 1
    )
)


candidate_df.to_csv(
    ALL_CANDIDATES_CSV,
    index=False
)


# ============================================================
# SPATIALLY THIN TOP CANDIDATES
#
# This avoids giving the next endmember stage hundreds of
# immediately adjacent pixels from one object.
# ============================================================

selected_rows = []

blocked = set()


for row in candidate_df.itertuples(
    index=False
):

    key = (
        int(
            row.row
        ),
        int(
            row.col
        )
    )


    if key in blocked:

        continue


    selected_rows.append(
        row
    )


    rr = int(
        row.row
    )

    cc = int(
        row.col
    )


    for dr in range(

        -THIN_RADIUS_PIXELS,

        THIN_RADIUS_PIXELS
        +
        1
    ):

        for dc in range(

            -THIN_RADIUS_PIXELS,

            THIN_RADIUS_PIXELS
            +
            1
        ):

            blocked.add(
                (
                    rr + dr,
                    cc + dc
                )
            )


    if (
        len(
            selected_rows
        )
        >=
        THIN_MAX_CANDIDATES
    ):

        break


thinned_df = pd.DataFrame(
    selected_rows
)


thinned_df.insert(

    0,

    "thinned_rank",

    np.arange(
        1,
        len(thinned_df) + 1
    )
)


thinned_df.to_csv(
    THINNED_CANDIDATES_CSV,
    index=False
)


print(
    f"\nSpatially thinned candidate list: "
    f"{len(thinned_df)} pixels"
)


# ============================================================
# CREATE PPI COUNT RASTER
#
# Valid non-candidate pixels = 0
# Candidate pixels = PPI count
# Outside PPI mask = 65535 NoData
# ============================================================

ppi_array = np.zeros(
    (
        height,
        width
    ),
    dtype=np.uint16
)


safe_counts = np.minimum(
    fine_counts,
    PPI_NODATA - 1
).astype(
    np.uint16
)


ppi_array[
    candidate_rows,
    candidate_cols
] = safe_counts


ppi_output = np.full(
    (
        height,
        width
    ),
    PPI_NODATA,
    dtype=np.uint16
)


ppi_output[
    ppi_mask
] = (
    ppi_array[
        ppi_mask
    ]
)


ppi_profile = {

    "driver":
        "GTiff",

    "width":
        width,

    "height":
        height,

    "count":
        1,

    "dtype":
        "uint16",

    "crs":
        crs,

    "transform":
        transform,

    "nodata":
        PPI_NODATA,

    "compress":
        "deflate",

    "tiled":
        True,
}


with rasterio.open(
    PPI_COUNTS_TIF,
    "w",
    **ppi_profile
) as dst:

    dst.write(
        ppi_output,
        1
    )

    dst.set_band_description(
        1,
        (
            "Two-stage MNF1-25 "
            "Pixel Purity Index count"
        )
    )


# ============================================================
# PPI QUICKLOOK
# ============================================================

display = np.full(
    (
        height,
        width
    ),
    np.nan,
    dtype=np.float32
)


display[
    ppi_mask
] = np.log1p(
    ppi_array[
        ppi_mask
    ].astype(
        np.float32
    )
)


plt.figure(
    figsize=(15, 7)
)

plt.imshow(
    display
)

plt.title(
    "AVIRIS-NG MNF1–25 Pixel Purity Index\n"
    "log(1 + refined PPI count)"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    PPI_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TOP-300 LOCATION QUICKLOOK
# ============================================================

plt.figure(
    figsize=(15, 7)
)

plt.imshow(
    ppi_mask,
    cmap="gray"
)

plt.scatter(

    thinned_df[
        "col"
    ],

    thinned_df[
        "row"
    ],

    s=8,

    alpha=0.7
)

plt.title(
    "AVIRIS-NG Spatially Thinned "
    "Top PPI Candidate Pixels"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.savefig(
    TOP_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# STATISTICS
# ============================================================

positive_candidate_count = int(
    np.sum(
        fine_counts > 0
    )
)


max_ppi_count = int(
    np.max(
        fine_counts
    )
)


median_positive_ppi = float(

    np.median(
        fine_counts[
            fine_counts > 0
        ]
    )

    if positive_candidate_count > 0

    else 0
)


p95_positive_ppi = float(

    np.percentile(

        fine_counts[
            fine_counts > 0
        ],

        95
    )

    if positive_candidate_count > 0

    else 0
)


p99_positive_ppi = float(

    np.percentile(

        fine_counts[
            fine_counts > 0
        ],

        99
    )

    if positive_candidate_count > 0

    else 0
)


# ============================================================
# SAVE SETTINGS
# ============================================================

settings = {

    "PPI_components":
        "MNF1-MNF25",

    "PPI_dimensions":
        PPI_COMPONENTS,

    "edge_erosion_pixels":
        EDGE_EROSION_PIXELS,

    "edge_erosion_m":
        EDGE_EROSION_PIXELS
        *
        4,

    "coarse_skewers":
        COARSE_SKEWERS,

    "coarse_extreme_K_each_side":
        COARSE_EXTREME_K,

    "coarse_random_seed":
        RANDOM_SEED_COARSE,

    "fine_skewers":
        FINE_SKEWERS,

    "fine_extreme_K_each_side":
        FINE_EXTREME_K,

    "fine_random_seed":
        RANDOM_SEED_FINE,

    "spatial_thinning_radius_pixels":
        THIN_RADIUS_PIXELS,

    "spatial_thinning_radius_m":
        THIN_RADIUS_PIXELS
        *
        4,

    "maximum_thinned_candidates":
        THIN_MAX_CANDIDATES,

    "PPI_method":
        (
            "full-scene coarse random-projection "
            "candidate screening followed by "
            "10,000-skewer refined PPI"
        ),

    "thematic_mask":
        "none",

    "source_modified":
        False,
}


with SETTINGS_JSON.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        settings,
        f,
        indent=2
    )


# ============================================================
# SUMMARY
# ============================================================

total_elapsed = (
    time.time()
    -
    copy_start
)


summary_df = pd.DataFrame([{

    "source_MNF_cube":
        str(
            SOURCE_DAT
        ),

    "PPI_input_cube":
        str(
            PPI_INPUT_DAT
        ),

    "PPI_components":
        "MNF1-MNF25",

    "PPI_dimensions":
        PPI_COMPONENTS,

    "width":
        width,

    "height":
        height,

    "pixel_size_m":
        4.0,

    "crs":
        str(
            crs
        ),

    "original_valid_pixels":
        base_valid_count,

    "PPI_mask_valid_pixels":
        ppi_mask_count,

    "edge_pixels_removed":
        removed_edge_pixels,

    "edge_erosion_pixels":
        EDGE_EROSION_PIXELS,

    "edge_erosion_m":
        EDGE_EROSION_PIXELS
        *
        4,

    "coarse_skewers":
        COARSE_SKEWERS,

    "coarse_extreme_K":
        COARSE_EXTREME_K,

    "unique_coarse_candidates":
        candidate_count,

    "fine_skewers":
        FINE_SKEWERS,

    "fine_extreme_K":
        FINE_EXTREME_K,

    "candidates_with_positive_PPI_count":
        positive_candidate_count,

    "maximum_refined_PPI_count":
        max_ppi_count,

    "median_positive_PPI_count":
        median_positive_ppi,

    "p95_positive_PPI_count":
        p95_positive_ppi,

    "p99_positive_PPI_count":
        p99_positive_ppi,

    "spatially_thinned_candidates":
        len(
            thinned_df
        ),

    "thematic_mask_applied":
        False,

    "source_MNF1_40_modified":
        False,

    "source_scientific_cube_modified":
        False,

    "step11_status":
        "PASS",
}])


summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 100)

print(
    "STEP 11 COMPLETE — MNF1–25 PPI"
)

print("=" * 100)


print(
    f"\nPPI mask pixels:"
)

print(
    f"  {ppi_mask_count:,}"
)


print(
    f"\nCoarse candidate pool:"
)

print(
    f"  {candidate_count:,}"
)


print(
    f"\nCandidates receiving refined votes:"
)

print(
    f"  {positive_candidate_count:,}"
)


print(
    f"\nMaximum refined PPI count:"
)

print(
    f"  {max_ppi_count}"
)


print(
    f"\nMedian positive PPI count:"
)

print(
    f"  {median_positive_ppi:.2f}"
)


print(
    f"\n95th percentile:"
)

print(
    f"  {p95_positive_ppi:.2f}"
)


print(
    f"\n99th percentile:"
)

print(
    f"  {p99_positive_ppi:.2f}"
)


print(
    f"\nSpatially thinned top candidates:"
)

print(
    f"  {len(thinned_df)}"
)


print("\nOutputs:")


print(
    f"\nPPI input:"
)

print(
    PPI_INPUT_DAT
)


print(
    f"\nPPI valid mask:"
)

print(
    MASK_TIF
)


print(
    f"\nPPI count raster:"
)

print(
    PPI_COUNTS_TIF
)


print(
    f"\nAll candidates:"
)

print(
    ALL_CANDIDATES_CSV
)


print(
    f"\nTop spatially thinned candidates:"
)

print(
    THINNED_CANDIDATES_CSV
)


print(
    f"\nPPI quicklook:"
)

print(
    PPI_PNG
)


print(
    f"\nTop-candidate map:"
)

print(
    TOP_PNG
)


print(
    f"\nSummary:"
)

print(
    SUMMARY_CSV
)


print("\n")
print("=" * 100)
print("IMPORTANT")
print("=" * 100)


print(
    "\n1. The 326-band reflectance master "
    "was NOT modified."
)

print(
    "2. The MNF1–40 diagnostic cube "
    "was NOT modified."
)

print(
    "3. Only MNF1–25 were used for "
    "the primary PPI."
)

print(
    "4. MNF35–40 striping components "
    "were excluded."
)

print(
    "5. An 8-pixel / 32-m inward edge "
    "mask was applied before PPI."
)

print(
    "6. No vegetation/water thematic "
    "mask was applied."
)

print(
    "7. PPI counts identify spectral extremes, "
    "not mineral identities."
)

print(
    "8. Mineral identification must return "
    "to the original 326-band reflectance "
    "spectra at these candidate coordinates."
)

print(
    "\nNEXT:"
)

print(
    "Step 12 — inspect/rank the PPI candidate "
    "population, extract original reflectance "
    "spectra, cluster/n-D candidate spectra, "
    "and derive candidate endmembers."
)
