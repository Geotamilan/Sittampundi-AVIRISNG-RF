from pathlib import Path
import csv
import json
import re
import shutil
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path.cwd()

SCIENCE_DIR = (
    ROOT
    / "03_Mosaic"
    / "Scientific_326Band_Subset"
)

SOURCE_DAT = (
    SCIENCE_DIR
    / "AVIRIS_NG_mosaic_science_326band_4m.dat"
)

SOURCE_HDR = (
    SCIENCE_DIR
    / "AVIRIS_NG_mosaic_science_326band_4m.hdr"
)


MNF_MODEL_DIR = (
    ROOT
    / "05_MNF"
    / "Step10B_MNF_Statistics"
)

MNF_MEAN_FILE = (
    MNF_MODEL_DIR
    / "MNF_mean_326.npy"
)

MNF_TRANSFORM_FILE = (
    MNF_MODEL_DIR
    / "MNF_transform_326x326.npy"
)

MNF_EIGENVALUES_FILE = (
    MNF_MODEL_DIR
    / "MNF_eigenvalues_326.npy"
)

MNF_MODEL_JSON = (
    MNF_MODEL_DIR
    / "MNF_model_metadata.json"
)

STEP10B_SUMMARY = (
    MNF_MODEL_DIR
    / "AVIRIS_Step10B_MNF_summary.csv"
)

MNF_EIGEN_CSV = (
    MNF_MODEL_DIR
    / "AVIRIS_MNF_eigenvalues_SNR.csv"
)


OUTDIR = (
    ROOT
    / "05_MNF"
    / "Step10C_MNF1_40"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DAT = (
    OUTDIR
    / "AVIRIS_NG_MNF1_40_4m.dat"
)

OUTPUT_HDR = (
    OUTDIR
    / "AVIRIS_NG_MNF1_40_4m.hdr"
)

COMPONENT_STATS_CSV = (
    OUTDIR
    / "AVIRIS_MNF1_40_component_statistics.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step10C_MNF1_40_summary.csv"
)

QC_CSV = (
    OUTDIR
    / "AVIRIS_Step10C_numerical_QC.csv"
)

QUICKLOOK_DIR = (
    OUTDIR
    / "Quicklooks"
)

QUICKLOOK_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_INPUT_BANDS = 326

OUTPUT_COMPONENTS = 40

# Keep row blocks intentionally small.
#
# 16 rows × 4620 pixels × 326 input bands is manageable
# on a Mac without loading a huge fraction of the cube.
ROW_CHUNK = 16

FORCE_OVERWRITE = False

OUTPUT_NODATA_FALLBACK = -9999.0


# Components we want immediately for visual QC.
QUICKLOOK_COMPONENTS = [
    1,
    2,
    3,
    5,
    10,
    15,
    19,
    20,
    25,
    30,
    35,
    40,
]


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

        text += replacement + "\n"

    return text


def format_envi_list(
    values,
    values_per_line=12
):

    values = list(
        values
    )

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
# START
# ============================================================

print("=" * 98)

print(
    "AVIRIS-NG STEP 10C — "
    "CREATE MNF1–MNF40 BLOCK-WISE"
)

print("=" * 98)

print("\nIMPORTANT:")

print(
    "The 326-band scientific cube will NOT be modified."
)

print(
    "Only MNF components 1–40 will be written."
)

print(
    "The approved Step 10B transform will be used exactly."
)

print(
    "No new noise estimation will occur."
)

print(
    "No spatial resampling will occur."
)

print(
    "No radiometric normalization will occur."
)


# ============================================================
# INPUT CHECK
# ============================================================

required = [
    SOURCE_DAT,
    SOURCE_HDR,
    MNF_MEAN_FILE,
    MNF_TRANSFORM_FILE,
    MNF_EIGENVALUES_FILE,
    STEP10B_SUMMARY,
    MNF_EIGEN_CSV,
]

for path in required:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# CHECK STEP 10B STATUS
# ============================================================

step10b = pd.read_csv(
    STEP10B_SUMMARY
)


if len(step10b) != 1:

    raise RuntimeError(
        "Unexpected Step 10B summary."
    )


status = str(
    step10b.iloc[0][
        "step10b_status"
    ]
).strip().upper()


if status != "PASS":

    raise RuntimeError(
        f"Step 10B status is {status}; "
        "MNF raster creation is not authorized."
    )


print(
    "\nStep 10B model status: PASS"
)


# ============================================================
# LOAD SAVED MNF MODEL
# ============================================================

mean_vector = np.load(
    MNF_MEAN_FILE
).astype(
    np.float64
)


full_transform = np.load(
    MNF_TRANSFORM_FILE
).astype(
    np.float64
)


mnf_eigenvalues = np.load(
    MNF_EIGENVALUES_FILE
).astype(
    np.float64
)


if mean_vector.shape != (
    EXPECTED_INPUT_BANDS,
):

    raise RuntimeError(
        "MNF mean vector shape is incorrect."
    )


if full_transform.shape != (
    EXPECTED_INPUT_BANDS,
    EXPECTED_INPUT_BANDS
):

    raise RuntimeError(
        "MNF transform matrix shape is incorrect."
    )


if mnf_eigenvalues.shape != (
    EXPECTED_INPUT_BANDS,
):

    raise RuntimeError(
        "MNF eigenvalue array shape is incorrect."
    )


# Only first 40 columns.
transform_40 = (
    full_transform[
        :,
        :OUTPUT_COMPONENTS
    ]
)


eigenvalues_40 = (
    mnf_eigenvalues[
        :OUTPUT_COMPONENTS
    ]
)


estimated_snr_40 = (
    eigenvalues_40
    -
    1.0
)


print("\nMODEL")

print("-" * 98)

print(
    "Input dimensions of transform :",
    full_transform.shape
)

print(
    "Output transform             :",
    transform_40.shape
)

print(
    f"MNF1 SNR  : "
    f"{estimated_snr_40[0]:.4f}"
)

print(
    f"MNF19 SNR : "
    f"{estimated_snr_40[18]:.4f}"
)

print(
    f"MNF40 SNR : "
    f"{estimated_snr_40[39]:.4f}"
)


# ============================================================
# SOURCE VALIDATION
# ============================================================

with rasterio.open(
    SOURCE_DAT
) as src:


    if src.count != EXPECTED_INPUT_BANDS:

        raise RuntimeError(
            f"Expected 326 source bands; "
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
            "Input scientific cube is not 4 m."
        )


    width = src.width
    height = src.height
    crs = src.crs
    transform = src.transform
    nodata = src.nodata
    dtype = src.dtypes[0]


if nodata is None:

    output_nodata = (
        OUTPUT_NODATA_FALLBACK
    )

else:

    output_nodata = float(
        nodata
    )


print("\nSOURCE")

print("-" * 98)

print(
    f"Dimensions : "
    f"{width} x {height}"
)

print(
    f"Bands      : "
    f"{EXPECTED_INPUT_BANDS}"
)

print(
    f"Resolution : "
    f"4 m"
)

print(
    f"CRS        : "
    f"{crs}"
)

print(
    f"Input dtype: "
    f"{dtype}"
)

print(
    f"NoData     : "
    f"{output_nodata}"
)


# ============================================================
# DISK SPACE CHECK
# ============================================================

expected_bytes = (
    width
    *
    height
    *
    OUTPUT_COMPONENTS
    *
    4
)


expected_gib = (
    expected_bytes
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


print("\n")
print("=" * 98)
print("DISK SPACE")
print("=" * 98)

print(
    f"\nEstimated MNF1–40 cube : "
    f"{expected_gib:.2f} GiB"
)

print(
    f"Available space        : "
    f"{free_gib:.2f} GiB"
)


if free_bytes < (
    expected_bytes
    *
    1.25
):

    raise RuntimeError(
        "Insufficient free disk space "
        "for safe MNF processing."
    )


# ============================================================
# OUTPUT PROTECTION
# ============================================================

if (
    OUTPUT_DAT.exists()
    and
    not FORCE_OVERWRITE
):

    raise RuntimeError(
        "\nMNF1–40 output already exists:\n"
        f"{OUTPUT_DAT}\n\n"
        "It was NOT overwritten."
    )


if FORCE_OVERWRITE:

    for path in [

        OUTPUT_DAT,
        OUTPUT_HDR,
        Path(
            str(OUTPUT_DAT)
            + ".aux.xml"
        ),
        COMPONENT_STATS_CSV,
        SUMMARY_CSV,
        QC_CSV,

    ]:

        if path.exists():

            path.unlink()


# ============================================================
# CREATE OUTPUT PROFILE
# ============================================================

profile = {

    "driver":
        "ENVI",

    "width":
        width,

    "height":
        height,

    "count":
        OUTPUT_COMPONENTS,

    "dtype":
        "float32",

    "crs":
        crs,

    "transform":
        transform,

    "nodata":
        output_nodata,

    "interleave":
        "bil",
}


# ============================================================
# STATISTIC ACCUMULATORS
# ============================================================

valid_counts = np.zeros(
    OUTPUT_COMPONENTS,
    dtype=np.int64
)


sum_values = np.zeros(
    OUTPUT_COMPONENTS,
    dtype=np.float64
)


sum_squares = np.zeros(
    OUTPUT_COMPONENTS,
    dtype=np.float64
)


minimum_values = np.full(
    OUTPUT_COMPONENTS,
    np.inf,
    dtype=np.float64
)


maximum_values = np.full(
    OUTPUT_COMPONENTS,
    -np.inf,
    dtype=np.float64
)


# ============================================================
# BLOCK-WISE MNF TRANSFORM
# ============================================================

print("\n")
print("=" * 98)
print("CREATING MNF1–40 CUBE")
print("=" * 98)


start_time = time.time()


with rasterio.open(
    SOURCE_DAT
) as src, \
     rasterio.open(
         OUTPUT_DAT,
         "w",
         **profile
     ) as dst:


    total_blocks = int(
        np.ceil(
            height
            /
            ROW_CHUNK
        )
    )


    for block_number, row_off in enumerate(
        range(
            0,
            height,
            ROW_CHUNK
        ),
        start=1
    ):


        rows = min(
            ROW_CHUNK,
            height - row_off
        )


        window = Window(

            col_off=0,

            row_off=row_off,

            width=width,

            height=rows,
        )


        # Shape:
        # 326 × rows × width
        cube = src.read(
            window=window
        ).astype(
            np.float64,
            copy=False
        )


        # Valid only if ALL 326 scientific bands are valid.
        valid = np.all(
            np.isfinite(
                cube
            ),
            axis=0
        )


        if nodata is not None:

            valid &= np.all(
                cube
                !=
                nodata,
                axis=0
            )


        valid_count = int(
            np.sum(
                valid
            )
        )


        output = np.full(

            (
                OUTPUT_COMPONENTS,
                rows,
                width
            ),

            output_nodata,

            dtype=np.float32
        )


        if valid_count > 0:


            # -----------------------------------------------
            # Extract valid pixel spectra:
            #
            # cube[:, valid] = 326 × N
            # transpose      = N × 326
            # -----------------------------------------------

            spectra = (
                cube[
                    :,
                    valid
                ]
                .T
            )


            # Center using the exact Step 10B saved mean.
            spectra -= (
                mean_vector
            )


            # -----------------------------------------------
            # Apply saved MNF model.
            #
            # N × 326 @ 326 × 40
            # =
            # N × 40
            # -----------------------------------------------

            scores = (
                spectra
                @
                transform_40
            )


            if not np.all(
                np.isfinite(
                    scores
                )
            ):

                raise RuntimeError(
                    f"Non-finite MNF scores "
                    f"detected in block {block_number}."
                )


            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

            valid_counts += (
                valid_count
            )


            sum_values += np.sum(
                scores,
                axis=0,
                dtype=np.float64
            )


            sum_squares += np.sum(
                scores ** 2,
                axis=0,
                dtype=np.float64
            )


            minimum_values = np.minimum(
                minimum_values,
                np.min(
                    scores,
                    axis=0
                )
            )


            maximum_values = np.maximum(
                maximum_values,
                np.max(
                    scores,
                    axis=0
                )
            )


            # -----------------------------------------------
            # Place scores into output image block.
            # -----------------------------------------------

            for component in range(
                OUTPUT_COMPONENTS
            ):


                component_array = np.full(
                    (
                        rows,
                        width
                    ),
                    output_nodata,
                    dtype=np.float32
                )


                component_array[
                    valid
                ] = (
                    scores[
                        :,
                        component
                    ]
                    .astype(
                        np.float32
                    )
                )


                output[
                    component
                ] = (
                    component_array
                )


        dst.write(
            output,
            window=window
        )


        # Release large arrays quickly.
        del cube
        del output


        elapsed = (
            time.time()
            -
            start_time
        )


        if (
            block_number == 1
            or
            block_number % 10 == 0
            or
            block_number
            ==
            total_blocks
        ):


            fraction = (
                block_number
                /
                total_blocks
            )


            if fraction > 0:

                estimated_total = (
                    elapsed
                    /
                    fraction
                )

                remaining = max(
                    0,
                    estimated_total
                    -
                    elapsed
                )

            else:

                remaining = np.nan


            print(
                f"Block "
                f"{block_number:3d}/"
                f"{total_blocks} "
                f"| rows "
                f"{row_off:4d}–"
                f"{row_off+rows-1:4d} "
                f"| valid="
                f"{valid_count:,} "
                f"| elapsed="
                f"{elapsed/60:.1f} min "
                f"| ETA="
                f"{remaining/60:.1f} min"
            )


# ============================================================
# FIND GENERATED HEADER
# ============================================================

if not OUTPUT_HDR.exists():

    alternate = Path(
        str(
            OUTPUT_DAT
        )
        +
        ".hdr"
    )


    if alternate.exists():

        OUTPUT_HDR = (
            alternate
        )

    else:

        raise RuntimeError(
            "MNF ENVI header was not generated."
        )


# ============================================================
# PATCH MNF HEADER
#
# MNF components are NOT wavelength bands.
#
# Therefore:
# DO NOT attach the original wavelengths/FWHM to this cube.
# ============================================================

header_text = (
    OUTPUT_HDR.read_text(
        encoding="utf-8",
        errors="ignore"
    )
)


description = (
    "{AVIRIS-NG MNF diagnostic cube; "
    "MNF components 1-40 generated from "
    "326-band scientific mosaic using "
    "Step 10B full-resolution local-noise model; "
    "source scientific cube unchanged}"
)


header_text = set_envi_field(
    header_text,
    "description",
    description
)


band_names = [
    f"MNF{component}"
    for component
    in range(
        1,
        OUTPUT_COMPONENTS + 1
    )
]


header_text = set_envi_field(

    header_text,

    "band names",

    format_envi_list(
        band_names,
        values_per_line=10
    )
)


header_text = set_envi_field(

    header_text,

    "bbl",

    format_envi_list(
        [1]
        *
        OUTPUT_COMPONENTS
    )
)


header_text = set_envi_field(

    header_text,

    "default bands",

    "{1, 2, 3}"
)


header_text = set_envi_field(

    header_text,

    "data ignore value",

    str(
        output_nodata
    )
)


header_text = set_envi_field(

    header_text,

    "source scientific cube",

    "{AVIRIS_NG_mosaic_science_326band_4m.dat}"
)


header_text = set_envi_field(

    header_text,

    "source scientific bands",

    "326"
)


header_text = set_envi_field(

    header_text,

    "mnf output components",

    "40"
)


header_text = set_envi_field(

    header_text,

    "mnf component range",

    "{1-40}"
)


header_text = set_envi_field(

    header_text,

    "mnf noise model",

    "{Step 10B local adjacent-pixel noise covariance}"
)


header_text = set_envi_field(

    header_text,

    "mnf transform equation",

    "{(pixel - mean) multiplied by saved MNF transform}"
)


header_text = set_envi_field(

    header_text,

    "mnf spatial resampling",

    "{none}"
)


header_text = set_envi_field(

    header_text,

    "mnf radiometric transformation",

    "{none beyond MNF linear transform}"
)


header_text = set_envi_field(

    header_text,

    "mnf eigenvalues",

    format_envi_list(
        [
            f"{value:.10g}"
            for value
            in eigenvalues_40
        ]
    )
)


header_text = set_envi_field(

    header_text,

    "mnf estimated snr",

    format_envi_list(
        [
            f"{value:.10g}"
            for value
            in estimated_snr_40
        ]
    )
)


OUTPUT_HDR.write_text(
    header_text,
    encoding="utf-8"
)


# ============================================================
# STRUCTURAL VALIDATION
# ============================================================

print("\n")
print("=" * 98)
print("STRUCTURAL VALIDATION")
print("=" * 98)


with rasterio.open(
    OUTPUT_DAT
) as dst:


    if dst.count != OUTPUT_COMPONENTS:

        raise RuntimeError(
            "MNF output does not contain 40 components."
        )


    if (
        dst.width != width
        or
        dst.height != height
    ):

        raise RuntimeError(
            "MNF output dimensions changed."
        )


    if dst.crs != crs:

        raise RuntimeError(
            "MNF output CRS changed."
        )


    if dst.transform != transform:

        raise RuntimeError(
            "MNF output transform changed."
        )


    if dst.nodata != output_nodata:

        raise RuntimeError(
            "MNF output NoData changed."
        )


print(
    "\n40 MNF components     : PASS"
)

print(
    "Dimensions unchanged : PASS"
)

print(
    "CRS unchanged        : PASS"
)

print(
    "4 m grid unchanged   : PASS"
)

print(
    "No spatial resampling: PASS"
)


# ============================================================
# FILE SIZE VALIDATION
# ============================================================

actual_bytes = (
    OUTPUT_DAT.stat().st_size
)


if actual_bytes != expected_bytes:

    raise RuntimeError(
        "\nUnexpected MNF binary size.\n"
        f"Expected: {expected_bytes:,}\n"
        f"Actual  : {actual_bytes:,}"
    )


actual_gib = (
    actual_bytes
    /
    (1024 ** 3)
)


print(
    f"\nOutput file size: "
    f"{actual_gib:.2f} GiB"
)


# ============================================================
# COMPONENT STATISTICS
# ============================================================

means = np.divide(

    sum_values,

    valid_counts,

    out=np.full(
        OUTPUT_COMPONENTS,
        np.nan
    ),

    where=
        valid_counts > 0
)


variance = np.divide(

    sum_squares,

    valid_counts,

    out=np.full(
        OUTPUT_COMPONENTS,
        np.nan
    ),

    where=
        valid_counts > 0

) - (
    means ** 2
)


variance = np.maximum(
    variance,
    0
)


stds = np.sqrt(
    variance
)


stats_df = pd.DataFrame({

    "MNF_component":
        np.arange(
            1,
            OUTPUT_COMPONENTS + 1
        ),

    "MNF_eigenvalue":
        eigenvalues_40,

    "estimated_SNR":
        estimated_snr_40,

    "valid_pixels":
        valid_counts,

    "mean":
        means,

    "standard_deviation":
        stds,

    "minimum":
        minimum_values,

    "maximum":
        maximum_values,
})


stats_df.to_csv(
    COMPONENT_STATS_CSV,
    index=False
)


# ============================================================
# NUMERICAL QC
#
# Independently recalculate selected MNF components from
# source spectra and compare against written Float32 values.
# ============================================================

print("\n")
print("=" * 98)
print("NUMERICAL TRANSFORM QC")
print("=" * 98)


test_components = [
    1,
    2,
    3,
    10,
    19,
    25,
    30,
    40,
]


qc_rows = []


# Try several locations until we find a valid 128x128 area.
candidate_centers = [

    (0.50, 0.50),
    (0.50, 0.35),
    (0.50, 0.65),
    (0.35, 0.50),
    (0.65, 0.50),
    (0.35, 0.35),
    (0.65, 0.65),
]


test_window = None


with rasterio.open(
    SOURCE_DAT
) as src:


    for row_fraction, col_fraction in candidate_centers:


        test_h = min(
            128,
            height
        )

        test_w = min(
            128,
            width
        )


        center_row = int(
            height
            *
            row_fraction
        )

        center_col = int(
            width
            *
            col_fraction
        )


        row0 = int(
            np.clip(
                center_row
                -
                test_h // 2,
                0,
                height - test_h
            )
        )


        col0 = int(
            np.clip(
                center_col
                -
                test_w // 2,
                0,
                width - test_w
            )
        )


        window = Window(
            col0,
            row0,
            test_w,
            test_h
        )


        source_test = src.read(
            window=window
        ).astype(
            np.float64
        )


        valid_test = np.all(
            np.isfinite(
                source_test
            ),
            axis=0
        )


        if nodata is not None:

            valid_test &= np.all(
                source_test
                !=
                nodata,
                axis=0
            )


        if np.sum(
            valid_test
        ) >= 500:


            test_window = (
                window,
                source_test,
                valid_test
            )

            break


if test_window is None:

    raise RuntimeError(
        "Could not find a suitable valid window "
        "for numerical QC."
    )


window, source_test, valid_test = (
    test_window
)


spectra_test = (
    source_test[
        :,
        valid_test
    ]
    .T
)


expected_scores = (
    (
        spectra_test
        -
        mean_vector
    )
    @
    transform_40
)


with rasterio.open(
    OUTPUT_DAT
) as dst:


    for component in test_components:


        written = dst.read(
            component,
            window=window
        ).astype(
            np.float64
        )


        written_valid = (
            written[
                valid_test
            ]
        )


        expected = (
            expected_scores[
                :,
                component - 1
            ]
        )


        absolute_error = np.abs(
            written_valid
            -
            expected
        )


        max_abs_error = float(
            np.max(
                absolute_error
            )
        )


        median_abs_error = float(
            np.median(
                absolute_error
            )
        )


        scale = max(
            1.0,
            float(
                np.max(
                    np.abs(
                        expected
                    )
                )
            )
        )


        relative_max_error = (
            max_abs_error
            /
            scale
        )


        qc_rows.append({

            "MNF_component":
                component,

            "test_pixels":
                len(
                    expected
                ),

            "maximum_absolute_error":
                max_abs_error,

            "median_absolute_error":
                median_abs_error,

            "relative_max_error":
                relative_max_error,
        })


        print(
            f"MNF{component:02d}: "
            f"max abs error="
            f"{max_abs_error:.6e} "
            f"| relative="
            f"{relative_max_error:.6e}"
        )


        # Float64 model -> Float32 output.
        # Relative error should remain tiny.
        if relative_max_error > 1e-5:

            raise RuntimeError(
                f"MNF{component} numerical QC failed."
            )


qc_df = pd.DataFrame(
    qc_rows
)


qc_df.to_csv(
    QC_CSV,
    index=False
)


print(
    "\nNumerical transform QC: PASS"
)


# ============================================================
# CREATE SELECTED COMPONENT QUICKLOOKS
#
# These are diagnostic only.
#
# They do NOT alter the scientific MNF cube.
# ============================================================

print("\n")
print("=" * 98)
print("CREATING MNF COMPONENT QUICKLOOKS")
print("=" * 98)


quicklook_rows = []


with rasterio.open(
    OUTPUT_DAT
) as src:


    for component in QUICKLOOK_COMPONENTS:


        array = src.read(
            component
        ).astype(
            np.float32
        )


        valid = np.isfinite(
            array
        )


        if src.nodata is not None:

            valid &= (
                array
                !=
                src.nodata
            )


        values = (
            array[
                valid
            ]
        )


        if values.size == 0:

            continue


        low, high = np.percentile(
            values,
            [
                2,
                98
            ]
        )


        display = np.zeros(
            array.shape,
            dtype=np.float32
        )


        if high > low:


            display[
                valid
            ] = np.clip(

                (
                    array[
                        valid
                    ]
                    -
                    low
                )

                /

                (
                    high
                    -
                    low
                ),

                0,
                1
            )


        png_path = (

            QUICKLOOK_DIR
            /
            f"MNF{component:02d}_quicklook.png"
        )


        plt.figure(
            figsize=(
                14,
                7
            )
        )


        plt.imshow(
            display,
            cmap="gray"
        )


        plt.title(

            f"AVIRIS-NG MNF{component}\n"
            f"Eigenvalue="
            f"{eigenvalues_40[component-1]:.3f}, "
            f"Estimated SNR="
            f"{estimated_snr_40[component-1]:.3f}"
        )


        plt.axis(
            "off"
        )


        plt.tight_layout()


        plt.savefig(
            png_path,
            dpi=180,
            bbox_inches="tight"
        )


        plt.close()


        quicklook_rows.append({

            "MNF_component":
                component,

            "eigenvalue":
                eigenvalues_40[
                    component - 1
                ],

            "estimated_SNR":
                estimated_snr_40[
                    component - 1
                ],

            "stretch_low_2percent":
                low,

            "stretch_high_98percent":
                high,

            "quicklook":
                str(
                    png_path
                ),
        })


        print(
            f"MNF{component:02d} quicklook created"
        )


pd.DataFrame(
    quicklook_rows
).to_csv(

    QUICKLOOK_DIR
    /
    "AVIRIS_MNF_quicklook_stretch_values.csv",

    index=False
)


# ============================================================
# ARCHIVE MODEL REFERENCES
# ============================================================

for source in [

    STEP10B_SUMMARY,
    MNF_EIGEN_CSV,
    MNF_MODEL_JSON,

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
# SUMMARY
# ============================================================

elapsed = (
    time.time()
    -
    start_time
)


summary = pd.DataFrame([{

    "source_cube":
        str(
            SOURCE_DAT
        ),

    "source_bands":
        EXPECTED_INPUT_BANDS,

    "output_cube":
        str(
            OUTPUT_DAT
        ),

    "output_components":
        OUTPUT_COMPONENTS,

    "first_component":
        1,

    "last_component":
        40,

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

    "dtype":
        "Float32",

    "interleave":
        "BIL",

    "nodata":
        output_nodata,

    "MNF1_eigenvalue":
        float(
            eigenvalues_40[
                0
            ]
        ),

    "MNF1_estimated_SNR":
        float(
            estimated_snr_40[
                0
            ]
        ),

    "MNF19_eigenvalue":
        float(
            eigenvalues_40[
                18
            ]
        ),

    "MNF19_estimated_SNR":
        float(
            estimated_snr_40[
                18
            ]
        ),

    "MNF40_eigenvalue":
        float(
            eigenvalues_40[
                39
            ]
        ),

    "MNF40_estimated_SNR":
        float(
            estimated_snr_40[
                39
            ]
        ),

    "output_size_GiB":
        actual_gib,

    "spatial_resampling":
        "none",

    "new_noise_estimation":
        "none",

    "step10b_transform_reused":
        True,

    "source_raster_modified":
        False,

    "numerical_transform_QC":
        "PASS",

    "quicklook_components":
        ",".join(
            str(x)
            for x
            in QUICKLOOK_COMPONENTS
        ),

    "processing_minutes":
        elapsed
        /
        60.0,

    "step10c_status":
        "PASS",
}])


summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 98)

print(
    "STEP 10C COMPLETE — MNF1–MNF40 CREATED"
)

print("=" * 98)


print(
    f"\nOutput:"
)

print(
    OUTPUT_DAT
)


print(
    "\nHeader:"
)

print(
    OUTPUT_HDR
)


print(
    "\nComponent statistics:"
)

print(
    COMPONENT_STATS_CSV
)


print(
    "\nNumerical QC:"
)

print(
    QC_CSV
)


print(
    "\nSummary:"
)

print(
    SUMMARY_CSV
)


print(
    "\nQuicklooks:"
)

print(
    QUICKLOOK_DIR
)


print(
    f"\nDimensions      : "
    f"{width} x {height}"
)


print(
    f"Components      : "
    f"{OUTPUT_COMPONENTS}"
)


print(
    f"File size       : "
    f"{actual_gib:.2f} GiB"
)


print(
    f"Processing time : "
    f"{elapsed/60:.1f} min"
)


print("\nKey components:")


for component in [
    1,
    5,
    10,
    15,
    19,
    20,
    25,
    30,
    35,
    40
]:

    print(
        f"  MNF{component:02d}: "
        f"eigenvalue="
        f"{eigenvalues_40[component-1]:.3f}, "
        f"SNR="
        f"{estimated_snr_40[component-1]:.3f}"
    )


print("\n")
print("=" * 98)
print("IMPORTANT")
print("=" * 98)


print(
    "\n1. The 326-band scientific master "
    "was NOT modified."
)


print(
    "2. The Step 10B noise model was "
    "NOT recalculated."
)


print(
    "3. Only MNF1–40 were generated."
)


print(
    "4. No spatial resampling occurred."
)


print(
    "5. MNF components do NOT represent "
    "individual wavelengths."
)


print(
    "6. The output header therefore does "
    "not assign wavelength/FWHM values "
    "to MNF components."
)


print(
    "7. Do NOT start PPI solely from "
    "the numerical SNR cutoff."
)


print(
    "8. First inspect MNF component imagery "
    "and determine the coherent-to-noise "
    "transition."
)


print(
    "\nNEXT:"
)

print(
    "Step 10D — MNF1–40 visual/structural QC "
    "and final component selection for PPI."
)
