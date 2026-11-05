from pathlib import Path
import csv
import re
import shutil
import time

import numpy as np
import pandas as pd
import rasterio
import spectral as spy


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path.cwd()

MASTER_DIR = (
    ROOT
    / "03_Mosaic"
    / "Final_425Band_Mosaic"
)

SOURCE_DAT = (
    MASTER_DIR
    / "AVIRIS_NG_mosaic_master_425band_4m.dat"
)

SOURCE_HDR = (
    MASTER_DIR
    / "AVIRIS_NG_mosaic_master_425band_4m.hdr"
)

MASK_FILE = (
    MASTER_DIR
    / "AVIRIS_FINAL_326band_scientific_mask.csv"
)

# Fallback location if the mask copy is not present
MASK_FALLBACK = (
    ROOT
    / "03_Mosaic"
    / "Post_Normalization_QC"
    / "AVIRIS_FINAL_326band_scientific_mask.csv"
)

OUTDIR = (
    ROOT
    / "03_Mosaic"
    / "Scientific_326Band_Subset"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FINAL OUTPUTS
# ============================================================

OUTPUT_DAT = (
    OUTDIR
    / "AVIRIS_NG_mosaic_science_326band_4m.dat"
)

OUTPUT_HDR = (
    OUTDIR
    / "AVIRIS_NG_mosaic_science_326band_4m.hdr"
)

BAND_MAP_CSV = (
    OUTDIR
    / "AVIRIS_326band_to_original_425band_mapping.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step9_326band_subset_summary.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_SOURCE_BANDS = 425
EXPECTED_OUTPUT_BANDS = 326

FORCE_OVERWRITE = False


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


def format_envi_list(values, values_per_line=12):

    values = list(values)

    lines = []

    for start in range(
        0,
        len(values),
        values_per_line
    ):

        section = values[
            start:start + values_per_line
        ]

        lines.append(
            ", ".join(
                str(v)
                for v in section
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

print("=" * 94)

print(
    "AVIRIS-NG STEP 9 — "
    "CREATE 326-BAND SCIENTIFIC SUBSET"
)

print("=" * 94)

print("\nSOURCE:")

print(
    SOURCE_DAT
)

print("\nIMPORTANT:")

print(
    "The 425-band archival master will NOT be modified."
)

print(
    "This step performs band selection only."
)

print(
    "No spatial resampling."
)

print(
    "No radiometric correction."
)

print(
    "No spectral smoothing."
)

print(
    "No reflectance clipping."
)

print(
    "No PCA or MNF is performed in Step 9."
)


# ============================================================
# INPUT VALIDATION
# ============================================================

if not SOURCE_DAT.exists():

    raise FileNotFoundError(
        SOURCE_DAT
    )

if not SOURCE_HDR.exists():

    raise FileNotFoundError(
        SOURCE_HDR
    )


if MASK_FILE.exists():

    mask_path = MASK_FILE

elif MASK_FALLBACK.exists():

    mask_path = MASK_FALLBACK

else:

    raise FileNotFoundError(
        "Final 326-band mask was not found."
    )


print("\nScientific mask:")

print(
    mask_path
)


# ============================================================
# LOAD FINAL BAND MASK
# ============================================================

mask_df = pd.read_csv(
    mask_path
)


if len(mask_df) != EXPECTED_SOURCE_BANDS:

    raise RuntimeError(
        f"Scientific mask contains "
        f"{len(mask_df)} rows; expected 425."
    )


required_columns = [
    "band",
    "wavelength_nm",
    "final_recommended_for_scientific_analysis",
]


for column in required_columns:

    if column not in mask_df.columns:

        raise RuntimeError(
            f"Missing mask column: {column}"
        )


science_mask = (
    mask_df[
        "final_recommended_for_scientific_analysis"
    ]
    .astype(bool)
    .to_numpy()
)


selected_original_bands = (
    mask_df.loc[
        science_mask,
        "band"
    ]
    .astype(int)
    .to_numpy()
)


selected_mask_wavelengths = (
    mask_df.loc[
        science_mask,
        "wavelength_nm"
    ]
    .astype(float)
    .to_numpy()
)


if (
    len(selected_original_bands)
    != EXPECTED_OUTPUT_BANDS
):

    raise RuntimeError(
        "\nExpected 326 scientific bands, "
        f"found {len(selected_original_bands)}."
    )


print(
    "\nSelected scientific bands:",
    len(selected_original_bands)
)

print(
    "Excluded bands:",
    EXPECTED_SOURCE_BANDS
    -
    len(selected_original_bands)
)

print(
    "\nFirst selected original band:",
    selected_original_bands[0]
)

print(
    "Last selected original band:",
    selected_original_bands[-1]
)

print(
    "\nScientific wavelength range:"
)

print(
    f"{selected_mask_wavelengths.min():.3f} "
    f"to "
    f"{selected_mask_wavelengths.max():.3f} nm"
)


# ============================================================
# LOAD SOURCE SPECTRAL METADATA
# ============================================================

source_spectral = spy.open_image(
    str(
        SOURCE_HDR
    )
)


source_metadata = (
    source_spectral.metadata
)


if "wavelength" not in source_metadata:

    raise RuntimeError(
        "Source wavelength metadata missing."
    )


source_wavelengths = np.asarray(
    source_metadata[
        "wavelength"
    ],
    dtype=float
)


if len(source_wavelengths) != EXPECTED_SOURCE_BANDS:

    raise RuntimeError(
        "Source wavelength count is not 425."
    )


# Check mask wavelengths against source header.
header_selected_wavelengths = (
    source_wavelengths[
        selected_original_bands - 1
    ]
)


if not np.allclose(
    header_selected_wavelengths,
    selected_mask_wavelengths,
    atol=0.01
):

    raise RuntimeError(
        "Scientific mask wavelengths do not "
        "match the 425-band master header."
    )


# ============================================================
# OPTIONAL FWHM
# ============================================================

selected_fwhm = None


if "fwhm" in source_metadata:

    source_fwhm = np.asarray(
        source_metadata[
            "fwhm"
        ],
        dtype=float
    )


    if len(source_fwhm) == EXPECTED_SOURCE_BANDS:

        selected_fwhm = (
            source_fwhm[
                selected_original_bands - 1
            ]
        )


# ============================================================
# SOURCE RASTER VALIDATION
# ============================================================

with rasterio.open(
    SOURCE_DAT
) as src:


    if src.count != EXPECTED_SOURCE_BANDS:

        raise RuntimeError(
            f"Source raster contains "
            f"{src.count} bands, not 425."
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
            "Source mosaic is not 4 m."
        )


    source_width = src.width
    source_height = src.height
    source_transform = src.transform
    source_crs = src.crs
    source_nodata = src.nodata
    source_dtype = src.dtypes[0]


    print("\nSOURCE MASTER VALIDATION")

    print("-" * 94)

    print(
        "Dimensions :",
        source_width,
        "x",
        source_height
    )

    print(
        "Bands      :",
        src.count
    )

    print(
        "CRS        :",
        source_crs
    )

    print(
        "Resolution :",
        src.res
    )

    print(
        "NoData     :",
        source_nodata
    )

    print(
        "Data type  :",
        source_dtype
    )

    print(
        "Transform:"
    )

    print(
        source_transform
    )


# ============================================================
# DISK SPACE CHECK
# ============================================================

estimated_bytes = (
    source_width
    *
    source_height
    *
    EXPECTED_OUTPUT_BANDS
    *
    4
)


estimated_gib = (
    estimated_bytes
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
print("=" * 94)
print("DISK SPACE")
print("=" * 94)

print(
    f"\nEstimated 326-band cube: "
    f"{estimated_gib:.2f} GiB"
)

print(
    f"Free space            : "
    f"{free_gib:.2f} GiB"
)


if free_gib < (
    estimated_gib * 1.20
):

    raise RuntimeError(
        "Insufficient free disk space."
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
        "\n326-band scientific cube already exists:\n"
        f"{OUTPUT_DAT}\n\n"
        "The file was NOT overwritten."
    )


if FORCE_OVERWRITE:

    for path in [

        OUTPUT_DAT,
        OUTPUT_HDR,
        Path(
            str(OUTPUT_DAT)
            + ".aux.xml"
        ),

    ]:

        if path.exists():

            path.unlink()


# ============================================================
# CREATE 326-BAND ENVI CUBE
#
# IMPORTANT:
#
# Every output band is copied directly from the corresponding
# original band in the final 425-band mosaic.
#
# No numerical operation is performed.
# ============================================================

print("\n")
print("=" * 94)

print(
    "CREATING 326-BAND SCIENTIFIC CUBE"
)

print("=" * 94)


profile = {

    "driver":
        "ENVI",

    "width":
        source_width,

    "height":
        source_height,

    "count":
        EXPECTED_OUTPUT_BANDS,

    "dtype":
        "float32",

    "crs":
        source_crs,

    "transform":
        source_transform,

    "nodata":
        source_nodata,

    "interleave":
        "bil",
}


start_time = time.time()


with rasterio.open(
    SOURCE_DAT
) as src, \
     rasterio.open(
         OUTPUT_DAT,
         "w",
         **profile
     ) as dst:


    for output_band, original_band in enumerate(
        selected_original_bands,
        start=1
    ):


        data = src.read(
            int(
                original_band
            )
        )


        dst.write(
            data,
            output_band
        )


        if (
            output_band == 1
            or
            output_band % 25 == 0
            or
            output_band
            ==
            EXPECTED_OUTPUT_BANDS
        ):

            elapsed = (
                time.time()
                -
                start_time
            )


            wavelength_nm = (
                source_wavelengths[
                    original_band - 1
                ]
            )


            print(
                f"  Output band "
                f"{output_band:3d}/326 "
                f"<- original band "
                f"{original_band:3d} "
                f"({wavelength_nm:.2f} nm) "
                f"complete "
                f"({elapsed/60:.1f} min)"
            )


# ============================================================
# FIND GENERATED HEADER
# ============================================================

if not OUTPUT_HDR.exists():

    alternate_hdr = Path(
        str(OUTPUT_DAT)
        + ".hdr"
    )


    if alternate_hdr.exists():

        OUTPUT_HDR = (
            alternate_hdr
        )

    else:

        raise RuntimeError(
            "Output ENVI header not generated."
        )


# ============================================================
# PATCH 326-BAND HEADER
#
# Keep spatial metadata written by Rasterio.
# Replace/add only spectral and provenance metadata.
# ============================================================

header_text = (
    OUTPUT_HDR.read_text(
        encoding="utf-8",
        errors="ignore"
    )
)


description = (
    "{AVIRIS-NG final scientific subset; "
    "326 recommended bands selected from "
    "AVIRIS_NG_mosaic_master_425band_4m; "
    "for PCA, MNF, PPI, endmember extraction "
    "and mineral mapping; "
    "no resampling or spectral transformation "
    "performed during subset creation}"
)


header_text = set_envi_field(
    header_text,
    "description",
    description
)


# ============================================================
# SELECTED WAVELENGTHS
# ============================================================

header_text = set_envi_field(

    header_text,

    "wavelength",

    format_envi_list(
        [
            f"{value:.6f}"
            for value
            in header_selected_wavelengths
        ]
    )
)


# ============================================================
# FWHM
# ============================================================

if selected_fwhm is not None:

    header_text = set_envi_field(

        header_text,

        "fwhm",

        format_envi_list(
            [
                f"{value:.6f}"
                for value
                in selected_fwhm
            ]
        )
    )


# ============================================================
# ALL BANDS IN THIS SUBSET ARE SCIENTIFICALLY ACCEPTED
# ============================================================

header_text = set_envi_field(

    header_text,

    "bbl",

    format_envi_list(
        [1]
        *
        EXPECTED_OUTPUT_BANDS
    )
)


# ============================================================
# ORIGINAL AVIRIS BAND NUMBER MAPPING
# ============================================================

header_text = set_envi_field(

    header_text,

    "original 425-band indices",

    format_envi_list(
        [
            int(x)
            for x
            in selected_original_bands
        ]
    )
)


# ============================================================
# BAND NAMES
# ============================================================

band_names = [

    (
        f"ScienceBand_{new_band:03d}_"
        f"OrigBand_{original_band:03d}_"
        f"{source_wavelengths[original_band-1]:.2f}nm"
    )

    for new_band, original_band
    in enumerate(
        selected_original_bands,
        start=1
    )
]


header_text = set_envi_field(

    header_text,

    "band names",

    format_envi_list(
        band_names,
        values_per_line=4
    )
)


# ============================================================
# SCALAR SPECTRAL METADATA
# ============================================================

for key in [

    "wavelength units",
    "sensor type",
    "reflectance scale factor",

]:

    if key in source_metadata:

        header_text = set_envi_field(

            header_text,

            key,

            str(
                source_metadata[
                    key
                ]
            )
        )


# ============================================================
# DATA IGNORE VALUE
# ============================================================

if source_nodata is not None:

    header_text = set_envi_field(

        header_text,

        "data ignore value",

        str(
            source_nodata
        )
    )


# ============================================================
# PROVENANCE
# ============================================================

header_text = set_envi_field(

    header_text,

    "source archival master",

    "{AVIRIS_NG_mosaic_master_425band_4m.dat}"
)


header_text = set_envi_field(

    header_text,

    "source band count",

    "425"
)


header_text = set_envi_field(

    header_text,

    "scientific subset band count",

    "326"
)


header_text = set_envi_field(

    header_text,

    "subset creation method",

    "{direct band selection using final Step 7C scientific mask}"
)


header_text = set_envi_field(

    header_text,

    "subset spatial resampling",

    "{none}"
)


header_text = set_envi_field(

    header_text,

    "subset spectral transformation",

    "{none}"
)


header_text = set_envi_field(

    header_text,

    "subset radiometric transformation",

    "{none}"
)


OUTPUT_HDR.write_text(
    header_text,
    encoding="utf-8"
)


# ============================================================
# BAND MAPPING CSV
# ============================================================

mapping_rows = []


for new_band, original_band in enumerate(
    selected_original_bands,
    start=1
):


    row = {

        "science_band":
            new_band,

        "original_425_band":
            int(
                original_band
            ),

        "wavelength_nm":
            float(
                source_wavelengths[
                    original_band - 1
                ]
            ),

        "scientific_bbl":
            1,
    }


    if selected_fwhm is not None:

        row[
            "fwhm_nm"
        ] = float(
            selected_fwhm[
                new_band - 1
            ]
        )


    mapping_rows.append(
        row
    )


mapping_df = pd.DataFrame(
    mapping_rows
)


mapping_df.to_csv(
    BAND_MAP_CSV,
    index=False
)


# ============================================================
# OUTPUT STRUCTURAL VALIDATION
# ============================================================

print("\n")
print("=" * 94)
print("OUTPUT VALIDATION")
print("=" * 94)


with rasterio.open(
    OUTPUT_DAT
) as dst:


    if dst.count != EXPECTED_OUTPUT_BANDS:

        raise RuntimeError(
            f"Output has {dst.count} bands."
        )


    if (
        dst.width != source_width
        or
        dst.height != source_height
    ):

        raise RuntimeError(
            "Output dimensions changed."
        )


    if dst.transform != source_transform:

        raise RuntimeError(
            "Output transform changed."
        )


    if dst.crs != source_crs:

        raise RuntimeError(
            "Output CRS changed."
        )


    if dst.nodata != source_nodata:

        raise RuntimeError(
            "Output NoData changed."
        )


print(
    "\n326 bands             : PASS"
)

print(
    "Dimensions unchanged : PASS"
)

print(
    "Transform unchanged  : PASS"
)

print(
    "CRS unchanged        : PASS"
)

print(
    "NoData unchanged     : PASS"
)


# ============================================================
# HEADER VALIDATION
# ============================================================

output_spectral = spy.open_image(
    str(
        OUTPUT_HDR
    )
)


output_metadata = (
    output_spectral.metadata
)


if "wavelength" not in output_metadata:

    raise RuntimeError(
        "Output wavelength metadata missing."
    )


output_wavelengths = np.asarray(
    output_metadata[
        "wavelength"
    ],
    dtype=float
)


if len(output_wavelengths) != 326:

    raise RuntimeError(
        "Output wavelength count is not 326."
    )


if not np.allclose(
    output_wavelengths,
    header_selected_wavelengths,
    atol=0.001
):

    raise RuntimeError(
        "Output wavelength sequence changed."
    )


if "bbl" not in output_metadata:

    raise RuntimeError(
        "Output BBL missing."
    )


output_bbl = np.asarray(
    output_metadata[
        "bbl"
    ],
    dtype=int
)


if not (
    len(output_bbl) == 326
    and
    np.all(
        output_bbl == 1
    )
):

    raise RuntimeError(
        "Output BBL is incorrect."
    )


print(
    "326 wavelengths       : PASS"
)

print(
    "Wavelength order      : PASS"
)

print(
    "326 BBL=1             : PASS"
)


# ============================================================
# NUMERICAL IDENTITY QC
#
# Verify that selected output bands are numerically identical
# to the corresponding bands in the 425-band archival master.
# ============================================================

test_science_bands = [

    1,
    25,
    75,
    125,
    175,
    225,
    275,
    326,
]


identity_rows = []


with rasterio.open(
    SOURCE_DAT
) as src, \
     rasterio.open(
         OUTPUT_DAT
     ) as dst:


    for science_band in test_science_bands:


        original_band = int(
            selected_original_bands[
                science_band - 1
            ]
        )


        # Central region avoids reading entire ~10M-pixel band.
        test_width = min(
            500,
            src.width
        )

        test_height = min(
            500,
            src.height
        )


        col0 = (
            src.width
            -
            test_width
        ) // 2


        row0 = (
            src.height
            -
            test_height
        ) // 2


        window = rasterio.windows.Window(

            col_off=
                col0,

            row_off=
                row0,

            width=
                test_width,

            height=
                test_height,
        )


        original = src.read(
            original_band,
            window=window
        )


        subset = dst.read(
            science_band,
            window=window
        )


        difference = (
            subset.astype(
                np.float64
            )
            -
            original.astype(
                np.float64
            )
        )


        max_abs_difference = float(
            np.nanmax(
                np.abs(
                    difference
                )
            )
        )


        identity_rows.append({

            "science_band":
                science_band,

            "original_band":
                original_band,

            "wavelength_nm":
                source_wavelengths[
                    original_band - 1
                ],

            "maximum_absolute_difference":
                max_abs_difference,
        })


        print(
            f"\nScience band "
            f"{science_band:3d} "
            f"<- Original band "
            f"{original_band:3d} "
            f"({source_wavelengths[original_band-1]:.2f} nm)"
        )


        print(
            "  Max absolute difference:",
            max_abs_difference
        )


        if max_abs_difference != 0.0:

            raise RuntimeError(
                "Band-selection identity "
                "test failed."
            )


print(
    "\nNumerical identity QC : PASS"
)


# ============================================================
# FINAL FILE SIZE CHECK
# ============================================================

actual_bytes = (
    OUTPUT_DAT.stat().st_size
)


expected_bytes = (
    source_width
    *
    source_height
    *
    EXPECTED_OUTPUT_BANDS
    *
    4
)


if actual_bytes != expected_bytes:

    raise RuntimeError(
        "\nUnexpected output binary size.\n"
        f"Expected: {expected_bytes:,}\n"
        f"Actual  : {actual_bytes:,}"
    )


actual_gib = (
    actual_bytes
    /
    (1024 ** 3)
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

    "source_master":
        str(
            SOURCE_DAT
        ),

    "output_scientific_cube":
        str(
            OUTPUT_DAT
        ),

    "source_bands":
        425,

    "output_bands":
        326,

    "excluded_bands":
        99,

    "width":
        source_width,

    "height":
        source_height,

    "pixel_size_m":
        4.0,

    "crs":
        str(
            source_crs
        ),

    "dtype":
        "Float32",

    "interleave":
        "BIL",

    "nodata":
        source_nodata,

    "wavelength_min_nm":
        float(
            output_wavelengths.min()
        ),

    "wavelength_max_nm":
        float(
            output_wavelengths.max()
        ),

    "output_size_GiB":
        actual_gib,

    "spatial_resampling":
        "none",

    "radiometric_transformation":
        "none",

    "spectral_transformation":
        "none",

    "band_selection_only":
        True,

    "numerical_identity_qc":
        "PASS",

    "processing_minutes":
        elapsed / 60.0,

    "step9_status":
        "PASS",
}])


summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# ARCHIVE MASK WITH OUTPUT
# ============================================================

mask_archive = (
    OUTDIR
    /
    "AVIRIS_FINAL_326band_scientific_mask.csv"
)


if (
    mask_path.resolve()
    !=
    mask_archive.resolve()
):

    shutil.copy2(
        mask_path,
        mask_archive
    )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 94)

print(
    "STEP 9 COMPLETE — "
    "326-BAND SCIENTIFIC MASTER CREATED"
)

print("=" * 94)


print(
    f"\nSource archival master:"
)

print(
    SOURCE_DAT
)


print(
    "\nScientific working cube:"
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
    "\nBand mapping:"
)

print(
    BAND_MAP_CSV
)


print(
    "\nSummary:"
)

print(
    SUMMARY_CSV
)


print(
    f"\nDimensions      : "
    f"{source_width} x "
    f"{source_height}"
)


print(
    "Source bands    : 425"
)


print(
    "Scientific bands: 326"
)


print(
    "Excluded bands  : 99"
)


print(
    f"Wavelength range: "
    f"{output_wavelengths.min():.2f} "
    f"to "
    f"{output_wavelengths.max():.2f} nm"
)


print(
    f"File size       : "
    f"{actual_gib:.2f} GiB"
)


print(
    f"Processing time : "
    f"{elapsed/60:.1f} minutes"
)


print("\n")
print("=" * 94)
print("IMPORTANT")
print("=" * 94)


print(
    "\n1. The 425-band archival mosaic "
    "was NOT modified."
)


print(
    "2. The 326-band cube contains "
    "direct copies of selected source bands."
)


print(
    "3. No resampling occurred."
)


print(
    "4. No normalization occurred."
)


print(
    "5. No spectral smoothing occurred."
)


print(
    "6. No reflectance values changed."
)


print(
    "7. All 326 bands have BBL=1."
)


print(
    "8. Original 425-band indices are "
    "preserved in the header and mapping CSV."
)


print(
    "\nThe 326-band cube is now the working "
    "input for PCA and MNF."
)
