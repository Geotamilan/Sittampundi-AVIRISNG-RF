from pathlib import Path
import csv
import shutil
import time
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window


# ============================================================
# PROJECT SETTINGS
# ============================================================

ROOT = Path.cwd()

INDIR = (
    ROOT
    / "02_Registered_Scenes"
    / "Registered_425Band_4m"
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
    / "Normalized_425Band_4m"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


MASTER_SCENE = "ang20160110t053932"

SCENES = [
    "ang20160110t053216",
    "ang20160110t053932",
    "ang20160110t054703",
    "ang20160110t055421",
]

EXPECTED_BANDS = 425

# Number of rows read at one time.
# Keeps RAM use very low.
ROW_CHUNK = 256

# Recreate completed outputs only if you deliberately change
# this to True.
FORCE_OVERWRITE = False


# ============================================================
# OUTPUT NAMES
# ============================================================

def source_dat(scene):
    return (
        INDIR
        / f"{scene}_registered_4m.dat"
    )


def source_hdr(scene):
    return (
        INDIR
        / f"{scene}_registered_4m.hdr"
    )


def output_dat(scene):
    return (
        OUTDIR
        / f"{scene}_normalized_to_053932_4m.dat"
    )


def output_hdr(scene):
    return (
        OUTDIR
        / f"{scene}_normalized_to_053932_4m.hdr"
    )


# ============================================================
# ENVI HEADER PATCHING
# ============================================================

def set_envi_field(text, key, value):

    """
    Replace an existing ENVI header field, including
    multiline {...} lists. Append if the key does not exist.
    """

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

        text += (
            replacement
            + "\n"
        )

    return text


def format_envi_list(values):

    values = list(values)

    lines = []

    # Keep lines reasonably short.
    for start in range(
        0,
        len(values),
        20
    ):

        section = values[
            start:start + 20
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


def patch_header(
    source_header,
    destination_header,
    scene,
    bbl,
    factors
):

    # Copy the REGISTERED header first.
    #
    # Geometry, wavelength, FWHM, map info and all other
    # spectral metadata are therefore preserved exactly.
    shutil.copy2(
        source_header,
        destination_header
    )

    text = destination_header.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    description = (
        "{AVIRIS-NG L2 reflectance; "
        "registered to common 4 m grid; "
        "radiometrically normalized by wavelength-dependent "
        "multiplicative factors; "
        f"master scene = {MASTER_SCENE}; "
        f"source scene = {scene}}}"
    )

    text = set_envi_field(
        text,
        "description",
        description
    )

    # --------------------------------------------------------
    # BBL / scientific analysis mask
    #
    # 1 = recommended for subsequent analysis
    # 0 = excluded from PCA/MNF/mineral analysis
    # --------------------------------------------------------

    text = set_envi_field(
        text,
        "bbl",
        format_envi_list(
            [int(x) for x in bbl]
        )
    )

    # --------------------------------------------------------
    # Documentation fields
    # --------------------------------------------------------

    text = set_envi_field(
        text,
        "radiometric normalization master",
        MASTER_SCENE
    )

    text = set_envi_field(
        text,
        "radiometric normalization method",
        "{wavelength-dependent multiplicative normalization}"
    )

    text = set_envi_field(
        text,
        "radiometric normalization coefficient version",
        "{Step 7A-v3 FINAL}"
    )

    text = set_envi_field(
        text,
        "radiometric normalization coefficient file",
        "{AVIRIS_425band_FINAL_normalization_coefficients_TO_MASTER_v3.csv}"
    )

    text = set_envi_field(
        text,
        "radiometric normalization additive offset",
        "0"
    )

    text = set_envi_field(
        text,
        "radiometric normalization factor median",
        f"{np.median(factors):.8f}"
    )

    text = set_envi_field(
        text,
        "radiometric normalization factor minimum",
        f"{np.min(factors):.8f}"
    )

    text = set_envi_field(
        text,
        "radiometric normalization factor maximum",
        f"{np.max(factors):.8f}"
    )

    destination_header.write_text(
        text,
        encoding="utf-8"
    )


# ============================================================
# CHECK EXISTING OUTPUT
# ============================================================

def valid_completed_output(
    dat,
    hdr,
    expected_width,
    expected_height
):

    if not dat.exists() or not hdr.exists():
        return False

    expected_bytes = (
        expected_width
        * expected_height
        * EXPECTED_BANDS
        * 4
    )

    if dat.stat().st_size != expected_bytes:
        return False

    try:

        with rasterio.open(dat) as src:

            if (
                src.count != EXPECTED_BANDS
                or
                src.width != expected_width
                or
                src.height != expected_height
            ):
                return False

    except Exception:
        return False

    return True


# ============================================================
# LOAD FINAL V3 COEFFICIENTS
# ============================================================

print("=" * 92)
print("AVIRIS-NG STEP 7B")
print("APPLY FINAL v3 RADIOMETRIC NORMALIZATION")
print("=" * 92)

print(
    "\nRegistered source cubes will NOT be modified."
)

print(
    "Normalized data will be written to:"
)

print(
    OUTDIR
)


if not COEFF_FILE.exists():

    raise FileNotFoundError(
        "\nFinal v3 coefficient file not found:\n"
        f"{COEFF_FILE}"
    )


coeff = pd.read_csv(
    COEFF_FILE
)


if len(coeff) != EXPECTED_BANDS:

    raise RuntimeError(
        f"Coefficient file contains "
        f"{len(coeff)} rows; expected 425."
    )


if "recommended_for_scientific_analysis" not in coeff.columns:

    raise RuntimeError(
        "Scientific-analysis mask column is missing."
    )


# ============================================================
# SCIENTIFIC ANALYSIS MASK
# ============================================================

analysis_mask = (
    coeff[
        "recommended_for_scientific_analysis"
    ]
    .astype(bool)
    .to_numpy()
)


bbl = np.where(
    analysis_mask,
    1,
    0
).astype(int)


print(
    "\nBands retained for later scientific analysis:",
    int(
        np.sum(bbl)
    )
)


print(
    "Bands flagged BBL=0:",
    int(
        np.sum(
            bbl == 0
        )
    )
)


# ============================================================
# LOAD FACTORS FOR EACH SCENE
# ============================================================

factor_columns = {

    "ang20160110t053216":
        "ang20160110t053216_factor_to_master",

    "ang20160110t053932":
        "ang20160110t053932_factor_to_master",

    "ang20160110t054703":
        "ang20160110t054703_factor_to_master",

    "ang20160110t055421":
        "ang20160110t055421_factor_to_master",
}


scene_factors = {}


for scene in SCENES:

    column = factor_columns[
        scene
    ]

    if column not in coeff.columns:

        raise RuntimeError(
            f"Missing coefficient column:\n"
            f"{column}"
        )


    factors = (
        coeff[
            column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )


    if (
        len(factors)
        != EXPECTED_BANDS
    ):

        raise RuntimeError(
            f"{scene}: coefficient count "
            "is not 425."
        )


    if not np.all(
        np.isfinite(factors)
    ):

        raise RuntimeError(
            f"{scene}: coefficient file "
            "contains NaN or infinite values."
        )


    if np.any(
        factors <= 0
    ):

        raise RuntimeError(
            f"{scene}: non-positive "
            "normalization coefficient found."
        )


    scene_factors[
        scene
    ] = factors


    print(
        f"\n{scene}"
    )

    print(
        f"  Minimum factor : "
        f"{np.min(factors):.6f}"
    )

    print(
        f"  Median factor  : "
        f"{np.median(factors):.6f}"
    )

    print(
        f"  Maximum factor : "
        f"{np.max(factors):.6f}"
    )


# ============================================================
# LARGE-FACTOR WARNING
# ============================================================

f4 = scene_factors[
    "ang20160110t055421"
]

large4 = (
    f4 > 1.8
)

if np.any(large4):

    wave = (
        coeff[
            "wavelength_nm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    print("\n" + "-" * 92)

    print(
        "CAUTION — SCENE 055421 BLUE-VNIR"
    )

    print("-" * 92)

    print(
        f"Bands with cumulative factor > 1.8: "
        f"{np.sum(large4)}"
    )

    print(
        "Approximate wavelength range:"
    )

    print(
        f"{wave[large4].min():.2f} – "
        f"{wave[large4].max():.2f} nm"
    )

    print(
        "These bands are preserved but must remain "
        "flagged for later spectral/SNR QC."
    )


# ============================================================
# SOURCE DATA VALIDATION + SPACE REQUIREMENT
# ============================================================

source_info = {}

required_bytes = 0


print("\n")
print("=" * 92)
print("SOURCE VALIDATION")
print("=" * 92)


for scene in SCENES:

    dat = source_dat(
        scene
    )

    hdr = source_hdr(
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
                f"found {src.count}"
            )


        if str(src.crs) != "EPSG:32643":

            raise RuntimeError(
                f"{scene}: unexpected CRS "
                f"{src.crs}"
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
                f"{scene}: registered "
                "resolution is not 4 m."
            )


        expected_bytes = (
            src.width
            * src.height
            * src.count
            * 4
        )


        required_bytes += (
            expected_bytes
        )


        source_info[
            scene
        ] = {

            "width":
                src.width,

            "height":
                src.height,

            "count":
                src.count,

            "crs":
                src.crs,

            "transform":
                src.transform,

            "nodata":
                src.nodata,

            "dtype":
                src.dtypes[0],

            "expected_bytes":
                expected_bytes,
        }


        print(
            f"\n{scene}"
        )

        print(
            f"  Dimensions : "
            f"{src.width} x {src.height}"
        )

        print(
            f"  Bands      : "
            f"{src.count}"
        )

        print(
            f"  CRS        : "
            f"{src.crs}"
        )

        print(
            f"  Resolution : "
            f"{src.res}"
        )

        print(
            f"  Nodata     : "
            f"{src.nodata}"
        )


# ============================================================
# DISK SPACE CHECK
# ============================================================

disk = shutil.disk_usage(
    OUTDIR
)

free_bytes = disk.free

required_gib = (
    required_bytes
    / (1024 ** 3)
)

free_gib = (
    free_bytes
    / (1024 ** 3)
)


print("\n")
print("=" * 92)
print("DISK SPACE")
print("=" * 92)

print(
    f"\nEstimated normalized output: "
    f"{required_gib:.2f} GiB"
)

print(
    f"Currently free: "
    f"{free_gib:.2f} GiB"
)


if free_bytes < (
    required_bytes
    * 1.15
):

    raise RuntimeError(
        "Not enough free disk space "
        "for safe processing."
    )


# ============================================================
# NORMALIZATION
# ============================================================

summary = []

overall_start = time.time()


print("\n")
print("=" * 92)
print("STARTING STEP 7B")
print("=" * 92)


for scene_index, scene in enumerate(
    SCENES,
    start=1
):


    src_dat = source_dat(
        scene
    )

    src_hdr = source_hdr(
        scene
    )

    dst_dat = output_dat(
        scene
    )

    dst_hdr = output_hdr(
        scene
    )

    factors = scene_factors[
        scene
    ]

    info = source_info[
        scene
    ]


    print("\n")
    print("=" * 92)

    print(
        f"SCENE {scene_index}/"
        f"{len(SCENES)}"
    )

    print(
        scene
    )

    print("=" * 92)


    # ========================================================
    # SKIP VALID COMPLETED OUTPUT
    # ========================================================

    if (
        not FORCE_OVERWRITE
        and
        valid_completed_output(
            dst_dat,
            dst_hdr,
            info["width"],
            info["height"]
        )
    ):

        print(
            "\nValid normalized output "
            "already exists."
        )

        print(
            "Skipping."
        )


        summary.append({

            "scene":
                scene,

            "status":
                "existing",

            "minimum_factor":
                float(
                    np.min(factors)
                ),

            "median_factor":
                float(
                    np.median(factors)
                ),

            "maximum_factor":
                float(
                    np.max(factors)
                ),

            "output":
                str(dst_dat),
        })

        continue


    # Remove incomplete outputs from a prior failed run.
    for path in [

        dst_dat,
        dst_hdr,
        Path(
            str(dst_dat)
            + ".aux.xml"
        ),

    ]:

        if path.exists():

            path.unlink()


    scene_start = (
        time.time()
    )


    # ========================================================
    # MASTER SCENE
    #
    # Factor = exactly 1.
    #
    # Copy the binary byte-for-byte rather than recalculating.
    # ========================================================

    if scene == MASTER_SCENE:

        print(
            "\nMASTER SCENE"
        )

        print(
            "Normalization factor = 1.0."
        )

        print(
            "Creating untouched binary copy..."
        )


        shutil.copy2(
            src_dat,
            dst_dat
        )


        patch_header(
            src_hdr,
            dst_hdr,
            scene,
            bbl,
            factors
        )


    # ========================================================
    # NORMALIZE NON-MASTER SCENES
    # ========================================================

    else:

        print(
            "\nApplying multiplicative "
            "normalization:"
        )

        print(
            "    corrected reflectance "
            "= input reflectance × factor"
        )

        print(
            "\nNo additive offset."
        )

        print(
            "No spectral clipping."
        )

        print(
            "No spatial resampling."
        )


        with rasterio.open(
            src_dat
        ) as src:


            profile = {

                "driver":
                    "ENVI",

                "width":
                    src.width,

                "height":
                    src.height,

                "count":
                    src.count,

                "dtype":
                    "float32",

                "crs":
                    src.crs,

                "transform":
                    src.transform,

                "nodata":
                    src.nodata,

                "interleave":
                    "bil",
            }


            nodata = (
                src.nodata
            )


            with rasterio.open(
                dst_dat,
                "w",
                **profile
            ) as dst:


                for band in range(
                    1,
                    EXPECTED_BANDS + 1
                ):


                    factor = np.float32(
                        factors[
                            band - 1
                        ]
                    )


                    for row_start in range(
                        0,
                        src.height,
                        ROW_CHUNK
                    ):


                        rows = min(
                            ROW_CHUNK,
                            src.height
                            - row_start
                        )


                        window = Window(

                            col_off=0,

                            row_off=
                                row_start,

                            width=
                                src.width,

                            height=
                                rows,
                        )


                        data = src.read(
                            band,
                            window=window
                        ).astype(
                            np.float32,
                            copy=False
                        )


                        # ====================================
                        # VALID-PIXEL MASK
                        # ====================================

                        valid = np.isfinite(
                            data
                        )


                        if nodata is not None:

                            valid &= (
                                data
                                != nodata
                            )


                        # ====================================
                        # MULTIPLICATIVE NORMALIZATION
                        # ====================================

                        corrected = (
                            data.copy()
                        )


                        corrected[
                            valid
                        ] *= factor


                        # Nodata remains exactly unchanged.
                        if nodata is not None:

                            corrected[
                                ~valid
                            ] = nodata


                        dst.write(
                            corrected,
                            band,
                            window=window
                        )


                    # Progress message.
                    if (
                        band == 1
                        or
                        band % 25 == 0
                        or
                        band
                        == EXPECTED_BANDS
                    ):

                        elapsed = (
                            time.time()
                            -
                            scene_start
                        )


                        print(
                            f"  Band "
                            f"{band:3d}/425 "
                            f"factor="
                            f"{factor:.6f} "
                            f"complete "
                            f"({elapsed/60:.1f} min)"
                        )


        # ----------------------------------------------------
        # Replace Rasterio's minimal ENVI header with a copy
        # of the registered source header, then patch it.
        #
        # Geometry is identical because Step 7B performs no
        # spatial transformation.
        # ----------------------------------------------------

        patch_header(
            src_hdr,
            dst_hdr,
            scene,
            bbl,
            factors
        )


    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    expected_bytes = (
        info[
            "expected_bytes"
        ]
    )


    actual_bytes = (
        dst_dat.stat().st_size
    )


    if (
        actual_bytes
        != expected_bytes
    ):

        raise RuntimeError(
            f"{scene}: output binary size "
            f"is incorrect.\n"
            f"Expected {expected_bytes:,} bytes\n"
            f"Found    {actual_bytes:,} bytes"
        )


    with rasterio.open(
        dst_dat
    ) as check:


        if (
            check.count
            != EXPECTED_BANDS
        ):

            raise RuntimeError(
                f"{scene}: output band "
                "count is incorrect."
            )


        if (
            check.width
            != info["width"]
            or
            check.height
            != info["height"]
        ):

            raise RuntimeError(
                f"{scene}: dimensions changed."
            )


        if (
            check.transform
            != info["transform"]
        ):

            raise RuntimeError(
                f"{scene}: transform changed."
            )


        if (
            check.crs
            != info["crs"]
        ):

            raise RuntimeError(
                f"{scene}: CRS changed."
            )


    # ========================================================
    # NUMERICAL SPOT CHECK
    #
    # Test several bands from the center region and verify
    # output/input ratio against the expected coefficient.
    # ========================================================

    test_bands = [
        20,     # ~471.6 nm
        54,     # ~641.9 nm
        125,
        200,
        300,
        400,
    ]


    ratio_errors = []


    with rasterio.open(
        src_dat
    ) as src, \
         rasterio.open(
             dst_dat
         ) as dst:


        test_width = min(
            300,
            src.width
        )

        test_height = min(
            300,
            src.height
        )


        col0 = max(
            0,
            (
                src.width
                -
                test_width
            )
            // 2
        )


        row0 = max(
            0,
            (
                src.height
                -
                test_height
            )
            // 2
        )


        window = Window(
            col0,
            row0,
            test_width,
            test_height
        )


        for band in test_bands:


            src_data = src.read(
                band,
                window=window
            ).astype(
                np.float64
            )


            dst_data = dst.read(
                band,
                window=window
            ).astype(
                np.float64
            )


            valid = (
                np.isfinite(
                    src_data
                )
                &
                np.isfinite(
                    dst_data
                )
                &
                (
                    np.abs(
                        src_data
                    )
                    > 1e-8
                )
            )


            if src.nodata is not None:

                valid &= (
                    src_data
                    != src.nodata
                )


            if dst.nodata is not None:

                valid &= (
                    dst_data
                    != dst.nodata
                )


            if np.sum(valid) > 100:


                measured_ratio = float(
                    np.median(
                        dst_data[
                            valid
                        ]
                        /
                        src_data[
                            valid
                        ]
                    )
                )


                expected_ratio = float(
                    factors[
                        band - 1
                    ]
                )


                error = abs(
                    measured_ratio
                    -
                    expected_ratio
                )


                ratio_errors.append(
                    error
                )


                print(
                    f"  QC Band {band:3d}: "
                    f"expected="
                    f"{expected_ratio:.6f}, "
                    f"measured="
                    f"{measured_ratio:.6f}"
                )


    if ratio_errors:

        maximum_ratio_error = (
            max(
                ratio_errors
            )
        )

    else:

        maximum_ratio_error = (
            np.nan
        )


    if (
        np.isfinite(
            maximum_ratio_error
        )
        and
        maximum_ratio_error
        > 1e-4
    ):

        raise RuntimeError(
            f"{scene}: numerical normalization "
            "spot check failed."
        )


    elapsed_scene = (
        time.time()
        -
        scene_start
    )


    size_gib = (
        actual_bytes
        /
        (1024 ** 3)
    )


    print(
        "\nVALIDATION PASSED"
    )

    print(
        f"  Bands              : "
        f"{EXPECTED_BANDS}"
    )

    print(
        f"  Dimensions         : "
        f"{info['width']} x "
        f"{info['height']}"
    )

    print(
        f"  Pixel size         : "
        f"4.0 m"
    )

    print(
        f"  CRS                : "
        f"{info['crs']}"
    )

    print(
        f"  Minimum factor     : "
        f"{np.min(factors):.6f}"
    )

    print(
        f"  Median factor      : "
        f"{np.median(factors):.6f}"
    )

    print(
        f"  Maximum factor     : "
        f"{np.max(factors):.6f}"
    )

    print(
        f"  File size          : "
        f"{size_gib:.2f} GiB"
    )

    print(
        f"  Processing time    : "
        f"{elapsed_scene/60:.1f} min"
    )

    print(
        f"  Output             : "
        f"{dst_dat}"
    )


    summary.append({

        "scene":
            scene,

        "master_scene":
            MASTER_SCENE,

        "minimum_factor":
            float(
                np.min(
                    factors
                )
            ),

        "median_factor":
            float(
                np.median(
                    factors
                )
            ),

        "maximum_factor":
            float(
                np.max(
                    factors
                )
            ),

        "scientific_analysis_bands":
            int(
                np.sum(
                    bbl
                )
            ),

        "excluded_bbl_bands":
            int(
                np.sum(
                    bbl == 0
                )
            ),

        "maximum_spotcheck_ratio_error":
            (
                float(
                    maximum_ratio_error
                )
                if
                np.isfinite(
                    maximum_ratio_error
                )
                else
                np.nan
            ),

        "width":
            info["width"],

        "height":
            info["height"],

        "bands":
            EXPECTED_BANDS,

        "crs":
            str(
                info["crs"]
            ),

        "resolution_m":
            4.0,

        "output_size_GiB":
            size_gib,

        "processing_minutes":
            elapsed_scene / 60.0,

        "status":
            "created",

        "output":
            str(
                dst_dat
            ),
    })


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv = (
    OUTDIR
    / "AVIRIS_Step7B_normalized_425band_summary.csv"
)


fieldnames = sorted(
    {
        key
        for row in summary
        for key in row.keys()
    }
)


with summary_csv.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:


    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )


    writer.writeheader()


    writer.writerows(
        summary
    )


# ============================================================
# COPY COEFFICIENT FILE INTO OUTPUT DIRECTORY
#
# Makes the processing result self-documenting.
# ============================================================

coefficient_copy = (
    OUTDIR
    / COEFF_FILE.name
)


shutil.copy2(
    COEFF_FILE,
    coefficient_copy
)


# ============================================================
# FINAL REPORT
# ============================================================

elapsed_total = (
    time.time()
    -
    overall_start
)


print("\n")
print("=" * 92)
print("STEP 7B COMPLETE")
print("=" * 92)


print(
    f"\nTotal processing time: "
    f"{elapsed_total/60:.1f} minutes"
)


print(
    "\nNormalized datasets:"
)


for result in summary:

    print(
        f"\n  {result['scene']}"
    )

    print(
        f"    {result['output']}"
    )


print(
    "\nSummary CSV:"
)

print(
    summary_csv
)


print(
    "\nCoefficient archive:"
)

print(
    coefficient_copy
)


print("\n")
print("=" * 92)
print("IMPORTANT")
print("=" * 92)


print(
    "\n1. Original AVIRIS-NG L2 scenes "
    "were NOT modified."
)


print(
    "2. Registered 425-band scenes "
    "were NOT modified."
)


print(
    "3. New normalized copies were "
    "created in 03_Mosaic/"
    "Normalized_425Band_4m."
)


print(
    "4. Only multiplicative "
    "normalization was used."
)


print(
    "5. NO additive offset was used."
)


print(
    "6. NO spatial resampling occurred "
    "during Step 7B."
)


print(
    "7. NO reflectance clipping was "
    "performed."
)


print(
    "8. All 425 bands were preserved."
)


print(
    "9. Wavelength/FWHM/map metadata "
    "were preserved from the registered "
    "ENVI headers."
)


print(
    "10. The BBL field now identifies "
    "bands recommended for subsequent "
    "scientific analysis."
)


print(
    "\nDO NOT MOSAIC YET."
)


print(
    "\nNext step:"
)


print(
    "STEP 7C — post-normalization "
    "425-band overlap QC."
)
