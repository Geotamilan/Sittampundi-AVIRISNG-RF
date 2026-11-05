from pathlib import Path
import csv
import math
import os
import shutil
import time

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from affine import Affine
import spectral as spy


# ============================================================
# USER / PROJECT SETTINGS
# ============================================================

ROOT = Path.cwd()

REGISTRATION_CSV = (
    ROOT
    / "02_Registered_Scenes"
    / "Registration_Check"
    / "AVIRIS_pairwise_registration_results.csv"
)

OUTDIR = (
    ROOT
    / "02_Registered_Scenes"
    / "Registered_425Band_4m"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_RESOLUTION = 4.0

REFERENCE_SCENE = "ang20160110t053932"

SCENES = [
    "ang20160110t053216",
    "ang20160110t053932",
    "ang20160110t054703",
    "ang20160110t055421",
]

EXPECTED_BANDS = 425

FALLBACK_NODATA = -9999.0

# Number of GDAL warp threads.
# 4 is conservative and appropriate for a MacBook.
NUM_THREADS = 4

# Change to True only if you intentionally want to recreate
# already completed outputs.
FORCE_OVERWRITE = False


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_source_header(scene):

    expected = (
        ROOT
        / f"{scene}_rfl_v2m2"
        / f"{scene}_rfl_v2m2"
        / f"{scene}_corr_v2m2_img.hdr"
    )

    if expected.exists():
        return expected

    matches = list(
        ROOT.glob(
            f"{scene}_rfl_v2m2/"
            f"{scene}_rfl_v2m2/"
            f"{scene}_corr_v2m2_img.hdr"
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            f"Could not uniquely locate source header for {scene}. "
            f"Found {len(matches)} matches."
        )

    return matches[0]


def transformed_bounds(transform, width, height):

    corners = [
        transform * (0, 0),
        transform * (width, 0),
        transform * (0, height),
        transform * (width, height),
    ]

    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def aligned_grid(bounds, anchor_left, anchor_top, resolution):

    left, bottom, right, top = bounds

    col_start = math.floor(
        (left - anchor_left) / resolution
    )

    col_end = math.ceil(
        (right - anchor_left) / resolution
    )

    row_start = math.floor(
        (anchor_top - top) / resolution
    )

    row_end = math.ceil(
        (anchor_top - bottom) / resolution
    )

    grid_left = (
        anchor_left
        + col_start * resolution
    )

    grid_right = (
        anchor_left
        + col_end * resolution
    )

    grid_top = (
        anchor_top
        - row_start * resolution
    )

    grid_bottom = (
        anchor_top
        - row_end * resolution
    )

    width = int(col_end - col_start)
    height = int(row_end - row_start)

    transform = from_origin(
        grid_left,
        grid_top,
        resolution,
        resolution,
    )

    return {
        "left": grid_left,
        "bottom": grid_bottom,
        "right": grid_right,
        "top": grid_top,
        "width": width,
        "height": height,
        "transform": transform,
    }


def format_envi_list(values):

    values = list(values)

    return (
        "{\n"
        + ",\n".join(str(v) for v in values)
        + "\n}"
    )


def append_spectral_metadata(
    output_hdr,
    source_metadata,
    east_shift,
    north_shift,
):

    text = output_hdr.read_text(
        encoding="utf-8",
        errors="ignore"
    )

    lower_text = text.lower()

    additions = []

    # --------------------------------------------------------
    # IMPORTANT SPECTRAL METADATA
    # --------------------------------------------------------

    list_keys = [
        "wavelength",
        "fwhm",
        "bbl",
        "default bands",
    ]

    scalar_keys = [
        "wavelength units",
        "sensor type",
        "reflectance scale factor",
    ]

    for key in scalar_keys:

        if key in source_metadata:

            if f"{key.lower()} =" not in lower_text:

                value = source_metadata[key]

                additions.append(
                    f"{key} = {value}"
                )

    for key in list_keys:

        if key in source_metadata:

            if f"{key.lower()} =" not in lower_text:

                value = source_metadata[key]

                if isinstance(value, (list, tuple, np.ndarray)):

                    additions.append(
                        f"{key} = "
                        f"{format_envi_list(value)}"
                    )

    # --------------------------------------------------------
    # REGISTRATION DOCUMENTATION
    # --------------------------------------------------------

    additions.append(
        "registration reference scene = "
        f"{REFERENCE_SCENE}"
    )

    additions.append(
        "registration east shift meters = "
        f"{east_shift:.6f}"
    )

    additions.append(
        "registration north shift meters = "
        f"{north_shift:.6f}"
    )

    additions.append(
        "registration target pixel size meters = "
        f"{TARGET_RESOLUTION:.6f}"
    )

    if additions:

        with output_hdr.open(
            "a",
            encoding="utf-8"
        ) as f:

            f.write("\n")

            for item in additions:
                f.write(item)
                f.write("\n")


# ============================================================
# READ REGISTRATION CSV
# ============================================================

print("=" * 82)
print("AVIRIS-NG FULL 425-BAND REGISTRATION")
print("=" * 82)

if not REGISTRATION_CSV.exists():

    raise FileNotFoundError(
        "\nRegistration CSV not found:\n"
        f"{REGISTRATION_CSV}\n\n"
        "Run the registration measurement step first."
    )


rows = []

with REGISTRATION_CSV.open(
    "r",
    encoding="utf-8"
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        row["east_shift_m"] = float(
            row["east_shift_m"]
        )

        row["north_shift_m"] = float(
            row["north_shift_m"]
        )

        rows.append(row)


def find_pair(moving, reference):

    for row in rows:

        if (
            row["moving_scene"] == moving
            and row["reference_scene"] == reference
        ):

            return row

    raise RuntimeError(
        f"Registration pair not found:\n"
        f"{moving} -> {reference}"
    )


# ============================================================
# CALCULATE EXACT CUMULATIVE CORRECTIONS
# ============================================================

pair_053216 = find_pair(
    "ang20160110t053216",
    "ang20160110t053932"
)

pair_054703 = find_pair(
    "ang20160110t054703",
    "ang20160110t053932"
)

pair_055421 = find_pair(
    "ang20160110t055421",
    "ang20160110t054703"
)


corrections = {

    "ang20160110t053932": {
        "east": 0.0,
        "north": 0.0,
    },

    "ang20160110t053216": {
        "east": pair_053216["east_shift_m"],
        "north": pair_053216["north_shift_m"],
    },

    "ang20160110t054703": {
        "east": pair_054703["east_shift_m"],
        "north": pair_054703["north_shift_m"],
    },
}


corrections[
    "ang20160110t055421"
] = {

    "east": (
        corrections[
            "ang20160110t054703"
        ]["east"]
        +
        pair_055421["east_shift_m"]
    ),

    "north": (
        corrections[
            "ang20160110t054703"
        ]["north"]
        +
        pair_055421["north_shift_m"]
    ),
}


print("\nCUMULATIVE REGISTRATION CORRECTIONS")
print("-" * 82)

for scene in SCENES:

    east = corrections[scene]["east"]
    north = corrections[scene]["north"]

    total = math.sqrt(
        east ** 2 + north ** 2
    )

    print(f"\n{scene}")

    print(
        f"  East shift  : "
        f"{east:+.3f} m"
    )

    print(
        f"  North shift : "
        f"{north:+.3f} m"
    )

    print(
        f"  Total       : "
        f"{total:.3f} m"
    )


# ============================================================
# READ SOURCE INFORMATION
# ============================================================

scene_info = {}

print("\n")
print("=" * 82)
print("SOURCE DATA CHECK")
print("=" * 82)

for scene in SCENES:

    hdr = find_source_header(scene)

    binary = hdr.with_suffix("")

    if not binary.exists():

        raise FileNotFoundError(
            f"Binary file missing:\n{binary}"
        )

    spectral_img = spy.open_image(
        str(hdr)
    )

    metadata = spectral_img.metadata

    with rasterio.open(binary) as src:

        if src.count != EXPECTED_BANDS:

            raise RuntimeError(
                f"{scene}: expected "
                f"{EXPECTED_BANDS} bands, "
                f"found {src.count}"
            )

        if src.crs is None:

            raise RuntimeError(
                f"{scene}: CRS is missing."
            )

        east = corrections[scene]["east"]
        north = corrections[scene]["north"]

        # ----------------------------------------------------
        # APPLY RIGID MAP TRANSLATION
        #
        # This changes geolocation only.
        # Original reflectance values are not altered here.
        # ----------------------------------------------------

        shifted_transform = (
            Affine.translation(
                east,
                north
            )
            *
            src.transform
        )

        bounds = transformed_bounds(
            shifted_transform,
            src.width,
            src.height,
        )

        nodata = src.nodata

        if nodata is None:

            try:

                nodata = float(
                    metadata.get(
                        "data ignore value",
                        FALLBACK_NODATA,
                    )
                )

            except Exception:

                nodata = FALLBACK_NODATA

        scene_info[scene] = {

            "hdr": hdr,
            "binary": binary,
            "metadata": metadata,

            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs,
            "nodata": nodata,

            "original_transform":
                src.transform,

            "shifted_transform":
                shifted_transform,

            "shifted_bounds":
                bounds,
        }

        print("\n" + scene)

        print(
            f"  Size       : "
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
            f"  Original resolution : "
            f"{src.res}"
        )

        print(
            f"  Nodata     : "
            f"{nodata}"
        )

        print(
            "  Corrected bounds:"
        )

        print(
            "   ",
            bounds
        )


# ============================================================
# CONFIRM ALL CRS ARE IDENTICAL
# ============================================================

crs_values = {
    str(scene_info[s]["crs"])
    for s in SCENES
}

if len(crs_values) != 1:

    raise RuntimeError(
        "The four scenes do not share "
        "the same CRS."
    )


# ============================================================
# CREATE COMMON 4-M GRID ALIGNMENT ANCHOR
# ============================================================

all_left = [
    scene_info[s]["shifted_bounds"][0]
    for s in SCENES
]

all_bottom = [
    scene_info[s]["shifted_bounds"][1]
    for s in SCENES
]

all_right = [
    scene_info[s]["shifted_bounds"][2]
    for s in SCENES
]

all_top = [
    scene_info[s]["shifted_bounds"][3]
    for s in SCENES
]


union_left = min(all_left)
union_bottom = min(all_bottom)
union_right = max(all_right)
union_top = max(all_top)


anchor_left = (
    math.floor(
        union_left
        / TARGET_RESOLUTION
    )
    * TARGET_RESOLUTION
)

anchor_top = (
    math.ceil(
        union_top
        / TARGET_RESOLUTION
    )
    * TARGET_RESOLUTION
)


print("\n")
print("=" * 82)
print("COMMON GRID")
print("=" * 82)

print(
    f"\nTarget resolution : "
    f"{TARGET_RESOLUTION} m"
)

print(
    f"Grid anchor X     : "
    f"{anchor_left}"
)

print(
    f"Grid anchor Y     : "
    f"{anchor_top}"
)

print(
    "\nAll four outputs will share "
    "the same 4-m pixel lattice."
)


# ============================================================
# CALCULATE OUTPUT GRIDS AND STORAGE REQUIREMENT
# ============================================================

estimated_bytes = 0

for scene in SCENES:

    grid = aligned_grid(

        scene_info[
            scene
        ]["shifted_bounds"],

        anchor_left,
        anchor_top,
        TARGET_RESOLUTION,
    )

    scene_info[
        scene
    ]["output_grid"] = grid

    bytes_scene = (
        grid["width"]
        *
        grid["height"]
        *
        EXPECTED_BANDS
        *
        4
    )

    estimated_bytes += bytes_scene


estimated_gb = (
    estimated_bytes
    / (1024 ** 3)
)

free_bytes = shutil.disk_usage(
    OUTDIR
).free

free_gb = (
    free_bytes
    / (1024 ** 3)
)


print(
    f"\nEstimated output size : "
    f"{estimated_gb:.2f} GB"
)

print(
    f"Currently free space  : "
    f"{free_gb:.2f} GB"
)


if free_bytes < (
    estimated_bytes * 1.10
):

    raise RuntimeError(
        "\nInsufficient free disk space.\n"
        f"Estimated registered data: "
        f"{estimated_gb:.2f} GB\n"
        f"Available: {free_gb:.2f} GB\n\n"
        "Free additional space or use "
        "an external SSD before continuing."
    )


# ============================================================
# REGISTER EACH 425-BAND SCENE
# ============================================================

summary = []

print("\n")
print("=" * 82)
print("STARTING FULL HYPERSPECTRAL REGISTRATION")
print("=" * 82)

overall_start = time.time()


for scene_index, scene in enumerate(
    SCENES,
    start=1
):

    info = scene_info[scene]

    grid = info["output_grid"]

    out_dat = (
        OUTDIR
        / f"{scene}_registered_4m.dat"
    )

    out_hdr = (
        OUTDIR
        / f"{scene}_registered_4m.hdr"
    )

    print("\n")
    print("=" * 82)

    print(
        f"SCENE {scene_index}/"
        f"{len(SCENES)}"
    )

    print(scene)

    print("=" * 82)

    if (
        out_dat.exists()
        and out_hdr.exists()
        and not FORCE_OVERWRITE
    ):

        try:

            with rasterio.open(
                out_dat
            ) as check:

                if (
                    check.count
                    == EXPECTED_BANDS
                    and
                    check.width
                    == grid["width"]
                    and
                    check.height
                    == grid["height"]
                ):

                    print(
                        "\nValid completed output "
                        "already exists."
                    )

                    print(
                        "Skipping this scene."
                    )

                    summary.append({

                        "scene": scene,

                        "east_shift_m":
                            corrections[
                                scene
                            ]["east"],

                        "north_shift_m":
                            corrections[
                                scene
                            ]["north"],

                        "width":
                            check.width,

                        "height":
                            check.height,

                        "bands":
                            check.count,

                        "output":
                            str(out_dat),

                        "status":
                            "existing",
                    })

                    continue

        except Exception:

            pass

    # Delete incomplete/old versions.
    for p in [
        out_dat,
        out_hdr,
        Path(str(out_dat) + ".aux.xml"),
    ]:

        if p.exists():
            p.unlink()

    profile = {

        "driver": "ENVI",

        "width":
            grid["width"],

        "height":
            grid["height"],

        "count":
            EXPECTED_BANDS,

        "dtype":
            "float32",

        "crs":
            info["crs"],

        "transform":
            grid["transform"],

        "nodata":
            info["nodata"],

        "interleave":
            "bil",
    }

    print(
        f"\nOutput dimensions : "
        f"{grid['width']} x "
        f"{grid['height']}"
    )

    print(
        f"Output bands      : "
        f"{EXPECTED_BANDS}"
    )

    print(
        f"Output resolution : "
        f"{TARGET_RESOLUTION} m"
    )

    print(
        "Output transform:"
    )

    print(
        grid["transform"]
    )

    print(
        "\nRegistration correction:"
    )

    print(
        f"  East  : "
        f"{corrections[scene]['east']:+.3f} m"
    )

    print(
        f"  North : "
        f"{corrections[scene]['north']:+.3f} m"
    )

    scene_start = time.time()

    with rasterio.open(
        info["binary"]
    ) as src:

        with rasterio.open(
            out_dat,
            "w",
            **profile
        ) as dst:

            for band in range(
                1,
                EXPECTED_BANDS + 1
            ):

                reproject(

                    source=rasterio.band(
                        src,
                        band
                    ),

                    destination=rasterio.band(
                        dst,
                        band
                    ),

                    src_transform=
                        info[
                            "shifted_transform"
                        ],

                    src_crs=
                        src.crs,

                    src_nodata=
                        info["nodata"],

                    dst_transform=
                        grid["transform"],

                    dst_crs=
                        src.crs,

                    dst_nodata=
                        info["nodata"],

                    resampling=
                        Resampling.bilinear,

                    init_dest_nodata=True,

                    num_threads=
                        NUM_THREADS,
                )

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
                        - scene_start
                    )

                    print(
                        f"  Band "
                        f"{band:3d}/"
                        f"{EXPECTED_BANDS}"
                        f" complete "
                        f"({elapsed/60:.1f} min)"
                    )

    # ========================================================
    # ADD SPECTRAL METADATA TO OUTPUT ENVI HEADER
    # ========================================================

    if not out_hdr.exists():

        alternate_hdr = Path(
            str(out_dat) + ".hdr"
        )

        if alternate_hdr.exists():

            out_hdr = alternate_hdr

        else:

            raise RuntimeError(
                "ENVI header was not created "
                f"for {out_dat}"
            )

    append_spectral_metadata(

        out_hdr,

        info["metadata"],

        corrections[scene]["east"],

        corrections[scene]["north"],
    )


    # ========================================================
    # VALIDATE OUTPUT
    # ========================================================

    with rasterio.open(
        out_dat
    ) as check:

        if check.count != EXPECTED_BANDS:

            raise RuntimeError(
                f"{scene}: output has "
                f"{check.count} bands."
            )

        if (
            abs(
                abs(check.transform.a)
                - TARGET_RESOLUTION
            ) > 1e-6
        ):

            raise RuntimeError(
                f"{scene}: incorrect "
                "X resolution."
            )

        if (
            abs(
                abs(check.transform.e)
                - TARGET_RESOLUTION
            ) > 1e-6
        ):

            raise RuntimeError(
                f"{scene}: incorrect "
                "Y resolution."
            )

        output_size_gb = (
            out_dat.stat().st_size
            / (1024 ** 3)
        )

        print(
            "\nVALIDATION PASSED"
        )

        print(
            f"  Dimensions : "
            f"{check.width} x "
            f"{check.height}"
        )

        print(
            f"  Bands      : "
            f"{check.count}"
        )

        print(
            f"  CRS        : "
            f"{check.crs}"
        )

        print(
            f"  Resolution : "
            f"{check.res}"
        )

        print(
            f"  Nodata     : "
            f"{check.nodata}"
        )

        print(
            f"  File size  : "
            f"{output_size_gb:.2f} GB"
        )

        print(
            f"  Output     : "
            f"{out_dat}"
        )

        summary.append({

            "scene": scene,

            "east_shift_m":
                corrections[
                    scene
                ]["east"],

            "north_shift_m":
                corrections[
                    scene
                ]["north"],

            "width":
                check.width,

            "height":
                check.height,

            "bands":
                check.count,

            "resolution_x":
                check.res[0],

            "resolution_y":
                check.res[1],

            "crs":
                str(check.crs),

            "output_size_GB":
                output_size_gb,

            "output":
                str(out_dat),

            "status":
                "created",
        })


# ============================================================
# SAVE SUMMARY CSV
# ============================================================

summary_csv = (
    OUTDIR
    / "AVIRIS_registered_425band_summary.csv"
)

fieldnames = sorted(
    {
        key
        for item in summary
        for key in item.keys()
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
# FINAL REPORT
# ============================================================

elapsed_total = (
    time.time()
    - overall_start
)

print("\n")
print("=" * 82)
print("FULL 425-BAND REGISTRATION COMPLETE")
print("=" * 82)

print(
    f"\nTotal processing time: "
    f"{elapsed_total/60:.1f} minutes"
)

print(
    "\nRegistered datasets:"
)

for item in summary:

    print(
        f"\n  {item['scene']}"
    )

    print(
        f"    {item['output']}"
    )

print(
    "\nSummary CSV:"
)

print(summary_csv)

print(
    "\nIMPORTANT:"
)

print(
    "1. Original AVIRIS-NG cubes "
    "were not modified."
)

print(
    "2. All outputs contain "
    "425 bands."
)

print(
    "3. All outputs are EPSG:32643."
)

print(
    "4. All outputs use a common "
    "north-up 4.0 m pixel grid."
)

print(
    "5. Wavelength/FWHM metadata "
    "were copied to the ENVI headers "
    "when available."
)

print(
    "6. Bilinear interpolation was "
    "used during the single spatial "
    "resampling operation."
)

print(
    "\nDo NOT mosaic yet."
)

print(
    "First perform the post-registration "
    "overlap QC."
)
