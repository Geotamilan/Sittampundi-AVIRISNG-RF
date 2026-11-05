from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
import spectral as spy
import matplotlib.pyplot as plt

ROOT = Path.cwd()

MOSAIC_DIR = (
    ROOT / "03_Mosaic" / "Final_425Band_Mosaic"
)

DAT = MOSAIC_DIR / "AVIRIS_NG_mosaic_master_425band_4m.dat"
HDR = MOSAIC_DIR / "AVIRIS_NG_mosaic_master_425band_4m.hdr"

COVERAGE = MOSAIC_DIR / "AVIRIS_NG_mosaic_coverage_count.tif"
SCENE_INDEX = MOSAIC_DIR / "AVIRIS_NG_mosaic_scene_index.tif"

OUTDIR = (
    MOSAIC_DIR / "Step8B_Final_QC"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)

EXPECTED_BANDS = 425
EXPECTED_SCIENCE_BANDS = 326
EXPECTED_RESOLUTION = 4.0

RGB_BANDS = [54, 36, 20]

# ============================================================
# START
# ============================================================

print("=" * 90)
print("AVIRIS-NG STEP 8B — FINAL MOSAIC QC")
print("=" * 90)

print("\nREAD-ONLY QC")
print("No raster will be modified.")

for f in [DAT, HDR, COVERAGE, SCENE_INDEX]:
    if not f.exists():
        raise FileNotFoundError(f)

# ============================================================
# RASTER VALIDATION
# ============================================================

with rasterio.open(DAT) as src:

    print("\nFINAL MOSAIC")
    print("-" * 90)

    print("Dimensions :", src.width, "x", src.height)
    print("Bands      :", src.count)
    print("CRS        :", src.crs)
    print("Resolution :", src.res)
    print("NoData     :", src.nodata)
    print("Bounds     :", src.bounds)
    print("Transform  :")
    print(src.transform)

    if src.count != EXPECTED_BANDS:
        raise RuntimeError("Final mosaic does not contain 425 bands.")

    if str(src.crs) != "EPSG:32643":
        raise RuntimeError("Unexpected final mosaic CRS.")

    if not (
        np.isclose(src.res[0], EXPECTED_RESOLUTION)
        and
        np.isclose(src.res[1], EXPECTED_RESOLUTION)
    ):
        raise RuntimeError("Unexpected final mosaic resolution.")

    width = src.width
    height = src.height
    transform = src.transform
    bounds = src.bounds
    nodata = src.nodata

# ============================================================
# ENVI SPECTRAL METADATA
# ============================================================

img = spy.open_image(
    str(HDR)
)

md = img.metadata

if "wavelength" not in md:
    raise RuntimeError("Wavelength metadata missing.")

if "bbl" not in md:
    raise RuntimeError("BBL metadata missing.")

wavelengths = np.asarray(
    md["wavelength"],
    dtype=float
)

bbl = np.asarray(
    md["bbl"],
    dtype=int
)

print("\nSPECTRAL METADATA")
print("-" * 90)

print("Wavelength count :", len(wavelengths))
print(
    "Range            :",
    f"{wavelengths.min():.2f}",
    "to",
    f"{wavelengths.max():.2f}",
    "nm"
)

print("BBL=1 bands      :", int(np.sum(bbl == 1)))
print("BBL=0 bands      :", int(np.sum(bbl == 0)))

if len(wavelengths) != 425:
    raise RuntimeError("Wavelength count is not 425.")

if len(bbl) != 425:
    raise RuntimeError("BBL count is not 425.")

if int(np.sum(bbl == 1)) != EXPECTED_SCIENCE_BANDS:
    raise RuntimeError("Scientific BBL count is not 326.")

# ============================================================
# COVERAGE VALIDATION
# ============================================================

with rasterio.open(COVERAGE) as src:

    coverage = src.read(1)

    if (
        src.width != width
        or src.height != height
        or src.transform != transform
    ):
        raise RuntimeError(
            "Coverage raster grid does not match mosaic."
        )

unique, counts = np.unique(
    coverage,
    return_counts=True
)

coverage_stats = dict(
    zip(
        unique.tolist(),
        counts.tolist()
    )
)

print("\nCOVERAGE")
print("-" * 90)

for value in sorted(coverage_stats):

    pixels = coverage_stats[value]

    area = (
        pixels
        * 16
        / 1_000_000
    )

    print(
        f"Coverage {value}: "
        f"{pixels:,} pixels "
        f"= {area:.3f} km²"
    )

if np.max(coverage) > 2:

    print(
        "\nWARNING: Coverage greater than two scenes exists."
    )

# ============================================================
# DOMINANT SCENE VALIDATION
# ============================================================

with rasterio.open(SCENE_INDEX) as src:

    scene_index = src.read(1)

    if (
        src.width != width
        or src.height != height
        or src.transform != transform
    ):
        raise RuntimeError(
            "Scene-index grid does not match mosaic."
        )

print("\nDOMINANT SCENE PIXELS")
print("-" * 90)

scene_counts = {}

for scene_id in [1, 2, 3, 4]:

    count = int(
        np.sum(
            scene_index == scene_id
        )
    )

    scene_counts[scene_id] = count

    print(
        f"Scene {scene_id}: "
        f"{count:,} pixels"
    )

# ============================================================
# MOSAIC VALIDITY VS COVERAGE
#
# Test representative science bands.
# ============================================================

TEST_BANDS = [
    21,    # ~476 nm
    54,    # red
    100,
    180,
    220,
    250,
    324,
    350,
    400,
    414,
]

validity_rows = []

with rasterio.open(DAT) as src:

    coverage_valid = (
        coverage > 0
    )

    for band in TEST_BANDS:

        array = src.read(
            band
        )

        valid = np.isfinite(
            array
        )

        if nodata is not None:

            valid &= (
                array != nodata
            )

        missing_inside_coverage = int(
            np.sum(
                coverage_valid
                &
                (~valid)
            )
        )

        unexpected_valid_outside = int(
            np.sum(
                (~coverage_valid)
                &
                valid
            )
        )

        validity_rows.append({

            "band":
                band,

            "wavelength_nm":
                wavelengths[
                    band - 1
                ],

            "missing_pixels_inside_coverage":
                missing_inside_coverage,

            "valid_pixels_outside_coverage":
                unexpected_valid_outside,
        })

        print(
            f"\nBand {band:3d} "
            f"{wavelengths[band-1]:.2f} nm"
        )

        print(
            "  Missing inside expected footprint:",
            missing_inside_coverage
        )

        print(
            "  Valid outside expected footprint :",
            unexpected_valid_outside
        )

validity_df = pd.DataFrame(
    validity_rows
)

validity_csv = (
    OUTDIR
    / "AVIRIS_Step8B_band_validity_QC.csv"
)

validity_df.to_csv(
    validity_csv,
    index=False
)

# ============================================================
# COVERAGE TRANSITION QC
#
# Check whether overlap boundaries contain spatial holes.
# ============================================================

valid_footprint = (
    coverage > 0
)

# Test internal changes in coverage.
horizontal_changes = np.sum(
    coverage[:, 1:]
    !=
    coverage[:, :-1]
)

vertical_changes = np.sum(
    coverage[1:, :]
    !=
    coverage[:-1, :]
)

print("\nCOVERAGE TRANSITIONS")
print("-" * 90)

print(
    "Horizontal coverage transitions:",
    f"{horizontal_changes:,}"
)

print(
    "Vertical coverage transitions:",
    f"{vertical_changes:,}"
)

# ============================================================
# FINAL SCIENTIFIC BAND LIST
# ============================================================

band_table = pd.DataFrame({

    "band":
        np.arange(
            1,
            426
        ),

    "wavelength_nm":
        wavelengths,

    "bbl":
        bbl,

    "scientific_use":
        np.where(
            bbl == 1,
            "USE",
            "EXCLUDE"
        ),
})

band_table_csv = (
    OUTDIR
    / "AVIRIS_Step8B_FINAL_band_table.csv"
)

band_table.to_csv(
    band_table_csv,
    index=False
)

# ============================================================
# RGB QC WITH COVERAGE BOUNDARIES
# ============================================================

def byte_stretch(
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

    if values.size == 0:
        return output

    lo, hi = np.percentile(
        values,
        [2, 98]
    )

    if hi <= lo:
        return output

    scaled = (
        array - lo
    ) / (
        hi - lo
    )

    scaled = np.clip(
        scaled,
        0,
        1
    )

    output = (
        scaled
        * 255
    ).astype(
        np.uint8
    )

    output[
        ~valid
    ] = 0

    return output


with rasterio.open(DAT) as src:

    rgb = src.read(
        RGB_BANDS
    ).astype(
        np.float32
    )

    rgb8 = np.zeros(
        rgb.shape,
        dtype=np.uint8
    )

    for i in range(3):

        valid = np.isfinite(
            rgb[i]
        )

        if src.nodata is not None:

            valid &= (
                rgb[i]
                !=
                src.nodata
            )

        rgb8[i] = byte_stretch(
            rgb[i],
            valid
        )


display = np.moveaxis(
    rgb8,
    0,
    -1
)

# Coverage boundaries.
boundary = np.zeros(
    coverage.shape,
    dtype=bool
)

boundary[:, 1:] |= (
    coverage[:, 1:]
    !=
    coverage[:, :-1]
)

boundary[1:, :] |= (
    coverage[1:, :]
    !=
    coverage[:-1, :]
)


plt.figure(
    figsize=(16, 9)
)

plt.imshow(
    display
)

# Overlay coverage transitions.
yy, xx = np.where(
    boundary
)

# Thin points to avoid overplotting.
if len(xx) > 100000:

    step = int(
        np.ceil(
            len(xx)
            /
            100000
        )
    )

    xx = xx[::step]
    yy = yy[::step]

plt.scatter(
    xx,
    yy,
    s=0.2,
    alpha=0.35
)

plt.title(
    "AVIRIS-NG Final Mosaic Step 8B QC\n"
    "RGB with coverage-transition locations"
)

plt.axis(
    "off"
)

plt.tight_layout()

overlay_png = (
    OUTDIR
    / "AVIRIS_Step8B_RGB_with_coverage_boundaries.png"
)

plt.savefig(
    overlay_png,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# GLOBAL SUMMARY
# ============================================================

missing_max = int(
    validity_df[
        "missing_pixels_inside_coverage"
    ].max()
)

outside_max = int(
    validity_df[
        "valid_pixels_outside_coverage"
    ].max()
)

status = (
    "PASS"
    if (
        missing_max == 0
        and
        outside_max == 0
        and
        len(wavelengths) == 425
        and
        np.sum(bbl) == 326
    )
    else
    "REVIEW"
)

summary = pd.DataFrame([{

    "width":
        width,

    "height":
        height,

    "bands":
        425,

    "scientific_bands":
        int(
            np.sum(bbl)
        ),

    "excluded_bands":
        int(
            np.sum(bbl == 0)
        ),

    "pixel_size_m":
        4.0,

    "crs":
        "EPSG:32643",

    "wavelength_min_nm":
        float(
            wavelengths.min()
        ),

    "wavelength_max_nm":
        float(
            wavelengths.max()
        ),

    "coverage_1_pixels":
        int(
            np.sum(
                coverage == 1
            )
        ),

    "coverage_2_pixels":
        int(
            np.sum(
                coverage == 2
            )
        ),

    "maximum_missing_pixels_inside_coverage":
        missing_max,

    "maximum_valid_pixels_outside_coverage":
        outside_max,

    "step8b_status":
        status,
}])


summary_csv = (
    OUTDIR
    / "AVIRIS_Step8B_FINAL_QC_summary.csv"
)

summary.to_csv(
    summary_csv,
    index=False
)


print("\n")
print("=" * 90)
print("STEP 8B FINAL RESULT")
print("=" * 90)

print(
    summary.to_string(
        index=False
    )
)

print("\nFinal QC:")
print(summary_csv)

print("\nBand table:")
print(band_table_csv)

print("\nValidity QC:")
print(validity_csv)

print("\nRGB coverage QC:")
print(overlay_png)

print("\nNo raster was modified.")

if status == "PASS":

    print(
        "\nFINAL MOSAIC STRUCTURAL QC: PASS"
    )

else:

    print(
        "\nFINAL MOSAIC STRUCTURAL QC: REVIEW REQUIRED"
    )
