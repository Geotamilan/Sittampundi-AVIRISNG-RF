from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter, shift as ndi_shift
import matplotlib.pyplot as plt
import spectral as spy

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path.cwd()

INDIR = (
    ROOT
    / "02_Registered_Scenes"
    / "Registered_425Band_4m"
)

OUTDIR = (
    ROOT
    / "02_Registered_Scenes"
    / "Post_Registration_QC"
)

OUTDIR.mkdir(parents=True, exist_ok=True)

SCENES = [
    "ang20160110t053216",
    "ang20160110t053932",
    "ang20160110t054703",
    "ang20160110t055421",
]

PAIRS = [
    ("ang20160110t053932", "ang20160110t053216"),
    ("ang20160110t053932", "ang20160110t054703"),
    ("ang20160110t054703", "ang20160110t055421"),
]

RGB_BANDS = [54, 36, 20]

RESOLUTION = 4.0
EXPECTED_BANDS = 425

# QC criterion
PASS_LIMIT_PIXELS = 1.0


# ============================================================
# HELPERS
# ============================================================

def dat_path(scene):
    return INDIR / f"{scene}_registered_4m.dat"


def hdr_path(scene):
    return INDIR / f"{scene}_registered_4m.hdr"


def overlap_bounds(a, b):

    left = max(a.left, b.left)
    right = min(a.right, b.right)
    bottom = max(a.bottom, b.bottom)
    top = min(a.top, b.top)

    if left >= right or bottom >= top:
        return None

    return left, bottom, right, top


def robust_byte(arr, mask):

    out = np.zeros(arr.shape, dtype=np.uint8)

    values = arr[mask]

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        return out

    lo, hi = np.percentile(
        values,
        [2, 98]
    )

    if hi <= lo:
        return out

    scaled = (
        arr - lo
    ) / (
        hi - lo
    )

    scaled = np.clip(
        scaled,
        0,
        1
    )

    out = (
        scaled * 255
    ).astype(np.uint8)

    out[~mask] = 0

    return out


def prepare_registration_image(rgb, valid):

    gray = np.mean(
        rgb,
        axis=0
    ).astype(np.float32)

    gray[~valid] = np.nan

    median = np.nanmedian(gray)

    gray = np.where(
        np.isfinite(gray),
        gray,
        median
    )

    # Remove low-frequency illumination differences
    smooth = gaussian_filter(
        gray,
        sigma=8
    )

    highpass = gray - smooth

    mean = np.mean(highpass)
    std = np.std(highpass)

    if std > 0:
        highpass = (
            highpass - mean
        ) / std

    # Reduce FFT edge effects
    wy = np.hanning(
        highpass.shape[0]
    )

    wx = np.hanning(
        highpass.shape[1]
    )

    window = np.outer(
        wy,
        wx
    )

    return (
        highpass * window
    )


def correlation(a, b, mask):

    valid = (
        mask
        & np.isfinite(a)
        & np.isfinite(b)
    )

    if np.sum(valid) < 100:
        return np.nan

    return np.corrcoef(
        a[valid],
        b[valid]
    )[0, 1]


# ============================================================
# PART 1 — DATASET VALIDATION
# ============================================================

print("=" * 84)
print("AVIRIS-NG POST-REGISTRATION QC")
print("=" * 84)

print("\nDATASET VALIDATION")
print("-" * 84)

scene_metadata = {}

reference_wavelengths = None

for scene in SCENES:

    dat = dat_path(scene)
    hdr = hdr_path(scene)

    if not dat.exists():
        raise FileNotFoundError(dat)

    if not hdr.exists():
        raise FileNotFoundError(hdr)

    print("\n" + scene)

    with rasterio.open(dat) as src:

        print(
            "  Dimensions :",
            src.width,
            "x",
            src.height
        )

        print(
            "  Bands      :",
            src.count
        )

        print(
            "  CRS        :",
            src.crs
        )

        print(
            "  Resolution :",
            src.res
        )

        print(
            "  Nodata     :",
            src.nodata
        )

        print(
            "  Transform  :"
        )

        print(
            " ",
            src.transform
        )

        if src.count != EXPECTED_BANDS:
            raise RuntimeError(
                f"{scene} does not contain "
                f"{EXPECTED_BANDS} bands."
            )

        if str(src.crs) != "EPSG:32643":
            raise RuntimeError(
                f"{scene} CRS is not EPSG:32643."
            )

        if not (
            np.isclose(
                src.res[0],
                RESOLUTION
            )
            and
            np.isclose(
                src.res[1],
                RESOLUTION
            )
        ):
            raise RuntimeError(
                f"{scene} is not 4 m."
            )

        # Common-grid lattice test
        x_phase = (
            src.transform.c
            % RESOLUTION
        )

        y_phase = (
            src.transform.f
            % RESOLUTION
        )

        print(
            f"  Grid phase X: "
            f"{x_phase:.6f}"
        )

        print(
            f"  Grid phase Y: "
            f"{y_phase:.6f}"
        )

        scene_metadata[scene] = {
            "bounds": src.bounds,
            "transform": src.transform,
            "nodata": src.nodata
        }

    # --------------------------------------------------------
    # SPECTRAL METADATA CHECK
    # --------------------------------------------------------

    img = spy.open_image(
        str(hdr)
    )

    md = img.metadata

    if "wavelength" not in md:
        raise RuntimeError(
            f"Wavelength metadata missing "
            f"from {scene}"
        )

    wavelengths = np.asarray(
        md["wavelength"],
        dtype=float
    )

    print(
        "  Wavelengths:",
        len(wavelengths)
    )

    print(
        f"  Spectral range: "
        f"{wavelengths.min():.3f} - "
        f"{wavelengths.max():.3f} nm"
    )

    print(
        f"  RGB wavelengths: "
        f"{wavelengths[53]:.2f}, "
        f"{wavelengths[35]:.2f}, "
        f"{wavelengths[19]:.2f} nm"
    )

    if len(wavelengths) != EXPECTED_BANDS:
        raise RuntimeError(
            f"{scene}: wavelength count "
            f"is not 425."
        )

    if reference_wavelengths is None:

        reference_wavelengths = wavelengths

    else:

        if not np.allclose(
            wavelengths,
            reference_wavelengths,
            atol=0.001
        ):

            raise RuntimeError(
                f"{scene}: wavelengths differ "
                f"from reference."
            )


print("\n")
print("DATASET VALIDATION: PASS")


# ============================================================
# PART 2 — PAIRWISE OVERLAP REGISTRATION QC
# ============================================================

print("\n")
print("=" * 84)
print("PAIRWISE OVERLAP QC")
print("=" * 84)

results = []


for reference_scene, moving_scene in PAIRS:

    print("\n")
    print("=" * 84)

    print(
        "REFERENCE:",
        reference_scene
    )

    print(
        "MOVING   :",
        moving_scene
    )

    print("=" * 84)

    ref_file = dat_path(
        reference_scene
    )

    mov_file = dat_path(
        moving_scene
    )

    with rasterio.open(ref_file) as ref, \
         rasterio.open(mov_file) as mov:

        overlap = overlap_bounds(
            ref.bounds,
            mov.bounds
        )

        if overlap is None:
            raise RuntimeError(
                "No overlap between scenes."
            )

        left, bottom, right, top = overlap

        # Snap to common 4-m lattice
        left = np.ceil(
            left / RESOLUTION
        ) * RESOLUTION

        right = np.floor(
            right / RESOLUTION
        ) * RESOLUTION

        bottom = np.ceil(
            bottom / RESOLUTION
        ) * RESOLUTION

        top = np.floor(
            top / RESOLUTION
        ) * RESOLUTION

        width = int(
            round(
                (right - left)
                / RESOLUTION
            )
        )

        height = int(
            round(
                (top - bottom)
                / RESOLUTION
            )
        )

        transform = from_origin(
            left,
            top,
            RESOLUTION,
            RESOLUTION
        )

        print(
            "\nOverlap size:",
            width,
            "x",
            height
        )

        print(
            "Overlap bounds:",
            (
                left,
                bottom,
                right,
                top
            )
        )

        ref_rgb = np.full(
            (3, height, width),
            np.nan,
            dtype=np.float32
        )

        mov_rgb = np.full(
            (3, height, width),
            np.nan,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # READ BANDS 54, 36, 20 FROM THE REGISTERED CUBES
        # ----------------------------------------------------

        for channel, band in enumerate(
            RGB_BANDS
        ):

            reproject(

                source=rasterio.band(
                    ref,
                    band
                ),

                destination=
                    ref_rgb[channel],

                src_transform=
                    ref.transform,

                src_crs=
                    ref.crs,

                dst_transform=
                    transform,

                dst_crs=
                    ref.crs,

                src_nodata=
                    ref.nodata,

                dst_nodata=
                    np.nan,

                resampling=
                    Resampling.bilinear
            )

            reproject(

                source=rasterio.band(
                    mov,
                    band
                ),

                destination=
                    mov_rgb[channel],

                src_transform=
                    mov.transform,

                src_crs=
                    mov.crs,

                dst_transform=
                    transform,

                dst_crs=
                    mov.crs,

                src_nodata=
                    mov.nodata,

                dst_nodata=
                    np.nan,

                resampling=
                    Resampling.bilinear
            )

    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    valid = (
        np.all(
            np.isfinite(ref_rgb),
            axis=0
        )
        &
        np.all(
            np.isfinite(mov_rgb),
            axis=0
        )
    )

    valid_count = np.sum(valid)

    print(
        "Valid overlapping pixels:",
        f"{valid_count:,}"
    )

    if valid_count < 1000:
        raise RuntimeError(
            "Too few valid overlap pixels."
        )

    # --------------------------------------------------------
    # CREATE NORMALIZED REGISTRATION IMAGES
    # --------------------------------------------------------

    ref_reg = prepare_registration_image(
        ref_rgb,
        valid
    )

    mov_reg = prepare_registration_image(
        mov_rgb,
        valid
    )

    # --------------------------------------------------------
    # MEASURE RESIDUAL SHIFT
    # --------------------------------------------------------

    shift_pixels, error, phasediff = (
        phase_cross_correlation(
            ref_reg,
            mov_reg,
            upsample_factor=50
        )
    )

    row_shift = float(
        shift_pixels[0]
    )

    col_shift = float(
        shift_pixels[1]
    )

    east_m = (
        col_shift
        * RESOLUTION
    )

    north_m = (
        -row_shift
        * RESOLUTION
    )

    total_pixels = np.sqrt(
        row_shift ** 2
        +
        col_shift ** 2
    )

    total_m = (
        total_pixels
        * RESOLUTION
    )

    # --------------------------------------------------------
    # CORRELATION
    # --------------------------------------------------------

    before_corr = correlation(
        ref_reg,
        mov_reg,
        valid
    )

    corrected_reg = ndi_shift(
        mov_reg,
        shift=(
            row_shift,
            col_shift
        ),
        order=1,
        mode="constant",
        cval=0
    )

    after_corr = correlation(
        ref_reg,
        corrected_reg,
        valid
    )

    # --------------------------------------------------------
    # QC STATUS
    # --------------------------------------------------------

    if total_pixels <= PASS_LIMIT_PIXELS:

        qc_status = "PASS"

    else:

        qc_status = "REVIEW"


    print("\nRESIDUAL REGISTRATION")
    print("-" * 50)

    print(
        f"Row residual     : "
        f"{row_shift:+.3f} pixels"
    )

    print(
        f"Column residual  : "
        f"{col_shift:+.3f} pixels"
    )

    print(
        f"Easting residual : "
        f"{east_m:+.3f} m"
    )

    print(
        f"Northing residual: "
        f"{north_m:+.3f} m"
    )

    print(
        f"Total residual   : "
        f"{total_pixels:.3f} pixels "
        f"({total_m:.3f} m)"
    )

    print(
        f"Correlation      : "
        f"{before_corr:.4f}"
    )

    print(
        f"QC status        : "
        f"{qc_status}"
    )

    # ========================================================
    # RGB DISPLAY
    # ========================================================

    ref_byte = np.zeros(
        ref_rgb.shape,
        dtype=np.uint8
    )

    mov_byte = np.zeros(
        mov_rgb.shape,
        dtype=np.uint8
    )

    for channel in range(3):

        ref_byte[channel] = robust_byte(
            ref_rgb[channel],
            valid
        )

        mov_byte[channel] = robust_byte(
            mov_rgb[channel],
            valid
        )

    ref_display = np.moveaxis(
        ref_byte,
        0,
        -1
    )

    mov_display = np.moveaxis(
        mov_byte,
        0,
        -1
    )

    # --------------------------------------------------------
    # BLEND
    # --------------------------------------------------------

    blend = (
        0.5 * ref_display.astype(
            np.float32
        )
        +
        0.5 * mov_display.astype(
            np.float32
        )
    ).astype(np.uint8)

    # --------------------------------------------------------
    # CHECKERBOARD
    # --------------------------------------------------------

    checker = ref_display.copy()

    tile = 100

    yy, xx = np.indices(
        valid.shape
    )

    checker_mask = (
        (
            xx // tile
            +
            yy // tile
        )
        % 2
        == 1
    )

    checker[
        checker_mask
    ] = mov_display[
        checker_mask
    ]

    # --------------------------------------------------------
    # DIFFERENCE IMAGE
    # --------------------------------------------------------

    ref_gray = np.mean(
        ref_display.astype(
            np.float32
        ),
        axis=2
    )

    mov_gray = np.mean(
        mov_display.astype(
            np.float32
        ),
        axis=2
    )

    difference = (
        ref_gray
        -
        mov_gray
    )

    # ========================================================
    # SAVE QC FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(16, 11)
    )

    ax1 = fig.add_subplot(
        2,
        3,
        1
    )

    ax1.imshow(
        ref_display
    )

    ax1.set_title(
        f"Reference\n"
        f"{reference_scene}"
    )

    ax1.axis("off")


    ax2 = fig.add_subplot(
        2,
        3,
        2
    )

    ax2.imshow(
        mov_display
    )

    ax2.set_title(
        f"Moving\n"
        f"{moving_scene}"
    )

    ax2.axis("off")


    ax3 = fig.add_subplot(
        2,
        3,
        3
    )

    ax3.imshow(
        blend
    )

    ax3.set_title(
        "50/50 Blend"
    )

    ax3.axis("off")


    ax4 = fig.add_subplot(
        2,
        3,
        4
    )

    ax4.imshow(
        checker
    )

    ax4.set_title(
        "Checkerboard"
    )

    ax4.axis("off")


    ax5 = fig.add_subplot(
        2,
        3,
        5
    )

    ax5.imshow(
        difference,
        cmap="gray"
    )

    ax5.set_title(
        "RGB Intensity Difference"
    )

    ax5.axis("off")


    ax6 = fig.add_subplot(
        2,
        3,
        6
    )

    ax6.text(
        0.05,
        0.85,
        "Residual Registration",
        fontsize=14,
        weight="bold"
    )

    ax6.text(
        0.05,
        0.70,
        f"Row: {row_shift:+.3f} px"
    )

    ax6.text(
        0.05,
        0.60,
        f"Column: {col_shift:+.3f} px"
    )

    ax6.text(
        0.05,
        0.50,
        f"East: {east_m:+.2f} m"
    )

    ax6.text(
        0.05,
        0.40,
        f"North: {north_m:+.2f} m"
    )

    ax6.text(
        0.05,
        0.30,
        f"Total: {total_m:.2f} m"
    )

    ax6.text(
        0.05,
        0.20,
        f"Correlation: {before_corr:.3f}"
    )

    ax6.text(
        0.05,
        0.08,
        f"QC: {qc_status}",
        fontsize=16,
        weight="bold"
    )

    ax6.axis("off")

    fig.suptitle(
        f"AVIRIS-NG Post-Registration QC\n"
        f"{moving_scene} → "
        f"{reference_scene}",
        fontsize=16
    )

    plt.tight_layout()

    figure_path = (
        OUTDIR
        /
        f"{moving_scene}_to_"
        f"{reference_scene}_QC.png"
    )

    plt.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "\nQC image saved:"
    )

    print(
        figure_path
    )

    results.append({

        "reference_scene":
            reference_scene,

        "moving_scene":
            moving_scene,

        "row_residual_pixels":
            row_shift,

        "column_residual_pixels":
            col_shift,

        "east_residual_m":
            east_m,

        "north_residual_m":
            north_m,

        "total_residual_pixels":
            total_pixels,

        "total_residual_m":
            total_m,

        "correlation":
            before_corr,

        "phase_error":
            float(error),

        "qc_status":
            qc_status,
    })


# ============================================================
# WRITE CSV
# ============================================================

csv_output = (
    OUTDIR
    /
    "AVIRIS_post_registration_QC.csv"
)

with csv_output.open(
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=
            results[0].keys()
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 84)
print("POST-REGISTRATION QC SUMMARY")
print("=" * 84)

all_pass = True

for result in results:

    print(
        f"\n{result['moving_scene']}"
        f" → "
        f"{result['reference_scene']}"
    )

    print(
        f"  Residual: "
        f"{result['total_residual_pixels']:.3f} px "
        f"= "
        f"{result['total_residual_m']:.3f} m"
    )

    print(
        f"  Correlation: "
        f"{result['correlation']:.4f}"
    )

    print(
        f"  Status: "
        f"{result['qc_status']}"
    )

    if (
        result["qc_status"]
        != "PASS"
    ):
        all_pass = False


print("\nQC CSV:")
print(csv_output)

print("\n")

if all_pass:

    print(
        "OVERALL REGISTRATION QC: PASS"
    )

    print(
        "All adjacent scene pairs have "
        "residual displacement <= "
        f"{PASS_LIMIT_PIXELS} pixel."
    )

    print(
        "\nThe four registered 425-band "
        "cubes are ready for mosaic preparation."
    )

else:

    print(
        "OVERALL REGISTRATION QC: REVIEW REQUIRED"
    )

    print(
        "At least one overlap has residual "
        "displacement > "
        f"{PASS_LIMIT_PIXELS} pixel."
    )

    print(
        "\nDo NOT mosaic until the flagged "
        "overlap has been inspected."
    )
