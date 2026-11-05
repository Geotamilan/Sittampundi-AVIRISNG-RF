from pathlib import Path
import csv
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from skimage.registration import phase_cross_correlation
from scipy.ndimage import gaussian_filter, shift as ndi_shift
import matplotlib.pyplot as plt

# ============================================================
# SETTINGS
# ============================================================

ROOT = Path.cwd()

RGBDIR = ROOT / "02_Registered_Scenes" / "RGB_NorthUp_4m"
OUTDIR = ROOT / "02_Registered_Scenes" / "Registration_Check"

OUTDIR.mkdir(parents=True, exist_ok=True)

RES = 4.0

# Central scene 053932 is the main reference.
#
# Pair 1:
# 053216 will be aligned to 053932
#
# Pair 2:
# 054703 will be aligned to 053932
#
# Pair 3:
# 055421 will be aligned to 054703
#
PAIRS = [
    ("ang20160110t053932", "ang20160110t053216"),
    ("ang20160110t053932", "ang20160110t054703"),
    ("ang20160110t054703", "ang20160110t055421"),
]

# ============================================================
# FUNCTIONS
# ============================================================

def open_rgb(scene):
    path = RGBDIR / f"{scene}_RGB_NorthUp_4m.tif"

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def intersection_bounds(a, b):

    left = max(a.left, b.left)
    right = min(a.right, b.right)
    bottom = max(a.bottom, b.bottom)
    top = min(a.top, b.top)

    if left >= right or bottom >= top:
        return None

    return left, bottom, right, top


def read_on_common_overlap(reference_path, moving_path):

    with rasterio.open(reference_path) as ref, \
         rasterio.open(moving_path) as mov:

        if ref.crs != mov.crs:
            raise RuntimeError(
                f"CRS mismatch: {ref.crs} vs {mov.crs}"
            )

        overlap = intersection_bounds(
            ref.bounds,
            mov.bounds
        )

        if overlap is None:
            raise RuntimeError("No spatial overlap.")

        left, bottom, right, top = overlap

        # Snap overlap to 4 m grid.
        left = np.ceil(left / RES) * RES
        right = np.floor(right / RES) * RES
        bottom = np.ceil(bottom / RES) * RES
        top = np.floor(top / RES) * RES

        width = int(round((right - left) / RES))
        height = int(round((top - bottom) / RES))

        transform = from_origin(
            left,
            top,
            RES,
            RES
        )

        ref_array = np.zeros(
            (3, height, width),
            dtype=np.float32
        )

        mov_array = np.zeros(
            (3, height, width),
            dtype=np.float32
        )

        for band in range(1, 4):

            reproject(
                source=rasterio.band(ref, band),
                destination=ref_array[band - 1],
                src_transform=ref.transform,
                src_crs=ref.crs,
                dst_transform=transform,
                dst_crs=ref.crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.bilinear
            )

            reproject(
                source=rasterio.band(mov, band),
                destination=mov_array[band - 1],
                src_transform=mov.transform,
                src_crs=mov.crs,
                dst_transform=transform,
                dst_crs=mov.crs,
                src_nodata=0,
                dst_nodata=0,
                resampling=Resampling.bilinear
            )

    return (
        ref_array,
        mov_array,
        transform,
        overlap
    )


def make_registration_image(rgb):

    # Convert RGB to mean intensity.
    gray = np.mean(rgb, axis=0)

    valid = np.all(rgb > 0, axis=0)

    gray[~valid] = np.nan

    if not np.any(np.isfinite(gray)):
        raise RuntimeError("No valid overlap pixels.")

    # Replace invalid pixels with median.
    median = np.nanmedian(gray)

    gray = np.where(
        np.isfinite(gray),
        gray,
        median
    )

    # Remove broad illumination differences.
    smooth = gaussian_filter(
        gray,
        sigma=8
    )

    highpass = gray - smooth

    # Normalize.
    mean = np.mean(highpass)
    std = np.std(highpass)

    if std > 0:
        highpass = (
            highpass - mean
        ) / std

    # Hanning window suppresses edge artefacts.
    wy = np.hanning(highpass.shape[0])
    wx = np.hanning(highpass.shape[1])

    window = np.outer(wy, wx)

    return highpass * window


def correlation(a, b):

    a = a.ravel()
    b = b.ravel()

    mask = (
        np.isfinite(a) &
        np.isfinite(b)
    )

    if mask.sum() < 100:
        return np.nan

    return np.corrcoef(
        a[mask],
        b[mask]
    )[0, 1]


# ============================================================
# REGISTRATION
# ============================================================

results = []

print("=" * 80)
print("AVIRIS-NG RESIDUAL REGISTRATION ANALYSIS")
print("=" * 80)

for reference_scene, moving_scene in PAIRS:

    print("\n" + "=" * 80)
    print("REFERENCE :", reference_scene)
    print("MOVING    :", moving_scene)
    print("=" * 80)

    ref_path = open_rgb(reference_scene)
    mov_path = open_rgb(moving_scene)

    ref_rgb, mov_rgb, transform, overlap = \
        read_on_common_overlap(
            ref_path,
            mov_path
        )

    print("\nOverlap array:")
    print(
        ref_rgb.shape[2],
        "columns x",
        ref_rgb.shape[1],
        "rows"
    )

    print("Overlap geographic bounds:")
    print(overlap)

    ref_reg = make_registration_image(
        ref_rgb
    )

    mov_reg = make_registration_image(
        mov_rgb
    )

    # --------------------------------------------------------
    # SUBPIXEL PHASE CORRELATION
    # --------------------------------------------------------

    shift_pixels, error, phasediff = \
        phase_cross_correlation(
            ref_reg,
            mov_reg,
            upsample_factor=20
        )

    row_shift = float(shift_pixels[0])
    col_shift = float(shift_pixels[1])

    # Convert pixel displacement to map displacement.
    #
    # Positive column shift = EAST
    # Negative column shift = WEST
    #
    # Positive row shift means image moves downward,
    # therefore SOUTH.
    #
    east_shift_m = col_shift * RES
    north_shift_m = -row_shift * RES

    total_shift_m = np.sqrt(
        east_shift_m**2 +
        north_shift_m**2
    )

    print("\nREGISTRATION RESULT")
    print("-------------------")

    print(
        f"Row shift to apply    : "
        f"{row_shift:.3f} pixels"
    )

    print(
        f"Column shift to apply : "
        f"{col_shift:.3f} pixels"
    )

    print(
        f"Easting correction    : "
        f"{east_shift_m:.3f} m"
    )

    print(
        f"Northing correction   : "
        f"{north_shift_m:.3f} m"
    )

    print(
        f"Total displacement    : "
        f"{total_shift_m:.3f} m"
    )

    print(
        f"Phase-correlation error: "
        f"{error:.6f}"
    )

    # --------------------------------------------------------
    # CORRELATION BEFORE / AFTER
    # --------------------------------------------------------

    before_corr = correlation(
        ref_reg,
        mov_reg
    )

    corrected = ndi_shift(
        mov_reg,
        shift=(row_shift, col_shift),
        order=1,
        mode="constant",
        cval=0
    )

    after_corr = correlation(
        ref_reg,
        corrected
    )

    print(
        f"Correlation before    : "
        f"{before_corr:.4f}"
    )

    print(
        f"Correlation after     : "
        f"{after_corr:.4f}"
    )

    # --------------------------------------------------------
    # SAVE DIAGNOSTIC FIGURE
    # --------------------------------------------------------

    reference_display = np.moveaxis(
        ref_rgb,
        0,
        -1
    )

    moving_display = np.moveaxis(
        mov_rgb,
        0,
        -1
    )

    # Normalize independently for display.
    reference_display = np.clip(
        reference_display / 255.0,
        0,
        1
    )

    moving_display = np.clip(
        moving_display / 255.0,
        0,
        1
    )

    fig = plt.figure(
        figsize=(15, 10)
    )

    ax1 = fig.add_subplot(2, 2, 1)

    ax1.imshow(reference_display)
    ax1.set_title(
        f"Reference\n{reference_scene}"
    )
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)

    ax2.imshow(moving_display)
    ax2.set_title(
        f"Moving\n{moving_scene}"
    )
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)

    before_difference = (
        ref_reg - mov_reg
    )

    ax3.imshow(
        before_difference,
        cmap="gray"
    )

    ax3.set_title(
        f"Before registration\n"
        f"Correlation = {before_corr:.3f}"
    )

    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)

    after_difference = (
        ref_reg - corrected
    )

    ax4.imshow(
        after_difference,
        cmap="gray"
    )

    ax4.set_title(
        f"After registration\n"
        f"Correlation = {after_corr:.3f}"
    )

    ax4.axis("off")

    fig.suptitle(
        f"{moving_scene} → {reference_scene}\n"
        f"E = {east_shift_m:.2f} m, "
        f"N = {north_shift_m:.2f} m, "
        f"Total = {total_shift_m:.2f} m"
    )

    plt.tight_layout()

    figure_path = (
        OUTDIR /
        f"{moving_scene}_to_{reference_scene}_registration.png"
    )

    plt.savefig(
        figure_path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nDiagnostic saved:")
    print(figure_path)

    results.append({
        "reference_scene": reference_scene,
        "moving_scene": moving_scene,
        "row_shift_pixels": row_shift,
        "column_shift_pixels": col_shift,
        "east_shift_m": east_shift_m,
        "north_shift_m": north_shift_m,
        "total_shift_m": total_shift_m,
        "correlation_before": before_corr,
        "correlation_after": after_corr,
        "phase_error": float(error)
    })


# ============================================================
# CUMULATIVE CORRECTIONS
# ============================================================

print("\n")
print("=" * 80)
print("CUMULATIVE CORRECTIONS TO CENTRAL REFERENCE 053932")
print("=" * 80)

lookup = {
    row["moving_scene"]: row
    for row in results
}

# Central reference
cumulative = {
    "ang20160110t053932": (0.0, 0.0)
}

# 053216 -> 053932
r = lookup["ang20160110t053216"]

cumulative[
    "ang20160110t053216"
] = (
    r["east_shift_m"],
    r["north_shift_m"]
)

# 054703 -> 053932
r = lookup["ang20160110t054703"]

cumulative[
    "ang20160110t054703"
] = (
    r["east_shift_m"],
    r["north_shift_m"]
)

# 055421 -> 054703 -> 053932
r55 = lookup["ang20160110t055421"]

e54, n54 = cumulative[
    "ang20160110t054703"
]

cumulative[
    "ang20160110t055421"
] = (
    e54 + r55["east_shift_m"],
    n54 + r55["north_shift_m"]
)

for scene, (east, north) in cumulative.items():

    total = np.sqrt(
        east**2 +
        north**2
    )

    print(
        f"\n{scene}"
    )

    print(
        f"  East correction : {east:.3f} m"
    )

    print(
        f"  North correction: {north:.3f} m"
    )

    print(
        f"  Total correction: {total:.3f} m"
    )


# ============================================================
# SAVE PAIRWISE CSV
# ============================================================

csv_path = (
    OUTDIR /
    "AVIRIS_pairwise_registration_results.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=results[0].keys()
    )

    writer.writeheader()
    writer.writerows(results)

print("\n")
print("=" * 80)
print("REGISTRATION MEASUREMENT COMPLETE")
print("=" * 80)

print("\nResults saved:")
print(csv_path)

print("\nIMPORTANT:")
print(
    "No shift has yet been applied to the original "
    "425-band AVIRIS-NG cubes."
)
print(
    "Review the measured offsets and diagnostic "
    "figures before applying corrections."
)
