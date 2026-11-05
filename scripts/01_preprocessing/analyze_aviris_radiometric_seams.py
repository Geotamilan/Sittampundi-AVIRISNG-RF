from pathlib import Path
import csv
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import spectral as spy
import matplotlib.pyplot as plt


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
    / "03_Mosaic"
    / "Radiometric_Seam_Analysis"
)

OUTDIR.mkdir(parents=True, exist_ok=True)


PAIRS = [
    (
        "ang20160110t053932",
        "ang20160110t053216"
    ),
    (
        "ang20160110t053932",
        "ang20160110t054703"
    ),
    (
        "ang20160110t054703",
        "ang20160110t055421"
    ),
]


EXPECTED_BANDS = 425

# Maximum paired samples used for each band.
# This keeps processing fast while retaining strong statistics.
MAX_SAMPLES_PER_BAND = 150000

# For regression, ignore extremely small datasets.
MIN_REGRESSION_SAMPLES = 100

# Ratio is only calculated where both values are positive.
MIN_POSITIVE_VALUE = 1e-8


# ============================================================
# FILE FUNCTIONS
# ============================================================

def dat_path(scene):

    return (
        INDIR
        / f"{scene}_registered_4m.dat"
    )


def hdr_path(scene):

    return (
        INDIR
        / f"{scene}_registered_4m.hdr"
    )


def intersection_bounds(a, b):

    left = max(
        a.left,
        b.left
    )

    right = min(
        a.right,
        b.right
    )

    bottom = max(
        a.bottom,
        b.bottom
    )

    top = min(
        a.top,
        b.top
    )

    if (
        left >= right
        or
        bottom >= top
    ):

        return None

    return (
        left,
        bottom,
        right,
        top
    )


def get_window(src, bounds):

    left, bottom, right, top = bounds

    window = from_bounds(
        left,
        bottom,
        right,
        top,
        transform=src.transform
    )

    window = (
        window
        .round_offsets()
        .round_lengths()
    )

    return window


# ============================================================
# VALID PIXELS
# ============================================================

def valid_mask(
    ref,
    mov,
    ref_nodata,
    mov_nodata
):

    mask = (
        np.isfinite(ref)
        &
        np.isfinite(mov)
    )

    if ref_nodata is not None:

        mask &= (
            ref != ref_nodata
        )

    if mov_nodata is not None:

        mask &= (
            mov != mov_nodata
        )

    return mask


# ============================================================
# SYSTEMATIC SAMPLE
# ============================================================

def sample_pairs(
    ref_values,
    mov_values,
    max_samples
):

    n = ref_values.size

    if n <= max_samples:

        return (
            ref_values,
            mov_values
        )

    step = int(
        math.ceil(
            n / max_samples
        )
    )

    return (
        ref_values[::step],
        mov_values[::step]
    )


# ============================================================
# CORRELATION
# ============================================================

def pearson_r(x, y):

    if x.size < 3:
        return np.nan

    sx = np.std(x)
    sy = np.std(y)

    if (
        sx == 0
        or
        sy == 0
    ):

        return np.nan

    return float(
        np.corrcoef(
            x,
            y
        )[0, 1]
    )


# ============================================================
# LINEAR NORMALIZATION
#
# reference = gain * moving + offset
# ============================================================

def fit_gain_offset(
    moving,
    reference
):

    if (
        moving.size
        < MIN_REGRESSION_SAMPLES
    ):

        return (
            np.nan,
            np.nan
        )

    variance = np.var(
        moving
    )

    if variance <= 0:

        return (
            np.nan,
            np.nan
        )

    covariance = np.mean(
        (
            moving
            - np.mean(moving)
        )
        *
        (
            reference
            - np.mean(reference)
        )
    )

    gain = (
        covariance
        / variance
    )

    offset = (
        np.mean(reference)
        -
        gain
        *
        np.mean(moving)
    )

    return (
        float(gain),
        float(offset)
    )


# ============================================================
# LOAD WAVELENGTHS
# ============================================================

reference_hdr = hdr_path(
    "ang20160110t053932"
)

if not reference_hdr.exists():

    raise FileNotFoundError(
        reference_hdr
    )


spectral_img = spy.open_image(
    str(reference_hdr)
)

if (
    "wavelength"
    not in spectral_img.metadata
):

    raise RuntimeError(
        "Wavelength metadata missing."
    )


wavelengths = np.asarray(
    spectral_img.metadata[
        "wavelength"
    ],
    dtype=float
)


if len(wavelengths) != EXPECTED_BANDS:

    raise RuntimeError(
        "Expected 425 wavelengths, "
        f"found {len(wavelengths)}."
    )


print("=" * 88)

print(
    "AVIRIS-NG 425-BAND "
    "RADIOMETRIC / SEAM ANALYSIS"
)

print("=" * 88)

print(
    f"\nSpectral range: "
    f"{wavelengths.min():.3f} "
    f"to "
    f"{wavelengths.max():.3f} nm"
)

print(
    f"Bands: "
    f"{len(wavelengths)}"
)

print(
    "\nIMPORTANT:"
)

print(
    "This program performs analysis only."
)

print(
    "No registered hyperspectral "
    "dataset will be modified."
)


# ============================================================
# GLOBAL STORAGE
# ============================================================

all_results = []


# ============================================================
# PROCESS EACH OVERLAP
# ============================================================

for pair_number, (
    reference_scene,
    moving_scene
) in enumerate(
    PAIRS,
    start=1
):

    print("\n")
    print("=" * 88)

    print(
        f"PAIR {pair_number}/"
        f"{len(PAIRS)}"
    )

    print(
        "REFERENCE :",
        reference_scene
    )

    print(
        "MOVING    :",
        moving_scene
    )

    print("=" * 88)


    reference_file = dat_path(
        reference_scene
    )

    moving_file = dat_path(
        moving_scene
    )


    if not reference_file.exists():

        raise FileNotFoundError(
            reference_file
        )


    if not moving_file.exists():

        raise FileNotFoundError(
            moving_file
        )


    pair_results = []


    with rasterio.open(
        reference_file
    ) as ref, \
         rasterio.open(
             moving_file
         ) as mov:


        # ====================================================
        # BASIC COMPATIBILITY
        # ====================================================

        if (
            ref.count != EXPECTED_BANDS
            or
            mov.count != EXPECTED_BANDS
        ):

            raise RuntimeError(
                "Band count mismatch."
            )


        if ref.crs != mov.crs:

            raise RuntimeError(
                "CRS mismatch."
            )


        overlap = intersection_bounds(
            ref.bounds,
            mov.bounds
        )


        if overlap is None:

            raise RuntimeError(
                "No spatial overlap."
            )


        ref_window = get_window(
            ref,
            overlap
        )

        mov_window = get_window(
            mov,
            overlap
        )


        print(
            "\nOverlap bounds:"
        )

        print(
            overlap
        )


        print(
            "\nReference window:"
        )

        print(
            ref_window
        )


        print(
            "\nMoving window:"
        )

        print(
            mov_window
        )


        ref_height = int(
            ref_window.height
        )

        ref_width = int(
            ref_window.width
        )

        mov_height = int(
            mov_window.height
        )

        mov_width = int(
            mov_window.width
        )


        # Grid alignment should make these equal.
        height = min(
            ref_height,
            mov_height
        )

        width = min(
            ref_width,
            mov_width
        )


        print(
            f"\nAnalysis overlap: "
            f"{width} x {height} pixels"
        )


        print(
            f"Potential paired pixels: "
            f"{width * height:,}"
        )


        # ====================================================
        # BAND LOOP
        # ====================================================

        for band in range(
            1,
            EXPECTED_BANDS + 1
        ):


            ref_array = ref.read(
                band,
                window=ref_window,
                out_shape=(
                    height,
                    width
                ),
                resampling=
                    rasterio.enums
                    .Resampling
                    .nearest
            ).astype(
                np.float32
            )


            mov_array = mov.read(
                band,
                window=mov_window,
                out_shape=(
                    height,
                    width
                ),
                resampling=
                    rasterio.enums
                    .Resampling
                    .nearest
            ).astype(
                np.float32
            )


            mask = valid_mask(
                ref_array,
                mov_array,
                ref.nodata,
                mov.nodata
            )


            ref_values = (
                ref_array[mask]
                .astype(
                    np.float64
                )
            )


            mov_values = (
                mov_array[mask]
                .astype(
                    np.float64
                )
            )


            total_valid = (
                ref_values.size
            )


            if total_valid == 0:

                record = {

                    "reference_scene":
                        reference_scene,

                    "moving_scene":
                        moving_scene,

                    "band":
                        band,

                    "wavelength_nm":
                        wavelengths[
                            band - 1
                        ],

                    "valid_pixels":
                        0,
                }

                pair_results.append(
                    record
                )

                all_results.append(
                    record
                )

                continue


            # ================================================
            # SYSTEMATIC SAMPLE
            # ================================================

            (
                ref_sample,
                mov_sample
            ) = sample_pairs(
                ref_values,
                mov_values,
                MAX_SAMPLES_PER_BAND
            )


            n = ref_sample.size


            # ================================================
            # BASIC STATISTICS
            # ================================================

            ref_mean = float(
                np.mean(
                    ref_sample
                )
            )

            mov_mean = float(
                np.mean(
                    mov_sample
                )
            )


            ref_median = float(
                np.median(
                    ref_sample
                )
            )

            mov_median = float(
                np.median(
                    mov_sample
                )
            )


            difference = (
                mov_sample
                - ref_sample
            )


            mean_bias = float(
                np.mean(
                    difference
                )
            )


            median_bias = float(
                np.median(
                    difference
                )
            )


            mae = float(
                np.mean(
                    np.abs(
                        difference
                    )
                )
            )


            rmse = float(
                np.sqrt(
                    np.mean(
                        difference ** 2
                    )
                )
            )


            correlation = pearson_r(
                ref_sample,
                mov_sample
            )


            # ================================================
            # NORMALIZED RMSE
            # ================================================

            ref_range = (
                np.percentile(
                    ref_sample,
                    98
                )
                -
                np.percentile(
                    ref_sample,
                    2
                )
            )


            if ref_range != 0:

                nrmse_percent = (
                    rmse
                    /
                    ref_range
                    *
                    100.0
                )

            else:

                nrmse_percent = np.nan


            # ================================================
            # RELATIVE MEDIAN BIAS
            # ================================================

            if (
                abs(ref_median)
                > MIN_POSITIVE_VALUE
            ):

                median_bias_percent = (
                    median_bias
                    /
                    abs(
                        ref_median
                    )
                    *
                    100.0
                )

            else:

                median_bias_percent = (
                    np.nan
                )


            # ================================================
            # SPECTRAL RATIO
            #
            # moving / reference
            # ================================================

            ratio_mask = (
                (
                    ref_sample
                    > MIN_POSITIVE_VALUE
                )
                &
                (
                    mov_sample
                    > MIN_POSITIVE_VALUE
                )
            )


            if np.any(
                ratio_mask
            ):

                ratio = (
                    mov_sample[
                        ratio_mask
                    ]
                    /
                    ref_sample[
                        ratio_mask
                    ]
                )


                median_ratio = float(
                    np.median(
                        ratio
                    )
                )


                mean_ratio = float(
                    np.mean(
                        ratio
                    )
                )


            else:

                median_ratio = np.nan
                mean_ratio = np.nan


            # ================================================
            # GAIN / OFFSET
            #
            # reference =
            # gain * moving + offset
            # ================================================

            (
                gain_to_reference,
                offset_to_reference
            ) = fit_gain_offset(
                mov_sample,
                ref_sample
            )


            record = {

                "reference_scene":
                    reference_scene,

                "moving_scene":
                    moving_scene,

                "band":
                    band,

                "wavelength_nm":
                    float(
                        wavelengths[
                            band - 1
                        ]
                    ),

                "valid_pixels":
                    int(
                        total_valid
                    ),

                "sample_pixels":
                    int(
                        n
                    ),

                "reference_mean":
                    ref_mean,

                "moving_mean":
                    mov_mean,

                "reference_median":
                    ref_median,

                "moving_median":
                    mov_median,

                "mean_bias_moving_minus_reference":
                    mean_bias,

                "median_bias_moving_minus_reference":
                    median_bias,

                "median_bias_percent":
                    median_bias_percent,

                "mae":
                    mae,

                "rmse":
                    rmse,

                "nrmse_percent":
                    nrmse_percent,

                "pearson_r":
                    correlation,

                "mean_ratio_moving_reference":
                    mean_ratio,

                "median_ratio_moving_reference":
                    median_ratio,

                "gain_to_reference":
                    gain_to_reference,

                "offset_to_reference":
                    offset_to_reference,
            }


            pair_results.append(
                record
            )


            all_results.append(
                record
            )


            if (
                band == 1
                or
                band % 25 == 0
                or
                band
                == EXPECTED_BANDS
            ):

                print(
                    f"Band "
                    f"{band:3d}/425 "
                    f"({wavelengths[band-1]:7.2f} nm)"
                    f" | "
                    f"r={correlation:6.3f}"
                    f" | "
                    f"bias={median_bias:+.5f}"
                    f" | "
                    f"ratio={median_ratio:.4f}"
                )


    # ========================================================
    # PAIR DATAFRAME
    # ========================================================

    pair_df = pd.DataFrame(
        pair_results
    )


    pair_name = (
        f"{moving_scene}_to_"
        f"{reference_scene}"
    )


    pair_csv = (
        OUTDIR
        /
        f"{pair_name}"
        "_radiometric_stats.csv"
    )


    pair_df.to_csv(
        pair_csv,
        index=False
    )


    # ========================================================
    # SUMMARY STATISTICS
    # ========================================================

    valid_df = pair_df[
        pair_df[
            "valid_pixels"
        ]
        > 0
    ].copy()


    median_abs_bias = np.nanmedian(
        np.abs(
            valid_df[
                "median_bias_moving_minus_reference"
            ]
        )
    )


    median_abs_bias_percent = np.nanmedian(
        np.abs(
            valid_df[
                "median_bias_percent"
            ]
        )
    )


    median_r = np.nanmedian(
        valid_df[
            "pearson_r"
        ]
    )


    median_ratio_all = np.nanmedian(
        valid_df[
            "median_ratio_moving_reference"
        ]
    )


    median_nrmse = np.nanmedian(
        valid_df[
            "nrmse_percent"
        ]
    )


    print("\n")
    print("-" * 88)

    print(
        "PAIR SUMMARY"
    )

    print("-" * 88)


    print(
        f"Median |reflectance bias| : "
        f"{median_abs_bias:.6f}"
    )


    print(
        f"Median |bias| percent     : "
        f"{median_abs_bias_percent:.2f}%"
    )


    print(
        f"Median spectral ratio     : "
        f"{median_ratio_all:.4f}"
    )


    print(
        f"Median band correlation   : "
        f"{median_r:.4f}"
    )


    print(
        f"Median NRMSE              : "
        f"{median_nrmse:.2f}%"
    )


    print(
        "\nPer-band CSV:"
    )

    print(
        pair_csv
    )


    # ========================================================
    # PLOT 1 — MEDIAN SPECTRA
    # ========================================================

    plt.figure(
        figsize=(13, 6)
    )


    plt.plot(
        valid_df[
            "wavelength_nm"
        ],
        valid_df[
            "reference_median"
        ],
        label="Reference"
    )


    plt.plot(
        valid_df[
            "wavelength_nm"
        ],
        valid_df[
            "moving_median"
        ],
        label="Moving"
    )


    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Median reflectance"
    )

    plt.title(
        f"Overlap Median Spectra\n"
        f"{moving_scene} → "
        f"{reference_scene}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()


    plot1 = (
        OUTDIR
        /
        f"{pair_name}"
        "_01_median_spectra.png"
    )


    plt.savefig(
        plot1,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PLOT 2 — BIAS
    # ========================================================

    plt.figure(
        figsize=(13, 6)
    )


    plt.axhline(
        0,
        linewidth=1
    )


    plt.plot(
        valid_df[
            "wavelength_nm"
        ],
        valid_df[
            "median_bias_moving_minus_reference"
        ]
    )


    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Median bias\n"
        "(Moving - Reference)"
    )

    plt.title(
        f"Band-by-Band Reflectance Bias\n"
        f"{moving_scene} → "
        f"{reference_scene}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plot2 = (
        OUTDIR
        /
        f"{pair_name}"
        "_02_bias.png"
    )


    plt.savefig(
        plot2,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PLOT 3 — SPECTRAL RATIO
    # ========================================================

    plt.figure(
        figsize=(13, 6)
    )


    plt.axhline(
        1.0,
        linewidth=1
    )


    plt.plot(
        valid_df[
            "wavelength_nm"
        ],
        valid_df[
            "median_ratio_moving_reference"
        ]
    )


    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Median ratio\n"
        "(Moving / Reference)"
    )

    plt.title(
        f"Spectral Ratio Across Overlap\n"
        f"{moving_scene} → "
        f"{reference_scene}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plot3 = (
        OUTDIR
        /
        f"{pair_name}"
        "_03_ratio.png"
    )


    plt.savefig(
        plot3,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PLOT 4 — CORRELATION
    # ========================================================

    plt.figure(
        figsize=(13, 6)
    )


    plt.plot(
        valid_df[
            "wavelength_nm"
        ],
        valid_df[
            "pearson_r"
        ]
    )


    plt.axhline(
        0.8,
        linestyle="--",
        linewidth=1
    )


    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Pearson correlation"
    )

    plt.ylim(
        -0.1,
        1.05
    )

    plt.title(
        f"Band-by-Band Spatial Correlation\n"
        f"{moving_scene} → "
        f"{reference_scene}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plot4 = (
        OUTDIR
        /
        f"{pair_name}"
        "_04_correlation.png"
    )


    plt.savefig(
        plot4,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


    # ========================================================
    # PLOT 5 — GAIN
    # ========================================================

    plt.figure(
        figsize=(13, 6)
    )


    plt.axhline(
        1.0,
        linewidth=1
    )


    plt.plot(
        valid_df[
            "wavelength_nm"
        ],
        valid_df[
            "gain_to_reference"
        ]
    )


    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Gain to reference"
    )

    plt.title(
        f"Potential Radiometric Gain\n"
        f"Reference = Gain × Moving + Offset\n"
        f"{moving_scene} → "
        f"{reference_scene}"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()


    plot5 = (
        OUTDIR
        /
        f"{pair_name}"
        "_05_gain.png"
    )


    plt.savefig(
        plot5,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# COMBINED CSV
# ============================================================

combined_df = pd.DataFrame(
    all_results
)


combined_csv = (
    OUTDIR
    /
    "AVIRIS_all_pairs_425band_"
    "radiometric_statistics.csv"
)


combined_df.to_csv(
    combined_csv,
    index=False
)


# ============================================================
# PAIR SUMMARY TABLE
# ============================================================

summary_rows = []


for (
    reference_scene,
    moving_scene
) in PAIRS:


    subset = combined_df[
        (
            combined_df[
                "reference_scene"
            ]
            ==
            reference_scene
        )
        &
        (
            combined_df[
                "moving_scene"
            ]
            ==
            moving_scene
        )
        &
        (
            combined_df[
                "valid_pixels"
            ]
            > 0
        )
    ]


    summary_rows.append({

        "reference_scene":
            reference_scene,

        "moving_scene":
            moving_scene,

        "median_abs_bias":
            np.nanmedian(
                np.abs(
                    subset[
                        "median_bias_moving_minus_reference"
                    ]
                )
            ),

        "median_abs_bias_percent":
            np.nanmedian(
                np.abs(
                    subset[
                        "median_bias_percent"
                    ]
                )
            ),

        "median_ratio_moving_reference":
            np.nanmedian(
                subset[
                    "median_ratio_moving_reference"
                ]
            ),

        "median_pearson_r":
            np.nanmedian(
                subset[
                    "pearson_r"
                ]
            ),

        "median_nrmse_percent":
            np.nanmedian(
                subset[
                    "nrmse_percent"
                ]
            ),

        "median_gain_to_reference":
            np.nanmedian(
                subset[
                    "gain_to_reference"
                ]
            ),

        "median_offset_to_reference":
            np.nanmedian(
                subset[
                    "offset_to_reference"
                ]
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_csv = (
    OUTDIR
    /
    "AVIRIS_radiometric_pair_summary.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 88)

print(
    "425-BAND RADIOMETRIC / "
    "SEAM ANALYSIS COMPLETE"
)

print("=" * 88)


print(
    "\nPAIR SUMMARY"
)

print("-" * 88)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n\nCombined 425-band statistics:"
)

print(
    combined_csv
)


print(
    "\nPair summary:"
)

print(
    summary_csv
)


print(
    "\nPlots and individual pair CSVs:"
)

print(
    OUTDIR
)


print(
    "\nIMPORTANT:"
)

print(
    "No registered AVIRIS-NG "
    "raster was modified."
)

print(
    "No radiometric correction "
    "has been applied."
)

print(
    "No mosaic has been created."
)

print(
    "\nNext decision:"
)

print(
    "Use these statistics to determine "
    "whether scene-to-scene radiometric "
    "normalization is necessary before "
    "the final mosaic."
)
