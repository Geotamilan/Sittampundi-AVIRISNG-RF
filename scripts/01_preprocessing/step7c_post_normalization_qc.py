from pathlib import Path
import csv
import math

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

INDIR = (
    ROOT
    / "03_Mosaic"
    / "Normalized_425Band_4m"
)

STEP6_DIR = (
    ROOT
    / "03_Mosaic"
    / "Radiometric_Seam_Analysis"
)

STEP6_CSV = (
    STEP6_DIR
    / "AVIRIS_all_pairs_425band_radiometric_statistics.csv"
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
    / "Post_Normalization_QC"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SCENES / PAIRS
# ============================================================

MASTER = "ang20160110t053932"

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

RESOLUTION = 4.0

MAX_SAMPLES_PER_BAND = 200000

MIN_POSITIVE = 1e-8


# ============================================================
# QC PASS THRESHOLDS
#
# Applied to the conservative 326-band scientific subset.
# ============================================================

MAX_MEDIAN_ABS_BIAS_PERCENT = 1.0

MAX_P95_ABS_BIAS_PERCENT = 2.0

MIN_MEDIAN_RATIO = 0.98

MAX_MEDIAN_RATIO = 1.02


# ============================================================
# FILE HELPERS
# ============================================================

def dat_path(scene):

    return (
        INDIR
        /
        f"{scene}_normalized_to_053932_4m.dat"
    )


def overlap_bounds(a, b):

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


def aligned_window(
    src,
    bounds
):

    window = from_bounds(
        *bounds,
        transform=src.transform
    )

    return (
        window
        .round_offsets()
        .round_lengths()
    )


def systematic_sample(
    a,
    b,
    max_samples
):

    n = a.size

    if n <= max_samples:

        return a, b

    step = int(
        math.ceil(
            n
            /
            max_samples
        )
    )

    return (
        a[::step],
        b[::step]
    )


def pearson_r(
    a,
    b
):

    if a.size < 3:

        return np.nan

    sa = np.std(a)
    sb = np.std(b)

    if (
        sa <= 0
        or
        sb <= 0
    ):

        return np.nan

    return float(
        np.corrcoef(
            a,
            b
        )[0, 1]
    )


# ============================================================
# START
# ============================================================

print("=" * 94)

print(
    "AVIRIS-NG STEP 7C — "
    "ACTUAL POST-NORMALIZATION OVERLAP QC"
)

print("=" * 94)

print(
    "\nREAD-ONLY QC"
)

print(
    "No raster will be modified."
)

print(
    "No further normalization will be applied."
)

print(
    "No mosaic will be created."
)


# ============================================================
# LOAD FINAL COEFFICIENT MODEL
# ============================================================

if not COEFF_FILE.exists():

    raise FileNotFoundError(
        COEFF_FILE
    )


coeff = pd.read_csv(
    COEFF_FILE
)


if len(coeff) != EXPECTED_BANDS:

    raise RuntimeError(
        "Expected 425 coefficient rows."
    )


bands = (
    coeff[
        "band"
    ]
    .to_numpy(
        dtype=int
    )
)


wavelength = (
    coeff[
        "wavelength_nm"
    ]
    .to_numpy(
        dtype=float
    )
)


v3_science_mask = (
    coeff[
        "recommended_for_scientific_analysis"
    ]
    .astype(bool)
    .to_numpy()
)


# ============================================================
# FINAL CONSERVATIVE SCIENTIFIC MASK
#
# Remove any otherwise-recommended band for which scene
# 055421 required cumulative correction > 1.8.
#
# This removes bands 16-20 in addition to the original
# v3 exclusions.
# ============================================================

factor_055421 = (
    coeff[
        "ang20160110t055421_factor_to_master"
    ]
    .to_numpy(
        dtype=float
    )
)


high_correction_mask = (
    factor_055421
    >
    1.8
)


final_science_mask = (
    v3_science_mask
    &
    (~high_correction_mask)
)


excluded_mask = (
    ~final_science_mask
)


print(
    "\nV3 recommended bands:",
    int(
        np.sum(
            v3_science_mask
        )
    )
)


print(
    "Additional high-correction bands removed:",
    int(
        np.sum(
            v3_science_mask
            &
            high_correction_mask
        )
    )
)


print(
    "FINAL conservative scientific bands:",
    int(
        np.sum(
            final_science_mask
        )
    )
)


print(
    "FINAL excluded bands:",
    int(
        np.sum(
            excluded_mask
        )
    )
)


# ============================================================
# SAVE FINAL SCIENTIFIC MASK
# ============================================================

mask_df = pd.DataFrame({

    "band":
        bands,

    "wavelength_nm":
        wavelength,

    "v3_recommended":
        v3_science_mask,

    "scene055421_factor_to_master":
        factor_055421,

    "high_correction_gt_1p8":
        high_correction_mask,

    "final_recommended_for_scientific_analysis":
        final_science_mask,

    "final_bbl":
        np.where(
            final_science_mask,
            1,
            0
        )
})


mask_csv = (
    OUTDIR
    /
    "AVIRIS_FINAL_326band_scientific_mask.csv"
)


mask_df.to_csv(
    mask_csv,
    index=False
)


# ============================================================
# LOAD STEP 6 PRE-NORMALIZATION STATISTICS
# ============================================================

before_df = None


if STEP6_CSV.exists():

    before_df = pd.read_csv(
        STEP6_CSV
    )

    print(
        "\nStep 6 before-normalization statistics loaded."
    )

else:

    print(
        "\nWARNING:"
    )

    print(
        "Step 6 statistics not found."
    )

    print(
        "Actual post-normalization QC will still run, "
        "but before/after comparison will be unavailable."
    )


# ============================================================
# VALIDATE NORMALIZED DATASETS
# ============================================================

print("\n")
print("=" * 94)
print("NORMALIZED DATASET VALIDATION")
print("=" * 94)


all_scenes = sorted(
    {
        scene
        for pair in PAIRS
        for scene in pair
    }
)


scene_info = {}


for scene in all_scenes:

    path = dat_path(
        scene
    )

    if not path.exists():

        raise FileNotFoundError(
            path
        )


    with rasterio.open(
        path
    ) as src:

        if src.count != EXPECTED_BANDS:

            raise RuntimeError(
                f"{scene}: band count "
                f"is {src.count}, not 425."
            )


        if str(src.crs) != "EPSG:32643":

            raise RuntimeError(
                f"{scene}: unexpected CRS "
                f"{src.crs}"
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
                f"{scene}: pixel size "
                "is not 4 m."
            )


        scene_info[
            scene
        ] = {

            "width":
                src.width,

            "height":
                src.height,

            "bounds":
                src.bounds,

            "transform":
                src.transform,

            "nodata":
                src.nodata,
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
# STORAGE
# ============================================================

all_records = []

pair_summaries = []


# ============================================================
# PAIRWISE ACTUAL QC
# ============================================================

for (
    reference_scene,
    moving_scene
) in PAIRS:


    print("\n")
    print("=" * 94)

    print(
        "REFERENCE :",
        reference_scene
    )

    print(
        "MOVING    :",
        moving_scene
    )

    print("=" * 94)


    ref_path = dat_path(
        reference_scene
    )

    mov_path = dat_path(
        moving_scene
    )


    pair_records = []


    with rasterio.open(
        ref_path
    ) as ref, \
         rasterio.open(
             mov_path
         ) as mov:


        overlap = overlap_bounds(
            ref.bounds,
            mov.bounds
        )


        if overlap is None:

            raise RuntimeError(
                f"No overlap: "
                f"{moving_scene} / "
                f"{reference_scene}"
            )


        ref_window = aligned_window(
            ref,
            overlap
        )

        mov_window = aligned_window(
            mov,
            overlap
        )


        height = int(
            min(
                ref_window.height,
                mov_window.height
            )
        )

        width = int(
            min(
                ref_window.width,
                mov_window.width
            )
        )


        print(
            "\nOverlap:"
        )

        print(
            overlap
        )


        print(
            f"Analysis window: "
            f"{width} x {height}"
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
                )
            ).astype(
                np.float32
            )


            mov_array = mov.read(
                band,
                window=mov_window,
                out_shape=(
                    height,
                    width
                )
            ).astype(
                np.float32
            )


            valid = (
                np.isfinite(
                    ref_array
                )
                &
                np.isfinite(
                    mov_array
                )
            )


            if ref.nodata is not None:

                valid &= (
                    ref_array
                    !=
                    ref.nodata
                )


            if mov.nodata is not None:

                valid &= (
                    mov_array
                    !=
                    mov.nodata
                )


            ref_values = (
                ref_array[
                    valid
                ]
                .astype(
                    np.float64
                )
            )


            mov_values = (
                mov_array[
                    valid
                ]
                .astype(
                    np.float64
                )
            )


            valid_pixels = (
                ref_values.size
            )


            if valid_pixels == 0:

                record = {

                    "reference_scene":
                        reference_scene,

                    "moving_scene":
                        moving_scene,

                    "band":
                        band,

                    "wavelength_nm":
                        wavelength[
                            band - 1
                        ],

                    "valid_pixels":
                        0,

                    "final_scientific_band":
                        bool(
                            final_science_mask[
                                band - 1
                            ]
                        ),
                }


                pair_records.append(
                    record
                )

                all_records.append(
                    record
                )

                continue


            (
                ref_sample,
                mov_sample
            ) = systematic_sample(
                ref_values,
                mov_values,
                MAX_SAMPLES_PER_BAND
            )


            # ================================================
            # ACTUAL STATISTICS
            # ================================================

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


            difference = (
                mov_sample
                -
                ref_sample
            )


            median_bias = float(
                np.median(
                    difference
                )
            )


            mean_bias = float(
                np.mean(
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


            if (
                abs(
                    ref_median
                )
                >
                MIN_POSITIVE
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


            positive = (
                (
                    ref_sample
                    >
                    MIN_POSITIVE
                )
                &
                (
                    mov_sample
                    >
                    MIN_POSITIVE
                )
            )


            if np.any(
                positive
            ):

                ratio_values = (
                    mov_sample[
                        positive
                    ]
                    /
                    ref_sample[
                        positive
                    ]
                )


                median_ratio = float(
                    np.median(
                        ratio_values
                    )
                )


            else:

                median_ratio = np.nan


            # ================================================
            # PRE-NORMALIZATION VALUES FROM STEP 6
            # ================================================

            before_bias_percent = np.nan
            before_ratio = np.nan
            before_r = np.nan


            if before_df is not None:

                before_match = before_df[
                    (
                        before_df[
                            "reference_scene"
                        ]
                        ==
                        reference_scene
                    )
                    &
                    (
                        before_df[
                            "moving_scene"
                        ]
                        ==
                        moving_scene
                    )
                    &
                    (
                        before_df[
                            "band"
                        ]
                        ==
                        band
                    )
                ]


                if len(
                    before_match
                ) == 1:

                    row = (
                        before_match.iloc[0]
                    )


                    before_bias_percent = (
                        row.get(
                            "median_bias_percent",
                            np.nan
                        )
                    )


                    before_ratio = (
                        row.get(
                            "median_ratio_moving_reference",
                            np.nan
                        )
                    )


                    before_r = (
                        row.get(
                            "pearson_r",
                            np.nan
                        )
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
                        wavelength[
                            band - 1
                        ]
                    ),

                "final_scientific_band":
                    bool(
                        final_science_mask[
                            band - 1
                        ]
                    ),

                "valid_pixels":
                    int(
                        valid_pixels
                    ),

                "sample_pixels":
                    int(
                        ref_sample.size
                    ),

                "reference_mean_after":
                    ref_mean,

                "moving_mean_after":
                    mov_mean,

                "reference_median_after":
                    ref_median,

                "moving_median_after":
                    mov_median,

                "median_bias_after":
                    median_bias,

                "median_bias_percent_after":
                    median_bias_percent,

                "mean_bias_after":
                    mean_bias,

                "mae_after":
                    mae,

                "rmse_after":
                    rmse,

                "median_ratio_after":
                    median_ratio,

                "pearson_r_after":
                    correlation,

                "median_bias_percent_before":
                    before_bias_percent,

                "median_ratio_before":
                    before_ratio,

                "pearson_r_before":
                    before_r,
            }


            pair_records.append(
                record
            )


            all_records.append(
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
                    f"{wavelength[band-1]:7.2f} nm"
                    f" | "
                    f"bias="
                    f"{median_bias_percent:+7.3f}%"
                    f" | "
                    f"ratio="
                    f"{median_ratio:7.4f}"
                    f" | "
                    f"r="
                    f"{correlation:6.3f}"
                )


    # ========================================================
    # PAIR DATAFRAME
    # ========================================================

    pair_df = pd.DataFrame(
        pair_records
    )


    pair_name = (
        f"{moving_scene}_to_"
        f"{reference_scene}"
    )


    pair_csv = (
        OUTDIR
        /
        f"{pair_name}"
        "_actual_post_normalization_425band_QC.csv"
    )


    pair_df.to_csv(
        pair_csv,
        index=False
    )


    # ========================================================
    # FINAL 326-BAND SCIENTIFIC SUBSET
    # ========================================================

    science = pair_df[
        (
            pair_df[
                "final_scientific_band"
            ]
            ==
            True
        )
        &
        (
            pair_df[
                "valid_pixels"
            ]
            >
            0
        )
    ].copy()


    excluded = pair_df[
        (
            pair_df[
                "final_scientific_band"
            ]
            ==
            False
        )
        &
        (
            pair_df[
                "valid_pixels"
            ]
            >
            0
        )
    ].copy()


    abs_after = np.abs(
        science[
            "median_bias_percent_after"
        ]
    )


    abs_before = np.abs(
        science[
            "median_bias_percent_before"
        ]
    )


    median_abs_after = float(
        np.nanmedian(
            abs_after
        )
    )


    p95_abs_after = float(
        np.nanpercentile(
            abs_after,
            95
        )
    )


    maximum_abs_after = float(
        np.nanmax(
            abs_after
        )
    )


    median_ratio_after = float(
        np.nanmedian(
            science[
                "median_ratio_after"
            ]
        )
    )


    median_r_after = float(
        np.nanmedian(
            science[
                "pearson_r_after"
            ]
        )
    )


    median_abs_before = float(
        np.nanmedian(
            abs_before
        )
    )


    excluded_median_abs_after = float(
        np.nanmedian(
            np.abs(
                excluded[
                    "median_bias_percent_after"
                ]
            )
        )
    )


    # ========================================================
    # PASS / REVIEW
    # ========================================================

    pass_bias = (
        median_abs_after
        <=
        MAX_MEDIAN_ABS_BIAS_PERCENT
    )


    pass_p95 = (
        p95_abs_after
        <=
        MAX_P95_ABS_BIAS_PERCENT
    )


    pass_ratio = (
        MIN_MEDIAN_RATIO
        <=
        median_ratio_after
        <=
        MAX_MEDIAN_RATIO
    )


    if (
        pass_bias
        and
        pass_p95
        and
        pass_ratio
    ):

        qc_status = "PASS"

    else:

        qc_status = "REVIEW"


    pair_summaries.append({

        "reference_scene":
            reference_scene,

        "moving_scene":
            moving_scene,

        "scientific_bands":
            int(
                len(
                    science
                )
            ),

        "median_abs_bias_percent_before":
            median_abs_before,

        "median_abs_bias_percent_after":
            median_abs_after,

        "p95_abs_bias_percent_after":
            p95_abs_after,

        "maximum_abs_bias_percent_after":
            maximum_abs_after,

        "median_ratio_after":
            median_ratio_after,

        "median_pearson_r_after":
            median_r_after,

        "excluded_band_median_abs_bias_percent_after":
            excluded_median_abs_after,

        "qc_status":
            qc_status,
    })


    # ========================================================
    # PLOT 1 — BEFORE / AFTER BIAS
    # ========================================================

    plt.figure(
        figsize=(15, 7)
    )


    plt.axhline(
        0,
        linewidth=1
    )


    plt.plot(
        pair_df[
            "wavelength_nm"
        ],
        pair_df[
            "median_bias_percent_before"
        ],
        label="Before normalization",
        alpha=0.65
    )


    plt.plot(
        pair_df[
            "wavelength_nm"
        ],
        pair_df[
            "median_bias_percent_after"
        ],
        label="Actual after normalization",
        linewidth=1.5
    )


    # Shade excluded wavelengths.
    ymax = 25

    excluded_band = (
        ~final_science_mask
    )

    start = None

    for i, excluded_value in enumerate(
        excluded_band
    ):

        if (
            excluded_value
            and
            start is None
        ):

            start = i


        if (
            start is not None
            and
            (
                not excluded_value
                or
                i
                ==
                len(
                    excluded_band
                ) - 1
            )
        ):

            end = (
                i
                if excluded_value
                else i - 1
            )


            plt.axvspan(
                wavelength[
                    start
                ],
                wavelength[
                    end
                ],
                alpha=0.08
            )


            start = None


    plt.xlabel(
        "Wavelength (nm)"
    )


    plt.ylabel(
        "Median bias (%)"
    )


    plt.title(
        "AVIRIS-NG Actual Seam Bias — "
        "Before vs After Normalization\n"
        + pair_name
    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTDIR
        /
        f"{pair_name}"
        "_01_before_after_bias.png",
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # PLOT 2 — ACTUAL RATIO
    # ========================================================

    plt.figure(
        figsize=(15, 7)
    )


    plt.axhline(
        1.0,
        linewidth=1
    )


    plt.plot(
        pair_df[
            "wavelength_nm"
        ],
        pair_df[
            "median_ratio_after"
        ]
    )


    plt.xlabel(
        "Wavelength (nm)"
    )


    plt.ylabel(
        "Moving / reference median ratio"
    )


    plt.title(
        "Actual Post-Normalization "
        "Spectral Ratio\n"
        + pair_name
    )


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(
        OUTDIR
        /
        f"{pair_name}"
        "_02_actual_ratio.png",
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # PLOT 3 — MEDIAN SPECTRA
    # ========================================================

    plt.figure(
        figsize=(15, 7)
    )


    plt.plot(
        pair_df[
            "wavelength_nm"
        ],
        pair_df[
            "reference_median_after"
        ],
        label="Reference"
    )


    plt.plot(
        pair_df[
            "wavelength_nm"
        ],
        pair_df[
            "moving_median_after"
        ],
        label="Normalized moving"
    )


    plt.xlabel(
        "Wavelength (nm)"
    )


    plt.ylabel(
        "Median reflectance"
    )


    plt.title(
        "Actual Normalized Overlap "
        "Median Spectra\n"
        + pair_name
    )


    plt.grid(
        alpha=0.25
    )


    plt.legend()


    plt.tight_layout()


    plt.savefig(
        OUTDIR
        /
        f"{pair_name}"
        "_03_actual_median_spectra.png",
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    # ========================================================
    # PLOT 4 — CORRELATION
    # ========================================================

    plt.figure(
        figsize=(15, 7)
    )


    plt.plot(
        pair_df[
            "wavelength_nm"
        ],
        pair_df[
            "pearson_r_after"
        ]
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
        "Actual Post-Normalization "
        "Spatial Correlation\n"
        + pair_name
    )


    plt.grid(
        alpha=0.25
    )


    plt.tight_layout()


    plt.savefig(
        OUTDIR
        /
        f"{pair_name}"
        "_04_correlation.png",
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print("\n")
    print("-" * 94)
    print("PAIR ACTUAL QC SUMMARY")
    print("-" * 94)


    print(
        f"Final scientific bands       : "
        f"{len(science)}"
    )


    print(
        f"Median |bias| BEFORE         : "
        f"{median_abs_before:.3f}%"
    )


    print(
        f"Median |bias| AFTER          : "
        f"{median_abs_after:.3f}%"
    )


    print(
        f"95th percentile |bias| AFTER : "
        f"{p95_abs_after:.3f}%"
    )


    print(
        f"Maximum |bias| AFTER         : "
        f"{maximum_abs_after:.3f}%"
    )


    print(
        f"Median ratio AFTER           : "
        f"{median_ratio_after:.5f}"
    )


    print(
        f"Median correlation AFTER     : "
        f"{median_r_after:.4f}"
    )


    print(
        f"Excluded bands median |bias| : "
        f"{excluded_median_abs_after:.3f}%"
    )


    print(
        f"QC STATUS                    : "
        f"{qc_status}"
    )


# ============================================================
# SAVE ALL-BAND TABLE
# ============================================================

all_df = pd.DataFrame(
    all_records
)


all_csv = (
    OUTDIR
    /
    "AVIRIS_Step7C_actual_post_normalization_ALL_425band_QC.csv"
)


all_df.to_csv(
    all_csv,
    index=False
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    pair_summaries
)


summary_csv = (
    OUTDIR
    /
    "AVIRIS_Step7C_pair_summary.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# GLOBAL DECISION
# ============================================================

all_pass = bool(
    np.all(
        summary_df[
            "qc_status"
        ]
        ==
        "PASS"
    )
)


print("\n")
print("=" * 94)
print("STEP 7C FINAL SUMMARY")
print("=" * 94)


print(
    "\nFINAL SCIENTIFIC MASK:"
)

print(
    f"  Recommended bands : "
    f"{np.sum(final_science_mask)}"
)

print(
    f"  Excluded bands    : "
    f"{np.sum(~final_science_mask)}"
)


print("\n")
print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n\nPair summary:"
)

print(
    summary_csv
)


print(
    "\nAll 425-band statistics:"
)

print(
    all_csv
)


print(
    "\nFinal scientific mask:"
)

print(
    mask_csv
)


print(
    "\nDiagnostic plots:"
)

print(
    OUTDIR
)


print("\n")
print("=" * 94)


if all_pass:

    print(
        "OVERALL STEP 7C QC: PASS"
    )

    print(
        "\nThe actual normalized raster overlaps "
        "meet the quantitative seam criteria."
    )

    print(
        "\nThe data may proceed to the final "
        "mosaic/seamline design stage."
    )

else:

    print(
        "OVERALL STEP 7C QC: REVIEW REQUIRED"
    )

    print(
        "\nAt least one actual normalized overlap "
        "does not meet the seam criteria."
    )

    print(
        "\nDO NOT MOSAIC."
    )


print("=" * 94)

print(
    "\nNo raster was modified during Step 7C."
)
