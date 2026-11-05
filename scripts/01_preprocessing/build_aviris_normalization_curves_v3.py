from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

INDIR = (
    ROOT
    / "03_Mosaic"
    / "Radiometric_Normalization_Model_v2"
)

OUTDIR = (
    ROOT
    / "03_Mosaic"
    / "Radiometric_Normalization_Model_v3_FINAL"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


MASTER = "ang20160110t053932"

S1 = "ang20160110t053216"
S2 = "ang20160110t053932"
S3 = "ang20160110t054703"
S4 = "ang20160110t055421"

EXPECTED_BANDS = 425


# ============================================================
# INPUT FILES
# ============================================================

FILES = {

    "s1":
        INDIR /
        "ang20160110t053216_to_ang20160110t053932_v2.csv",

    "s3":
        INDIR /
        "ang20160110t054703_to_ang20160110t053932_v2.csv",

    "s4pair":
        INDIR /
        "ang20160110t055421_to_ang20160110t054703_v2.csv",
}


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("AVIRIS-NG STEP 7A-v3")
print("FINAL CONTINUOUS NORMALIZATION COEFFICIENT MODEL")
print("=" * 90)

print("\nNO raster will be modified.")
print("NO normalization will be applied.")
print("NO mosaic will be created.")


data = {}

for key, path in FILES.items():

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    df = pd.read_csv(
        path
    )

    df = df.sort_values(
        "band"
    ).reset_index(
        drop=True
    )

    if len(df) != EXPECTED_BANDS:

        raise RuntimeError(
            f"{path.name}: expected 425 bands, "
            f"found {len(df)}"
        )

    data[key] = df


# ============================================================
# VERIFY WAVELENGTHS
# ============================================================

wavelength = (
    data["s1"][
        "wavelength_nm"
    ]
    .to_numpy(
        dtype=float
    )
)

bands = (
    data["s1"][
        "band"
    ]
    .to_numpy(
        dtype=int
    )
)


for key in [
    "s3",
    "s4pair"
]:

    other = (
        data[key][
            "wavelength_nm"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.allclose(
        wavelength,
        other,
        atol=0.001
    ):

        raise RuntimeError(
            "Wavelength mismatch."
        )


# ============================================================
# ANALYSIS MASK
#
# These bands genuinely participate in determining the
# normalization curve.
#
# Other bands receive interpolated/edge-held coefficients,
# but are NOT recommended for later mineral analysis.
# ============================================================

analysis_mask = (
    data["s1"][
        "normalization_model_region"
    ]
    .astype(bool)
    .to_numpy()
)


print(
    "\nBands used for radiometric modelling:",
    int(
        analysis_mask.sum()
    )
)

print(
    "Bands excluded from modelling:",
    int(
        (~analysis_mask).sum()
    )
)


# ============================================================
# BUILD CONTINUOUS FACTOR
# ============================================================

def continuous_factor(df, name):

    factor_v2 = (
        df[
            "smoothed_factor"
        ]
        .to_numpy(
            dtype=float
        )
    )

    model_mask = (
        df[
            "normalization_model_region"
        ]
        .astype(bool)
        .to_numpy()
    )

    valid = (
        model_mask
        &
        np.isfinite(
            factor_v2
        )
        &
        (
            factor_v2 > 0
        )
    )


    if np.sum(valid) < 20:

        raise RuntimeError(
            f"{name}: insufficient "
            "trusted factors."
        )


    # ========================================================
    # LOG-SPACE INTERPOLATION
    #
    # Atmospheric bands do NOT influence the curve.
    #
    # np.interp behaviour:
    #
    # between trusted windows:
    #     interpolation
    #
    # below first trusted wavelength:
    #     nearest reliable factor is held
    #
    # above last trusted wavelength:
    #     nearest reliable factor is held
    # ========================================================

    log_factor = np.log(
        factor_v2[
            valid
        ]
    )


    continuous_log = np.interp(

        wavelength,

        wavelength[
            valid
        ],

        log_factor
    )


    final_factor = np.exp(
        continuous_log
    )


    source = np.full(
        EXPECTED_BANDS,
        "interpolated_or_edgeheld",
        dtype=object
    )


    source[
        model_mask
    ] = "modelled"


    # ========================================================
    # ADJACENT-BAND SMOOTHNESS CHECK
    # ========================================================

    percent_step = np.zeros(
        EXPECTED_BANDS,
        dtype=float
    )


    percent_step[
        1:
    ] = (
        (
            final_factor[
                1:
            ]
            /
            final_factor[
                :-1
            ]
        )
        -
        1.0
    ) * 100.0


    print(
        f"\n{name}"
    )

    print(
        "  Median trusted factor:",
        f"{np.median(final_factor[model_mask]):.6f}"
    )

    print(
        "  Minimum full factor:",
        f"{np.min(final_factor):.6f}"
    )

    print(
        "  Maximum full factor:",
        f"{np.max(final_factor):.6f}"
    )

    print(
        "  Maximum adjacent-band change:",
        f"{np.max(np.abs(percent_step[1:])):.3f}%"
    )


    return (
        final_factor,
        source,
        percent_step
    )


# ============================================================
# PAIR FACTORS
# ============================================================

(
    f1,
    source1,
    step1
) = continuous_factor(
    data["s1"],
    "053216 -> 053932"
)


(
    f3,
    source3,
    step3
) = continuous_factor(
    data["s3"],
    "054703 -> 053932"
)


(
    f4_pair,
    source4,
    step4pair
) = continuous_factor(
    data["s4pair"],
    "055421 -> 054703"
)


# ============================================================
# MASTER
# ============================================================

f2 = np.ones(
    EXPECTED_BANDS,
    dtype=float
)


# ============================================================
# CUMULATIVE SCENE 4 FACTOR
#
# 055421 → 054703 → 053932
# ============================================================

f4 = (
    f4_pair
    *
    f3
)


step4 = np.zeros(
    EXPECTED_BANDS,
    dtype=float
)

step4[
    1:
] = (
    (
        f4[
            1:
        ]
        /
        f4[
            :-1
        ]
    )
    -
    1.0
) * 100.0


# ============================================================
# COEFFICIENT SOURCE
# ============================================================

coefficient_source = np.where(
    analysis_mask,
    "modelled_from_overlap",
    "interpolated_or_edgeheld_not_used_for_fit"
)


# ============================================================
# FINAL MASTER COEFFICIENT TABLE
# ============================================================

coefficients = pd.DataFrame({

    "band":
        bands,

    "wavelength_nm":
        wavelength,

    "recommended_for_scientific_analysis":
        analysis_mask,

    "coefficient_source":
        coefficient_source,

    f"{S1}_factor_to_master":
        f1,

    f"{S2}_factor_to_master":
        f2,

    f"{S3}_factor_to_master":
        f3,

    f"{S4}_factor_to_master":
        f4,

    f"{S4}_pair_factor_to_{S3}":
        f4_pair,

    f"{S1}_adjacent_factor_change_percent":
        step1,

    f"{S3}_adjacent_factor_change_percent":
        step3,

    f"{S4}_adjacent_factor_change_percent":
        step4,
})


# ============================================================
# LARGE FACTOR FLAGS
#
# These are inspection flags, not automatic rejection.
# ============================================================

coefficients[
    f"{S1}_large_factor_flag"
] = (
    (f1 < 0.70)
    |
    (f1 > 1.80)
)


coefficients[
    f"{S3}_large_factor_flag"
] = (
    (f3 < 0.70)
    |
    (f3 > 1.80)
)


coefficients[
    f"{S4}_large_factor_flag"
] = (
    (f4 < 0.70)
    |
    (f4 > 1.80)
)


coeff_csv = (
    OUTDIR
    /
    "AVIRIS_425band_FINAL_normalization_coefficients_TO_MASTER_v3.csv"
)


coefficients.to_csv(
    coeff_csv,
    index=False
)


# ============================================================
# PREDICTED PAIRWISE PERFORMANCE
# ============================================================

def predicted_pair_qc(
    df,
    factor,
    name
):

    ref = (
        df[
            "reference_median"
        ]
        .to_numpy(
            dtype=float
        )
    )


    mov = (
        df[
            "moving_median"
        ]
        .to_numpy(
            dtype=float
        )
    )


    valid = (
        np.isfinite(ref)
        &
        np.isfinite(mov)
        &
        (
            np.abs(ref)
            > 1e-8
        )
    )


    corrected = (
        mov
        *
        factor
    )


    before_percent = np.full(
        EXPECTED_BANDS,
        np.nan
    )


    after_percent = np.full(
        EXPECTED_BANDS,
        np.nan
    )


    before_percent[
        valid
    ] = (
        (
            mov[
                valid
            ]
            -
            ref[
                valid
            ]
        )
        /
        np.abs(
            ref[
                valid
            ]
        )
        *
        100.0
    )


    after_percent[
        valid
    ] = (
        (
            corrected[
                valid
            ]
            -
            ref[
                valid
            ]
        )
        /
        np.abs(
            ref[
                valid
            ]
        )
        *
        100.0
    )


    analysis_valid = (
        valid
        &
        analysis_mask
    )


    excluded_valid = (
        valid
        &
        (~analysis_mask)
    )


    return {

        "pair":
            name,

        "analysis_bands":
            int(
                analysis_valid.sum()
            ),

        "median_abs_bias_before_analysis_percent":
            float(
                np.nanmedian(
                    np.abs(
                        before_percent[
                            analysis_valid
                        ]
                    )
                )
            ),

        "median_abs_bias_after_analysis_percent":
            float(
                np.nanmedian(
                    np.abs(
                        after_percent[
                            analysis_valid
                        ]
                    )
                )
            ),

        "p95_abs_bias_after_analysis_percent":
            float(
                np.nanpercentile(
                    np.abs(
                        after_percent[
                            analysis_valid
                        ]
                    ),
                    95
                )
            ),

        "max_abs_bias_after_analysis_percent":
            float(
                np.nanmax(
                    np.abs(
                        after_percent[
                            analysis_valid
                        ]
                    )
                )
            ),

        "median_abs_bias_after_excluded_percent":
            float(
                np.nanmedian(
                    np.abs(
                        after_percent[
                            excluded_valid
                        ]
                    )
                )
            ),

        "median_abs_bias_after_all_425_percent":
            float(
                np.nanmedian(
                    np.abs(
                        after_percent[
                            valid
                        ]
                    )
                )
            ),
    }


qc_rows = [

    predicted_pair_qc(
        data["s1"],
        f1,
        "053216_to_053932"
    ),

    predicted_pair_qc(
        data["s3"],
        f3,
        "054703_to_053932"
    ),

    predicted_pair_qc(
        data["s4pair"],
        f4_pair,
        "055421_to_054703"
    ),
]


qc_df = pd.DataFrame(
    qc_rows
)


qc_csv = (
    OUTDIR
    /
    "AVIRIS_predicted_post_normalization_QC_v3.csv"
)


qc_df.to_csv(
    qc_csv,
    index=False
)


# ============================================================
# FACTOR SUMMARY
# ============================================================

summary_rows = []


for scene, factor in [

    (S1, f1),
    (S2, f2),
    (S3, f3),
    (S4, f4),

]:

    trusted_values = (
        factor[
            analysis_mask
        ]
    )


    summary_rows.append({

        "scene":
            scene,

        "master":
            MASTER,

        "median_factor_analysis_bands":
            float(
                np.median(
                    trusted_values
                )
            ),

        "minimum_factor_analysis_bands":
            float(
                np.min(
                    trusted_values
                )
            ),

        "maximum_factor_analysis_bands":
            float(
                np.max(
                    trusted_values
                )
            ),

        "minimum_factor_all_425":
            float(
                np.min(
                    factor
                )
            ),

        "maximum_factor_all_425":
            float(
                np.max(
                    factor
                )
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_csv = (
    OUTDIR
    /
    "AVIRIS_normalization_factor_summary_v3.csv"
)


summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# PLOT — ALL SCENES TO MASTER
# ============================================================

plt.figure(
    figsize=(15, 8)
)


plt.plot(
    wavelength,
    f1,
    label=S1
)


plt.plot(
    wavelength,
    f2,
    label=S2 + " MASTER"
)


plt.plot(
    wavelength,
    f3,
    label=S3
)


plt.plot(
    wavelength,
    f4,
    label=S4
)


plt.axhline(
    1.0,
    linewidth=1
)


# Mark atmospheric / excluded areas
excluded = ~analysis_mask

start = None

for i, is_excluded in enumerate(
    excluded
):

    if is_excluded and start is None:
        start = i

    if (
        start is not None
        and
        (
            not is_excluded
            or
            i == len(excluded) - 1
        )
    ):

        end = (
            i
            if is_excluded
            else i - 1
        )

        plt.axvspan(
            wavelength[start],
            wavelength[end],
            alpha=0.08
        )

        start = None


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Multiplicative factor to master"
)


plt.title(
    "AVIRIS-NG Final Continuous "
    "Radiometric Normalization Model\n"
    "Master: ang20160110t053932"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plot_path = (
    OUTDIR
    /
    "AVIRIS_FINAL_all_scene_factors_to_master_v3.png"
)


plt.savefig(
    plot_path,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT — ADJACENT BAND COEFFICIENT CHANGE
# ============================================================

plt.figure(
    figsize=(15, 7)
)


plt.plot(
    wavelength,
    np.abs(step1),
    label=S1
)


plt.plot(
    wavelength,
    np.abs(step3),
    label=S3
)


plt.plot(
    wavelength,
    np.abs(step4),
    label=S4
)


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Absolute adjacent-band factor change (%)"
)


plt.title(
    "Normalization Coefficient Smoothness QC"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


smooth_plot = (
    OUTDIR
    /
    "AVIRIS_FINAL_coefficient_smoothness_QC_v3.png"
)


plt.savefig(
    smooth_plot,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 90)
print("STEP 7A-v3 COMPLETE")
print("=" * 90)


print(
    "\nFACTOR SUMMARY"
)

print("-" * 90)


print(
    summary_df.to_string(
        index=False
    )
)


print(
    "\n\nPREDICTED NORMALIZATION QC"
)

print("-" * 90)


print(
    qc_df.to_string(
        index=False
    )
)


print(
    "\n\nFinal coefficient file:"
)

print(
    coeff_csv
)


print(
    "\nQC:"
)

print(
    qc_csv
)


print(
    "\nSummary:"
)

print(
    summary_csv
)


print(
    "\nFactor plot:"
)

print(
    plot_path
)


print(
    "\nSmoothness plot:"
)

print(
    smooth_plot
)


print(
    "\nIMPORTANT:"
)

print(
    "1. No raster was modified."
)

print(
    "2. Coefficients in atmospheric/noisy bands "
    "were NOT estimated from those bands."
)

print(
    "3. They were filled only by interpolation "
    "or nearest-edge continuation."
)

print(
    "4. The scientific-analysis mask remains "
    "restricted to the trusted Step 7A-v2 bands."
)

print(
    "5. Scene 055421 remains chained through "
    "054703 to master 053932."
)

print(
    "\nDo NOT apply normalization until "
    "the v3 QC is reviewed."
)
