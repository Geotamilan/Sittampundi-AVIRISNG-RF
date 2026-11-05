from pathlib import Path
import math
import json
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import spectral as spy
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path.cwd()

INDIR = (
    ROOT
    / "03_Mosaic"
    / "Scientific_326Band_Subset"
)

SOURCE_DAT = (
    INDIR
    / "AVIRIS_NG_mosaic_science_326band_4m.dat"
)

SOURCE_HDR = (
    INDIR
    / "AVIRIS_NG_mosaic_science_326band_4m.hdr"
)

MAPPING_CSV = (
    INDIR
    / "AVIRIS_326band_to_original_425band_mapping.csv"
)

OUTDIR = (
    ROOT
    / "04_PCA"
    / "Step10A_PCA_Statistics"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

EIGEN_CSV = (
    OUTDIR
    / "AVIRIS_PCA_eigenvalues_explained_variance.csv"
)

MEAN_CSV = (
    OUTDIR
    / "AVIRIS_PCA_mean_std_spectrum.csv"
)

LOADINGS_CSV = (
    OUTDIR
    / "AVIRIS_PCA_all_component_loadings.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step10A_PCA_summary.csv"
)

SAMPLE_SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_PCA_sampling_QC.csv"
)

PLOT_SCREE = (
    OUTDIR
    / "AVIRIS_PCA_scree_plot.png"
)

PLOT_CUMULATIVE = (
    OUTDIR
    / "AVIRIS_PCA_cumulative_variance.png"
)

PLOT_MEAN = (
    OUTDIR
    / "AVIRIS_PCA_mean_spectrum.png"
)

PLOT_LOADINGS = (
    OUTDIR
    / "AVIRIS_PCA_first6_loadings.png"
)


# ============================================================
# PCA MODEL FILES
#
# These will be reused later to create the block-wise PCA cube.
# ============================================================

MODEL_MEAN = (
    OUTDIR
    / "PCA_mean_326.npy"
)

MODEL_COVARIANCE = (
    OUTDIR
    / "PCA_covariance_326x326.npy"
)

MODEL_EIGENVALUES = (
    OUTDIR
    / "PCA_eigenvalues_326.npy"
)

MODEL_EIGENVECTORS = (
    OUTDIR
    / "PCA_eigenvectors_326x326.npy"
)

MODEL_JSON = (
    OUTDIR
    / "PCA_model_metadata.json"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_BANDS = 326

# Systematic sampling interval.
#
# With your 4620 x 2198 mosaic:
#
# about 462 x 220 ≈ 101,640 potential sample positions.
#
# After NoData removal, expected usable sample size is
# around 80,000–90,000 pixels.
#
# This is much larger than 326 variables and is sufficient
# for stable first-stage PCA covariance statistics.
SAMPLE_STRIDE = 10

# Scree plot display.
SCREE_PCS_TO_PLOT = 80

# Loading plots.
LOADINGS_TO_PLOT = 6

# Explained-variance thresholds.
VARIANCE_THRESHOLDS = [
    0.95,
    0.99,
    0.995,
    0.999,
]


# ============================================================
# START
# ============================================================

print("=" * 96)

print(
    "AVIRIS-NG STEP 10A — "
    "PCA STATISTICAL SAMPLING + EIGEN ANALYSIS"
)

print("=" * 96)

print("\nIMPORTANT:")

print(
    "The 326-band scientific raster will NOT be modified."
)

print(
    "No full PCA raster will be created in this step."
)

print(
    "PCA is covariance-based, not correlation-standardized."
)

print(
    "Only NoData/non-finite pixels are excluded."
)


# ============================================================
# INPUT CHECK
# ============================================================

for path in [
    SOURCE_DAT,
    SOURCE_HDR,
    MAPPING_CSV,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# SPECTRAL METADATA
# ============================================================

spectral_image = spy.open_image(
    str(
        SOURCE_HDR
    )
)

metadata = (
    spectral_image.metadata
)


if "wavelength" not in metadata:

    raise RuntimeError(
        "Wavelength metadata missing."
    )


wavelengths = np.asarray(
    metadata[
        "wavelength"
    ],
    dtype=float
)


if len(wavelengths) != EXPECTED_BANDS:

    raise RuntimeError(
        f"Expected 326 wavelengths, "
        f"found {len(wavelengths)}."
    )


mapping = pd.read_csv(
    MAPPING_CSV
)


if len(mapping) != EXPECTED_BANDS:

    raise RuntimeError(
        "326-to-425 band mapping "
        "does not contain 326 rows."
    )


original_bands = (
    mapping[
        "original_425_band"
    ]
    .astype(int)
    .to_numpy()
)


mapping_wavelengths = (
    mapping[
        "wavelength_nm"
    ]
    .astype(float)
    .to_numpy()
)


if not np.allclose(
    wavelengths,
    mapping_wavelengths,
    atol=0.01
):

    raise RuntimeError(
        "Header wavelengths and mapping "
        "CSV do not match."
    )


# ============================================================
# RASTER VALIDATION
# ============================================================

with rasterio.open(
    SOURCE_DAT
) as src:


    if src.count != EXPECTED_BANDS:

        raise RuntimeError(
            f"Expected 326 bands, "
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
            "Input cube is not 4 m."
        )


    width = src.width
    height = src.height
    nodata = src.nodata
    transform = src.transform
    crs = src.crs


    print("\nSOURCE")

    print("-" * 96)

    print(
        "Dimensions :",
        width,
        "x",
        height
    )

    print(
        "Bands      :",
        src.count
    )

    print(
        "CRS        :",
        crs
    )

    print(
        "Resolution :",
        src.res
    )

    print(
        "NoData     :",
        nodata
    )

    print(
        "Spectral range:",
        f"{wavelengths.min():.2f}",
        "to",
        f"{wavelengths.max():.2f} nm"
    )


# ============================================================
# SYSTEMATIC SPATIAL SAMPLE
# ============================================================

sample_width = int(
    math.ceil(
        width
        /
        SAMPLE_STRIDE
    )
)

sample_height = int(
    math.ceil(
        height
        /
        SAMPLE_STRIDE
    )
)


potential_samples = (
    sample_width
    *
    sample_height
)


print("\n")
print("=" * 96)
print("SYSTEMATIC PCA SAMPLING")
print("=" * 96)


print(
    f"\nSampling stride : "
    f"{SAMPLE_STRIDE} original pixels"
)

print(
    f"Approximate spacing : "
    f"{SAMPLE_STRIDE * 4.0:.1f} m"
)

print(
    f"Sample grid      : "
    f"{sample_width} x "
    f"{sample_height}"
)

print(
    f"Potential samples: "
    f"{potential_samples:,}"
)


sample_start = time.time()


with rasterio.open(
    SOURCE_DAT
) as src:


    # --------------------------------------------------------
    # Read a nearest-neighbour decimated representation
    # ONLY for statistical sampling.
    #
    # This does not create or alter any raster.
    # --------------------------------------------------------

    sample_cube = src.read(

        indexes=list(
            range(
                1,
                EXPECTED_BANDS + 1
            )
        ),

        out_shape=(
            EXPECTED_BANDS,
            sample_height,
            sample_width
        ),

        resampling=
            Resampling.nearest
    ).astype(
        np.float32,
        copy=False
    )


print(
    f"\nSampling read completed in "
    f"{(time.time()-sample_start)/60:.2f} min"
)


# ============================================================
# VALID SAMPLE MASK
#
# A PCA sample is accepted only if ALL 326 scientific bands
# are finite and non-NoData.
# ============================================================

valid = np.all(
    np.isfinite(
        sample_cube
    ),
    axis=0
)


if nodata is not None:

    valid &= np.all(
        sample_cube
        !=
        nodata,
        axis=0
    )


valid_samples = int(
    np.sum(
        valid
    )
)


invalid_samples = int(
    potential_samples
    -
    valid_samples
)


print(
    f"\nValid samples   : "
    f"{valid_samples:,}"
)

print(
    f"Invalid samples : "
    f"{invalid_samples:,}"
)

print(
    f"Valid fraction  : "
    f"{100 * valid_samples / potential_samples:.2f}%"
)


if valid_samples < 10000:

    raise RuntimeError(
        "Too few valid pixels for robust PCA statistics."
    )


# ============================================================
# MATRIX X
#
# rows    = sampled pixels
# columns = 326 wavelengths
# ============================================================

X = (
    sample_cube[
        :,
        valid
    ]
    .T
    .astype(
        np.float64,
        copy=True
    )
)


# Sample cube no longer needed.
del sample_cube


print(
    "\nPCA sample matrix:"
)

print(
    f"  {X.shape[0]:,} pixels "
    f"x {X.shape[1]} bands"
)


# ============================================================
# BASIC DATA DISTRIBUTION
# ============================================================

global_min = float(
    np.min(
        X
    )
)

global_max = float(
    np.max(
        X
    )
)

global_median = float(
    np.median(
        X
    )
)


print(
    f"\nSample global minimum : "
    f"{global_min:.6f}"
)

print(
    f"Sample global median  : "
    f"{global_median:.6f}"
)

print(
    f"Sample global maximum : "
    f"{global_max:.6f}"
)


# ============================================================
# MEAN + STANDARD DEVIATION
# ============================================================

mean_spectrum = np.mean(
    X,
    axis=0,
    dtype=np.float64
)


std_spectrum = np.std(
    X,
    axis=0,
    ddof=1
)


# ============================================================
# COVARIANCE MATRIX
#
# PCA here uses reflectance covariance.
#
# We intentionally do NOT standardize every wavelength to
# unit variance because doing so can overweight intrinsically
# low-variance/noisy bands.
# ============================================================

print("\n")
print("=" * 96)
print("COMPUTING 326 x 326 COVARIANCE MATRIX")
print("=" * 96)


cov_start = time.time()


X -= mean_spectrum


covariance = (
    X.T
    @
    X
) / (
    X.shape[0]
    -
    1
)


print(
    f"\nCovariance completed in "
    f"{(time.time()-cov_start)/60:.2f} min"
)


# Centered sample matrix no longer needed.
del X


# ============================================================
# COVARIANCE QC
# ============================================================

symmetry_error = float(
    np.max(
        np.abs(
            covariance
            -
            covariance.T
        )
    )
)


print(
    f"Maximum covariance symmetry error: "
    f"{symmetry_error:.3e}"
)


# ============================================================
# EIGEN DECOMPOSITION
# ============================================================

print("\n")
print("=" * 96)
print("PCA EIGEN DECOMPOSITION")
print("=" * 96)


eig_start = time.time()


eigenvalues, eigenvectors = np.linalg.eigh(
    covariance
)


# np.linalg.eigh returns ascending eigenvalues.
order = np.argsort(
    eigenvalues
)[::-1]


eigenvalues = (
    eigenvalues[
        order
    ]
)


eigenvectors = (
    eigenvectors[
        :,
        order
    ]
)


# Small negative eigenvalues from floating point rounding
# are not physical variances.
tiny_negative_count = int(
    np.sum(
        eigenvalues < 0
    )
)


if tiny_negative_count > 0:

    print(
        f"\nWARNING: "
        f"{tiny_negative_count} tiny negative "
        "eigenvalues detected."
    )

    print(
        "They will be set to zero for "
        "explained-variance calculations."
    )


eigenvalues_for_variance = np.clip(
    eigenvalues,
    0,
    None
)


total_variance = float(
    np.sum(
        eigenvalues_for_variance
    )
)


if total_variance <= 0:

    raise RuntimeError(
        "Total PCA variance is non-positive."
    )


explained_fraction = (
    eigenvalues_for_variance
    /
    total_variance
)


explained_percent = (
    explained_fraction
    *
    100.0
)


cumulative_fraction = np.cumsum(
    explained_fraction
)


cumulative_percent = (
    cumulative_fraction
    *
    100.0
)


print(
    f"\nEigen decomposition completed in "
    f"{(time.time()-eig_start):.2f} sec"
)


# ============================================================
# EIGENVECTOR ORTHONORMALITY QC
# ============================================================

identity = (
    eigenvectors.T
    @
    eigenvectors
)


orthogonality_error = float(
    np.max(
        np.abs(
            identity
            -
            np.eye(
                EXPECTED_BANDS
            )
        )
    )
)


print(
    f"Maximum eigenvector orthogonality error: "
    f"{orthogonality_error:.3e}"
)


# ============================================================
# EXPLAINED VARIANCE THRESHOLDS
# ============================================================

threshold_results = {}


print("\n")
print("=" * 96)
print("EXPLAINED VARIANCE THRESHOLDS")
print("=" * 96)


for threshold in VARIANCE_THRESHOLDS:


    n_components = int(
        np.searchsorted(
            cumulative_fraction,
            threshold
        )
        +
        1
    )


    threshold_results[
        threshold
    ] = n_components


    print(
        f"\n{threshold*100:.1f}% variance:"
    )

    print(
        f"  first {n_components} PCs"
    )


# ============================================================
# HEURISTIC SCREE ELBOW
#
# This is a descriptive heuristic only.
#
# It is NOT a final signal/noise cutoff.
#
# MNF will be used later for that decision.
# ============================================================

n_knee = min(
    100,
    EXPECTED_BANDS
)


positive_eigs = np.maximum(
    eigenvalues_for_variance[
        :n_knee
    ],
    np.finfo(
        np.float64
    ).tiny
)


x = np.linspace(
    0,
    1,
    n_knee
)


y = np.log10(
    positive_eigs
)


# Normalize y.
if np.max(y) > np.min(y):

    y_norm = (
        y - np.min(y)
    ) / (
        np.max(y)
        -
        np.min(y)
    )

else:

    y_norm = np.zeros_like(
        y
    )


# Straight line connecting first/last points.
line = (
    y_norm[0]
    +
    (
        y_norm[-1]
        -
        y_norm[0]
    )
    *
    x
)


distance = np.abs(
    y_norm
    -
    line
)


heuristic_elbow_pc = int(
    np.argmax(
        distance
    )
    +
    1
)


print("\n")
print("=" * 96)
print("SCREE ELBOW HEURISTIC")
print("=" * 96)


print(
    f"\nHeuristic elbow near PC "
    f"{heuristic_elbow_pc}"
)


print(
    "\nIMPORTANT:"
)

print(
    "This is only a variance-curve heuristic."
)

print(
    "Do not use this alone to choose PPI/MNF components."
)


# ============================================================
# EFFECTIVE VARIANCE DIMENSION
#
# Participation ratio:
#
# (sum lambda)^2 / sum(lambda^2)
#
# Useful descriptive measure of how concentrated variance is.
# ============================================================

effective_dimension = float(

    (
        np.sum(
            eigenvalues_for_variance
        )
        **
        2
    )
    /
    np.sum(
        eigenvalues_for_variance
        **
        2
    )
)


print(
    f"\nEffective variance dimension "
    f"(participation ratio): "
    f"{effective_dimension:.2f}"
)


# ============================================================
# SAVE PCA MODEL
# ============================================================

np.save(
    MODEL_MEAN,
    mean_spectrum
)


np.save(
    MODEL_COVARIANCE,
    covariance
)


np.save(
    MODEL_EIGENVALUES,
    eigenvalues
)


np.save(
    MODEL_EIGENVECTORS,
    eigenvectors
)


# ============================================================
# EIGENVALUE TABLE
# ============================================================

eigen_df = pd.DataFrame({

    "PC":
        np.arange(
            1,
            EXPECTED_BANDS + 1
        ),

    "eigenvalue":
        eigenvalues,

    "explained_variance_percent":
        explained_percent,

    "cumulative_variance_percent":
        cumulative_percent,
})


eigen_df.to_csv(
    EIGEN_CSV,
    index=False
)


# ============================================================
# MEAN / STD SPECTRAL TABLE
# ============================================================

mean_df = pd.DataFrame({

    "science_band":
        np.arange(
            1,
            EXPECTED_BANDS + 1
        ),

    "original_425_band":
        original_bands,

    "wavelength_nm":
        wavelengths,

    "mean_reflectance":
        mean_spectrum,

    "std_reflectance":
        std_spectrum,
})


mean_df.to_csv(
    MEAN_CSV,
    index=False
)


# ============================================================
# PCA LOADINGS
#
# eigenvectors[:, 0] = PC1 spectral loading vector.
# ============================================================

loadings_data = {

    "science_band":
        np.arange(
            1,
            EXPECTED_BANDS + 1
        ),

    "original_425_band":
        original_bands,

    "wavelength_nm":
        wavelengths,
}


for pc in range(
    EXPECTED_BANDS
):

    loadings_data[
        f"PC{pc+1}"
    ] = (
        eigenvectors[
            :,
            pc
        ]
    )


loadings_df = pd.DataFrame(
    loadings_data
)


loadings_df.to_csv(
    LOADINGS_CSV,
    index=False
)


# ============================================================
# SAMPLING QC
# ============================================================

sampling_qc = pd.DataFrame([{

    "source_width":
        width,

    "source_height":
        height,

    "source_bands":
        EXPECTED_BANDS,

    "sampling_stride_pixels":
        SAMPLE_STRIDE,

    "sampling_spacing_m":
        SAMPLE_STRIDE * 4.0,

    "sample_grid_width":
        sample_width,

    "sample_grid_height":
        sample_height,

    "potential_samples":
        potential_samples,

    "valid_samples":
        valid_samples,

    "invalid_samples":
        invalid_samples,

    "valid_sample_fraction_percent":
        (
            100
            *
            valid_samples
            /
            potential_samples
        ),

    "sample_global_minimum":
        global_min,

    "sample_global_median":
        global_median,

    "sample_global_maximum":
        global_max,

    "covariance_symmetry_error":
        symmetry_error,

    "eigenvector_orthogonality_error":
        orthogonality_error,
}])


sampling_qc.to_csv(
    SAMPLE_SUMMARY_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "source_cube":
        str(
            SOURCE_DAT
        ),

    "bands":
        EXPECTED_BANDS,

    "valid_sample_pixels":
        valid_samples,

    "sampling_stride_pixels":
        SAMPLE_STRIDE,

    "sampling_spacing_m":
        SAMPLE_STRIDE * 4.0,

    "total_variance":
        total_variance,

    "PC1_variance_percent":
        float(
            explained_percent[
                0
            ]
        ),

    "PC2_variance_percent":
        float(
            explained_percent[
                1
            ]
        ),

    "PC3_variance_percent":
        float(
            explained_percent[
                2
            ]
        ),

    "PC1_to_PC3_cumulative_percent":
        float(
            cumulative_percent[
                2
            ]
        ),

    "PCs_for_95_percent":
        threshold_results[
            0.95
        ],

    "PCs_for_99_percent":
        threshold_results[
            0.99
        ],

    "PCs_for_99p5_percent":
        threshold_results[
            0.995
        ],

    "PCs_for_99p9_percent":
        threshold_results[
            0.999
        ],

    "heuristic_scree_elbow_PC":
        heuristic_elbow_pc,

    "effective_variance_dimension":
        effective_dimension,

    "covariance_symmetry_error":
        symmetry_error,

    "eigenvector_orthogonality_error":
        orthogonality_error,

    "pca_basis":
        "covariance",

    "full_PCA_raster_created":
        False,

    "source_raster_modified":
        False,

    "step10a_status":
        "PASS",
}


summary_df = pd.DataFrame(
    [
        summary
    ]
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# MODEL JSON
# ============================================================

model_metadata = {

    "source_cube":
        str(
            SOURCE_DAT
        ),

    "source_header":
        str(
            SOURCE_HDR
        ),

    "number_of_bands":
        EXPECTED_BANDS,

    "sample_stride_pixels":
        SAMPLE_STRIDE,

    "sample_spacing_m":
        SAMPLE_STRIDE * 4.0,

    "valid_sample_pixels":
        valid_samples,

    "PCA_type":
        "covariance PCA",

    "centering":
        "sample mean spectrum",

    "standardization":
        "none",

    "eigenvector_orientation":
        "columns are PC eigenvectors",

    "transformation_equation":
        "(pixel_spectrum - PCA_mean) @ PCA_eigenvectors",

    "PCs_for_95_percent":
        threshold_results[
            0.95
        ],

    "PCs_for_99_percent":
        threshold_results[
            0.99
        ],

    "PCs_for_99p5_percent":
        threshold_results[
            0.995
        ],

    "PCs_for_99p9_percent":
        threshold_results[
            0.999
        ],

    "heuristic_scree_elbow_PC":
        heuristic_elbow_pc,

    "effective_variance_dimension":
        effective_dimension,

    "note":
        (
            "PCA variance thresholds are descriptive. "
            "MNF signal-to-noise analysis should determine "
            "components used for PPI/endmember extraction."
        ),
}


with MODEL_JSON.open(
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        model_metadata,
        f,
        indent=2
    )


# ============================================================
# PLOT 1 — SCREE
# ============================================================

n_plot = min(
    SCREE_PCS_TO_PLOT,
    EXPECTED_BANDS
)


plt.figure(
    figsize=(13, 7)
)


plt.semilogy(
    np.arange(
        1,
        n_plot + 1
    ),
    eigenvalues_for_variance[
        :n_plot
    ],
    marker=".",
    markersize=4
)


plt.axvline(
    heuristic_elbow_pc,
    linestyle="--",
    linewidth=1,
    label=(
        f"Heuristic elbow "
        f"PC {heuristic_elbow_pc}"
    )
)


plt.xlabel(
    "Principal component"
)


plt.ylabel(
    "Eigenvalue — log scale"
)


plt.title(
    "AVIRIS-NG PCA Scree Plot\n"
    "326-Band Scientific Mosaic"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_SCREE,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 2 — CUMULATIVE VARIANCE
# ============================================================

plt.figure(
    figsize=(13, 7)
)


plt.plot(
    np.arange(
        1,
        EXPECTED_BANDS + 1
    ),
    cumulative_percent
)


for threshold in [
    95,
    99,
    99.5,
    99.9,
]:

    plt.axhline(
        threshold,
        linestyle="--",
        linewidth=0.8
    )


plt.xlabel(
    "Number of principal components"
)


plt.ylabel(
    "Cumulative explained variance (%)"
)


plt.title(
    "AVIRIS-NG PCA Cumulative Explained Variance"
)


plt.ylim(
    0,
    100.1
)


plt.grid(
    alpha=0.25
)


plt.tight_layout()


plt.savefig(
    PLOT_CUMULATIVE,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 3 — SAMPLE MEAN SPECTRUM
# ============================================================

plt.figure(
    figsize=(14, 7)
)


plt.plot(
    wavelengths,
    mean_spectrum
)


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Mean reflectance"
)


plt.title(
    "AVIRIS-NG PCA Sample Mean Spectrum"
)


plt.grid(
    alpha=0.25
)


plt.tight_layout()


plt.savefig(
    PLOT_MEAN,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 4 — FIRST SIX PC LOADINGS
# ============================================================

plt.figure(
    figsize=(15, 8)
)


for pc in range(
    min(
        LOADINGS_TO_PLOT,
        EXPECTED_BANDS
    )
):

    plt.plot(
        wavelengths,
        eigenvectors[
            :,
            pc
        ],
        label=f"PC{pc+1}"
    )


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Eigenvector loading"
)


plt.title(
    "AVIRIS-NG PCA Spectral Loadings — "
    "First Six PCs"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_LOADINGS,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL CONSOLE REPORT
# ============================================================

print("\n")
print("=" * 96)

print(
    "STEP 10A PCA STATISTICAL ANALYSIS COMPLETE"
)

print("=" * 96)


print(
    f"\nValid PCA samples : "
    f"{valid_samples:,}"
)


print(
    f"\nPC1 variance      : "
    f"{explained_percent[0]:.6f}%"
)


print(
    f"PC2 variance      : "
    f"{explained_percent[1]:.6f}%"
)


print(
    f"PC3 variance      : "
    f"{explained_percent[2]:.6f}%"
)


print(
    f"\nPC1–PC3 cumulative: "
    f"{cumulative_percent[2]:.6f}%"
)


print(
    "\nVariance thresholds:"
)


print(
    f"  95.0%  -> "
    f"{threshold_results[0.95]} PCs"
)


print(
    f"  99.0%  -> "
    f"{threshold_results[0.99]} PCs"
)


print(
    f"  99.5%  -> "
    f"{threshold_results[0.995]} PCs"
)


print(
    f"  99.9%  -> "
    f"{threshold_results[0.999]} PCs"
)


print(
    f"\nHeuristic scree elbow: "
    f"PC {heuristic_elbow_pc}"
)


print(
    f"Effective variance dimension: "
    f"{effective_dimension:.2f}"
)


print(
    "\nOutputs:"
)


print(
    f"\n  Eigenvalues:"
)

print(
    f"  {EIGEN_CSV}"
)


print(
    f"\n  PCA summary:"
)

print(
    f"  {SUMMARY_CSV}"
)


print(
    f"\n  Loadings:"
)

print(
    f"  {LOADINGS_CSV}"
)


print(
    f"\n  PCA model:"
)

print(
    f"  {MODEL_JSON}"
)


print(
    f"\n  Scree plot:"
)

print(
    f"  {PLOT_SCREE}"
)


print(
    f"\n  Cumulative variance:"
)

print(
    f"  {PLOT_CUMULATIVE}"
)


print("\n")
print("=" * 96)
print("IMPORTANT")
print("=" * 96)


print(
    "\n1. No PCA raster was created."
)


print(
    "2. The 326-band scientific master "
    "was not modified."
)


print(
    "3. The PCA model has been saved "
    "for later block-wise transformation."
)


print(
    "4. PCA variance does NOT by itself "
    "identify signal-versus-noise components."
)


print(
    "5. Review eigenvalue decay, cumulative "
    "variance and spectral loadings first."
)


print(
    "6. MNF noise/SNR analysis remains "
    "the primary component-selection step "
    "for PPI/endmember extraction."
)
