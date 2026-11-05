from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import rasterio
import spectral as spy
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path.cwd()

SCIENCE_DIR = (
    ROOT
    / "03_Mosaic"
    / "Scientific_326Band_Subset"
)

SOURCE_DAT = (
    SCIENCE_DIR
    / "AVIRIS_NG_mosaic_science_326band_4m.dat"
)

SOURCE_HDR = (
    SCIENCE_DIR
    / "AVIRIS_NG_mosaic_science_326band_4m.hdr"
)

MAPPING_CSV = (
    SCIENCE_DIR
    / "AVIRIS_326band_to_original_425band_mapping.csv"
)

PCA_DIR = (
    ROOT
    / "04_PCA"
    / "Step10A_PCA_Statistics"
)

PCA_COV_FILE = (
    PCA_DIR
    / "PCA_covariance_326x326.npy"
)

PCA_MEAN_FILE = (
    PCA_DIR
    / "PCA_mean_326.npy"
)

COVERAGE_FILE = (
    ROOT
    / "03_Mosaic"
    / "Final_425Band_Mosaic"
    / "AVIRIS_NG_mosaic_coverage_count.tif"
)

OUTDIR = (
    ROOT
    / "05_MNF"
    / "Step10B_MNF_Statistics"
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
    / "AVIRIS_MNF_eigenvalues_SNR.csv"
)

BAND_NOISE_CSV = (
    OUTDIR
    / "AVIRIS_MNF_per_band_noise_QC.csv"
)

WEIGHTS_CSV = (
    OUTDIR
    / "AVIRIS_MNF_transform_weights.csv"
)

WINDOW_QC_CSV = (
    OUTDIR
    / "AVIRIS_MNF_noise_sampling_windows_QC.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step10B_MNF_summary.csv"
)

MODEL_JSON = (
    OUTDIR
    / "MNF_model_metadata.json"
)


# ============================================================
# MODEL FILES FOR LATER MNF IMAGE CREATION
# ============================================================

NOISE_COV_FILE = (
    OUTDIR
    / "MNF_noise_covariance_326x326.npy"
)

NOISE_COV_REG_FILE = (
    OUTDIR
    / "MNF_noise_covariance_regularized_326x326.npy"
)

WHITENING_FILE = (
    OUTDIR
    / "MNF_noise_whitening_326x326.npy"
)

MNF_EIGENVALUES_FILE = (
    OUTDIR
    / "MNF_eigenvalues_326.npy"
)

MNF_TRANSFORM_FILE = (
    OUTDIR
    / "MNF_transform_326x326.npy"
)

MNF_MEAN_FILE = (
    OUTDIR
    / "MNF_mean_326.npy"
)


# ============================================================
# PLOTS
# ============================================================

PLOT_EIGEN = (
    OUTDIR
    / "AVIRIS_MNF_eigenvalue_plot.png"
)

PLOT_SNR = (
    OUTDIR
    / "AVIRIS_MNF_estimated_SNR_plot.png"
)

PLOT_NOISE_FRACTION = (
    OUTDIR
    / "AVIRIS_MNF_per_band_noise_fraction.png"
)

PLOT_NOISE_STD = (
    OUTDIR
    / "AVIRIS_MNF_noise_std_spectrum.png"
)

PLOT_WEIGHTS = (
    OUTDIR
    / "AVIRIS_MNF_first6_transform_weights.png"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_BANDS = 326

# Number of spatial strata used to distribute local noise
# sampling over the mosaic footprint.
GRID_ROWS = 6
GRID_COLS = 8

# Full-resolution local window.
WINDOW_SIZE = 128

# Ignore windows mostly outside the valid mosaic footprint.
MIN_WINDOW_VALID_FRACTION = 0.20

# Remove strong spatial boundaries from the noise estimate.
#
# Among locally valid adjacent-pixel pairs, retain the lower
# 70% of gradients measured from several guide wavelengths.
#
# This reduces contamination from real geological/land-cover
# boundaries while preserving local detector/spatial noise.
HOMOGENEOUS_PERCENTILE = 70.0

# Maximum number of pair differences retained from each
# direction in each window.
#
# 48 windows × 800 × 2 directions gives a potential maximum
# of ~76,800 full 326-dimensional local noise samples.
MAX_PAIRS_PER_DIRECTION_PER_WINDOW = 800

RANDOM_SEED = 20260903

# Target condition number for numerical regularization of the
# noise covariance.
MAX_NOISE_CONDITION = 1e8

# Components shown prominently in diagnostic plots.
MNF_COMPONENTS_TO_PLOT = 80

TRANSFORM_WEIGHTS_TO_PLOT = 6


# ============================================================
# START
# ============================================================

print("=" * 98)

print(
    "AVIRIS-NG STEP 10B — "
    "MNF NOISE COVARIANCE + SIGNAL/NOISE ANALYSIS"
)

print("=" * 98)

print("\nIMPORTANT:")

print(
    "The 326-band scientific master will NOT be modified."
)

print(
    "No MNF raster will be created."
)

print(
    "Noise will be estimated from full-resolution "
    "4 m adjacent-pixel differences."
)

print(
    "Horizontal and vertical noise estimates will be "
    "combined equally."
)

print(
    "Strong local edges are suppressed using a "
    "homogeneous-pair gradient filter."
)


# ============================================================
# INPUT VALIDATION
# ============================================================

required_files = [
    SOURCE_DAT,
    SOURCE_HDR,
    MAPPING_CSV,
    PCA_COV_FILE,
    PCA_MEAN_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD SPECTRAL METADATA
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
        "326-band wavelength metadata missing."
    )


wavelengths = np.asarray(
    metadata[
        "wavelength"
    ],
    dtype=np.float64
)


if len(wavelengths) != EXPECTED_BANDS:

    raise RuntimeError(
        f"Expected 326 wavelengths; "
        f"found {len(wavelengths)}."
    )


mapping = pd.read_csv(
    MAPPING_CSV
)


if len(mapping) != EXPECTED_BANDS:

    raise RuntimeError(
        "Band mapping does not contain 326 rows."
    )


original_bands = (
    mapping[
        "original_425_band"
    ]
    .astype(int)
    .to_numpy()
)


if not np.allclose(
    wavelengths,
    mapping[
        "wavelength_nm"
    ].to_numpy(
        dtype=float
    ),
    atol=0.01
):

    raise RuntimeError(
        "Band mapping wavelengths do not match header."
    )


# ============================================================
# LOAD STEP 10A SIGNAL/DATA COVARIANCE
# ============================================================

data_covariance = np.load(
    PCA_COV_FILE
).astype(
    np.float64
)


data_mean = np.load(
    PCA_MEAN_FILE
).astype(
    np.float64
)


if data_covariance.shape != (
    EXPECTED_BANDS,
    EXPECTED_BANDS
):

    raise RuntimeError(
        "PCA covariance dimensions are incorrect."
    )


if data_mean.shape != (
    EXPECTED_BANDS,
):

    raise RuntimeError(
        "PCA mean dimensions are incorrect."
    )


# Enforce exact symmetry against tiny floating-point errors.
data_covariance = (
    data_covariance
    +
    data_covariance.T
) / 2.0


# ============================================================
# SOURCE RASTER VALIDATION
# ============================================================

with rasterio.open(
    SOURCE_DAT
) as src:


    if src.count != EXPECTED_BANDS:

        raise RuntimeError(
            f"Expected 326 bands; found {src.count}."
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
            "Scientific cube is not on the 4 m grid."
        )


    width = src.width
    height = src.height
    nodata = src.nodata
    transform = src.transform
    crs = src.crs


print("\nSOURCE")
print("-" * 98)

print(
    f"Dimensions     : "
    f"{width} x {height}"
)

print(
    f"Bands          : "
    f"{EXPECTED_BANDS}"
)

print(
    f"CRS            : "
    f"{crs}"
)

print(
    f"Pixel size     : "
    f"4 m"
)

print(
    f"NoData         : "
    f"{nodata}"
)

print(
    f"Spectral range : "
    f"{wavelengths.min():.2f} – "
    f"{wavelengths.max():.2f} nm"
)


# ============================================================
# GUIDE WAVELENGTHS
#
# Used only to identify strong spatial boundaries.
#
# Noise covariance itself still uses ALL 326 bands.
# ============================================================

GUIDE_TARGETS_NM = [
    550.0,
    850.0,
    1250.0,
    1650.0,
    2200.0,
]


guide_indices = []


print("\nGUIDE BANDS")
print("-" * 98)


for target in GUIDE_TARGETS_NM:


    index = int(
        np.argmin(
            np.abs(
                wavelengths
                -
                target
            )
        )
    )


    guide_indices.append(
        index
    )


    print(
        f"Target {target:7.1f} nm "
        f"-> science band "
        f"{index + 1:3d} "
        f"({wavelengths[index]:.2f} nm)"
    )


guide_indices = np.asarray(
    guide_indices,
    dtype=int
)


# ============================================================
# BUILD VALID FOOTPRINT
#
# Prefer Step 8 coverage map because it is inexpensive and
# exactly describes the valid mosaic footprint.
# ============================================================

print("\n")
print("=" * 98)
print("BUILDING SPATIALLY STRATIFIED NOISE SAMPLE")
print("=" * 98)


if COVERAGE_FILE.exists():


    print(
        "\nUsing Step 8 coverage map for "
        "spatial window selection."
    )


    with rasterio.open(
        COVERAGE_FILE
    ) as cov_src:


        if (
            cov_src.width != width
            or
            cov_src.height != height
            or
            cov_src.transform != transform
        ):

            raise RuntimeError(
                "Coverage map does not match "
                "the 326-band cube."
            )


        footprint = (
            cov_src.read(
                1
            )
            >
            0
        )


else:


    print(
        "\nCoverage map unavailable."
    )

    print(
        "Using science band 1 to derive footprint."
    )


    with rasterio.open(
        SOURCE_DAT
    ) as src:


        first_band = src.read(
            1
        )


        footprint = np.isfinite(
            first_band
        )


        if nodata is not None:

            footprint &= (
                first_band
                !=
                nodata
            )


valid_footprint_pixels = int(
    np.sum(
        footprint
    )
)


print(
    f"\nValid mosaic footprint pixels: "
    f"{valid_footprint_pixels:,}"
)


# ============================================================
# SELECT ONE REPRESENTATIVE WINDOW PER SPATIAL GRID CELL
# ============================================================

row_edges = np.linspace(
    0,
    height,
    GRID_ROWS + 1,
    dtype=int
)

col_edges = np.linspace(
    0,
    width,
    GRID_COLS + 1,
    dtype=int
)


window_locations = []


for grid_row in range(
    GRID_ROWS
):


    for grid_col in range(
        GRID_COLS
    ):


        r0_cell = row_edges[
            grid_row
        ]

        r1_cell = row_edges[
            grid_row + 1
        ]

        c0_cell = col_edges[
            grid_col
        ]

        c1_cell = col_edges[
            grid_col + 1
        ]


        local = footprint[
            r0_cell:r1_cell,
            c0_cell:c1_cell
        ]


        coordinates = np.argwhere(
            local
        )


        if coordinates.size == 0:

            continue


        # Desired cell center in local coordinates.
        target_r = (
            local.shape[0]
            -
            1
        ) / 2.0

        target_c = (
            local.shape[1]
            -
            1
        ) / 2.0


        distance2 = (

            (
                coordinates[:, 0]
                -
                target_r
            ) ** 2

            +

            (
                coordinates[:, 1]
                -
                target_c
            ) ** 2
        )


        best = coordinates[
            np.argmin(
                distance2
            )
        ]


        center_row = int(
            r0_cell
            +
            best[0]
        )

        center_col = int(
            c0_cell
            +
            best[1]
        )


        row_off = int(
            np.clip(
                center_row
                -
                WINDOW_SIZE // 2,
                0,
                max(
                    0,
                    height
                    -
                    WINDOW_SIZE
                )
            )
        )


        col_off = int(
            np.clip(
                center_col
                -
                WINDOW_SIZE // 2,
                0,
                max(
                    0,
                    width
                    -
                    WINDOW_SIZE
                )
            )
        )


        window_locations.append(
            (
                row_off,
                col_off,
                grid_row,
                grid_col
            )
        )


# Remove accidental duplicate windows.
unique_locations = []

seen = set()


for item in window_locations:


    key = (
        item[0],
        item[1]
    )


    if key not in seen:

        unique_locations.append(
            item
        )

        seen.add(
            key
        )


window_locations = unique_locations


print(
    f"\nCandidate stratified windows: "
    f"{len(window_locations)}"
)


# ============================================================
# LOCAL DIFFERENCE EXTRACTION FUNCTION
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


def extract_differences(
    data,
    valid,
    direction,
):


    """
    Estimate local noise samples.

    Noise difference:
        n ~= (pixel_2 - pixel_1) / sqrt(2)

    The sqrt(2) scaling makes the covariance of the
    differences an estimate of the single-pixel noise
    covariance when adjacent noise is independent.
    """


    bands, h, w = data.shape


    if direction == "horizontal":


        pair_valid = (
            valid[:, 1:]
            &
            valid[:, :-1]
        )


        guide_difference = (

            data[
                guide_indices,
                :,
                1:
            ]

            -

            data[
                guide_indices,
                :,
                :-1
            ]
        )


        gradient = np.median(
            np.abs(
                guide_difference
            ),
            axis=0
        )


        flat_left = (
            data[
                :,
                :,
                :-1
            ]
            .reshape(
                bands,
                -1
            )
        )


        flat_right = (
            data[
                :,
                :,
                1:
            ]
            .reshape(
                bands,
                -1
            )
        )


    elif direction == "vertical":


        pair_valid = (
            valid[1:, :]
            &
            valid[:-1, :]
        )


        guide_difference = (

            data[
                guide_indices,
                1:,
                :
            ]

            -

            data[
                guide_indices,
                :-1,
                :
            ]
        )


        gradient = np.median(
            np.abs(
                guide_difference
            ),
            axis=0
        )


        flat_left = (
            data[
                :,
                :-1,
                :
            ]
            .reshape(
                bands,
                -1
            )
        )


        flat_right = (
            data[
                :,
                1:,
                :
            ]
            .reshape(
                bands,
                -1
            )
        )


    else:

        raise ValueError(
            direction
        )


    valid_gradient_values = gradient[
        pair_valid
    ]


    if valid_gradient_values.size < 100:

        return (
            None,
            0,
            0,
            np.nan
        )


    threshold = float(
        np.percentile(
            valid_gradient_values,
            HOMOGENEOUS_PERCENTILE
        )
    )


    homogeneous = (
        pair_valid
        &
        (
            gradient
            <=
            threshold
        )
    )


    candidate_indices = np.flatnonzero(
        homogeneous.reshape(
            -1
        )
    )


    candidate_count = int(
        candidate_indices.size
    )


    if candidate_count == 0:

        return (
            None,
            0,
            0,
            threshold
        )


    sample_count = min(
        candidate_count,
        MAX_PAIRS_PER_DIRECTION_PER_WINDOW
    )


    if candidate_count > sample_count:


        selected = rng.choice(

            candidate_indices,

            size=
                sample_count,

            replace=
                False
        )


    else:

        selected = candidate_indices


    # Extract all 326 wavelength differences only after
    # spatial pair selection.
    differences = (

        (
            flat_right[
                :,
                selected
            ]

            -

            flat_left[
                :,
                selected
            ]
        )

        /
        np.sqrt(
            2.0
        )
    ).T.astype(
        np.float32,
        copy=False
    )


    return (
        differences,
        candidate_count,
        sample_count,
        threshold
    )


# ============================================================
# READ LOCAL WINDOWS + COLLECT NOISE SAMPLES
# ============================================================

horizontal_samples = []

vertical_samples = []

window_qc_rows = []

sampling_start = time.time()


with rasterio.open(
    SOURCE_DAT
) as src:


    for window_number, (
        row_off,
        col_off,
        grid_row,
        grid_col
    ) in enumerate(
        window_locations,
        start=1
    ):


        window = rasterio.windows.Window(

            col_off=
                col_off,

            row_off=
                row_off,

            width=
                WINDOW_SIZE,

            height=
                WINDOW_SIZE
        )


        data = src.read(
            window=window
        ).astype(
            np.float32,
            copy=False
        )


        valid = np.all(
            np.isfinite(
                data
            ),
            axis=0
        )


        if nodata is not None:

            valid &= np.all(
                data
                !=
                nodata,
                axis=0
            )


        valid_fraction = float(
            np.mean(
                valid
            )
        )


        if (
            valid_fraction
            <
            MIN_WINDOW_VALID_FRACTION
        ):


            window_qc_rows.append({

                "window_number":
                    window_number,

                "grid_row":
                    grid_row,

                "grid_col":
                    grid_col,

                "row_off":
                    row_off,

                "col_off":
                    col_off,

                "valid_fraction":
                    valid_fraction,

                "used":
                    False,

                "reason":
                    "insufficient valid coverage",
            })


            continue


        (
            h_diff,
            h_candidates,
            h_count,
            h_threshold
        ) = extract_differences(

            data,
            valid,
            "horizontal"
        )


        (
            v_diff,
            v_candidates,
            v_count,
            v_threshold
        ) = extract_differences(

            data,
            valid,
            "vertical"
        )


        if h_diff is not None:

            horizontal_samples.append(
                h_diff
            )


        if v_diff is not None:

            vertical_samples.append(
                v_diff
            )


        used = (
            h_diff is not None
            and
            v_diff is not None
        )


        window_qc_rows.append({

            "window_number":
                window_number,

            "grid_row":
                grid_row,

            "grid_col":
                grid_col,

            "row_off":
                row_off,

            "col_off":
                col_off,

            "valid_fraction":
                valid_fraction,

            "horizontal_homogeneous_candidates":
                h_candidates,

            "horizontal_samples":
                h_count,

            "horizontal_gradient_threshold":
                h_threshold,

            "vertical_homogeneous_candidates":
                v_candidates,

            "vertical_samples":
                v_count,

            "vertical_gradient_threshold":
                v_threshold,

            "used":
                used,

            "reason":
                "accepted"
                if used
                else
                "insufficient adjacent pairs",
        })


        print(
            f"Window "
            f"{window_number:2d}/"
            f"{len(window_locations)} "
            f"| valid="
            f"{100*valid_fraction:5.1f}% "
            f"| H={h_count:4d} "
            f"| V={v_count:4d}"
        )


window_qc_df = pd.DataFrame(
    window_qc_rows
)


window_qc_df.to_csv(
    WINDOW_QC_CSV,
    index=False
)


used_window_count = int(
    np.sum(
        window_qc_df[
            "used"
        ]
        ==
        True
    )
)


if used_window_count < 12:

    raise RuntimeError(
        "Too few valid spatial windows "
        "for a robust MNF noise estimate."
    )


if len(horizontal_samples) == 0:

    raise RuntimeError(
        "No horizontal noise samples obtained."
    )


if len(vertical_samples) == 0:

    raise RuntimeError(
        "No vertical noise samples obtained."
    )


H = np.concatenate(
    horizontal_samples,
    axis=0
).astype(
    np.float64
)


V = np.concatenate(
    vertical_samples,
    axis=0
).astype(
    np.float64
)


del horizontal_samples
del vertical_samples


horizontal_count = H.shape[0]
vertical_count = V.shape[0]


print("\n")
print("=" * 98)
print("NOISE SAMPLE SUMMARY")
print("=" * 98)


print(
    f"\nSpatial windows used : "
    f"{used_window_count}"
)

print(
    f"Horizontal samples   : "
    f"{horizontal_count:,}"
)

print(
    f"Vertical samples     : "
    f"{vertical_count:,}"
)

print(
    f"Total local pairs    : "
    f"{horizontal_count + vertical_count:,}"
)

print(
    f"Sampling completed in "
    f"{(time.time()-sampling_start)/60:.2f} min"
)


if horizontal_count < 10000:

    raise RuntimeError(
        "Too few horizontal noise samples."
    )


if vertical_count < 10000:

    raise RuntimeError(
        "Too few vertical noise samples."
    )


# ============================================================
# DIRECTIONAL NOISE COVARIANCES
# ============================================================

print("\n")
print("=" * 98)
print("COMPUTING 326 x 326 NOISE COVARIANCE")
print("=" * 98)


cov_start = time.time()


H_mean = np.mean(
    H,
    axis=0
)


H_centered = (
    H
    -
    H_mean
)


noise_cov_h = (

    H_centered.T
    @
    H_centered

) / (
    horizontal_count
    -
    1
)


del H
del H_centered


V_mean = np.mean(
    V,
    axis=0
)


V_centered = (
    V
    -
    V_mean
)


noise_cov_v = (

    V_centered.T
    @
    V_centered

) / (
    vertical_count
    -
    1
)


del V
del V_centered


# Equal weighting of horizontal and vertical directions.
noise_covariance = (

    noise_cov_h
    +
    noise_cov_v

) / 2.0


noise_covariance = (

    noise_covariance
    +
    noise_covariance.T

) / 2.0


print(
    f"\nNoise covariance completed in "
    f"{(time.time()-cov_start)/60:.2f} min"
)


# ============================================================
# DIRECTIONAL CONSISTENCY
# ============================================================

noise_variance_h = np.diag(
    noise_cov_h
)


noise_variance_v = np.diag(
    noise_cov_v
)


direction_ratio = np.divide(

    noise_variance_h,

    noise_variance_v,

    out=
        np.full(
            EXPECTED_BANDS,
            np.nan
        ),

    where=
        noise_variance_v > 0
)


median_direction_ratio = float(
    np.nanmedian(
        direction_ratio
    )
)


print(
    f"\nMedian H/V per-band noise variance ratio: "
    f"{median_direction_ratio:.4f}"
)


# ============================================================
# NOISE COVARIANCE EIGEN ANALYSIS + REGULARIZATION
# ============================================================

noise_eigenvalues, noise_eigenvectors = np.linalg.eigh(
    noise_covariance
)


noise_eigenvalues = noise_eigenvalues.astype(
    np.float64
)


max_noise_eigenvalue = float(
    np.max(
        noise_eigenvalues
    )
)


positive_noise = noise_eigenvalues[
    noise_eigenvalues
    >
    0
]


if positive_noise.size == 0:

    raise RuntimeError(
        "Noise covariance has no positive eigenvalues."
    )


min_positive_noise_eigenvalue = float(
    np.min(
        positive_noise
    )
)


raw_noise_condition = (

    max_noise_eigenvalue
    /
    min_positive_noise_eigenvalue
)


eigenvalue_floor = (

    max_noise_eigenvalue
    /
    MAX_NOISE_CONDITION
)


regularized_noise_eigenvalues = np.maximum(

    noise_eigenvalues,

    eigenvalue_floor
)


regularized_count = int(
    np.sum(
        noise_eigenvalues
        <
        eigenvalue_floor
    )
)


regularized_condition = float(

    np.max(
        regularized_noise_eigenvalues
    )

    /

    np.min(
        regularized_noise_eigenvalues
    )
)


print("\n")
print("=" * 98)
print("NOISE COVARIANCE CONDITIONING")
print("=" * 98)


print(
    f"\nRaw condition number       : "
    f"{raw_noise_condition:.3e}"
)

print(
    f"Target maximum condition   : "
    f"{MAX_NOISE_CONDITION:.3e}"
)

print(
    f"Regularized eigenvalues    : "
    f"{regularized_count} / 326"
)

print(
    f"Eigenvalue floor           : "
    f"{eigenvalue_floor:.6e}"
)

print(
    f"Regularized condition      : "
    f"{regularized_condition:.3e}"
)


# Cn_reg = U diag(d) U^T
noise_covariance_regularized = (

    noise_eigenvectors
    *
    regularized_noise_eigenvalues

) @ noise_eigenvectors.T


noise_covariance_regularized = (

    noise_covariance_regularized
    +
    noise_covariance_regularized.T

) / 2.0


# ============================================================
# NOISE WHITENING MATRIX
# ============================================================

inverse_sqrt_noise = (

    1.0
    /
    np.sqrt(
        regularized_noise_eigenvalues
    )
)


whitening = (

    noise_eigenvectors
    *
    inverse_sqrt_noise

) @ noise_eigenvectors.T


whitening = (

    whitening
    +
    whitening.T

) / 2.0


# ============================================================
# WHITENING QC
#
# W Cn W^T should be identity.
# ============================================================

noise_identity = (

    whitening
    @
    noise_covariance_regularized
    @
    whitening.T
)


whitening_error = float(

    np.max(
        np.abs(
            noise_identity
            -
            np.eye(
                EXPECTED_BANDS
            )
        )
    )
)


print(
    f"\nNoise-whitening identity error: "
    f"{whitening_error:.3e}"
)


# ============================================================
# MNF EIGEN ANALYSIS
#
# After noise whitening:
#
# Cw = W Cdata W^T
#
# If noise variance has been normalized to 1:
#
# MNF eigenvalue ~= 1 + SNR
#
# Therefore:
#
# estimated SNR ~= eigenvalue - 1
# ============================================================

print("\n")
print("=" * 98)
print("MNF EIGEN ANALYSIS")
print("=" * 98)


mnf_start = time.time()


whitened_data_covariance = (

    whitening
    @
    data_covariance
    @
    whitening.T
)


whitened_data_covariance = (

    whitened_data_covariance
    +
    whitened_data_covariance.T

) / 2.0


mnf_eigenvalues, whitened_eigenvectors = np.linalg.eigh(
    whitened_data_covariance
)


order = np.argsort(
    mnf_eigenvalues
)[::-1]


mnf_eigenvalues = (
    mnf_eigenvalues[
        order
    ]
)


whitened_eigenvectors = (
    whitened_eigenvectors[
        :,
        order
    ]
)


# Forward transform:
#
# score = (pixel - mean) @ MNF_transform
#
mnf_transform = (
    whitening
    @
    whitened_eigenvectors
)


# ============================================================
# STABILIZE COMPONENT SIGNS
#
# Eigenvector signs are mathematically arbitrary.
#
# Make the largest absolute transform weight positive for
# reproducible plots and later MNF output.
# ============================================================

for component in range(
    EXPECTED_BANDS
):


    strongest_band = int(
        np.argmax(
            np.abs(
                mnf_transform[
                    :,
                    component
                ]
            )
        )
    )


    if (
        mnf_transform[
            strongest_band,
            component
        ]
        <
        0
    ):


        mnf_transform[
            :,
            component
        ] *= -1.0


        whitened_eigenvectors[
            :,
            component
        ] *= -1.0


estimated_snr = (
    mnf_eigenvalues
    -
    1.0
)


print(
    f"\nMNF analysis completed in "
    f"{time.time()-mnf_start:.2f} sec"
)


# ============================================================
# MNF TRANSFORM QC
#
# A^T Cn A should be identity.
#
# A^T Cdata A should be diagonal with MNF eigenvalues.
# ============================================================

transformed_noise = (

    mnf_transform.T
    @
    noise_covariance_regularized
    @
    mnf_transform
)


transformed_data = (

    mnf_transform.T
    @
    data_covariance
    @
    mnf_transform
)


mnf_noise_identity_error = float(

    np.max(
        np.abs(
            transformed_noise
            -
            np.eye(
                EXPECTED_BANDS
            )
        )
    )
)


expected_diagonal = np.diag(
    mnf_eigenvalues
)


mnf_data_diagonalization_error = float(

    np.max(
        np.abs(
            transformed_data
            -
            expected_diagonal
        )
    )
)


print(
    f"\nMNF noise identity error       : "
    f"{mnf_noise_identity_error:.3e}"
)

print(
    f"MNF covariance diagonal error  : "
    f"{mnf_data_diagonalization_error:.3e}"
)


# ============================================================
# COMPONENT SNR THRESHOLD COUNTS
#
# These are diagnostic only.
#
# Final PPI component selection will occur AFTER we inspect
# the actual MNF eigenvalue/SNR behaviour.
# ============================================================

snr_thresholds = [
    5.0,
    3.0,
    2.0,
    1.0,
    0.5,
    0.2,
    0.0,
]


snr_counts = {}


print("\n")
print("=" * 98)
print("MNF ESTIMATED SNR COUNTS")
print("=" * 98)


for threshold in snr_thresholds:


    count = int(
        np.sum(
            estimated_snr
            >
            threshold
        )
    )


    snr_counts[
        threshold
    ] = count


    print(
        f"\nSNR > {threshold:4.1f} : "
        f"{count} components"
    )


# ============================================================
# PER-BAND NOISE DIAGNOSTICS
# ============================================================

data_variance = np.diag(
    data_covariance
)


noise_variance = np.diag(
    noise_covariance
)


noise_std = np.sqrt(
    np.maximum(
        noise_variance,
        0
    )
)


data_std = np.sqrt(
    np.maximum(
        data_variance,
        0
    )
)


noise_fraction_percent = np.divide(

    noise_variance,

    data_variance,

    out=
        np.full(
            EXPECTED_BANDS,
            np.nan
        ),

    where=
        data_variance > 0

) * 100.0


per_band_snr = np.divide(

    data_variance,

    noise_variance,

    out=
        np.full(
            EXPECTED_BANDS,
            np.nan
        ),

    where=
        noise_variance > 0

) - 1.0


band_noise_df = pd.DataFrame({

    "science_band":
        np.arange(
            1,
            EXPECTED_BANDS + 1
        ),

    "original_425_band":
        original_bands,

    "wavelength_nm":
        wavelengths,

    "data_variance":
        data_variance,

    "data_std":
        data_std,

    "noise_variance":
        noise_variance,

    "noise_std":
        noise_std,

    "noise_fraction_percent":
        noise_fraction_percent,

    "per_band_variance_SNR_estimate":
        per_band_snr,

    "horizontal_noise_variance":
        noise_variance_h,

    "vertical_noise_variance":
        noise_variance_v,

    "horizontal_vertical_noise_ratio":
        direction_ratio,
})


band_noise_df.to_csv(
    BAND_NOISE_CSV,
    index=False
)


median_noise_fraction = float(
    np.nanmedian(
        noise_fraction_percent
    )
)


p95_noise_fraction = float(
    np.nanpercentile(
        noise_fraction_percent,
        95
    )
)


max_noise_index = int(
    np.nanargmax(
        noise_fraction_percent
    )
)


max_noise_fraction = float(
    noise_fraction_percent[
        max_noise_index
    ]
)


print("\n")
print("=" * 98)
print("PER-BAND NOISE SUMMARY")
print("=" * 98)


print(
    f"\nMedian noise/data variance : "
    f"{median_noise_fraction:.4f}%"
)

print(
    f"95th percentile            : "
    f"{p95_noise_fraction:.4f}%"
)

print(
    f"Maximum                    : "
    f"{max_noise_fraction:.4f}%"
)

print(
    f"Maximum-noise wavelength   : "
    f"{wavelengths[max_noise_index]:.2f} nm"
)

print(
    f"Science band               : "
    f"{max_noise_index + 1}"
)

print(
    f"Original 425 band          : "
    f"{original_bands[max_noise_index]}"
)


# ============================================================
# REPORT TOP 15 NOISIEST RETAINED BANDS
# ============================================================

top_noise_order = np.argsort(
    np.nan_to_num(
        noise_fraction_percent,
        nan=-np.inf
    )
)[::-1]


print("\nTop 15 retained bands by noise fraction:")


for rank, index in enumerate(
    top_noise_order[:15],
    start=1
):


    print(
        f"{rank:2d}. "
        f"Science band "
        f"{index+1:3d} | "
        f"Original "
        f"{original_bands[index]:3d} | "
        f"{wavelengths[index]:8.2f} nm | "
        f"noise fraction="
        f"{noise_fraction_percent[index]:8.3f}%"
    )


# ============================================================
# MNF EIGENVALUE TABLE
# ============================================================

signal_fraction_estimate = np.divide(

    np.maximum(
        estimated_snr,
        0
    ),

    mnf_eigenvalues,

    out=
        np.zeros(
            EXPECTED_BANDS,
            dtype=float
        ),

    where=
        mnf_eigenvalues != 0
)


mnf_df = pd.DataFrame({

    "MNF_component":
        np.arange(
            1,
            EXPECTED_BANDS + 1
        ),

    "MNF_eigenvalue":
        mnf_eigenvalues,

    "estimated_SNR":
        estimated_snr,

    "estimated_signal_fraction":
        signal_fraction_estimate,

    "SNR_gt_5":
        estimated_snr > 5,

    "SNR_gt_3":
        estimated_snr > 3,

    "SNR_gt_2":
        estimated_snr > 2,

    "SNR_gt_1":
        estimated_snr > 1,

    "SNR_gt_0p5":
        estimated_snr > 0.5,

    "SNR_gt_0":
        estimated_snr > 0,
})


mnf_df.to_csv(
    EIGEN_CSV,
    index=False
)


# ============================================================
# SAVE TRANSFORM WEIGHTS
# ============================================================

weights_data = {

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


for component in range(
    EXPECTED_BANDS
):


    weights_data[
        f"MNF{component+1}"
    ] = (
        mnf_transform[
            :,
            component
        ]
    )


weights_df = pd.DataFrame(
    weights_data
)


weights_df.to_csv(
    WEIGHTS_CSV,
    index=False
)


# ============================================================
# SAVE NUMERICAL MODEL
# ============================================================

np.save(
    NOISE_COV_FILE,
    noise_covariance
)


np.save(
    NOISE_COV_REG_FILE,
    noise_covariance_regularized
)


np.save(
    WHITENING_FILE,
    whitening
)


np.save(
    MNF_EIGENVALUES_FILE,
    mnf_eigenvalues
)


np.save(
    MNF_TRANSFORM_FILE,
    mnf_transform
)


np.save(
    MNF_MEAN_FILE,
    data_mean
)


# ============================================================
# MODEL METADATA
# ============================================================

model_metadata = {

    "source_cube":
        str(
            SOURCE_DAT
        ),

    "bands":
        EXPECTED_BANDS,

    "noise_estimation":
        (
            "full-resolution 4 m horizontal and vertical "
            "adjacent-pixel differences divided by sqrt(2)"
        ),

    "spatial_stratification":
        (
            f"{GRID_ROWS} x {GRID_COLS} grid"
        ),

    "window_size_pixels":
        WINDOW_SIZE,

    "homogeneous_pair_percentile":
        HOMOGENEOUS_PERCENTILE,

    "horizontal_noise_samples":
        int(
            horizontal_count
        ),

    "vertical_noise_samples":
        int(
            vertical_count
        ),

    "used_windows":
        int(
            used_window_count
        ),

    "noise_covariance_combination":
        (
            "equal mean of horizontal and vertical "
            "noise covariance matrices"
        ),

    "noise_covariance_max_condition":
        MAX_NOISE_CONDITION,

    "regularized_noise_eigenvalues":
        regularized_count,

    "MNF_equation":
        (
            "MNF_score = "
            "(pixel_spectrum - mean_spectrum) @ MNF_transform"
        ),

    "MNF_eigenvalue_interpretation":
        (
            "approximately 1 + estimated signal-to-noise ratio"
        ),

    "MNF_raster_created":
        False,

    "source_modified":
        False,

    "note":
        (
            "SNR thresholds are diagnostics only. "
            "Final useful MNF component count must be chosen "
            "after review of eigenvalue/SNR behaviour and "
            "component imagery."
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
# STEP STATUS
# ============================================================

structural_pass = (

    used_window_count >= 12

    and

    horizontal_count >= 10000

    and

    vertical_count >= 10000

    and

    np.all(
        np.isfinite(
            mnf_eigenvalues
        )
    )

    and

    whitening_error < 1e-5

    and

    mnf_noise_identity_error < 1e-5
)


step_status = (
    "PASS"
    if structural_pass
    else
    "REVIEW"
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary = pd.DataFrame([{

    "source_cube":
        str(
            SOURCE_DAT
        ),

    "bands":
        EXPECTED_BANDS,

    "noise_window_candidates":
        len(
            window_locations
        ),

    "noise_windows_used":
        used_window_count,

    "horizontal_noise_samples":
        horizontal_count,

    "vertical_noise_samples":
        vertical_count,

    "total_noise_samples":
        horizontal_count
        +
        vertical_count,

    "homogeneous_pair_percentile":
        HOMOGENEOUS_PERCENTILE,

    "median_H_to_V_noise_variance_ratio":
        median_direction_ratio,

    "raw_noise_covariance_condition":
        raw_noise_condition,

    "regularized_noise_covariance_condition":
        regularized_condition,

    "noise_eigenvalues_regularized":
        regularized_count,

    "noise_whitening_error":
        whitening_error,

    "mnf_noise_identity_error":
        mnf_noise_identity_error,

    "mnf_covariance_diagonalization_error":
        mnf_data_diagonalization_error,

    "MNF1_eigenvalue":
        float(
            mnf_eigenvalues[0]
        ),

    "MNF1_estimated_SNR":
        float(
            estimated_snr[0]
        ),

    "MNF2_eigenvalue":
        float(
            mnf_eigenvalues[1]
        ),

    "MNF2_estimated_SNR":
        float(
            estimated_snr[1]
        ),

    "MNF3_eigenvalue":
        float(
            mnf_eigenvalues[2]
        ),

    "MNF3_estimated_SNR":
        float(
            estimated_snr[2]
        ),

    "components_SNR_gt_5":
        snr_counts[5.0],

    "components_SNR_gt_3":
        snr_counts[3.0],

    "components_SNR_gt_2":
        snr_counts[2.0],

    "components_SNR_gt_1":
        snr_counts[1.0],

    "components_SNR_gt_0p5":
        snr_counts[0.5],

    "components_SNR_gt_0p2":
        snr_counts[0.2],

    "components_SNR_gt_0":
        snr_counts[0.0],

    "median_per_band_noise_fraction_percent":
        median_noise_fraction,

    "p95_per_band_noise_fraction_percent":
        p95_noise_fraction,

    "maximum_per_band_noise_fraction_percent":
        max_noise_fraction,

    "maximum_noise_science_band":
        max_noise_index + 1,

    "maximum_noise_original_425_band":
        int(
            original_bands[
                max_noise_index
            ]
        ),

    "maximum_noise_wavelength_nm":
        float(
            wavelengths[
                max_noise_index
            ]
        ),

    "full_MNF_raster_created":
        False,

    "source_raster_modified":
        False,

    "step10b_status":
        step_status,
}])


summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# PLOT 1 — MNF EIGENVALUES
# ============================================================

n_plot = min(
    MNF_COMPONENTS_TO_PLOT,
    EXPECTED_BANDS
)


plt.figure(
    figsize=(13, 7)
)


plt.plot(

    np.arange(
        1,
        n_plot + 1
    ),

    mnf_eigenvalues[
        :n_plot
    ],

    marker="."
)


plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="Noise level: eigenvalue = 1"
)


plt.axhline(
    2.0,
    linestyle="--",
    linewidth=1,
    label="Estimated SNR = 1"
)


plt.xlabel(
    "MNF component"
)


plt.ylabel(
    "MNF eigenvalue"
)


plt.title(
    "AVIRIS-NG MNF Noise-Adjusted Eigenvalues"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_EIGEN,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 2 — ESTIMATED SNR
# ============================================================

plt.figure(
    figsize=(13, 7)
)


plt.plot(

    np.arange(
        1,
        n_plot + 1
    ),

    estimated_snr[
        :n_plot
    ],

    marker="."
)


plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1,
    label="SNR = 0"
)


plt.axhline(
    1.0,
    linestyle="--",
    linewidth=1,
    label="SNR = 1"
)


plt.axhline(
    2.0,
    linestyle="--",
    linewidth=1,
    label="SNR = 2"
)


plt.xlabel(
    "MNF component"
)


plt.ylabel(
    "Estimated SNR = eigenvalue - 1"
)


plt.title(
    "AVIRIS-NG MNF Estimated Component SNR"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_SNR,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 3 — PER-BAND NOISE FRACTION
# ============================================================

plt.figure(
    figsize=(15, 7)
)


plt.plot(
    wavelengths,
    noise_fraction_percent
)


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Noise variance / data variance (%)"
)


plt.title(
    "AVIRIS-NG Per-Band Local Noise Fraction"
)


plt.grid(
    alpha=0.25
)


plt.tight_layout()


plt.savefig(
    PLOT_NOISE_FRACTION,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 4 — NOISE STANDARD DEVIATION
# ============================================================

plt.figure(
    figsize=(15, 7)
)


plt.plot(
    wavelengths,
    noise_std,
    label="Estimated noise standard deviation"
)


plt.plot(
    wavelengths,
    data_std,
    label="Observed data standard deviation"
)


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Reflectance standard deviation"
)


plt.title(
    "AVIRIS-NG Data Variability vs Local Noise"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_NOISE_STD,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# PLOT 5 — FIRST SIX NORMALIZED MNF TRANSFORM WEIGHTS
#
# Raw whitening weights may differ greatly in magnitude.
# Normalize each component to max absolute weight = 1 only
# for visualization.
#
# Saved model retains the true numerical weights.
# ============================================================

plt.figure(
    figsize=(15, 8)
)


for component in range(
    min(
        TRANSFORM_WEIGHTS_TO_PLOT,
        EXPECTED_BANDS
    )
):


    weights = (
        mnf_transform[
            :,
            component
        ]
    )


    scale = np.max(
        np.abs(
            weights
        )
    )


    if scale > 0:

        plot_weights = (
            weights
            /
            scale
        )

    else:

        plot_weights = weights


    plt.plot(

        wavelengths,

        plot_weights,

        label=
            f"MNF{component+1}"
    )


plt.xlabel(
    "Wavelength (nm)"
)


plt.ylabel(
    "Normalized MNF transform weight"
)


plt.title(
    "AVIRIS-NG MNF Transform Weights — First Six Components"
)


plt.grid(
    alpha=0.25
)


plt.legend()


plt.tight_layout()


plt.savefig(
    PLOT_WEIGHTS,
    dpi=220,
    bbox_inches="tight"
)


plt.close()


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 98)

print(
    "STEP 10B MNF STATISTICAL ANALYSIS COMPLETE"
)

print("=" * 98)


print(
    f"\nStep status: "
    f"{step_status}"
)


print(
    f"\nMNF1 eigenvalue : "
    f"{mnf_eigenvalues[0]:.6f}"
)

print(
    f"MNF1 SNR        : "
    f"{estimated_snr[0]:.6f}"
)


print(
    f"\nMNF2 eigenvalue : "
    f"{mnf_eigenvalues[1]:.6f}"
)

print(
    f"MNF2 SNR        : "
    f"{estimated_snr[1]:.6f}"
)


print(
    f"\nMNF3 eigenvalue : "
    f"{mnf_eigenvalues[2]:.6f}"
)

print(
    f"MNF3 SNR        : "
    f"{estimated_snr[2]:.6f}"
)


print("\nEstimated component counts:")


for threshold in [
    5.0,
    3.0,
    2.0,
    1.0,
    0.5,
    0.2,
    0.0
]:

    print(
        f"  SNR > {threshold:4.1f}: "
        f"{snr_counts[threshold]}"
    )


print(
    f"\nMedian per-band noise fraction : "
    f"{median_noise_fraction:.4f}%"
)

print(
    f"95th percentile                : "
    f"{p95_noise_fraction:.4f}%"
)

print(
    f"Maximum                        : "
    f"{max_noise_fraction:.4f}% "
    f"at "
    f"{wavelengths[max_noise_index]:.2f} nm"
)


print("\nOutputs:")


print(
    f"\n  Summary:"
)

print(
    f"  {SUMMARY_CSV}"
)


print(
    f"\n  MNF eigenvalues/SNR:"
)

print(
    f"  {EIGEN_CSV}"
)


print(
    f"\n  Per-band noise:"
)

print(
    f"  {BAND_NOISE_CSV}"
)


print(
    f"\n  Noise-window QC:"
)

print(
    f"  {WINDOW_QC_CSV}"
)


print(
    f"\n  MNF eigenvalue plot:"
)

print(
    f"  {PLOT_EIGEN}"
)


print(
    f"\n  Estimated SNR plot:"
)

print(
    f"  {PLOT_SNR}"
)


print(
    f"\n  Per-band noise plot:"
)

print(
    f"  {PLOT_NOISE_FRACTION}"
)


print(
    f"\n  First-six MNF weights:"
)

print(
    f"  {PLOT_WEIGHTS}"
)


print("\n")
print("=" * 98)
print("IMPORTANT")
print("=" * 98)


print(
    "\n1. The 326-band scientific master "
    "was NOT modified."
)


print(
    "2. No MNF raster was created."
)


print(
    "3. Noise covariance was estimated "
    "from full-resolution 4 m local differences."
)


print(
    "4. MNF eigenvalue approximately equals "
    "1 + estimated SNR under this model."
)


print(
    "5. Do NOT choose the final component count "
    "from a single numerical threshold yet."
)


print(
    "6. Review the eigenvalue curve, SNR curve, "
    "per-band noise spectrum, and edge-band behaviour."
)


print(
    "7. Only after that review should we create "
    "selected MNF component rasters for PPI."
)
