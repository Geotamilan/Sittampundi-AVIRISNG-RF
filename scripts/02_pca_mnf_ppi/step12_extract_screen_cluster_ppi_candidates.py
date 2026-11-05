from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path.cwd()

PPI_DIR = (
    ROOT
    / "06_PPI_Endmembers"
    / "Step11_MNF1_25_PPI"
)

CANDIDATE_CSV = (
    PPI_DIR
    / "AVIRIS_PPI_top300_spatially_thinned_candidates.csv"
)


SCIENCE_DIR = (
    ROOT
    / "03_Mosaic"
    / "Scientific_326Band_Subset"
)

REFLECTANCE_DAT = (
    SCIENCE_DIR
    / "AVIRIS_NG_mosaic_science_326band_4m.dat"
)

REFLECTANCE_HDR = (
    SCIENCE_DIR
    / "AVIRIS_NG_mosaic_science_326band_4m.hdr"
)

BAND_MAPPING_CSV = (
    SCIENCE_DIR
    / "AVIRIS_326band_to_original_425band_mapping.csv"
)


OUTDIR = (
    ROOT
    / "06_PPI_Endmembers"
    / "Step12_Candidate_Spectra"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True
)


TOP_SPECTRA_DIR = (
    OUTDIR
    / "Top20_Spectra"
)

TOP_SPECTRA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUTS
# ============================================================

CANDIDATE_QC_CSV = (
    OUTDIR
    / "AVIRIS_PPI_candidate_spectral_QC.csv"
)

CENTER_LONG_CSV = (
    OUTDIR
    / "AVIRIS_PPI_candidate_center_spectra_long.csv"
)

MEDIAN_LONG_CSV = (
    OUTDIR
    / "AVIRIS_PPI_candidate_3x3_median_spectra_long.csv"
)

CLUSTER_ASSIGNMENT_CSV = (
    OUTDIR
    / "AVIRIS_PPI_candidate_cluster_assignments.csv"
)

CLUSTER_MEDIAN_CSV = (
    OUTDIR
    / "AVIRIS_PPI_cluster_median_spectra.csv"
)

CLUSTER_REPRESENTATIVES_CSV = (
    OUTDIR
    / "AVIRIS_PPI_cluster_representatives.csv"
)

CLUSTER_SELECTION_CSV = (
    OUTDIR
    / "AVIRIS_PPI_cluster_number_QC.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step12_candidate_spectra_summary.csv"
)

ALL_MEDIAN_PLOT = (
    OUTDIR
    / "AVIRIS_PPI_all_3x3_median_candidate_spectra.png"
)

CLUSTER_PLOT = (
    OUTDIR
    / "AVIRIS_PPI_cluster_median_spectra.png"
)

NDVI_PPI_PLOT = (
    OUTDIR
    / "AVIRIS_PPI_NDVI_vs_PPI_count.png"
)

LOCAL_STABILITY_PLOT = (
    OUTDIR
    / "AVIRIS_PPI_local_spectral_stability.png"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_BANDS = 326

LOCAL_RADIUS = 1       # 1 = 3 x 3 window

TOP_INDIVIDUAL_PLOTS = 20

# Spectral stability threshold.
CENTER_MEDIAN_SAM_WARNING_DEG = 12.0

# Heuristic screening thresholds.
#
# IMPORTANT:
# These do NOT delete spectra.
# They only add review flags.
VEGETATION_NDVI = 0.45

ROCK_SOIL_REVIEW_NDVI_MAX = 0.35

DARK_NIR_THRESHOLD = 0.06

DARK_SWIR_THRESHOLD = 0.05

VERY_DARK_MEDIAN_THRESHOLD = 0.045

HIGH_REFLECTANCE_P95 = 0.75

HIGH_REFLECTANCE_MAX = 1.20

NEGATIVE_FRACTION_WARNING = 0.03

MIN_VALID_LOCAL_PIXELS = 5


# Candidate cluster range.
MIN_CLUSTERS = 3
MAX_CLUSTERS = 10


# ============================================================
# UTILITIES
# ============================================================

def nearest_band(
    wavelengths,
    target_nm
):

    return int(
        np.argmin(
            np.abs(
                wavelengths
                -
                target_nm
            )
        )
    )


def spectral_angle_deg(
    a,
    b
):

    a = np.asarray(
        a,
        dtype=np.float64
    )

    b = np.asarray(
        b,
        dtype=np.float64
    )

    valid = (
        np.isfinite(a)
        &
        np.isfinite(b)
    )

    if np.sum(valid) < 3:

        return np.nan

    aa = a[valid]
    bb = b[valid]

    denom = (
        np.linalg.norm(aa)
        *
        np.linalg.norm(bb)
    )

    if denom <= 0:

        return np.nan

    cosine = (
        np.dot(
            aa,
            bb
        )
        /
        denom
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


def spectral_segments(
    wavelengths,
    values,
    gap_nm=30.0
):

    """
    Yield wavelength segments separately so matplotlib
    does not draw artificial lines across the removed
    atmospheric windows.
    """

    wavelengths = np.asarray(
        wavelengths
    )

    values = np.asarray(
        values
    )

    breaks = np.where(
        np.diff(
            wavelengths
        )
        >
        gap_nm
    )[0]


    start = 0

    for brk in breaks:

        end = brk + 1

        yield (
            wavelengths[
                start:end
            ],
            values[
                start:end
            ]
        )

        start = end


    yield (
        wavelengths[
            start:
        ],
        values[
            start:
        ]
    )


def make_agglomerative(
    n_clusters
):

    try:

        return AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average"
        )

    except TypeError:

        # Compatibility with older sklearn.
        return AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="cosine",
            linkage="average"
        )


# ============================================================
# START
# ============================================================

print("=" * 100)

print(
    "AVIRIS-NG STEP 12 — "
    "PPI CANDIDATE REFLECTANCE EXTRACTION + SCREENING"
)

print("=" * 100)

print("\nIMPORTANT:")

print(
    "MNF values will NOT be interpreted as mineral spectra."
)

print(
    "Every PPI candidate will be returned to the original "
    "326-band reflectance mosaic."
)

print(
    "Both center-pixel and 3x3 median spectra will be saved."
)

print(
    "Screening flags are diagnostic only."
)

print(
    "No candidate will be automatically deleted."
)

print(
    "No mineral name will be assigned in Step 12."
)


# ============================================================
# INPUT CHECK
# ============================================================

for path in [
    CANDIDATE_CSV,
    REFLECTANCE_DAT,
    REFLECTANCE_HDR,
    BAND_MAPPING_CSV,
]:

    if not path.exists():

        raise FileNotFoundError(
            path
        )


# ============================================================
# LOAD CANDIDATES
# ============================================================

candidates = pd.read_csv(
    CANDIDATE_CSV
)


required_candidate_columns = [
    "row",
    "col",
    "easting_m",
    "northing_m",
    "refined_PPI_count",
]


for column in required_candidate_columns:

    if column not in candidates.columns:

        raise RuntimeError(
            f"Candidate table missing column: {column}"
        )


candidate_count = len(
    candidates
)


print(
    f"\nPPI candidates to extract: "
    f"{candidate_count}"
)


if candidate_count < 10:

    raise RuntimeError(
        "Too few PPI candidates."
    )


# ============================================================
# LOAD BAND TABLE
# ============================================================

band_map = pd.read_csv(
    BAND_MAPPING_CSV
)


if len(
    band_map
) != EXPECTED_BANDS:

    raise RuntimeError(
        "Band mapping does not contain 326 bands."
    )


wavelengths = (
    band_map[
        "wavelength_nm"
    ]
    .to_numpy(
        dtype=np.float64
    )
)


original_bands = (
    band_map[
        "original_425_band"
    ]
    .to_numpy(
        dtype=int
    )
)


# ============================================================
# DIAGNOSTIC WAVELENGTHS
# ============================================================

green_i = nearest_band(
    wavelengths,
    560.0
)

red_i = nearest_band(
    wavelengths,
    670.0
)

nir_i = nearest_band(
    wavelengths,
    850.0
)

swir1_i = nearest_band(
    wavelengths,
    1650.0
)

swir2_i = nearest_band(
    wavelengths,
    2200.0
)


print("\nDIAGNOSTIC BANDS")

print("-" * 100)


for label, index in [
    ("Green", green_i),
    ("Red", red_i),
    ("NIR", nir_i),
    ("SWIR1", swir1_i),
    ("SWIR2", swir2_i),
]:

    print(
        f"{label:6s}: "
        f"Science band {index+1:3d}, "
        f"{wavelengths[index]:.2f} nm"
    )


# ============================================================
# RASTER VALIDATION
# ============================================================

with rasterio.open(
    REFLECTANCE_DAT
) as src:

    if src.count != EXPECTED_BANDS:

        raise RuntimeError(
            f"Expected 326 bands, "
            f"found {src.count}."
        )


    width = src.width
    height = src.height
    nodata = src.nodata
    transform = src.transform
    crs = src.crs


print("\nREFLECTANCE MASTER")

print("-" * 100)

print(
    f"Dimensions : {width} x {height}"
)

print(
    f"Bands      : {EXPECTED_BANDS}"
)

print(
    f"CRS        : {crs}"
)

print(
    f"NoData     : {nodata}"
)

print(
    f"Range      : "
    f"{wavelengths.min():.2f} – "
    f"{wavelengths.max():.2f} nm"
)


# ============================================================
# EXTRACTION
# ============================================================

print("\n")
print("=" * 100)

print(
    "EXTRACTING CENTER + 3x3 REFLECTANCE SPECTRA"
)

print("=" * 100)


center_spectra = []

median_spectra = []

candidate_qc_rows = []

extraction_start = time.time()


with rasterio.open(
    REFLECTANCE_DAT
) as src:

    for i, row in candidates.iterrows():

        center_row = int(
            row[
                "row"
            ]
        )

        center_col = int(
            row[
                "col"
            ]
        )


        if not (
            0
            <=
            center_row
            <
            height
            and
            0
            <=
            center_col
            <
            width
        ):

            raise RuntimeError(
                f"Candidate {i+1} lies outside raster."
            )


        # ----------------------------------------------------
        # Center spectrum
        # ----------------------------------------------------

        center_window = Window(
            center_col,
            center_row,
            1,
            1
        )


        center = (
            src.read(
                window=center_window
            )[
                :,
                0,
                0
            ]
            .astype(
                np.float64
            )
        )


        center_valid = np.all(
            np.isfinite(
                center
            )
        )


        if nodata is not None:

            center_valid &= not np.any(
                center
                ==
                nodata
            )


        # ----------------------------------------------------
        # 3x3 local neighborhood
        # ----------------------------------------------------

        r0 = max(
            0,
            center_row
            -
            LOCAL_RADIUS
        )

        r1 = min(
            height,
            center_row
            +
            LOCAL_RADIUS
            +
            1
        )

        c0 = max(
            0,
            center_col
            -
            LOCAL_RADIUS
        )

        c1 = min(
            width,
            center_col
            +
            LOCAL_RADIUS
            +
            1
        )


        local_window = Window(
            c0,
            r0,
            c1-c0,
            r1-r0
        )


        local_cube = (
            src.read(
                window=local_window
            )
            .astype(
                np.float64
            )
        )


        # band x local pixel
        local_matrix = (
            local_cube
            .reshape(
                EXPECTED_BANDS,
                -1
            )
            .T
        )


        local_valid = np.all(
            np.isfinite(
                local_matrix
            ),
            axis=1
        )


        if nodata is not None:

            local_valid &= np.all(
                local_matrix
                !=
                nodata,
                axis=1
            )


        valid_local_spectra = (
            local_matrix[
                local_valid
            ]
        )


        local_valid_count = (
            len(
                valid_local_spectra
            )
        )


        if local_valid_count > 0:

            median3 = np.median(
                valid_local_spectra,
                axis=0
            )

            mad3 = np.median(
                np.abs(
                    valid_local_spectra
                    -
                    median3[
                        None,
                        :
                    ]
                ),
                axis=0
            )

            local_median_mad = float(
                np.median(
                    mad3
                )
            )

        else:

            median3 = np.full(
                EXPECTED_BANDS,
                np.nan
            )

            local_median_mad = np.nan


        center_spectra.append(
            center
        )

        median_spectra.append(
            median3
        )


        # ====================================================
        # SPECTRAL QC
        # ====================================================

        center_median_angle = spectral_angle_deg(
            center,
            median3
        )


        if np.all(
            np.isfinite(
                median3
            )
        ):

            red = float(
                median3[
                    red_i
                ]
            )

            nir = float(
                median3[
                    nir_i
                ]
            )

            swir1 = float(
                median3[
                    swir1_i
                ]
            )

            swir2 = float(
                median3[
                    swir2_i
                ]
            )


            ndvi_denom = (
                nir
                +
                red
            )


            if abs(
                ndvi_denom
            ) > 1e-12:

                ndvi = (
                    nir
                    -
                    red
                ) / (
                    ndvi_denom
                )

            else:

                ndvi = np.nan


            spectrum_median = float(
                np.median(
                    median3
                )
            )


            spectrum_p95 = float(
                np.percentile(
                    median3,
                    95
                )
            )


            spectrum_min = float(
                np.min(
                    median3
                )
            )


            spectrum_max = float(
                np.max(
                    median3
                )
            )


            negative_fraction = float(
                np.mean(
                    median3 < 0
                )
            )


        else:

            red = np.nan
            nir = np.nan
            swir1 = np.nan
            swir2 = np.nan
            ndvi = np.nan

            spectrum_median = np.nan
            spectrum_p95 = np.nan
            spectrum_min = np.nan
            spectrum_max = np.nan
            negative_fraction = np.nan


        # ====================================================
        # REVIEW FLAGS
        # ====================================================

        invalid_spectrum = (
            not center_valid
            or
            not np.all(
                np.isfinite(
                    median3
                )
            )
        )


        local_unstable = (
            local_valid_count
            <
            MIN_VALID_LOCAL_PIXELS
            or
            (
                np.isfinite(
                    center_median_angle
                )
                and
                center_median_angle
                >
                CENTER_MEDIAN_SAM_WARNING_DEG
            )
        )


        vegetation_like = (
            np.isfinite(
                ndvi
            )
            and
            ndvi
            >
            VEGETATION_NDVI
        )


        dark_water_shadow_like = (
            np.isfinite(
                nir
            )
            and
            np.isfinite(
                swir1
            )
            and
            nir
            <
            DARK_NIR_THRESHOLD
            and
            swir1
            <
            DARK_SWIR_THRESHOLD
        )


        very_dark = (
            np.isfinite(
                spectrum_median
            )
            and
            spectrum_median
            <
            VERY_DARK_MEDIAN_THRESHOLD
        )


        high_reflectance_review = (
            np.isfinite(
                spectrum_p95
            )
            and
            (
                spectrum_p95
                >
                HIGH_REFLECTANCE_P95
                or
                spectrum_max
                >
                HIGH_REFLECTANCE_MAX
            )
        )


        negative_reflectance_warning = (
            np.isfinite(
                negative_fraction
            )
            and
            negative_fraction
            >
            NEGATIVE_FRACTION_WARNING
        )


        rock_soil_review_priority = (

            not invalid_spectrum

            and

            not local_unstable

            and

            not vegetation_like

            and

            not dark_water_shadow_like

            and

            not very_dark

            and

            (
                np.isfinite(
                    ndvi
                )
                and
                ndvi
                <
                ROCK_SOIL_REVIEW_NDVI_MAX
            )
        )


        # ----------------------------------------------------
        # Human-readable priority label.
        #
        # No spectrum is discarded.
        # ----------------------------------------------------

        if invalid_spectrum:

            screening_class = (
                "INVALID_REVIEW"
            )

        elif local_unstable:

            screening_class = (
                "LOCAL_UNSTABLE_REVIEW"
            )

        elif vegetation_like:

            screening_class = (
                "VEGETATION_LIKE_REVIEW"
            )

        elif dark_water_shadow_like or very_dark:

            screening_class = (
                "DARK_WATER_SHADOW_REVIEW"
            )

        elif rock_soil_review_priority:

            screening_class = (
                "ROCK_SOIL_PRIORITY_REVIEW"
            )

        else:

            screening_class = (
                "MIXED_SPECTRAL_REVIEW"
            )


        qc_row = {
            "candidate_number":
                i + 1,

            "thinned_rank":
                row.get(
                    "thinned_rank",
                    i + 1
                ),

            "PPI_rank":
                row.get(
                    "PPI_rank",
                    np.nan
                ),

            "row":
                center_row,

            "col":
                center_col,

            "easting_m":
                row[
                    "easting_m"
                ],

            "northing_m":
                row[
                    "northing_m"
                ],

            "refined_PPI_count":
                row[
                    "refined_PPI_count"
                ],

            "coarse_PPI_votes":
                row.get(
                    "coarse_PPI_votes",
                    np.nan
                ),

            "local_valid_pixels":
                local_valid_count,

            "center_vs_3x3_SAM_deg":
                center_median_angle,

            "local_median_spectral_MAD":
                local_median_mad,

            "NDVI_approx":
                ndvi,

            "red_reflectance":
                red,

            "NIR_reflectance":
                nir,

            "SWIR1_reflectance":
                swir1,

            "SWIR2_reflectance":
                swir2,

            "spectrum_median_reflectance":
                spectrum_median,

            "spectrum_p95_reflectance":
                spectrum_p95,

            "spectrum_min_reflectance":
                spectrum_min,

            "spectrum_max_reflectance":
                spectrum_max,

            "negative_band_fraction":
                negative_fraction,

            "invalid_spectrum":
                invalid_spectrum,

            "local_unstable":
                local_unstable,

            "vegetation_like":
                vegetation_like,

            "dark_water_shadow_like":
                dark_water_shadow_like,

            "very_dark":
                very_dark,

            "high_reflectance_review":
                high_reflectance_review,

            "negative_reflectance_warning":
                negative_reflectance_warning,

            "rock_soil_review_priority":
                rock_soil_review_priority,

            "screening_class":
                screening_class,
        }


        candidate_qc_rows.append(
            qc_row
        )


        if (
            i == 0
            or
            (i + 1) % 20 == 0
            or
            i + 1
            ==
            candidate_count
        ):

            print(
                f"  Candidate "
                f"{i+1:3d}/"
                f"{candidate_count} extracted"
            )


center_spectra = np.asarray(
    center_spectra,
    dtype=np.float64
)


median_spectra = np.asarray(
    median_spectra,
    dtype=np.float64
)


candidate_qc = pd.DataFrame(
    candidate_qc_rows
)


print(
    f"\nExtraction completed in "
    f"{(time.time()-extraction_start)/60:.2f} min"
)


# ============================================================
# SAVE QC TABLE
# ============================================================

candidate_qc.to_csv(
    CANDIDATE_QC_CSV,
    index=False
)


# ============================================================
# SAVE LONG-FORM SPECTRA
# ============================================================

print("\nSaving reflectance spectra...")


center_long_rows = []

median_long_rows = []


for candidate_index in range(
    candidate_count
):

    candidate_number = (
        candidate_index
        +
        1
    )


    for band_index in range(
        EXPECTED_BANDS
    ):

        common = {
            "candidate_number":
                candidate_number,

            "science_band":
                band_index
                +
                1,

            "original_425_band":
                int(
                    original_bands[
                        band_index
                    ]
                ),

            "wavelength_nm":
                float(
                    wavelengths[
                        band_index
                    ]
                ),
        }


        center_long_rows.append({
            **common,

            "reflectance":
                float(
                    center_spectra[
                        candidate_index,
                        band_index
                    ]
                ),
        })


        median_long_rows.append({
            **common,

            "reflectance":
                float(
                    median_spectra[
                        candidate_index,
                        band_index
                    ]
                ),
        })


pd.DataFrame(
    center_long_rows
).to_csv(
    CENTER_LONG_CSV,
    index=False
)


pd.DataFrame(
    median_long_rows
).to_csv(
    MEDIAN_LONG_CSV,
    index=False
)


# ============================================================
# EXPLORATORY SPECTRAL-SHAPE CLUSTERING
#
# Use 3x3 median reflectance.
#
# L2 normalization makes clustering largely insensitive to
# brightness magnitude and closer to spectral-angle grouping.
# ============================================================

print("\n")
print("=" * 100)

print(
    "EXPLORATORY SPECTRAL-SHAPE CLUSTERING"
)

print("=" * 100)


cluster_valid = np.all(
    np.isfinite(
        median_spectra
    ),
    axis=1
)


cluster_indices = np.where(
    cluster_valid
)[0]


X = median_spectra[
    cluster_valid
].copy()


norms = np.linalg.norm(
    X,
    axis=1
)


good_norm = (
    norms > 0
)


X = X[
    good_norm
]


cluster_indices = (
    cluster_indices[
        good_norm
    ]
)


norms = norms[
    good_norm
]


X_normalized = (
    X
    /
    norms[
        :,
        None
    ]
)


n_cluster_samples = len(
    X_normalized
)


if n_cluster_samples < 10:

    raise RuntimeError(
        "Too few valid candidate spectra for clustering."
    )


max_k = min(
    MAX_CLUSTERS,
    n_cluster_samples
    -
    1
)


cluster_qc_rows = []


best_score = -np.inf

best_k = None

best_labels = None


for k in range(
    MIN_CLUSTERS,
    max_k + 1
):

    model = make_agglomerative(
        k
    )


    labels = model.fit_predict(
        X_normalized
    )


    score = silhouette_score(
        X_normalized,
        labels,
        metric="cosine"
    )


    cluster_qc_rows.append({
        "clusters":
            k,

        "cosine_silhouette_score":
            score,
    })


    print(
        f"  k={k:2d} "
        f"| silhouette="
        f"{score:.4f}"
    )


    if score > best_score:

        best_score = score

        best_k = k

        best_labels = labels.copy()


cluster_qc_df = pd.DataFrame(
    cluster_qc_rows
)


cluster_qc_df.to_csv(
    CLUSTER_SELECTION_CSV,
    index=False
)


print(
    f"\nSelected exploratory cluster count: "
    f"{best_k}"
)

print(
    f"Best cosine silhouette score: "
    f"{best_score:.4f}"
)


# ============================================================
# ASSIGN CLUSTERS
# ============================================================

cluster_labels_all = np.full(
    candidate_count,
    -1,
    dtype=int
)


cluster_labels_all[
    cluster_indices
] = (
    best_labels
    +
    1
)


candidate_qc[
    "spectral_cluster"
] = (
    cluster_labels_all
)


candidate_qc.to_csv(
    CANDIDATE_QC_CSV,
    index=False
)


cluster_assignment = candidate_qc[[
    "candidate_number",
    "thinned_rank",
    "PPI_rank",
    "refined_PPI_count",
    "row",
    "col",
    "easting_m",
    "northing_m",
    "screening_class",
    "rock_soil_review_priority",
    "spectral_cluster",
]].copy()


cluster_assignment.to_csv(
    CLUSTER_ASSIGNMENT_CSV,
    index=False
)


# ============================================================
# CLUSTER MEDIAN SPECTRA
# ============================================================

cluster_median_rows = []

cluster_medians = {}


for cluster in range(
    1,
    best_k + 1
):

    members = np.where(
        cluster_labels_all
        ==
        cluster
    )[0]


    cluster_spectra = (
        median_spectra[
            members
        ]
    )


    cluster_median = np.median(
        cluster_spectra,
        axis=0
    )


    cluster_medians[
        cluster
    ] = cluster_median


    for band_index in range(
        EXPECTED_BANDS
    ):

        cluster_median_rows.append({

            "cluster":
                cluster,

            "cluster_members":
                len(
                    members
                ),

            "science_band":
                band_index
                +
                1,

            "original_425_band":
                int(
                    original_bands[
                        band_index
                    ]
                ),

            "wavelength_nm":
                wavelengths[
                    band_index
                ],

            "median_reflectance":
                cluster_median[
                    band_index
                ],
        })


pd.DataFrame(
    cluster_median_rows
).to_csv(
    CLUSTER_MEDIAN_CSV,
    index=False
)


# ============================================================
# CLUSTER REPRESENTATIVES
#
# For every cluster report:
#   1) cosine medoid candidate
#   2) highest-PPI candidate
#
# These are REVIEW representatives only.
# ============================================================

representative_rows = []


for cluster in range(
    1,
    best_k + 1
):

    members = np.where(
        cluster_labels_all
        ==
        cluster
    )[0]


    member_spectra = (
        median_spectra[
            members
        ]
    )


    member_norm = (
        member_spectra
        /
        np.linalg.norm(
            member_spectra,
            axis=1
        )[
            :,
            None
        ]
    )


    cosine_similarity = (
        member_norm
        @
        member_norm.T
    )


    mean_similarity = np.mean(
        cosine_similarity,
        axis=1
    )


    medoid_local = int(
        np.argmax(
            mean_similarity
        )
    )


    medoid_index = int(
        members[
            medoid_local
        ]
    )


    member_ppi = (
        candidate_qc.iloc[
            members
        ][
            "refined_PPI_count"
        ]
        .to_numpy()
    )


    highest_ppi_index = int(
        members[
            np.argmax(
                member_ppi
            )
        ]
    )


    medoid_row = (
        candidate_qc.iloc[
            medoid_index
        ]
    )


    high_ppi_row = (
        candidate_qc.iloc[
            highest_ppi_index
        ]
    )


    representative_rows.append({

        "cluster":
            cluster,

        "cluster_members":
            len(
                members
            ),

        "medoid_candidate_number":
            medoid_row[
                "candidate_number"
            ],

        "medoid_PPI_count":
            medoid_row[
                "refined_PPI_count"
            ],

        "medoid_easting_m":
            medoid_row[
                "easting_m"
            ],

        "medoid_northing_m":
            medoid_row[
                "northing_m"
            ],

        "medoid_screening_class":
            medoid_row[
                "screening_class"
            ],

        "highest_PPI_candidate_number":
            high_ppi_row[
                "candidate_number"
            ],

        "highest_PPI_count":
            high_ppi_row[
                "refined_PPI_count"
            ],

        "highest_PPI_easting_m":
            high_ppi_row[
                "easting_m"
            ],

        "highest_PPI_northing_m":
            high_ppi_row[
                "northing_m"
            ],

        "highest_PPI_screening_class":
            high_ppi_row[
                "screening_class"
            ],
    })


representatives_df = pd.DataFrame(
    representative_rows
)


representatives_df.to_csv(
    CLUSTER_REPRESENTATIVES_CSV,
    index=False
)


# ============================================================
# PLOT ALL 3x3 MEDIAN CANDIDATE SPECTRA
# ============================================================

plt.figure(
    figsize=(
        15,
        8
    )
)


for i in range(
    candidate_count
):

    for xseg, yseg in spectral_segments(
        wavelengths,
        median_spectra[
            i
        ]
    ):

        plt.plot(
            xseg,
            yseg,
            alpha=0.12,
            linewidth=0.8
        )


plt.xlabel(
    "Wavelength (nm)"
)

plt.ylabel(
    "Reflectance"
)

plt.title(
    "AVIRIS-NG PPI Candidate Spectra\n"
    "3×3 Median Reflectance"
)

plt.grid(
    alpha=0.2
)

plt.tight_layout()

plt.savefig(
    ALL_MEDIAN_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# CLUSTER MEDIAN PLOT
# ============================================================

plt.figure(
    figsize=(
        15,
        8
    )
)


for cluster in range(
    1,
    best_k + 1
):

    spectrum = (
        cluster_medians[
            cluster
        ]
    )


    first_segment = True


    for xseg, yseg in spectral_segments(
        wavelengths,
        spectrum
    ):

        plt.plot(
            xseg,
            yseg,
            linewidth=1.7,
            label=(
                f"Cluster {cluster}"
                if first_segment
                else None
            )
        )

        first_segment = False


plt.xlabel(
    "Wavelength (nm)"
)

plt.ylabel(
    "Median reflectance"
)

plt.title(
    "AVIRIS-NG PPI Candidate Spectral Clusters\n"
    f"{best_k} exploratory spectral-shape groups"
)

plt.grid(
    alpha=0.2
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CLUSTER_PLOT,
    dpi=220,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# NDVI vs PPI
# ============================================================

plt.figure(
    figsize=(
        10,
        7
    )
)


plt.scatter(
    candidate_qc[
        "NDVI_approx"
    ],
    candidate_qc[
        "refined_PPI_count"
    ],
    s=22,
    alpha=0.7
)


plt.xlabel(
    "Approximate NDVI"
)

plt.ylabel(
    "Refined PPI count"
)

plt.title(
    "PPI Count vs Vegetation Indicator"
)

plt.yscale(
    "symlog",
    linthresh=1
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    NDVI_PPI_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# LOCAL STABILITY
# ============================================================

plt.figure(
    figsize=(
        10,
        7
    )
)


plt.scatter(
    candidate_qc[
        "refined_PPI_count"
    ],
    candidate_qc[
        "center_vs_3x3_SAM_deg"
    ],
    s=22,
    alpha=0.7
)


plt.axhline(
    CENTER_MEDIAN_SAM_WARNING_DEG,
    linestyle="--"
)


plt.xlabel(
    "Refined PPI count"
)

plt.ylabel(
    "Center vs 3×3 median spectral angle (degrees)"
)

plt.title(
    "PPI Candidate Local Spectral Stability"
)

plt.xscale(
    "symlog",
    linthresh=1
)

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    LOCAL_STABILITY_PLOT,
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# TOP 20 INDIVIDUAL SPECTRA
# ============================================================

top_order = (
    candidate_qc
    .sort_values(
        "refined_PPI_count",
        ascending=False
    )
    .index[
        :TOP_INDIVIDUAL_PLOTS
    ]
)


for plot_rank, index in enumerate(
    top_order,
    start=1
):

    row = candidate_qc.loc[
        index
    ]


    plt.figure(
        figsize=(
            13,
            7
        )
    )


    first = True

    for xseg, yseg in spectral_segments(
        wavelengths,
        center_spectra[
            index
        ]
    ):

        plt.plot(
            xseg,
            yseg,
            linewidth=1.0,
            alpha=0.65,
            label=(
                "Center pixel"
                if first
                else None
            )
        )

        first = False


    first = True

    for xseg, yseg in spectral_segments(
        wavelengths,
        median_spectra[
            index
        ]
    ):

        plt.plot(
            xseg,
            yseg,
            linewidth=2.0,
            label=(
                "3×3 median"
                if first
                else None
            )
        )

        first = False


    plt.xlabel(
        "Wavelength (nm)"
    )

    plt.ylabel(
        "Reflectance"
    )

    plt.title(
        f"PPI Candidate {int(row['candidate_number'])} "
        f"| PPI={int(row['refined_PPI_count'])}\n"
        f"{row['screening_class']} "
        f"| SAM={row['center_vs_3x3_SAM_deg']:.2f}°"
    )

    plt.grid(
        alpha=0.2
    )

    plt.legend()

    plt.tight_layout()


    output = (
        TOP_SPECTRA_DIR
        /
        (
            f"Rank_{plot_rank:02d}_"
            f"Candidate_{int(row['candidate_number']):03d}_"
            f"PPI_{int(row['refined_PPI_count'])}.png"
        )
    )


    plt.savefig(
        output,
        dpi=190,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# SUMMARY COUNTS
# ============================================================

screen_counts = (
    candidate_qc[
        "screening_class"
    ]
    .value_counts()
    .to_dict()
)


rock_priority_count = int(
    np.sum(
        candidate_qc[
            "rock_soil_review_priority"
        ]
    )
)


vegetation_count = int(
    np.sum(
        candidate_qc[
            "vegetation_like"
        ]
    )
)


dark_count = int(
    np.sum(
        candidate_qc[
            "dark_water_shadow_like"
        ]
        |
        candidate_qc[
            "very_dark"
        ]
    )
)


unstable_count = int(
    np.sum(
        candidate_qc[
            "local_unstable"
        ]
    )
)


high_reflectance_count = int(
    np.sum(
        candidate_qc[
            "high_reflectance_review"
        ]
    )
)


# ============================================================
# SUMMARY CSV
# ============================================================

summary = pd.DataFrame([{

    "input_PPI_candidates":
        candidate_count,

    "center_spectra_extracted":
        len(
            center_spectra
        ),

    "median_3x3_spectra_extracted":
        len(
            median_spectra
        ),

    "spectral_bands":
        EXPECTED_BANDS,

    "wavelength_min_nm":
        float(
            wavelengths.min()
        ),

    "wavelength_max_nm":
        float(
            wavelengths.max()
        ),

    "rock_soil_priority_review":
        rock_priority_count,

    "vegetation_like":
        vegetation_count,

    "dark_water_shadow_like":
        dark_count,

    "local_unstable":
        unstable_count,

    "high_reflectance_review":
        high_reflectance_count,

    "exploratory_cluster_count":
        best_k,

    "best_cosine_silhouette_score":
        best_score,

    "mineral_identification_performed":
        False,

    "candidate_deleted_automatically":
        False,

    "source_reflectance_cube_modified":
        False,

    "step12_status":
        "PASS",
}])


summary.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n")
print("=" * 100)

print(
    "STEP 12 COMPLETE — "
    "CANDIDATE REFLECTANCE SPECTRA EXTRACTED"
)

print("=" * 100)


print(
    f"\nCandidates extracted           : "
    f"{candidate_count}"
)

print(
    f"326-band center spectra        : "
    f"{len(center_spectra)}"
)

print(
    f"326-band 3x3 median spectra    : "
    f"{len(median_spectra)}"
)


print("\nScreening summary:")

print(
    f"  Rock/soil priority review    : "
    f"{rock_priority_count}"
)

print(
    f"  Vegetation-like              : "
    f"{vegetation_count}"
)

print(
    f"  Dark/water/shadow-like       : "
    f"{dark_count}"
)

print(
    f"  Local spectral instability   : "
    f"{unstable_count}"
)

print(
    f"  High-reflectance review      : "
    f"{high_reflectance_count}"
)


print(
    f"\nExploratory clusters          : "
    f"{best_k}"
)

print(
    f"Cosine silhouette score       : "
    f"{best_score:.4f}"
)


print("\nOutputs:")

print(
    f"\nCandidate QC:"
)

print(
    CANDIDATE_QC_CSV
)

print(
    f"\n3x3 median spectra:"
)

print(
    MEDIAN_LONG_CSV
)

print(
    f"\nCluster representatives:"
)

print(
    CLUSTER_REPRESENTATIVES_CSV
)

print(
    f"\nCluster median spectra:"
)

print(
    CLUSTER_MEDIAN_CSV
)

print(
    f"\nCluster plot:"
)

print(
    CLUSTER_PLOT
)

print(
    f"\nTop-20 spectral plots:"
)

print(
    TOP_SPECTRA_DIR
)

print(
    f"\nSummary:"
)

print(
    SUMMARY_CSV
)


print("\n")
print("=" * 100)

print("IMPORTANT")

print("=" * 100)

print(
    "\n1. PPI counts were used only to locate "
    "spectral extremes."
)

print(
    "2. Mineral interpretation uses the original "
    "326-band reflectance spectra."
)

print(
    "3. Center pixels and 3x3 medians were both retained."
)

print(
    "4. No candidate was automatically discarded."
)

print(
    "5. NDVI/water/dark/high-reflectance labels "
    "are heuristic review flags only."
)

print(
    "6. Spectral clusters are exploratory and "
    "are NOT mineral classes."
)

print(
    "7. The 326-band reflectance master was untouched."
)

print(
    "\nNEXT AFTER REVIEW:"
)

print(
    "Step 13 — select representative geological "
    "candidate spectra, continuum-remove them, "
    "locate absorption centers/depth/FWHM, "
    "and begin mineral spectral-library comparison."
)
