import os
import os
from pathlib import Path
import re
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks, peak_widths


# ============================================================================
# AVIRIS-NG STEP 13C
# SITTAMPUNDI LOCAL ASD LABORATORY SPECTRAL-LIBRARY VALIDATION
# ============================================================================
#
# PRIMARY REFERENCE:
#   Actual SAC laboratory ASD spectra.
#
# SECONDARY REFERENCE:
#   Selected USGS minerals:
#       Antigorite
#       Actinolite
#       Clinochlore
#
# IMPORTANT:
#   - No new PCA / MNF / PPI.
#   - No repeat of Step 13A.
#   - No repeat of 870-spectrum Step13B USGS search.
#   - AVIRIS targets = five Step12C coherent median endmembers.
#   - Same 326 scientific AVIRIS bands.
#   - Same atmospheric-gap-separated windows.
#   - Same continuum-removal philosophy.
#   - Mineral/PGE detection is NOT claimed.
# ============================================================================


ROOT = Path.cwd()


# ============================================================================
# INPUT PATHS
# ============================================================================

STEP12C_DIR = (
    ROOT
    / "06_PPI_Endmembers"
    / "Step12C_Robust_Clusters4_5"
)

STEP12C_MEDIAN_CSV = (
    STEP12C_DIR
    / "AVIRIS_Step12C_subcluster_median_spectra.csv"
)


STEP13A_QC_DIR = (
    ROOT
    / "07_Spectra"
    / "Step13A_QC_Reproducible_Features"
)

PRIORITY_FEATURE_CSV = (
    STEP13A_QC_DIR
    / "AVIRIS_Step13A_QC_Step13B_priority_features.csv"
)


STEP13B_V2A_DIR = (
    ROOT
    / "07_Spectra"
    / "Step13B_v2A_Sittampundi_Geology_Metadata_Validation"
)

STEP13B_V2A_FINAL_CSV = Path(os.environ.get("SAC_ASD_ROOT", "ASD_Spectra/SAC_Spectra"))


# ============================================================================
# SAC ASD LIBRARY PATH AUTO-DETECTION
# ============================================================================

HOME = Path.home()

ASD_LIBRARY_CANDIDATES = [

    Path(
        os.environ.get(
            "SAC_ASD_ROOT",
            str(ROOT / "ASD Spectra" / "SAC Spectra")
        )
    ),

    (
        ROOT
        / "ASD Spectra"
        / "SAC Spectra"
    ),

    (
        HOME
        / "Documents"
        / "AVIRIS NG L2 Data"
        / "ASD Spectra"
        / "SAC Spectra"
    ),
]


# ============================================================================
# OPTIONAL USER SAMPLE→ROCK FAMILY MAPPING
# ============================================================================
#
# If later needed, create:
#
# ASD Spectra/SAC Spectra/sample_family_map.csv
#
# with:
#
# pattern,family
# SCr,CHROMITITE
# AN,ANORTHOSITE_METAANORTHOSITE
#
# "pattern" is treated as a case-insensitive regular expression.
# ============================================================================


# ============================================================================
# EXISTING LOCAL USGS VERSION-7 LIBRARY
# ============================================================================

USGS_ROOT = (
    ROOT
    / "10_Reference_Libraries"
    / "USGS_Spectral_Library_v7"
)

USGS_ASD_DIR = (
    USGS_ROOT
    / "ASCIIdata_splib07b_cvASD"
)


TARGET_USGS_MINERALS = [

    "antigorite",

    "actinolite",

    "clinochlore",

    # also catch alternate / mistyped spelling
    "clinoclore",
]


# ============================================================================
# OUTPUT
# ============================================================================

OUTDIR = (
    ROOT
    / "07_Spectra"
    / "Step13C_Sittampundi_Local_ASD_Validation"
)

OUTDIR.mkdir(
    parents=True,
    exist_ok=True,
)


PLOT_DIR = (
    OUTDIR
    / "Top_Local_ASD_Plots"
)

PLOT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


MANIFEST_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_local_ASD_manifest.csv"
)

ALL_LOCAL_SCORES_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_all_local_ASD_scores.csv"
)

TOP_LOCAL_MATCHES_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_top20_local_ASD_matches.csv"
)

FAMILY_CONSENSUS_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_local_lithology_family_consensus.csv"
)

USGS_TARGETED_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_targeted_USGS_scores.csv"
)

FEATURE_MATCH_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_feature_match_details.csv"
)

FINAL_VALIDATION_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_final_validation.csv"
)

SUMMARY_CSV = (
    OUTDIR
    / "AVIRIS_Step13C_summary.csv"
)

SETTINGS_JSON = (
    OUTDIR
    / "AVIRIS_Step13C_settings.json"
)

RANKING_PNG = (
    OUTDIR
    / "AVIRIS_Step13C_local_ASD_family_ranking.png"
)


# ============================================================================
# SCIENTIFIC SETTINGS
# ============================================================================

EXPECTED_BANDS = 326


WINDOW_LIMITS = {

    "VNIR_NIR":
        (
            476.61,
            1328.09,
        ),

    "SWIR1":
        (
            1473.34,
            1788.88,
        ),

    "SWIR2":
        (
            1994.24,
            2445.02,
        ),
}


MATCH_TOLERANCE = {

    "VNIR_NIR":
        15.0,

    "SWIR1":
        20.0,

    "SWIR2":
        20.0,
}


SAVGOL_WINDOW = 7

SAVGOL_POLYORDER = 2


LIBRARY_MIN_DEPTH = 0.005

LIBRARY_MIN_PROMINENCE = 0.0025

LIBRARY_PEAK_DISTANCE = 2

EDGE_MARGIN_NM = 12.0


# ============================================================================
# FEATURE-MORPHOLOGY WEIGHTS
# ============================================================================

W_CENTER = 0.40

W_DEPTH = 0.15

W_FWHM = 0.20

W_AREA = 0.15

W_PROMINENCE = 0.10


# ============================================================================
# FINAL LOCAL-LIBRARY SCORE
# ============================================================================

W_FEATURE_PATTERN = 0.55

W_CR_SHAPE = 0.30

W_RAW_SHAPE = 0.15


# ============================================================================
# TARGET FEATURE CLASS WEIGHTS
# ============================================================================

CLASS_WEIGHT = {

    "CONSENSUS_HIGH":
        3.0,

    "CONSENSUS_SUPPORTED":
        1.5,

    "MEDIAN_STRONG":
        1.0,
}


# ============================================================================
# LOCAL FAMILY SCORE THRESHOLDS
# ============================================================================

STRONG_LOCAL_SCORE = 70.0

MODERATE_LOCAL_SCORE = 55.0

MIN_FAMILY_MARGIN = 5.0


BAD_VALUE_THRESHOLD = -1.0e20


# ============================================================================
# PATH DETECTION
# ============================================================================

def choose_asd_library():

    existing = []


    for path in ASD_LIBRARY_CANDIDATES:

        if path.exists():

            number_files = sum(

                1

                for p
                in path.rglob("*")

                if p.is_file()
            )


            existing.append(
                (
                    number_files,
                    path,
                )
            )


    if not existing:

        raise FileNotFoundError(

            "Could not locate SAC Spectra folder.\n\n"
            "Checked:\n"
            +
            "\n".join(
                str(
                    p
                )
                for p
                in ASD_LIBRARY_CANDIDATES
            )
        )


    existing.sort(
        reverse=True,
        key=lambda x:
            x[0],
    )


    return existing[0][1]


ASD_LIBRARY = choose_asd_library()


# ============================================================================
# HELPERS
# ============================================================================

def safe_float(
    value
):

    try:

        return float(
            value
        )

    except Exception:

        return np.nan


def cosine_similarity(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )


    valid = (

        np.isfinite(
            a
        )

        &

        np.isfinite(
            b
        )
    )


    if np.sum(
        valid
    ) < 20:

        return np.nan


    aa = a[
        valid
    ]

    bb = b[
        valid
    ]


    na = np.linalg.norm(
        aa
    )

    nb = np.linalg.norm(
        bb
    )


    if (
        na <= 0
        or
        nb <= 0
    ):

        return np.nan


    return float(

        np.clip(

            np.dot(
                aa,
                bb,
            )
            /
            (
                na
                *
                nb
            ),

            -1.0,
            1.0,
        )
    )


def spectral_angle_deg(
    a,
    b,
):

    similarity = cosine_similarity(
        a,
        b,
    )


    if not np.isfinite(
        similarity
    ):

        return np.nan


    return float(

        np.degrees(

            np.arccos(

                np.clip(
                    similarity,
                    -1.0,
                    1.0,
                )
            )
        )
    )


def ratio_similarity(
    a,
    b,
):

    a = safe_float(
        a
    )

    b = safe_float(
        b
    )


    if (
        not np.isfinite(
            a
        )
        or
        not np.isfinite(
            b
        )
        or
        a <= 0
        or
        b <= 0
    ):

        return 0.0


    return float(

        np.exp(

            -
            abs(
                np.log(
                    a
                    /
                    b
                )
            )
        )
    )


# ============================================================================
# FLEXIBLE ASD TEXT PARSER
# ============================================================================

FLOAT_RE = re.compile(

    r"[-+]?"
    r"(?:\d+\.\d*|\.\d+|\d+)"
    r"(?:[eE][-+]?\d+)?"
)


def parse_numeric_tokens(
    line
):

    return [

        float(
            value
        )

        for value
        in FLOAT_RE.findall(
            line
        )
    ]


def parse_asd_text_spectrum(
    path
):

    """
    Flexible parser for exported ASD text / SCO text files.

    Preferred:
        wavelength + reflectance numeric columns.

    Fallback:
        single-column 2151-point spectrum interpreted as 350–2500 nm @ 1 nm.

    No general wavelength guessing is used.
    """

    try:

        lines = path.read_text(

            encoding="utf-8",

            errors="ignore",
        ).splitlines()


    except Exception as error:

        return {

            "status":
                "READ_ERROR",

            "message":
                str(
                    error
                ),
        }


    pair_rows = []

    single_values = []


    for line in lines:

        numbers = parse_numeric_tokens(
            line
        )


        if len(
            numbers
        ) >= 2:

            x = numbers[
                0
            ]

            y = numbers[
                1
            ]


            if (
                300.0
                <=
                x
                <=
                2600.0
            ):

                pair_rows.append(
                    (
                        x,
                        y,
                    )
                )


        elif len(
            numbers
        ) == 1:

            single_values.append(
                numbers[
                    0
                ]
            )


    # ------------------------------------------------------------------------
    # Preferred wavelength / reflectance pairs
    # ------------------------------------------------------------------------

    if len(
        pair_rows
    ) >= 200:

        array = np.asarray(
            pair_rows,
            dtype=float,
        )


        wavelength = array[
            :,
            0
        ]

        reflectance = array[
            :,
            1
        ]


        order = np.argsort(
            wavelength
        )


        wavelength = wavelength[
            order
        ]

        reflectance = reflectance[
            order
        ]


        # Collapse duplicate wavelengths.

        frame = pd.DataFrame(
            {

                "wavelength":
                    wavelength,

                "reflectance":
                    reflectance,
            }
        )


        frame = (

            frame
            .groupby(
                "wavelength",
                as_index=False,
            )[
                "reflectance"
            ]
            .median()
            .sort_values(
                "wavelength"
            )
        )


        wavelength = frame[
            "wavelength"
        ].to_numpy(
            dtype=float
        )


        reflectance = frame[
            "reflectance"
        ].to_numpy(
            dtype=float
        )


        wavelength_source = (
            "EXPLICIT_WAVELENGTH_COLUMN"
        )


    # ------------------------------------------------------------------------
    # Known standard ASD 350–2500 nm, one-nm exported vector
    # ------------------------------------------------------------------------

    elif len(
        single_values
    ) == 2151:

        wavelength = np.arange(

            350.0,

            2501.0,

            1.0,
        )


        reflectance = np.asarray(

            single_values,

            dtype=float,
        )


        wavelength_source = (
            "INFERRED_STANDARD_ASD_350_2500_1NM"
        )


    else:

        return {

            "status":
                "UNSUPPORTED_TEXT_FORMAT",

            "message":
                (
                    f"pair_rows={len(pair_rows)}, "
                    f"single_values={len(single_values)}"
                ),
        }


    # ------------------------------------------------------------------------
    # Remove obvious invalid values
    # ------------------------------------------------------------------------

    valid = (

        np.isfinite(
            wavelength
        )

        &

        np.isfinite(
            reflectance
        )

        &

        (
            reflectance
            >
            BAD_VALUE_THRESHOLD
        )
    )


    wavelength = wavelength[
        valid
    ]

    reflectance = reflectance[
        valid
    ]


    if len(
        wavelength
    ) < 200:

        return {

            "status":
                "INSUFFICIENT_VALID_SAMPLES",

            "message":
                str(
                    len(
                        wavelength
                    )
                ),
        }


    # ------------------------------------------------------------------------
    # Common percent-reflectance export detection.
    # Only scale if values clearly behave as percentages.
    # ------------------------------------------------------------------------

    scale_note = (
        "UNCHANGED"
    )


    median_refl = float(
        np.nanmedian(
            reflectance
        )
    )


    p95_refl = float(
        np.nanpercentile(
            reflectance,
            95,
        )
    )


    if (
        median_refl > 2.0
        and
        p95_refl <= 120.0
    ):

        reflectance = (

            reflectance
            /
            100.0
        )


        scale_note = (
            "PERCENT_TO_FRACTION"
        )


    return {

        "status":
            "PASS",

        "wavelength_nm":
            wavelength,

        "reflectance":
            reflectance,

        "wavelength_source":
            wavelength_source,

        "scale_note":
            scale_note,

        "original_points":
            len(
                wavelength
            ),
    }


# ============================================================================
# SAMPLE PREPARATION + ROCK FAMILY
# ============================================================================

def sample_preparation(
    filename
):

    lower = filename.lower()


    if (
        lower.startswith(
            "l_"
        )
        or
        "_lump"
        in
        lower
    ):

        return "LUMP"


    if (
        lower.startswith(
            "p_"
        )
        or
        "_powder"
        in
        lower
    ):

        return "POWDER"


    return "UNKNOWN"


def load_family_override():

    mapping_file = (
        ASD_LIBRARY
        /
        "sample_family_map.csv"
    )


    if not mapping_file.exists():

        return []


    mapping = pd.read_csv(
        mapping_file
    )


    if not {

        "pattern",

        "family",

    }.issubset(
        mapping.columns
    ):

        raise RuntimeError(

            "sample_family_map.csv requires columns: "
            "pattern,family"
        )


    return mapping.to_dict(
        orient="records"
    )


FAMILY_OVERRIDE = load_family_override()


def infer_family(
    filename
):

    # ------------------------------------------------------------------------
    # User-supplied mapping has highest priority.
    # ------------------------------------------------------------------------

    for record in FAMILY_OVERRIDE:

        try:

            if re.search(

                str(
                    record[
                        "pattern"
                    ]
                ),

                filename,

                flags=re.IGNORECASE,
            ):

                return str(
                    record[
                        "family"
                    ]
                )


        except re.error:

            continue


    text = filename.lower()


    # Known SAC Chromitite naming already used by the user:
    # L_SCr1 / P_SCr1 etc.

    if (
        "chromit"
        in
        text
        or
        "_scr"
        in
        text
        or
        re.search(
            r"(?:^|[_-])scr\d*",
            text,
        )
    ):

        return "CHROMITITE"


    if (
        "metaanorth"
        in
        text
        or
        "meta_anorth"
        in
        text
        or
        "meta-anorth"
        in
        text
        or
        "anorthosite"
        in
        text
    ):

        return (
            "ANORTHOSITE_METAANORTHOSITE"
        )


    if (
        "metagabbro"
        in
        text
        or
        "meta_gabbro"
        in
        text
        or
        "meta-gabbro"
        in
        text
        or
        "gabbro"
        in
        text
    ):

        return (
            "GABBRO_METAGABBRO"
        )


    if (
        "pyroxenite"
        in
        text
        or
        "pyrox"
        in
        text
    ):

        return (
            "PYROXENITE_MAFIC_CUMULATE"
        )


    if (
        "gneiss"
        in
        text
        or
        "gness"
        in
        text
        or
        "migmat"
        in
        text
    ):

        return (
            "GNEISS_MIGMATITE"
        )


    if (
        "granite"
        in
        text
        or
        "pegmat"
        in
        text
    ):

        return (
            "GRANITE_PEGMATITE"
        )


    if (
        "amphibolite"
        in
        text
        or
        "hornblendite"
        in
        text
    ):

        return (
            "AMPHIBOLITE_HORNBLENDITE"
        )


    if (
        "charnock"
        in
        text
    ):

        return "CHARNOCKITE"


    return "UNCLASSIFIED"


# ============================================================================
# UPPER CONVEX HULL CONTINUUM
# ============================================================================

def upper_hull_indices(
    wavelength,
    reflectance,
):

    hull = []


    for i in range(
        len(
            wavelength
        )
    ):

        hull.append(
            i
        )


        while len(
            hull
        ) >= 3:

            i1 = hull[
                -3
            ]

            i2 = hull[
                -2
            ]

            i3 = hull[
                -1
            ]


            dx1 = (
                wavelength[
                    i2
                ]
                -
                wavelength[
                    i1
                ]
            )


            dx2 = (
                wavelength[
                    i3
                ]
                -
                wavelength[
                    i2
                ]
            )


            if (
                dx1 <= 0
                or
                dx2 <= 0
            ):

                break


            slope1 = (

                reflectance[
                    i2
                ]
                -
                reflectance[
                    i1
                ]

            ) / dx1


            slope2 = (

                reflectance[
                    i3
                ]
                -
                reflectance[
                    i2
                ]

            ) / dx2


            if slope2 >= slope1:

                hull.pop(
                    -2
                )


            else:

                break


    return np.asarray(
        hull,
        dtype=int,
    )


def continuum_remove_window(
    wavelength,
    reflectance,
):

    smooth = np.asarray(
        reflectance,
        dtype=float,
    ).copy()


    if len(
        smooth
    ) >= SAVGOL_WINDOW:

        smooth = savgol_filter(

            smooth,

            SAVGOL_WINDOW,

            SAVGOL_POLYORDER,

            mode="interp",
        )


    hull = upper_hull_indices(

        wavelength,

        smooth,
    )


    continuum = np.interp(

        wavelength,

        wavelength[
            hull
        ],

        smooth[
            hull
        ],
    )


    result = np.full_like(
        smooth,
        np.nan,
    )


    valid = (

        np.isfinite(
            continuum
        )

        &

        (
            continuum > 0
        )
    )


    result[
        valid
    ] = (

        smooth[
            valid
        ]

        /

        continuum[
            valid
        ]
    )


    return result


def full_continuum_removed(
    wavelength,
    reflectance,
):

    result = np.full(

        len(
            wavelength
        ),

        np.nan,

        dtype=float,
    )


    for window, (
        low,
        high,
    ) in WINDOW_LIMITS.items():

        mask = (

            (
                wavelength
                >=
                low
            )

            &

            (
                wavelength
                <=
                high
            )
        )


        if np.sum(
            mask
        ) < 5:

            continue


        values = reflectance[
            mask
        ]


        if not np.all(
            np.isfinite(
                values
            )
        ):

            continue


        result[
            mask
        ] = continuum_remove_window(

            wavelength[
                mask
            ],

            values,
        )


    return result


# ============================================================================
# RESAMPLING
# ============================================================================

def resample_spectrum(
    source_wavelength,
    source_reflectance,
    target_wavelength,
):

    valid = (

        np.isfinite(
            source_wavelength
        )

        &

        np.isfinite(
            source_reflectance
        )
    )


    sw = source_wavelength[
        valid
    ]

    sr = source_reflectance[
        valid
    ]


    if len(
        sw
    ) < 200:

        return None


    order = np.argsort(
        sw
    )


    sw = sw[
        order
    ]

    sr = sr[
        order
    ]


    result = np.interp(

        target_wavelength,

        sw,

        sr,

        left=np.nan,

        right=np.nan,
    )


    if np.mean(
        np.isfinite(
            result
        )
    ) < 0.95:

        return None


    return result


# ============================================================================
# DETAILED ABSORPTION FEATURE DETECTION
# ============================================================================

def detect_features(
    wavelength,
    continuum_removed,
):

    features = []

    feature_number = 0


    for window, (
        low,
        high,
    ) in WINDOW_LIMITS.items():

        mask = (

            (
                wavelength >= low
            )

            &

            (
                wavelength <= high
            )
        )


        wl = wavelength[
            mask
        ]

        cr = continuum_removed[
            mask
        ]


        valid = np.isfinite(
            cr
        )


        wl = wl[
            valid
        ]

        cr = cr[
            valid
        ]


        if len(
            wl
        ) < 10:

            continue


        depth = np.maximum(

            1.0
            -
            cr,

            0.0,
        )


        peaks, properties = find_peaks(

            depth,

            height=
                LIBRARY_MIN_DEPTH,

            prominence=
                LIBRARY_MIN_PROMINENCE,

            distance=
                LIBRARY_PEAK_DISTANCE,
        )


        if len(
            peaks
        ) == 0:

            continue


        (
            widths,
            heights,
            left_ips,
            right_ips,
        ) = peak_widths(

            depth,

            peaks,

            rel_height=0.5,
        )


        index_axis = np.arange(
            len(
                wl
            ),
            dtype=float,
        )


        for j, peak in enumerate(
            peaks
        ):

            center = float(
                wl[
                    peak
                ]
            )


            if (
                center
                -
                low
                <
                EDGE_MARGIN_NM
            ):

                continue


            if (
                high
                -
                center
                <
                EDGE_MARGIN_NM
            ):

                continue


            left_nm = float(

                np.interp(

                    left_ips[
                        j
                    ],

                    index_axis,

                    wl,
                )
            )


            right_nm = float(

                np.interp(

                    right_ips[
                        j
                    ],

                    index_axis,

                    wl,
                )
            )


            fwhm = (

                right_nm
                -
                left_nm
            )


            left_base = int(

                properties[
                    "left_bases"
                ][j]
            )


            right_base = int(

                properties[
                    "right_bases"
                ][j]
            )


            if right_base > left_base:

                xx = wl[
                    left_base:
                    right_base + 1
                ]


                yy = depth[
                    left_base:
                    right_base + 1
                ]


                if hasattr(
                    np,
                    "trapezoid",
                ):

                    area = float(

                        np.trapezoid(
                            yy,
                            xx,
                        )
                    )


                else:

                    area = float(

                        np.trapz(
                            yy,
                            xx,
                        )
                    )


            else:

                area = np.nan


            feature_number += 1


            features.append(
                {

                    "feature_id":
                        feature_number,

                    "spectral_window":
                        window,

                    "center_nm":
                        center,

                    "depth_percent":
                        float(
                            100.0
                            *
                            depth[
                                peak
                            ]
                        ),

                    "prominence":
                        float(
                            properties[
                                "prominences"
                            ][j]
                        ),

                    "FWHM_nm":
                        fwhm,

                    "band_area_nm":
                        area,
                }
            )


    return features


# ============================================================================
# FEATURE-MORPHOLOGY MATCH
# ============================================================================

def target_to_library_feature_score(
    target,
    library
):

    window = str(
        target[
            "spectral_window"
        ]
    )


    tolerance = MATCH_TOLERANCE[
        window
    ]


    center_target = safe_float(

        target[
            "consensus_center_nm"
        ]
    )


    difference = abs(

        center_target

        -

        safe_float(
            library[
                "center_nm"
            ]
        )
    )


    if difference > tolerance:

        return None


    sigma = (

        tolerance
        /
        2.0
    )


    center_score = float(

        np.exp(

            -0.5

            *

            (
                difference
                /
                sigma
            )
            **
            2
        )
    )


    depth_score = ratio_similarity(

        target[
            "median_band_depth_percent"
        ],

        library[
            "depth_percent"
        ],
    )


    fwhm_score = ratio_similarity(

        target[
            "median_FWHM_nm"
        ],

        library[
            "FWHM_nm"
        ],
    )


    area_score = ratio_similarity(

        target[
            "median_band_area_nm"
        ],

        library[
            "band_area_nm"
        ],
    )


    prominence_score = ratio_similarity(

        target[
            "median_prominence"
        ],

        library[
            "prominence"
        ],
    )


    total = (

        W_CENTER
        *
        center_score

        +

        W_DEPTH
        *
        depth_score

        +

        W_FWHM
        *
        fwhm_score

        +

        W_AREA
        *
        area_score

        +

        W_PROMINENCE
        *
        prominence_score
    )


    return {

        "total":
            float(
                total
            ),

        "center_difference_nm":
            difference,

        "center_score":
            center_score,

        "depth_score":
            depth_score,

        "FWHM_score":
            fwhm_score,

        "area_score":
            area_score,

        "prominence_score":
            prominence_score,
    }


def feature_pattern_score(
    target_features,
    library_features,
):

    target_features = target_features.copy()


    target_features[
        "_weight"
    ] = target_features[
        "consensus_class"
    ].map(
        CLASS_WEIGHT
    ).fillna(
        0.5
    )


    target_features = target_features.sort_values(

        [
            "_weight",
            "ranking_score",
        ],

        ascending=[
            False,
            False,
        ],
    )


    used_library_features = set()

    details = []


    weighted_total = 0.0

    total_weight = 0.0

    matched_weight = 0.0


    for _, target in target_features.iterrows():

        weight = float(
            target[
                "_weight"
            ]
        )


        total_weight += weight


        best = None

        best_library = None


        for library in library_features:

            library_id = int(
                library[
                    "feature_id"
                ]
            )


            if library_id in used_library_features:

                continue


            if (
                library[
                    "spectral_window"
                ]
                !=
                target[
                    "spectral_window"
                ]
            ):

                continue


            result = (
                target_to_library_feature_score(

                    target,

                    library,
                )
            )


            if result is None:

                continue


            if (
                best is None
                or
                result[
                    "total"
                ]
                >
                best[
                    "total"
                ]
            ):

                best = result

                best_library = library


        if best is None:

            details.append(
                {

                    "target_feature_id":
                        target[
                            "feature_id"
                        ],

                    "target_center_nm":
                        target[
                            "consensus_center_nm"
                        ],

                    "target_class":
                        target[
                            "consensus_class"
                        ],

                    "matched":
                        False,

                    "library_center_nm":
                        np.nan,

                    "center_difference_nm":
                        np.nan,

                    "feature_score":
                        0.0,
                }
            )


        else:

            library_id = int(
                best_library[
                    "feature_id"
                ]
            )


            used_library_features.add(
                library_id
            )


            weighted_total += (

                weight

                *
                best[
                    "total"
                ]
            )


            matched_weight += weight


            details.append(
                {

                    "target_feature_id":
                        target[
                            "feature_id"
                        ],

                    "target_center_nm":
                        target[
                            "consensus_center_nm"
                        ],

                    "target_class":
                        target[
                            "consensus_class"
                        ],

                    "matched":
                        True,

                    "library_center_nm":
                        best_library[
                            "center_nm"
                        ],

                    "library_depth_percent":
                        best_library[
                            "depth_percent"
                        ],

                    "library_FWHM_nm":
                        best_library[
                            "FWHM_nm"
                        ],

                    "library_band_area_nm":
                        best_library[
                            "band_area_nm"
                        ],

                    "center_difference_nm":
                        best[
                            "center_difference_nm"
                        ],

                    "center_score":
                        best[
                            "center_score"
                        ],

                    "depth_score":
                        best[
                            "depth_score"
                        ],

                    "FWHM_score":
                        best[
                            "FWHM_score"
                        ],

                    "area_score":
                        best[
                            "area_score"
                        ],

                    "prominence_score":
                        best[
                            "prominence_score"
                        ],

                    "feature_score":
                        best[
                            "total"
                        ],
                }
            )


    if total_weight > 0:

        score = (

            weighted_total
            /
            total_weight
        )


        coverage = (

            matched_weight
            /
            total_weight
        )


    else:

        score = 0.0

        coverage = 0.0


    return (

        float(
            score
        ),

        float(
            coverage
        ),

        details,
    )


# ============================================================================
# LOAD AVIRIS TARGET END MEMBERS
# ============================================================================

median_df = pd.read_csv(
    STEP12C_MEDIAN_CSV
)


priority_df = pd.read_csv(
    PRIORITY_FEATURE_CSV
)


v2a_df = pd.read_csv(
    STEP13B_V2A_FINAL_CSV
)


aviris_wavelength = (

    median_df[
        [
            "wavelength_nm",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "wavelength_nm"
    )[
        "wavelength_nm"
    ]
    .to_numpy(
        dtype=float
    )
)


if len(
    aviris_wavelength
) != EXPECTED_BANDS:

    raise RuntimeError(

        f"Expected {EXPECTED_BANDS} AVIRIS bands, "
        f"found {len(aviris_wavelength)}."
    )


TARGETS = {}


for group in sorted(
    median_df[
        "final_group"
    ].unique()
):

    subset = (

        median_df[
            median_df[
                "final_group"
            ]
            ==
            group
        ]
        .sort_values(
            "wavelength_nm"
        )
    )


    reflectance = subset[
        "median_reflectance"
    ].to_numpy(
        dtype=float
    )


    continuum_removed = (
        full_continuum_removed(

            aviris_wavelength,

            reflectance,
        )
    )


    TARGETS[
        group
    ] = {

        "reflectance":
            reflectance,

        "continuum_removed":
            continuum_removed,
    }


# ============================================================================
# SCAN LOCAL SAC ASD LIBRARY
# ============================================================================

print(
    "=" * 110
)

print(
    "STEP 13C — SITTAMPUNDI LOCAL ASD LABORATORY VALIDATION"
)

print(
    "=" * 110
)

print(
    f"\nSAC ASD folder:\n{ASD_LIBRARY}"
)


all_files = [

    path

    for path
    in ASD_LIBRARY.rglob("*")

    if path.is_file()
]


print(
    f"\nTotal files/items detected: "
    f"{len(all_files)}"
)


TEXT_EXTENSIONS = {

    ".txt",

    ".sco",

    ".csv",

    ".asc",

    ".dat",
}


candidate_files = [

    path

    for path
    in all_files

    if (
        path.suffix.lower()
        in
        TEXT_EXTENSIONS

        or

        ".sco."
        in
        path.name.lower()
    )
]


print(
    f"Potential text spectra: "
    f"{len(candidate_files)}"
)


manifest_records = []

LOCAL_CACHE = {}


for number, path in enumerate(
    candidate_files,
    start=1,
):

    if (
        number == 1
        or
        number % 50 == 0
    ):

        print(
            f"Parsing local ASD file "
            f"{number}/{len(candidate_files)}"
        )


    parsed = parse_asd_text_spectrum(
        path
    )


    record = {

        "file_name":
            path.name,

        "file_path":
            str(
                path
            ),

        "parse_status":
            parsed[
                "status"
            ],

        "inferred_family":
            infer_family(
                path.name
            ),

        "sample_preparation":
            sample_preparation(
                path.name
            ),
    }


    if parsed[
        "status"
    ] == "PASS":

        resampled = resample_spectrum(

            parsed[
                "wavelength_nm"
            ],

            parsed[
                "reflectance"
            ],

            aviris_wavelength,
        )


        if resampled is None:

            record[
                "parse_status"
            ] = (
                "FAIL_AVIRIS_RESAMPLING"
            )


        else:

            cr = full_continuum_removed(

                aviris_wavelength,

                resampled,
            )


            features = detect_features(

                aviris_wavelength,

                cr,
            )


            LOCAL_CACHE[
                str(
                    path
                )
            ] = {

                "reflectance":
                    resampled,

                "continuum_removed":
                    cr,

                "features":
                    features,
            }


            record.update(
                {

                    "wavelength_source":
                        parsed[
                            "wavelength_source"
                        ],

                    "scale_note":
                        parsed[
                            "scale_note"
                        ],

                    "original_points":
                        parsed[
                            "original_points"
                        ],

                    "detected_features":
                        len(
                            features
                        ),

                    "AVIRIS_resampled":
                        True,
                }
            )


    else:

        record[
            "message"
        ] = parsed.get(
            "message",
            "",
        )


    manifest_records.append(
        record
    )


manifest = pd.DataFrame(
    manifest_records
)


manifest.to_csv(
    MANIFEST_CSV,
    index=False,
)


usable_files = [

    path

    for path
    in candidate_files

    if str(
        path
    )
    in
    LOCAL_CACHE
]


print(
    f"\nUsable local ASD spectra: "
    f"{len(usable_files)}"
)


if len(
    usable_files
) == 0:

    raise RuntimeError(

        "No local ASD spectra were parsed successfully.\n"
        "Inspect AVIRIS_Step13C_local_ASD_manifest.csv."
    )


# ============================================================================
# COMPARE EVERY LOCAL ASD SPECTRUM AGAINST 5 AVIRIS GROUPS
# ============================================================================

score_records = []

feature_records = []


for group in sorted(
    TARGETS.keys()
):

    print(
        f"\nComparing group {group}..."
    )


    target = TARGETS[
        group
    ]


    target_features = priority_df[
        priority_df[
            "final_group"
        ]
        ==
        group
    ].copy()


    for path in usable_files:

        library = LOCAL_CACHE[
            str(
                path
            )
        ]


        (
            feature_score,
            feature_coverage,
            details,
        ) = feature_pattern_score(

            target_features,

            library[
                "features"
            ],
        )


        target_depth_vector = np.maximum(

            1.0
            -
            target[
                "continuum_removed"
            ],

            0.0,
        )


        library_depth_vector = np.maximum(

            1.0
            -
            library[
                "continuum_removed"
            ],

            0.0,
        )


        cr_similarity = cosine_similarity(

            target_depth_vector,

            library_depth_vector,
        )


        if not np.isfinite(
            cr_similarity
        ):

            cr_similarity = 0.0


        cr_similarity = float(

            np.clip(

                cr_similarity,

                0.0,

                1.0,
            )
        )


        sam = spectral_angle_deg(

            target[
                "reflectance"
            ],

            library[
                "reflectance"
            ],
        )


        if np.isfinite(
            sam
        ):

            raw_score = float(

                np.exp(

                    -
                    (
                        sam
                        /
                        15.0
                    )
                    **
                    2
                )
            )


        else:

            raw_score = 0.0


        final_score = 100.0 * (

            W_FEATURE_PATTERN
            *
            feature_score

            +

            W_CR_SHAPE
            *
            cr_similarity

            +

            W_RAW_SHAPE
            *
            raw_score
        )


        score_records.append(
            {

                "final_group":
                    group,

                "local_file":
                    path.name,

                "local_path":
                    str(
                        path
                    ),

                "local_family":
                    infer_family(
                        path.name
                    ),

                "sample_preparation":
                    sample_preparation(
                        path.name
                    ),

                "feature_pattern_score":
                    feature_score,

                "feature_coverage":
                    feature_coverage,

                "continuum_shape_score":
                    cr_similarity,

                "raw_SAM_deg":
                    sam,

                "raw_shape_score":
                    raw_score,

                "Step13C_score":
                    final_score,
            }
        )


        for detail in details:

            detail.update(
                {

                    "final_group":
                        group,

                    "local_file":
                        path.name,

                    "local_family":
                        infer_family(
                            path.name
                        ),
                }
            )


            feature_records.append(
                detail
            )


scores = pd.DataFrame(
    score_records
)


scores = scores.sort_values(

    [
        "final_group",
        "Step13C_score",
    ],

    ascending=[
        True,
        False,
    ],
).reset_index(
    drop=True
)


scores[
    "rank_within_group"
] = (

    scores
    .groupby(
        "final_group"
    )[
        "Step13C_score"
    ]
    .rank(
        method="first",
        ascending=False,
    )
    .astype(int)
)


scores.to_csv(
    ALL_LOCAL_SCORES_CSV,
    index=False,
)


pd.DataFrame(
    feature_records
).to_csv(
    FEATURE_MATCH_CSV,
    index=False,
)


top20 = scores[
    scores[
        "rank_within_group"
    ]
    <=
    20
].copy()


top20.to_csv(
    TOP_LOCAL_MATCHES_CSV,
    index=False,
)


# ============================================================================
# LOCAL ROCK-FAMILY CONSENSUS
# ============================================================================

family_records = []


recognized = scores[
    scores[
        "local_family"
    ]
    !=
    "UNCLASSIFIED"
].copy()


for (
    group,
    family,
), subset in recognized.groupby(
    [
        "final_group",
        "local_family",
    ]
):

    subset = subset.sort_values(

        "Step13C_score",

        ascending=False,
    )


    top3 = subset.head(
        3
    )


    best_score = float(
        subset[
            "Step13C_score"
        ].max()
    )


    median_top3 = float(
        top3[
            "Step13C_score"
        ].median()
    )


    number_samples = int(
        len(
            subset
        )
    )


    replicate_support = min(

        number_samples
        /
        3.0,

        1.0,
    )


    family_score = (

        0.65
        *
        best_score

        +

        0.25
        *
        median_top3

        +

        10.0
        *
        replicate_support
    )


    best = subset.iloc[
        0
    ]


    family_records.append(
        {

            "final_group":
                group,

            "local_family":
                family,

            "number_local_spectra":
                number_samples,

            "best_sample":
                best[
                    "local_file"
                ],

            "best_sample_preparation":
                best[
                    "sample_preparation"
                ],

            "best_sample_score":
                best_score,

            "median_top3_score":
                median_top3,

            "family_consensus_score":
                family_score,

            "best_feature_coverage":
                best[
                    "feature_coverage"
                ],

            "best_CR_shape_score":
                best[
                    "continuum_shape_score"
                ],

            "best_raw_SAM_deg":
                best[
                    "raw_SAM_deg"
                ],
        }
    )


family_df = pd.DataFrame(
    family_records
)


if len(
    family_df
) > 0:

    family_df = family_df.sort_values(

        [
            "final_group",
            "family_consensus_score",
        ],

        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


    family_df[
        "family_rank"
    ] = (

        family_df
        .groupby(
            "final_group"
        )[
            "family_consensus_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )


family_df.to_csv(
    FAMILY_CONSENSUS_CSV,
    index=False,
)


# ============================================================================
# TARGETED USGS: ANTIGORITE / ACTINOLITE / CLINOCHLORE
# ============================================================================

def read_usgs_numeric_file(
    path
):

    numbers = []


    try:

        lines = path.read_text(

            encoding="utf-8",

            errors="ignore",
        ).splitlines()


    except Exception:

        return np.asarray(
            [],
            dtype=float,
        )


    for line in lines:

        try:

            numbers.append(
                float(
                    line.strip()
                    .split()[0]
                )
            )

        except Exception:

            continue


    return np.asarray(
        numbers,
        dtype=float,
    )


def find_usgs_wavelength_file():

    candidates = [

        path

        for path
        in USGS_ASD_DIR.rglob(
            "*.txt"
        )

        if (
            "wavelength"
            in
            path.name.lower()
        )
        and
        (
            "asd"
            in
            path.name.lower()
        )
    ]


    for path in candidates:

        values = read_usgs_numeric_file(
            path
        )


        if len(
            values
        ) < 1000:

            continue


        wavelengths = values.copy()


        if np.nanmedian(
            wavelengths
        ) < 20:

            wavelengths *= 1000.0


        if (
            np.nanmin(
                wavelengths
            )
            <=
            400
            and
            np.nanmax(
                wavelengths
            )
            >=
            2450
        ):

            return (

                path,

                wavelengths,
            )


    return (
        None,
        None,
    )


usgs_score_records = []


if USGS_ASD_DIR.exists():

    (
        usgs_wavelength_file,
        usgs_wavelength,
    ) = find_usgs_wavelength_file()


    if usgs_wavelength is not None:

        targeted_files = []


        for path in USGS_ASD_DIR.rglob(
            "*.txt"
        ):

            lower = path.name.lower()


            if (
                "aref"
                not in
                lower
            ):

                continue


            if any(

                mineral
                in
                lower

                for mineral
                in
                TARGET_USGS_MINERALS
            ):

                targeted_files.append(
                    path
                )


        print(
            f"\nTargeted USGS spectra found: "
            f"{len(targeted_files)}"
        )


        for path in targeted_files:

            source_values = read_usgs_numeric_file(
                path
            )


            if len(
                source_values
            ) != len(
                usgs_wavelength
            ):

                continue


            resampled = resample_spectrum(

                usgs_wavelength,

                source_values,

                aviris_wavelength,
            )


            if resampled is None:

                continue


            cr = full_continuum_removed(

                aviris_wavelength,

                resampled,
            )


            library_features = detect_features(

                aviris_wavelength,

                cr,
            )


            lower_name = path.name.lower()


            if "antigorite" in lower_name:

                mineral = "ANTIGORITE"


            elif "actinolite" in lower_name:

                mineral = "ACTINOLITE"


            elif (
                "clinochlore"
                in
                lower_name
                or
                "clinoclore"
                in
                lower_name
            ):

                mineral = "CLINOCHLORE"


            else:

                mineral = "OTHER"


            for group in sorted(
                TARGETS.keys()
            ):

                target = TARGETS[
                    group
                ]


                target_features = priority_df[
                    priority_df[
                        "final_group"
                    ]
                    ==
                    group
                ]


                (
                    feature_score,
                    feature_coverage,
                    details,
                ) = feature_pattern_score(

                    target_features,

                    library_features,
                )


                target_depth = np.maximum(

                    1
                    -
                    target[
                        "continuum_removed"
                    ],

                    0,
                )


                library_depth = np.maximum(

                    1
                    -
                    cr,

                    0,
                )


                cr_similarity = cosine_similarity(

                    target_depth,

                    library_depth,
                )


                if not np.isfinite(
                    cr_similarity
                ):

                    cr_similarity = 0.0


                sam = spectral_angle_deg(

                    target[
                        "reflectance"
                    ],

                    resampled,
                )


                if np.isfinite(
                    sam
                ):

                    raw_score = float(

                        np.exp(

                            -
                            (
                                sam
                                /
                                15
                            )
                            **
                            2
                        )
                    )


                else:

                    raw_score = 0.0


                final_score = 100.0 * (

                    W_FEATURE_PATTERN
                    *
                    feature_score

                    +

                    W_CR_SHAPE
                    *
                    cr_similarity

                    +

                    W_RAW_SHAPE
                    *
                    raw_score
                )


                usgs_score_records.append(
                    {

                        "final_group":
                            group,

                        "USGS_mineral":
                            mineral,

                        "USGS_file":
                            path.name,

                        "feature_pattern_score":
                            feature_score,

                        "feature_coverage":
                            feature_coverage,

                        "continuum_shape_score":
                            cr_similarity,

                        "raw_SAM_deg":
                            sam,

                        "targeted_USGS_score":
                            final_score,
                    }
                )


usgs_df = pd.DataFrame(
    usgs_score_records
)


if len(
    usgs_df
) > 0:

    usgs_df = usgs_df.sort_values(

        [
            "final_group",
            "targeted_USGS_score",
        ],

        ascending=[
            True,
            False,
        ],
    )


    usgs_df[
        "rank_within_group"
    ] = (

        usgs_df
        .groupby(
            "final_group"
        )[
            "targeted_USGS_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )


usgs_df.to_csv(
    USGS_TARGETED_CSV,
    index=False,
)


# ============================================================================
# FINAL STEP 13C VALIDATION
# ============================================================================

final_records = []


for group in sorted(
    TARGETS.keys()
):

    group_samples = (

        scores[
            scores[
                "final_group"
            ]
            ==
            group
        ]
        .sort_values(
            "rank_within_group"
        )
    )


    best_sample = group_samples.iloc[
        0
    ]


    if len(
        family_df
    ) > 0:

        group_families = (

            family_df[
                family_df[
                    "final_group"
                ]
                ==
                group
            ]
            .sort_values(
                "family_rank"
            )
        )


    else:

        group_families = pd.DataFrame()


    if len(
        group_families
    ) > 0:

        top_family = group_families.iloc[
            0
        ]


        family_name = str(
            top_family[
                "local_family"
            ]
        )


        family_score = float(
            top_family[
                "family_consensus_score"
            ]
        )


        if len(
            group_families
        ) > 1:

            second_family = group_families.iloc[
                1
            ]


            second_family_name = str(
                second_family[
                    "local_family"
                ]
            )


            family_margin = (

                family_score

                -

                float(
                    second_family[
                        "family_consensus_score"
                    ]
                )
            )


        else:

            second_family_name = ""

            family_margin = np.nan


    else:

        family_name = (
            "UNCLASSIFIED"
        )

        family_score = np.nan

        second_family_name = ""

        family_margin = np.nan


    # ------------------------------------------------------------------------
    # Family validation
    # ------------------------------------------------------------------------

    if (
        np.isfinite(
            family_score
        )
        and
        family_score
        >=
        STRONG_LOCAL_SCORE
        and
        (
            not np.isfinite(
                family_margin
            )
            or
            family_margin
            >=
            MIN_FAMILY_MARGIN
        )
    ):

        local_validation = (
            "SUPPORTED_BY_LOCAL_ASD_LIBRARY"
        )


    elif (
        np.isfinite(
            family_score
        )
        and
        family_score
        >=
        MODERATE_LOCAL_SCORE
    ):

        local_validation = (
            "POSSIBLE_LOCAL_ASD_ANALOG"
        )


    else:

        local_validation = (
            "LOCAL_ASD_MATCH_UNRESOLVED"
        )


    # ------------------------------------------------------------------------
    # Chromitite interpretation
    # ------------------------------------------------------------------------

    if (
        family_name
        ==
        "CHROMITITE"

        and

        local_validation
        in
        [
            "SUPPORTED_BY_LOCAL_ASD_LIBRARY",
            "POSSIBLE_LOCAL_ASD_ANALOG",
        ]
    ):

        chromitite_status = (
            "LOCAL_CHROMITITE_LIKE_SPECTRAL_ANALOG"
        )


    elif (
        family_name
        in
        [
            "ANORTHOSITE_METAANORTHOSITE",
            "GABBRO_METAGABBRO",
            "PYROXENITE_MAFIC_CUMULATE",
        ]

        and

        local_validation
        in
        [
            "SUPPORTED_BY_LOCAL_ASD_LIBRARY",
            "POSSIBLE_LOCAL_ASD_ANALOG",
        ]
    ):

        chromitite_status = (
            "POSSIBLE_CHROMITITE_HOST_LITHOLOGY_ASSOCIATION"
        )


    else:

        chromitite_status = (
            "NO_CHROMITITE_ASSIGNMENT"
        )


    # ------------------------------------------------------------------------
    # Targeted USGS support
    # ------------------------------------------------------------------------

    if len(
        usgs_df
    ) > 0:

        group_usgs = (

            usgs_df[
                usgs_df[
                    "final_group"
                ]
                ==
                group
            ]
            .sort_values(
                "targeted_USGS_score",
                ascending=False,
            )
        )


    else:

        group_usgs = pd.DataFrame()


    if len(
        group_usgs
    ) > 0:

        best_usgs = group_usgs.iloc[
            0
        ]


        usgs_mineral = best_usgs[
            "USGS_mineral"
        ]


        usgs_score = best_usgs[
            "targeted_USGS_score"
        ]


    else:

        usgs_mineral = ""

        usgs_score = np.nan


    v2a_row = v2a_df[
        v2a_df[
            "final_group"
        ]
        ==
        group
    ]


    previous_v2a_label = (

        v2a_row.iloc[
            0
        ][
            "final_endmember_label"
        ]

        if len(
            v2a_row
        )

        else
        ""
    )


    final_records.append(
        {

            "final_group":
                group,

            "Step13B_v2A_previous_label":
                previous_v2a_label,

            "best_local_ASD_sample":
                best_sample[
                    "local_file"
                ],

            "best_local_ASD_sample_family":
                best_sample[
                    "local_family"
                ],

            "best_local_ASD_preparation":
                best_sample[
                    "sample_preparation"
                ],

            "best_local_ASD_score":
                best_sample[
                    "Step13C_score"
                ],

            "best_local_ASD_feature_coverage":
                best_sample[
                    "feature_coverage"
                ],

            "best_local_ASD_CR_similarity":
                best_sample[
                    "continuum_shape_score"
                ],

            "best_local_ASD_raw_SAM_deg":
                best_sample[
                    "raw_SAM_deg"
                ],

            "best_family":
                family_name,

            "best_family_consensus_score":
                family_score,

            "second_family":
                second_family_name,

            "family_margin":
                family_margin,

            "local_ASD_validation":
                local_validation,

            "chromitite_interpretation":
                chromitite_status,

            "best_targeted_USGS_mineral":
                usgs_mineral,

            "best_targeted_USGS_score":
                usgs_score,

            "mineral_identification":
                "NOT_FINAL",

            "chromite_direct_detection":
                False,

            "PGE_direct_detection":
                False,

            "mapping_ready":
                False,
        }
    )


final_df = pd.DataFrame(
    final_records
)


final_df.to_csv(
    FINAL_VALIDATION_CSV,
    index=False,
)


# ============================================================================
# TOP LOCAL ASD PLOTS
# ============================================================================

for group in sorted(
    TARGETS.keys()
):

    target = TARGETS[
        group
    ]


    subset = (

        scores[
            scores[
                "final_group"
            ]
            ==
            group
        ]
        .sort_values(
            "rank_within_group"
        )
        .head(
            5
        )
    )


    plt.figure(
        figsize=(
            15,
            8,
        )
    )


    for window, (
        low,
        high,
    ) in WINDOW_LIMITS.items():

        mask = (

            (
                aviris_wavelength
                >=
                low
            )

            &

            (
                aviris_wavelength
                <=
                high
            )
        )


        plt.plot(

            aviris_wavelength[
                mask
            ],

            target[
                "continuum_removed"
            ][
                mask
            ],

            linewidth=2.3,

            label=(
                f"{group} AVIRIS median"
                if
                window
                ==
                "VNIR_NIR"
                else None
            ),
        )


    for row in subset.itertuples():

        library = LOCAL_CACHE[
            row.local_path
        ]


        for window, (
            low,
            high,
        ) in WINDOW_LIMITS.items():

            mask = (

                (
                    aviris_wavelength
                    >=
                    low
                )

                &

                (
                    aviris_wavelength
                    <=
                    high
                )
            )


            plt.plot(

                aviris_wavelength[
                    mask
                ],

                library[
                    "continuum_removed"
                ][
                    mask
                ],

                linewidth=1.0,

                alpha=0.75,

                label=(
                    (
                        f"#{row.rank_within_group} "
                        f"{row.local_file} "
                        f"({row.Step13C_score:.1f})"
                    )
                    if
                    window
                    ==
                    "VNIR_NIR"
                    else None
                ),
            )


    plt.axhline(

        1.0,

        linestyle="--",

        linewidth=0.8,
    )


    plt.xlabel(
        "Wavelength (nm)"
    )


    plt.ylabel(
        "Continuum-removed reflectance"
    )


    plt.title(
        f"Step 13C — {group} vs Top 5 SAC Laboratory ASD Spectra"
    )


    plt.grid(
        alpha=0.20
    )


    plt.legend(
        fontsize=8,
    )


    plt.tight_layout()


    plt.savefig(

        PLOT_DIR
        /
        f"{group.replace('.', '_')}_Top5_Local_ASD.png",

        dpi=220,

        bbox_inches="tight",
    )


    plt.close()


# ============================================================================
# FAMILY RANKING PLOT
# ============================================================================

if len(
    family_df
) > 0:

    plot_df = family_df[
        family_df[
            "family_rank"
        ]
        <=
        4
    ].copy()


    plot_df[
        "label"
    ] = (

        plot_df[
            "final_group"
        ]

        +

        " | "

        +

        plot_df[
            "local_family"
        ]
    )


    plot_df = plot_df.sort_values(

        "family_consensus_score",

        ascending=True,
    )


    plt.figure(
        figsize=(
            14,
            max(
                8,
                0.40
                *
                len(
                    plot_df
                ),
            ),
        )
    )


    plt.barh(

        np.arange(
            len(
                plot_df
            )
        ),

        plot_df[
            "family_consensus_score"
        ],
    )


    plt.yticks(

        np.arange(
            len(
                plot_df
            )
        ),

        plot_df[
            "label"
        ],
    )


    plt.xlabel(
        "Local SAC ASD family-consensus score"
    )


    plt.title(
        "AVIRIS-NG Step 13C — Sittampundi Laboratory Rock-Family Validation"
    )


    plt.grid(
        axis="x",
        alpha=0.20,
    )


    plt.tight_layout()


    plt.savefig(

        RANKING_PNG,

        dpi=220,

        bbox_inches="tight",
    )


    plt.close()


# ============================================================================
# SUMMARY
# ============================================================================

recognized_family_spectra = int(

    np.sum(

        manifest[
            "inferred_family"
        ]
        !=
        "UNCLASSIFIED"
    )
)


supported_groups = int(

    np.sum(

        final_df[
            "local_ASD_validation"
        ]
        ==
        "SUPPORTED_BY_LOCAL_ASD_LIBRARY"
    )
)


possible_groups = int(

    np.sum(

        final_df[
            "local_ASD_validation"
        ]
        ==
        "POSSIBLE_LOCAL_ASD_ANALOG"
    )
)


unresolved_groups = int(

    np.sum(

        final_df[
            "local_ASD_validation"
        ]
        ==
        "LOCAL_ASD_MATCH_UNRESOLVED"
    )
)


summary = pd.DataFrame(
    [
        {

            "SAC_ASD_library_folder":
                str(
                    ASD_LIBRARY
                ),

            "total_folder_items":
                len(
                    all_files
                ),

            "candidate_text_spectra":
                len(
                    candidate_files
                ),

            "usable_local_ASD_spectra":
                len(
                    usable_files
                ),

            "recognized_family_spectra":
                recognized_family_spectra,

            "AVIRIS_groups":
                len(
                    TARGETS
                ),

            "SUPPORTED_BY_LOCAL_ASD_LIBRARY":
                supported_groups,

            "POSSIBLE_LOCAL_ASD_ANALOG":
                possible_groups,

            "LOCAL_ASD_MATCH_UNRESOLVED":
                unresolved_groups,

            "targeted_USGS_Antigorite":
                True,

            "targeted_USGS_Actinolite":
                True,

            "targeted_USGS_Clinochlore":
                True,

            "generic_870_USGS_search_repeated":
                False,

            "chromite_direct_detection":
                False,

            "PGE_direct_detection":
                False,

            "mineral_mapping_performed":
                False,

            "step13c_status":
                "PASS",
        }
    ]
)


summary.to_csv(
    SUMMARY_CSV,
    index=False,
)


# ============================================================================
# SETTINGS JSON
# ============================================================================

settings = {

    "primary_reference":
        "actual Sittampundi SAC ASD laboratory spectra",

    "secondary_USGS_targets":
        [
            "Antigorite",
            "Actinolite",
            "Clinochlore",
        ],

    "AVIRIS_bands":
        EXPECTED_BANDS,

    "spectral_windows":
        WINDOW_LIMITS,

    "feature_score_weights":
        {

            "center":
                W_CENTER,

            "depth":
                W_DEPTH,

            "FWHM":
                W_FWHM,

            "area":
                W_AREA,

            "prominence":
                W_PROMINENCE,
        },

    "combined_score_weights":
        {

            "feature_pattern":
                W_FEATURE_PATTERN,

            "continuum_shape":
                W_CR_SHAPE,

            "raw_shape":
                W_RAW_SHAPE,
        },

    "direct_chromite_detection":
        False,

    "direct_PGE_detection":
        False,
}


with SETTINGS_JSON.open(

    "w",

    encoding="utf-8",

) as file:

    json.dump(
        settings,
        file,
        indent=2,
    )


# ============================================================================
# CONSOLE REPORT
# ============================================================================

print(
    "\n"
    +
    "=" * 110
)

print(
    "STEP 13C COMPLETE"
)

print(
    "=" * 110
)


print(
    f"\nTotal SAC folder items              : "
    f"{len(all_files)}"
)

print(
    f"Potential text spectra              : "
    f"{len(candidate_files)}"
)

print(
    f"Usable local ASD spectra            : "
    f"{len(usable_files)}"
)

print(
    f"Recognized lithology-family spectra : "
    f"{recognized_family_spectra}"
)


print(
    "\nFINAL LOCAL ASD VALIDATION:"
)


print(

    final_df[
        [

            "final_group",

            "best_local_ASD_sample",

            "best_local_ASD_sample_family",

            "best_local_ASD_score",

            "best_family",

            "best_family_consensus_score",

            "local_ASD_validation",

            "chromitite_interpretation",

            "best_targeted_USGS_mineral",

            "best_targeted_USGS_score",
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nIMPORTANT:"
)

print(
    "Local ASD similarity does not by itself constitute field-confirmed lithology."
)

print(
    "Chromitite-like or host-lithology similarity is not direct chromite/PGE detection."
)

print(
    "Mineral mapping remains disabled until Step13C results are reviewed."
)


print(
    "\nPrimary outputs:"
)

print(
    SUMMARY_CSV
)

print(
    FINAL_VALIDATION_CSV
)

print(
    TOP_LOCAL_MATCHES_CSV
)

print(
    FAMILY_CONSENSUS_CSV
)

print(
    USGS_TARGETED_CSV
)

print(
    RANKING_PNG
)

