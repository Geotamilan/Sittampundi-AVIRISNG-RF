# -*- coding: utf-8 -*-
import os

from pathlib import Path
import re
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

STEP15A = ROOT / (
    "09_Mineral_Mapping/"
    "Step15A_13_Lithology_AVIRIS_Endmembers"
)

ENDMEMBER_326 = STEP15A / (
    "AVIRIS_Step15A_13_Endmembers_326ScientificBand_Wide.csv"
)

ENDMEMBER_425 = STEP15A / (
    "AVIRIS_Step15A_13_Endmembers_425Band_Wide.csv"
)

LOCATION_CSV = STEP15A / (
    "AVIRIS_Step15A_13_Lithology_Endmember_Locations.csv"
)

AVIRIS_HDR = ROOT / (
    "03_Mosaic/Final_425Band_Mosaic/"
    "AVIRIS_NG_mosaic_master_425band_4m.hdr"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step15B_13_Endmember_Spectral_Library"
)

OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# SCIENTIFIC GOOD BANDS
# ============================================================

GOOD_RANGES = [
    (21, 191),
    (220, 283),
    (324, 414),
]

GOOD_BANDS = []

for lo, hi in GOOD_RANGES:
    GOOD_BANDS.extend(range(lo, hi + 1))

# ============================================================
# HELPERS
# ============================================================

def parse_wavelengths(hdr):

    text = hdr.read_text(errors="ignore")

    m = re.search(
        r"wavelength\s*=\s*\{(.*?)\}",
        text,
        flags=re.I | re.S
    )

    if not m:
        raise RuntimeError(
            "Could not parse AVIRIS wavelengths."
        )

    vals = []

    for item in m.group(1).replace("\n", " ").split(","):
        try:
            vals.append(float(item.strip()))
        except Exception:
            pass

    return np.asarray(vals, dtype=float)


def clean_name(name):

    return (
        str(name)
        .replace("_", " ")
        .title()
    )


def continuum_safe_normalize(x):

    x = np.asarray(x, dtype=float)

    good = np.isfinite(x)

    out = np.full_like(
        x,
        np.nan,
        dtype=float
    )

    if good.sum() < 10:
        return out

    lo = np.nanmin(x[good])
    hi = np.nanmax(x[good])

    if hi <= lo:
        return out

    out[good] = (
        x[good] - lo
    ) / (
        hi - lo
    )

    return out


# ============================================================
# INPUT CHECK
# ============================================================

print("=" * 90)
print("STEP15B - BUILD 13-LITHOLOGY AVIRIS-NG SPECTRAL LIBRARY")
print("=" * 90)

for p in [
    ENDMEMBER_326,
    ENDMEMBER_425,
    LOCATION_CSV,
    AVIRIS_HDR
]:
    if not p.exists():
        raise FileNotFoundError(p)

# ============================================================
# LOAD
# ============================================================

df326 = pd.read_csv(
    ENDMEMBER_326
)

df425 = pd.read_csv(
    ENDMEMBER_425
)

locations = pd.read_csv(
    LOCATION_CSV
)

wl425 = parse_wavelengths(
    AVIRIS_HDR
)

if len(wl425) != 425:
    raise RuntimeError(
        f"Expected 425 wavelengths, found {len(wl425)}"
    )

good_index = np.array(
    [b - 1 for b in GOOD_BANDS],
    dtype=int
)

wl326 = wl425[
    good_index
]

print("\nEndmembers found:", len(df326))
print("Scientific wavelengths:", len(wl326))

if len(df326) != 13:
    print(
        "WARNING: Expected 13 endmembers."
    )

# ============================================================
# EXTRACT 326-BAND MATRIX
# ============================================================

band_cols326 = [
    f"B{b:03d}"
    for b in GOOD_BANDS
]

missing = [
    c for c in band_cols326
    if c not in df326.columns
]

if missing:
    raise RuntimeError(
        f"Missing spectral columns: {missing[:10]}"
    )

spectra = (
    df326[band_cols326]
    .apply(
        pd.to_numeric,
        errors="coerce"
    )
    .to_numpy(
        dtype=np.float32
    )
)

names = (
    df326[
        "lithology_family"
    ]
    .astype(str)
    .tolist()
)

codes = (
    df326[
        "subtype_code"
    ]
    .astype(int)
    .tolist()
)

# ============================================================
# LONG-FORM SPECTRAL TABLE
# ============================================================

long_rows = []

for i, name in enumerate(names):

    for j, wl in enumerate(wl326):

        long_rows.append({
            "subtype_code":
                codes[i],

            "lithology_family":
                name,

            "scientific_band_index_1based":
                j + 1,

            "original_band_1based":
                GOOD_BANDS[j],

            "wavelength_nm":
                wl,

            "reflectance":
                spectra[i, j]
        })

long_df = pd.DataFrame(
    long_rows
)

LONG_CSV = OUT / (
    "AVIRIS_Step15B_13_Endmember_Spectra_Long.csv"
)

long_df.to_csv(
    LONG_CSV,
    index=False
)

# ============================================================
# COMBINED RAW REFLECTANCE PLOT
# ============================================================

fig = plt.figure(
    figsize=(16, 10)
)

ax = fig.add_subplot(111)

for i, name in enumerate(names):

    ax.plot(
        wl326,
        spectra[i],
        linewidth=1.5,
        label=f"{codes[i]} - {clean_name(name)}"
    )

ax.set_xlabel(
    "Wavelength (nm)",
    fontsize=13
)

ax.set_ylabel(
    "AVIRIS-NG Reflectance",
    fontsize=13
)

ax.set_title(
    "AVIRIS-NG Image-Derived Spectral Endmembers – "
    "Sittampundi Anorthosite Complex",
    fontsize=14
)

ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=8,
    ncol=2,
    loc="best"
)

fig.tight_layout()

RAW_PNG = OUT / (
    "AVIRIS_Step15B_13_Endmembers_Raw_Reflectance.png"
)

RAW_PDF = OUT / (
    "AVIRIS_Step15B_13_Endmembers_Raw_Reflectance.pdf"
)

fig.savefig(
    RAW_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    RAW_PDF,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# NORMALIZED SPECTRAL SHAPE PLOT
# ============================================================

normalized = np.vstack([
    continuum_safe_normalize(
        row
    )
    for row in spectra
])

fig = plt.figure(
    figsize=(16, 10)
)

ax = fig.add_subplot(111)

for i, name in enumerate(names):

    ax.plot(
        wl326,
        normalized[i],
        linewidth=1.5,
        label=f"{codes[i]} - {clean_name(name)}"
    )

ax.set_xlabel(
    "Wavelength (nm)",
    fontsize=13
)

ax.set_ylabel(
    "Normalized Spectral Shape (0–1)",
    fontsize=13
)

ax.set_title(
    "Normalized AVIRIS-NG Spectral Shapes – "
    "13 Reference-Guided Lithology Endmembers",
    fontsize=14
)

ax.grid(
    alpha=0.25
)

ax.legend(
    fontsize=8,
    ncol=2,
    loc="best"
)

fig.tight_layout()

NORM_PNG = OUT / (
    "AVIRIS_Step15B_13_Endmembers_Normalized.png"
)

NORM_PDF = OUT / (
    "AVIRIS_Step15B_13_Endmembers_Normalized.pdf"
)

fig.savefig(
    NORM_PNG,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    NORM_PDF,
    bbox_inches="tight"
)

plt.close(fig)

# ============================================================
# INDIVIDUAL END-MEMBER PLOTS
# ============================================================

INDIVIDUAL_DIR = OUT / (
    "Individual_Endmember_Plots"
)

INDIVIDUAL_DIR.mkdir(
    exist_ok=True
)

for i, name in enumerate(names):

    fig = plt.figure(
        figsize=(11, 6)
    )

    ax = fig.add_subplot(111)

    ax.plot(
        wl326,
        spectra[i],
        linewidth=1.7
    )

    ax.set_xlabel(
        "Wavelength (nm)"
    )

    ax.set_ylabel(
        "Reflectance"
    )

    ax.set_title(
        f"{codes[i]} - {clean_name(name)}"
    )

    ax.grid(
        alpha=0.25
    )

    fig.tight_layout()

    p = INDIVIDUAL_DIR / (
        f"{codes[i]:02d}_{name}.png"
    )

    fig.savefig(
        p,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

# ============================================================
# ENVI SPECTRAL LIBRARY
#
# ENVI spectral library:
# samples = wavelengths
# lines   = spectra
# bands   = 1
# ============================================================

SLI = OUT / (
    "AVIRIS_Step15B_13_Lithology_Endmembers_326Band.sli"
)

HDR = OUT / (
    "AVIRIS_Step15B_13_Lithology_Endmembers_326Band.hdr"
)

library_data = spectra.astype(
    "<f4"
)

library_data.tofile(
    SLI
)

spectra_names = ",\n".join(
    [
        f"  {code:02d}_{name}"
        for code, name
        in zip(codes, names)
    ]
)

wavelength_text = ",\n".join(
    [
        f"  {x:.6f}"
        for x in wl326
    ]
)

hdr_text = f"""ENVI
description = {{
AVIRIS-NG 13-lithology image-derived reference spectral library
Sittampundi Anorthosite Complex
Scientific 326-band subset.
}}
samples = {len(wl326)}
lines = {len(names)}
bands = 1
header offset = 0
file type = ENVI Spectral Library
data type = 4
interleave = bsq
sensor type = AVIRIS-NG
byte order = 0
wavelength units = Nanometers
spectra names = {{
{spectra_names}
}}
wavelength = {{
{wavelength_text}
}}
"""

HDR.write_text(
    hdr_text,
    encoding="utf-8"
)

# ============================================================
# ENDMEMBER LOCATION SUMMARY
# ============================================================

location_cols = [
    "subtype_code",
    "lithology_family",
    "image_row_1based",
    "image_col_1based",
    "UTM_Easting_m",
    "UTM_Northing_m",
    "longitude",
    "latitude",
    "broad_probability",
    "uncertainty",
    "subtype_similarity",
    "subtype_margin",
]

location_cols = [
    c for c in location_cols
    if c in locations.columns
]

location_summary = (
    locations[
        location_cols
    ]
    .sort_values(
        "subtype_code"
    )
)

LOCATION_SUMMARY_CSV = OUT / (
    "AVIRIS_Step15B_13_Endmember_Location_Summary.csv"
)

location_summary.to_csv(
    LOCATION_SUMMARY_CSV,
    index=False
)

# ============================================================
# ENDMEMBER MASTER TABLE
# ============================================================

master_rows = []

for i, name in enumerate(names):

    loc = locations[
        locations[
            "lithology_family"
        ] == name
    ]

    if len(loc):
        loc = loc.iloc[0]
    else:
        loc = None

    master_rows.append({
        "subtype_code":
            codes[i],

        "lithology_family":
            name,

        "display_name":
            clean_name(name),

        "number_of_spectral_bands":
            326,

        "minimum_wavelength_nm":
            float(
                np.nanmin(wl326)
            ),

        "maximum_wavelength_nm":
            float(
                np.nanmax(wl326)
            ),

        "longitude":
            (
                loc["longitude"]
                if loc is not None
                else np.nan
            ),

        "latitude":
            (
                loc["latitude"]
                if loc is not None
                else np.nan
            ),

        "UTM_Easting_m":
            (
                loc["UTM_Easting_m"]
                if loc is not None
                else np.nan
            ),

        "UTM_Northing_m":
            (
                loc["UTM_Northing_m"]
                if loc is not None
                else np.nan
            ),

        "interpretation_status":
            "IMAGE_DERIVED_REFERENCE_GUIDED_ENDMEMBER"
    })

master_df = pd.DataFrame(
    master_rows
)

MASTER_CSV = OUT / (
    "AVIRIS_Step15B_13_Endmember_Master_Table.csv"
)

master_df.to_csv(
    MASTER_CSV,
    index=False
)

# ============================================================
# EXCEL WORKBOOK
# ============================================================

XLSX = OUT / (
    "AVIRIS_Step15B_13_Endmember_Spectral_Library.xlsx"
)

wide326 = df326.copy()

with pd.ExcelWriter(
    XLSX,
    engine="openpyxl"
) as writer:

    master_df.to_excel(
        writer,
        sheet_name="Endmember_Master",
        index=False
    )

    location_summary.to_excel(
        writer,
        sheet_name="Locations",
        index=False
    )

    wide326.to_excel(
        writer,
        sheet_name="326Band_Spectra",
        index=False
    )

    long_df.to_excel(
        writer,
        sheet_name="Long_Spectra",
        index=False
    )

# ============================================================
# SETTINGS
# ============================================================

settings = {
    "step":
        "Step15B_13_Endmember_Spectral_Library",

    "endmembers":
        len(names),

    "scientific_bands":
        len(wl326),

    "ENVI_spectral_library":
        str(SLI),

    "ENVI_header":
        str(HDR),

    "important_interpretation":
        (
            "The 13 spectra are image-derived, reference-guided "
            "representative AVIRIS-NG endmembers. They require "
            "field/laboratory validation before being presented as "
            "definitive lithological endmembers."
        ),

    "chromitite_warning":
        (
            "The chromitite-associated endmember does not constitute "
            "direct detection of chromite or PGE mineralization."
        )
}

(
    OUT / "AVIRIS_Step15B_Settings.json"
).write_text(
    json.dumps(
        settings,
        indent=2
    ),
    encoding="utf-8"
)

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 90)
print("STEP15B COMPLETE")
print("=" * 90)

print("\nEndmembers:", len(names))
print("Scientific bands:", len(wl326))

print("\nENDMEMBER MASTER TABLE:")
print(
    master_df.to_string(
        index=False
    )
)

print("\nENVI spectral library:")
print(SLI)

print("\nENVI spectral library header:")
print(HDR)

print("\nCombined raw spectral plot:")
print(RAW_PNG)

print("\nCombined normalized plot:")
print(NORM_PNG)

print("\nIndividual plots:")
print(INDIVIDUAL_DIR)

print("\nExcel workbook:")
print(XLSX)

print("\nIMPORTANT:")
print(
    "Use the 326-band library for spectral comparison/mapping."
)

print(
    "Keep the 425-band original spectra as archival/raw endmember spectra."
)

print(
    "These are representative image-derived endmembers, "
    "not yet independent field-confirmed geological endmembers."
)

print("\nPASS")
