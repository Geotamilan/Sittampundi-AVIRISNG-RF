# -*- coding: utf-8 -*-
import os

from pathlib import Path
import re
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

AUDIT = ROOT / (
    "07_Spectra/Step13C_v2_Repaired_Classification/"
    "AVIRIS_Step13C_v2_family_code_audit.csv"
)

OUT = ROOT / (
    "09_Mineral_Mapping/"
    "Step14C2A_ASD_WhiteReference_Audit"
)

OUT.mkdir(parents=True, exist_ok=True)

RECOGNIZED = [
    "ANORTHOSITE_METAANORTHOSITE",
    "CHROMITITE",
    "GABBRO_METAGABBRO",
    "GRANITE_GRANITIC_ROCK",
    "HORNBLENDE_BIOTITE_GNEISS",
    "PYROXENITE_MAFIC_CUMULATE",
]

if not AUDIT.exists():
    raise FileNotFoundError(AUDIT)

df = pd.read_csv(AUDIT)

print("=" * 90)
print("STEP14C2-A - SAC ASD WHITE-REFERENCE CONTAMINATION AUDIT")
print("=" * 90)

print("\nTotal Step13C-v2 audit records:", len(df))

# ------------------------------------------------------------------
# White-reference path detection
# ------------------------------------------------------------------

def is_wr_path(p):
    s = str(p).replace("\\", "/")
    return bool(re.search(r"/WR\d*(?:/|$)", s, flags=re.I))

df["is_white_reference_path"] = df["local_path"].apply(is_wr_path)

df["is_recognized_lithology"] = df["local_family_v2"].isin(RECOGNIZED)

recognized = df[df["is_recognized_lithology"]].copy()

print("\nRecognized geological-family records:", len(recognized))

print("\nRecognized family counts BEFORE WR removal:")
print(
    recognized["local_family_v2"]
    .value_counts()
    .to_string()
)

print("\nWhite-reference path counts:")
print(
    recognized["is_white_reference_path"]
    .value_counts()
    .rename(index={True: "WR_PATH", False: "NON_WR_PATH"})
    .to_string()
)

summary = (
    recognized
    .groupby("local_family_v2")
    .agg(
        total_records=("local_file", "size"),
        WR_records=("is_white_reference_path", "sum")
    )
    .reset_index()
)

summary["non_WR_records"] = (
    summary["total_records"] -
    summary["WR_records"]
)

summary["WR_fraction"] = (
    summary["WR_records"] /
    summary["total_records"]
)

print("\nPER-FAMILY WR AUDIT:")
print(summary.to_string(index=False))

# ------------------------------------------------------------------
# Basic parser for 2-column / multi-column ASCII spectra
# ------------------------------------------------------------------

def parse_ascii(path):

    path = Path(path)

    if not path.exists():
        return None

    try:
        lines = path.read_text(errors="ignore").splitlines()
    except Exception:
        return None

    wavelengths = []
    values = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = re.split(r"[\t, ]+", line)

        nums = []

        for p in parts:
            try:
                nums.append(float(p))
            except Exception:
                pass

        if len(nums) < 2:
            continue

        wl = nums[0]

        if not (300 <= wl <= 2600):
            continue

        spectral_values = [
            x for x in nums[1:]
            if np.isfinite(x)
        ]

        if not spectral_values:
            continue

        # One file = one representative spectrum.
        # For multi-column ENVI plot files use median across columns.
        value = float(np.nanmedian(spectral_values))

        wavelengths.append(wl)
        values.append(value)

    if len(wavelengths) < 100:
        return None

    wl = np.asarray(wavelengths, dtype=float)
    val = np.asarray(values, dtype=float)

    order = np.argsort(wl)

    wl = wl[order]
    val = val[order]

    return wl, val


# ------------------------------------------------------------------
# Spectral QC
# ------------------------------------------------------------------

qc_rows = []

for _, r in recognized.iterrows():

    p = Path(r["local_path"])

    parsed = parse_ascii(p)

    rec = {
        "local_file": r["local_file"],
        "local_path": r["local_path"],
        "local_family_v2": r["local_family_v2"],
        "sample_preparation_v2": r["sample_preparation_v2"],
        "is_white_reference_path": r["is_white_reference_path"],
        "file_exists": p.exists(),
    }

    if parsed is None:

        rec.update({
            "parsed": False,
            "n_points": 0,
            "wl_min_nm": np.nan,
            "wl_max_nm": np.nan,
            "median_value": np.nan,
            "mean_value": np.nan,
            "std_value": np.nan,
            "near_unity_fraction": np.nan,
            "spectral_signature_hash": "",
        })

    else:

        wl, val = parsed

        finite = np.isfinite(val)

        v = val[finite]

        near_unity = (
            np.mean(
                (v >= 0.95) &
                (v <= 1.05)
            )
            if len(v)
            else np.nan
        )

        # Hash first 200 rounded values to detect exact/near-exact duplicates
        arr = np.round(v[:200], 8)

        sig_hash = hashlib.md5(
            arr.tobytes()
        ).hexdigest()

        rec.update({
            "parsed": True,
            "n_points": len(v),
            "wl_min_nm": float(np.nanmin(wl)),
            "wl_max_nm": float(np.nanmax(wl)),
            "median_value": float(np.nanmedian(v)),
            "mean_value": float(np.nanmean(v)),
            "std_value": float(np.nanstd(v)),
            "near_unity_fraction": float(near_unity),
            "spectral_signature_hash": sig_hash,
        })

    qc_rows.append(rec)

qc = pd.DataFrame(qc_rows)

# ------------------------------------------------------------------
# Duplicate signature detection
# ------------------------------------------------------------------

hash_counts = (
    qc[qc["parsed"]]
    .groupby("spectral_signature_hash")
    .size()
    .rename("duplicate_count")
)

qc = qc.merge(
    hash_counts,
    on="spectral_signature_hash",
    how="left"
)

qc["duplicate_count"] = (
    qc["duplicate_count"]
    .fillna(0)
    .astype(int)
)

# Flag obvious reference-like spectra.
# This does NOT classify minerals; it only flags suspicious near-unity spectra.
qc["near_unity_reference_like"] = (
    (qc["near_unity_fraction"] >= 0.80) &
    (qc["std_value"] <= 0.05)
)

# ------------------------------------------------------------------
# Recommended clean ASD candidates
# ------------------------------------------------------------------

clean = qc[
    (qc["parsed"]) &
    (~qc["is_white_reference_path"]) &
    (~qc["near_unity_reference_like"])
].copy()

print("\nPARSE SUMMARY:")
print("Parsed recognized records :", int(qc["parsed"].sum()))
print("Unparsed records          :", int((~qc["parsed"]).sum()))

print("\nREFERENCE-LIKE QC:")
print(
    qc[
        [
            "is_white_reference_path",
            "near_unity_reference_like"
        ]
    ]
    .value_counts()
    .to_string()
)

print("\nCLEAN NON-WR CANDIDATES BY FAMILY:")
if clean.empty:
    print("NONE")
else:
    print(
        clean["local_family_v2"]
        .value_counts()
        .to_string()
    )

print("\nEXACT/ROUNDED DUPLICATE SIGNATURES ACROSS MULTIPLE FILES:")

dup = (
    qc[qc["duplicate_count"] > 1]
    .sort_values(
        ["duplicate_count", "spectral_signature_hash"],
        ascending=[False, True]
    )
)

if dup.empty:
    print("None")
else:
    print(
        dup[
            [
                "local_family_v2",
                "local_file",
                "is_white_reference_path",
                "median_value",
                "std_value",
                "duplicate_count"
            ]
        ]
        .head(50)
        .to_string(index=False)
    )

# ------------------------------------------------------------------
# Show one clean non-WR example per family
# ------------------------------------------------------------------

print("\n" + "=" * 90)
print("ONE CLEAN NON-WR EXAMPLE PER FAMILY")
print("=" * 90)

for fam in RECOGNIZED:

    sub = clean[
        clean["local_family_v2"] == fam
    ]

    print(f"\n--- {fam} ---")

    if sub.empty:
        print("NO CLEAN NON-WR SPECTRUM FOUND")
        continue

    r = sub.iloc[0]

    print("FILE:", r["local_path"])
    print("median:", r["median_value"])
    print("std:", r["std_value"])
    print("near-unity fraction:", r["near_unity_fraction"])
    print("duplicates:", r["duplicate_count"])

# ------------------------------------------------------------------
# Save
# ------------------------------------------------------------------

summary.to_csv(
    OUT / "AVIRIS_Step14C2A_WR_Counts_By_Family.csv",
    index=False
)

qc.to_csv(
    OUT / "AVIRIS_Step14C2A_All_Recognized_ASD_QC.csv",
    index=False
)

clean.to_csv(
    OUT / "AVIRIS_Step14C2A_Clean_NonWR_ASD_Candidates.csv",
    index=False
)

with pd.ExcelWriter(
    OUT / "AVIRIS_Step14C2A_ASD_WhiteReference_Audit.xlsx",
    engine="openpyxl"
) as writer:

    summary.to_excel(
        writer,
        sheet_name="WR_Counts",
        index=False
    )

    qc.to_excel(
        writer,
        sheet_name="All_ASD_QC",
        index=False
    )

    clean.to_excel(
        writer,
        sheet_name="Clean_NonWR",
        index=False
    )

print("\n" + "=" * 90)
print("STEP14C2-A COMPLETE")
print("=" * 90)

print("\nRecognized ASD records :", len(recognized))
print("WR-path records        :", int(recognized["is_white_reference_path"].sum()))
print("Non-WR records         :", int((~recognized["is_white_reference_path"]).sum()))
print("Clean non-WR candidates:", len(clean))

print("\nOutput folder:")
print(OUT)

print("\nIMPORTANT:")
print("WR-path spectra are NOT used as rock training spectra.")
print("No lithology classification map is created in this audit.")

print("\nPASS")
