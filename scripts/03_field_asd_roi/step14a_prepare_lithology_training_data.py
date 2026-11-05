# -*- coding: utf-8 -*-
import os

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely.geometry import Point

warnings.filterwarnings("ignore")

ROOT = Path(os.environ.get("SAC_AVIRIS_ROOT", "."))

AVIRIS = ROOT / "03_Mosaic/Final_425Band_Mosaic/AVIRIS_NG_mosaic_master_425band_4m.dat"

FIELD_SHP = ROOT / "09_Mineral_Mapping/Step13E_Geological_Overlay_Inputs/Location_Geology_PGE_Sittampundi.shp"

OUT = ROOT / "09_Mineral_Mapping/Step14A_Lithology_Training_Data_Preparation"
OUT.mkdir(parents=True, exist_ok=True)

NODATA = -9999.0

# AVIRIS approximate diagnostic display bands, 1-based index
DISPLAY_BANDS = {
    "blue_band_1based": 20,
    "green_band_1based": 36,
    "red_band_1based": 54,
    "nir_band_1based": 98,
}

# Minimum sample rules for later Step14 classification
MIN_FOR_STRONG_SUPERVISED = 5
MIN_FOR_LIMITED_SUPERVISED = 3

# Lithology keyword rules.
# These rules only classify field sample labels into working families.
# They do not create a final lithology map in Step14A.
LITHOLOGY_RULES = [
    ("CHROMITITE", [
        "sicr", "chromitite", "chromite", "cr spinel", "cr-spinel", "spinelite"
    ]),
    ("ANORTHOSITE_METAANORTHOSITE", [
        "sian", "anorthosite", "meta anorthosite", "meta-anorthosite",
        "metaanorthosite", "anorthositic"
    ]),
    ("GABBRO_METAGABBRO", [
        "sigb", "gabbro", "metagabbro", "meta gabbro", "gabbroic"
    ]),
    ("PYROXENITE_MAFIC_CUMULATE", [
        "sipx", "pyroxenite", "clinopyroxenite", "orthopyroxenite",
        "mafic cumulate", "cumulate"
    ]),
    ("ULTRAMAFIC_DUNITE", [
        "dunite", "ultramafic", "peridotite", "serpentinite"
    ]),
    ("HORNBLENDE_BIOTITE_GNEISS", [
        "sihbg", "hbg", "hornblende biotite gneiss",
        "hornblende-biotite gneiss", "hb gneiss", "biotite gneiss"
    ]),
    ("GRANITE_GRANITIC_ROCK", [
        "sigr", "granite", "granitic", "pink granite", "white granite"
    ]),
    ("PEGMATITE", [
        "sipt", "pegmatite", "pegmatitic"
    ]),
    ("MIGMATITE", [
        "migmatite", "pink migmatite", "migmatitic"
    ]),
    ("QUARTZITE", [
        "siqzt", "quartzite", "quartzitic"
    ]),
    ("DOLERITE", [
        "dolerite", "diabase"
    ]),
    ("DOLOMITE_CARBONATE", [
        "dolomite", "carbonate", "limestone"
    ]),
    ("GNEISS_GRANITIC_COMPLEX", [
        "gneiss", "granitic gneiss", "granite gneiss"
    ]),
]

# Later Step14B/14C can use this grouping strategy.
# Step14A only reports it.
MERGED_CLASS_SUGGESTIONS = {
    "MAFIC_ULTRAMAFIC_HOST_ROCKS": [
        "CHROMITITE",
        "GABBRO_METAGABBRO",
        "PYROXENITE_MAFIC_CUMULATE",
        "ULTRAMAFIC_DUNITE",
    ],
    "FELSIC_GRANITIC_ROCKS": [
        "GRANITE_GRANITIC_ROCK",
        "PEGMATITE",
        "MIGMATITE",
        "GNEISS_GRANITIC_COMPLEX",
    ],
    "ANORTHOSITE_HOST": [
        "ANORTHOSITE_METAANORTHOSITE",
    ],
    "METAMORPHIC_GNEISS": [
        "HORNBLENDE_BIOTITE_GNEISS",
        "GNEISS_GRANITIC_COMPLEX",
    ],
}


def parse_envi_wavelengths(hdr_path):
    if not hdr_path.exists():
        return []
    text = hdr_path.read_text(errors="ignore")
    m = re.search(r"wavelength\s*=\s*\{(.*?)\}", text, flags=re.I | re.S)
    if not m:
        return []
    body = m.group(1).replace("\n", " ")
    vals = []
    for x in body.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            vals.append(float(x))
        except Exception:
            pass
    return vals


def one_text(row):
    parts = []
    for v in row.values:
        if pd.isna(v):
            continue
        parts.append(str(v))
    return " ".join(parts).lower()


def classify_lithology(row):
    text = one_text(row)

    # Special handling: granite gneiss should not become pure granite first
    if "granite gneiss" in text or "granitic gneiss" in text:
        return "GNEISS_GRANITIC_COMPLEX"

    for family, keywords in LITHOLOGY_RULES:
        for kw in keywords:
            if kw.lower() in text:
                return family

    return "UNCLASSIFIED"


def get_sample_name_column(gdf):
    candidates = [
        "sample", "sample_id", "sampleid", "name", "code", "id",
        "Sample", "Sample_ID", "SampleID", "Name", "Code", "ID"
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    # fallback: first non-geometry column
    for c in gdf.columns:
        if c != gdf.geometry.name:
            return c
    return None


def to_2d_points(gdf):
    records = []
    geoms = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            x, y = np.nan, np.nan
            p = None
        else:
            if geom.geom_type == "Point":
                x, y = float(geom.x), float(geom.y)
                p = Point(x, y)
            else:
                rp = geom.representative_point()
                x, y = float(rp.x), float(rp.y)
                p = Point(x, y)
        rr = row.drop(labels=[gdf.geometry.name]).to_dict()
        rr["field_E"] = x
        rr["field_N"] = y
        records.append(rr)
        geoms.append(p)
    out = gpd.GeoDataFrame(records, geometry=geoms, crs=gdf.crs)
    return out


def patch_median(src, row, col, radius=1):
    # radius 1 = 3x3, radius 2 = 5x5
    r0 = int(row) - radius
    c0 = int(col) - radius
    size = radius * 2 + 1
    win = Window(c0, r0, size, size)
    arr = src.read(window=win, boundless=True, fill_value=NODATA)
    # arr shape = bands, h, w
    arr = arr.astype("float64")
    arr[arr == NODATA] = np.nan
    arr[arr <= -9000] = np.nan
    med = np.nanmedian(arr.reshape(arr.shape[0], -1), axis=1)
    valid = np.isfinite(med)
    return med, int(valid.sum())


def safe_norm(x):
    x = np.asarray(x, dtype="float64")
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return x
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if not np.isfinite(mx - mn) or abs(mx - mn) < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def spectrum_features(spec, wavelengths):
    spec = np.asarray(spec, dtype="float64")
    out = {}

    finite = np.isfinite(spec)
    out["valid_band_count"] = int(finite.sum())
    out["reflectance_mean"] = float(np.nanmean(spec)) if finite.sum() else np.nan
    out["reflectance_std"] = float(np.nanstd(spec)) if finite.sum() else np.nan
    out["reflectance_min"] = float(np.nanmin(spec)) if finite.sum() else np.nan
    out["reflectance_max"] = float(np.nanmax(spec)) if finite.sum() else np.nan

    # Display band values
    for name, b1 in DISPLAY_BANDS.items():
        idx = b1 - 1
        if 0 <= idx < len(spec):
            out[name.replace("_band_1based", "_value")] = float(spec[idx]) if np.isfinite(spec[idx]) else np.nan

    red_idx = DISPLAY_BANDS["red_band_1based"] - 1
    nir_idx = DISPLAY_BANDS["nir_band_1based"] - 1
    if 0 <= red_idx < len(spec) and 0 <= nir_idx < len(spec):
        red = spec[red_idx]
        nir = spec[nir_idx]
        if np.isfinite(red) and np.isfinite(nir) and abs(nir + red) > 1e-12:
            out["ndvi_like"] = float((nir - red) / (nir + red))
        else:
            out["ndvi_like"] = np.nan
    else:
        out["ndvi_like"] = np.nan

    if wavelengths and len(wavelengths) == len(spec):
        wl = np.asarray(wavelengths)
        for label, lo, hi in [
            ("vis_400_700", 400, 700),
            ("nir_700_1300", 700, 1300),
            ("swir1_1500_1800", 1500, 1800),
            ("swir2_2000_2400", 2000, 2400),
            ("mafic_850_1300", 850, 1300),
            ("hydroxyl_2200_2350", 2200, 2350),
        ]:
            mask = (wl >= lo) & (wl <= hi) & np.isfinite(spec)
            out[f"{label}_mean"] = float(np.nanmean(spec[mask])) if mask.sum() else np.nan
            out[f"{label}_std"] = float(np.nanstd(spec[mask])) if mask.sum() else np.nan
    return out


def class_readiness(n):
    if n >= MIN_FOR_STRONG_SUPERVISED:
        return "READY_FOR_SUPERVISED_CLASSIFICATION"
    if n >= MIN_FOR_LIMITED_SUPERVISED:
        return "LIMITED_SUPERVISED_OR_REFERENCE_CLASS"
    if n >= 1:
        return "REFERENCE_ONLY_SAMPLE_COUNT_LOW"
    return "NO_SAMPLE"


def main():
    print("=" * 80)
    print("STEP14A - SAC LITHOLOGY TRAINING DATA PREPARATION")
    print("=" * 80)

    print("\nAVIRIS master:")
    print(AVIRIS)

    print("\nField sample shapefile:")
    print(FIELD_SHP)

    print("\nOutput folder:")
    print(OUT)

    if not AVIRIS.exists():
        raise FileNotFoundError(f"AVIRIS raster not found: {AVIRIS}")
    if not FIELD_SHP.exists():
        raise FileNotFoundError(f"Field shapefile not found: {FIELD_SHP}")

    hdr = AVIRIS.with_suffix(".hdr")
    wavelengths = parse_envi_wavelengths(hdr)
    print("\nWavelength count from HDR:", len(wavelengths))

    field_raw = gpd.read_file(FIELD_SHP)
    print("\nField samples loaded:", len(field_raw))
    print("Original CRS:", field_raw.crs)

    if field_raw.empty:
        raise ValueError("Field shapefile is empty.")

    with rasterio.open(AVIRIS) as src:
        print("\nAVIRIS raster info:")
        print("  size:", src.width, "x", src.height)
        print("  bands:", src.count)
        print("  CRS:", src.crs)
        print("  resolution:", src.res)

        if field_raw.crs is None:
            print("\nWARNING: field shapefile CRS is missing.")
            print("Assuming EPSG:32643 based on SAC AVIRIS project.")
            field_raw = field_raw.set_crs(src.crs)
        else:
            field_raw = field_raw.to_crs(src.crs)

        field_pts = to_2d_points(field_raw)
        field_pts = field_pts.set_crs(src.crs)

        name_col = get_sample_name_column(field_pts)
        print("\nDetected sample name/code column:", name_col)

        # Lithology classification from all field attributes
        field_pts["sample_label_for_step14a"] = (
            field_pts[name_col].astype(str) if name_col else ["SAMPLE_" + str(i + 1) for i in range(len(field_pts))]
        )
        non_geom_cols = [c for c in field_pts.columns if c != field_pts.geometry.name]
        field_pts["step14a_text_used_for_classification"] = field_pts[non_geom_cols].astype(str).agg(" | ".join, axis=1)
        field_pts["lithology_family"] = field_pts.apply(classify_lithology, axis=1)

        spectra_rows = []
        feature_rows = []

        print("\nExtracting AVIRIS spectra at field sample locations...")

        for i, row in field_pts.iterrows():
            x = row["field_E"]
            y = row["field_N"]
            if not np.isfinite(x) or not np.isfinite(y):
                center_spec = np.full(src.count, np.nan)
                med3 = np.full(src.count, np.nan)
                med5 = np.full(src.count, np.nan)
                px_row = np.nan
                px_col = np.nan
                in_raster = False
                valid_center = 0
                valid_3x3 = 0
                valid_5x5 = 0
            else:
                px_row, px_col = src.index(x, y)
                in_raster = (0 <= px_row < src.height) and (0 <= px_col < src.width)

                if in_raster:
                    center_spec = src.read(window=Window(px_col, px_row, 1, 1)).reshape(src.count).astype("float64")
                    center_spec[center_spec == NODATA] = np.nan
                    center_spec[center_spec <= -9000] = np.nan
                    valid_center = int(np.isfinite(center_spec).sum())

                    med3, valid_3x3 = patch_median(src, px_row, px_col, radius=1)
                    med5, valid_5x5 = patch_median(src, px_row, px_col, radius=2)
                else:
                    center_spec = np.full(src.count, np.nan)
                    med3 = np.full(src.count, np.nan)
                    med5 = np.full(src.count, np.nan)
                    valid_center = 0
                    valid_3x3 = 0
                    valid_5x5 = 0

            base = {
                "field_index": int(i),
                "sample_label": row["sample_label_for_step14a"],
                "lithology_family": row["lithology_family"],
                "field_E": x,
                "field_N": y,
                "raster_row": px_row,
                "raster_col": px_col,
                "inside_aviris": bool(in_raster),
                "valid_center_bands": valid_center,
                "valid_3x3_median_bands": valid_3x3,
                "valid_5x5_median_bands": valid_5x5,
            }

            # Use 3x3 median as preferred training spectrum
            feat = base.copy()
            feat.update(spectrum_features(med3, wavelengths))
            feature_rows.append(feat)

            # Wide spectra table
            wide = base.copy()
            for b in range(src.count):
                wl = wavelengths[b] if b < len(wavelengths) else np.nan
                wide[f"B{b+1:03d}"] = med3[b]
            spectra_rows.append(wide)

            if (i + 1) % 10 == 0:
                print("  samples processed:", i + 1)

    features = pd.DataFrame(feature_rows)
    spectra_wide = pd.DataFrame(spectra_rows)

    # Add wavelength mapping
    wave_rows = []
    band_count = spectra_wide.filter(regex=r"^B\d+").shape[1]
    for b in range(band_count):
        wave_rows.append({
            "band_1based": b + 1,
            "band_name": f"B{b+1:03d}",
            "wavelength_nm": wavelengths[b] if b < len(wavelengths) else np.nan,
            "display_role": (
                "BLUE_TCC" if b + 1 == DISPLAY_BANDS["blue_band_1based"] else
                "GREEN_TCC" if b + 1 == DISPLAY_BANDS["green_band_1based"] else
                "RED_TCC" if b + 1 == DISPLAY_BANDS["red_band_1based"] else
                "NIR_FCC" if b + 1 == DISPLAY_BANDS["nir_band_1based"] else
                ""
            )
        })
    wavelength_table = pd.DataFrame(wave_rows)

    # Family counts
    counts = (
        features.groupby("lithology_family", dropna=False)
        .agg(
            sample_count=("sample_label", "count"),
            inside_aviris_count=("inside_aviris", "sum"),
            median_valid_bands=("valid_3x3_median_bands", "median"),
            mean_reflectance=("reflectance_mean", "mean"),
            mean_ndvi_like=("ndvi_like", "mean"),
        )
        .reset_index()
        .sort_values(["sample_count", "lithology_family"], ascending=[False, True])
    )

    counts["step14_readiness"] = counts["sample_count"].apply(class_readiness)

    # Suggested merged classes summary
    merged_rows = []
    for merged_name, fams in MERGED_CLASS_SUGGESTIONS.items():
        n = int(counts[counts["lithology_family"].isin(fams)]["sample_count"].sum())
        merged_rows.append({
            "suggested_merged_class": merged_name,
            "included_families": ", ".join(fams),
            "sample_count": n,
            "step14_readiness": class_readiness(n),
        })
    merged_summary = pd.DataFrame(merged_rows)

    # Preferred training/reference role
    features = features.merge(
        counts[["lithology_family", "sample_count", "step14_readiness"]],
        on="lithology_family",
        how="left"
    )

    def role(row):
        fam = str(row["lithology_family"])
        n = int(row["sample_count"]) if pd.notna(row["sample_count"]) else 0
        if fam == "UNCLASSIFIED":
            return "DO_NOT_USE_UNCLASSIFIED"
        if not row["inside_aviris"] or row["valid_3x3_median_bands"] < 100:
            return "DO_NOT_USE_OUTSIDE_OR_INVALID"
        if n >= MIN_FOR_STRONG_SUPERVISED:
            return "PRIMARY_TRAINING"
        if n >= MIN_FOR_LIMITED_SUPERVISED:
            return "LIMITED_TRAINING_REFERENCE"
        return "REFERENCE_ONLY_LOW_SAMPLE"

    features["step14a_training_role"] = features.apply(role, axis=1)

    # Representative median spectra by lithology family
    band_cols = [c for c in spectra_wide.columns if re.match(r"^B\d{3}$", c)]
    spectra_with_family = spectra_wide.merge(
        features[["field_index", "lithology_family", "step14a_training_role"]],
        on=["field_index", "lithology_family"],
        how="left"
    )

    reps = []
    for fam, sub in spectra_with_family.groupby("lithology_family"):
        use = sub[sub["step14a_training_role"].str.contains("TRAINING|REFERENCE", na=False)]
        if use.empty:
            use = sub
        med = use[band_cols].median(axis=0, skipna=True)
        rec = {
            "lithology_family": fam,
            "sample_count_used": len(use),
        }
        for c in band_cols:
            rec[c] = med[c]
        reps.append(rec)
    representative_spectra = pd.DataFrame(reps)

    # Long spectra table, useful for plotting
    long_rows = []
    for _, r in spectra_wide.iterrows():
        for b, c in enumerate(band_cols):
            long_rows.append({
                "field_index": r["field_index"],
                "sample_label": r["sample_label"],
                "lithology_family": r["lithology_family"],
                "band_1based": b + 1,
                "band_name": c,
                "wavelength_nm": wavelengths[b] if b < len(wavelengths) else np.nan,
                "reflectance_3x3_median": r[c],
            })
    spectra_long = pd.DataFrame(long_rows)

    # GeoPackage output, keep only compact attributes
    compact_cols = [
        "field_index", "sample_label", "lithology_family", "step14a_training_role",
        "sample_count", "step14_readiness", "field_E", "field_N",
        "raster_row", "raster_col", "inside_aviris",
        "valid_center_bands", "valid_3x3_median_bands", "valid_5x5_median_bands",
        "reflectance_mean", "reflectance_std", "ndvi_like",
        "red_value", "green_value", "blue_value", "nir_value",
    ]
    compact_cols = [c for c in compact_cols if c in features.columns]

    gdf_out = gpd.GeoDataFrame(
        features[compact_cols].copy(),
        geometry=gpd.points_from_xy(features["field_E"], features["field_N"]),
        crs="EPSG:32643"
    )

    # Remove unsupported object/list fields if any
    for c in gdf_out.columns:
        if c != "geometry" and gdf_out[c].dtype == "object":
            gdf_out[c] = gdf_out[c].astype(str)

    gpkg = OUT / "AVIRIS_Step14A_Lithology_Training_Data.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    gdf_out.to_file(gpkg, layer="Step14A_Field_Lithology_Training_Points", driver="GPKG")

    # Output files
    features_csv = OUT / "AVIRIS_Step14A_Field_Samples_Classified.csv"
    counts_csv = OUT / "AVIRIS_Step14A_Lithology_Family_Counts.csv"
    merged_csv = OUT / "AVIRIS_Step14A_Suggested_Merged_Classes.csv"
    spectra_wide_csv = OUT / "AVIRIS_Step14A_Field_Spectra_3x3Median_Wide.csv"
    spectra_long_csv = OUT / "AVIRIS_Step14A_Field_Spectra_3x3Median_Long.csv"
    rep_csv = OUT / "AVIRIS_Step14A_Representative_Median_Spectra_By_Family.csv"
    wave_csv = OUT / "AVIRIS_Step14A_Band_Wavelength_Table.csv"
    settings_json = OUT / "AVIRIS_Step14A_Settings_and_QC.json"
    qc_txt = OUT / "AVIRIS_Step14A_QC_Report.txt"
    xlsx = OUT / "AVIRIS_Step14A_Lithology_Training_Data_Summary.xlsx"

    features.to_csv(features_csv, index=False)
    counts.to_csv(counts_csv, index=False)
    merged_summary.to_csv(merged_csv, index=False)
    spectra_wide.to_csv(spectra_wide_csv, index=False)
    spectra_long.to_csv(spectra_long_csv, index=False)
    representative_spectra.to_csv(rep_csv, index=False)
    wavelength_table.to_csv(wave_csv, index=False)

    settings = {
        "step": "Step14A_Lithology_Training_Data_Preparation",
        "root": str(ROOT),
        "aviris_master": str(AVIRIS),
        "field_shapefile": str(FIELD_SHP),
        "output_folder": str(OUT),
        "field_sample_count": int(len(features)),
        "wavelength_count": int(len(wavelengths)),
        "display_bands_1based": DISPLAY_BANDS,
        "minimum_for_strong_supervised": MIN_FOR_STRONG_SUPERVISED,
        "minimum_for_limited_supervised": MIN_FOR_LIMITED_SUPERVISED,
        "important_warning": "Step14A prepares lithology training data only. It is not a final lithology map and not direct chromitite/chromite/PGE detection."
    }
    settings_json.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        counts.to_excel(writer, sheet_name="Lithology_Counts", index=False)
        merged_summary.to_excel(writer, sheet_name="Merged_Class_Suggestions", index=False)
        features.to_excel(writer, sheet_name="Field_Samples_Classified", index=False)
        representative_spectra.to_excel(writer, sheet_name="Representative_Spectra", index=False)
        wavelength_table.to_excel(writer, sheet_name="Band_Wavelengths", index=False)

    with open(qc_txt, "w", encoding="utf-8") as f:
        f.write("STEP14A LITHOLOGY TRAINING DATA QC REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write("This step prepares field lithology training/reference data for later full-image lithology mapping.\n")
        f.write("This is not a final lithology map.\n")
        f.write("This is not direct chromitite/chromite/PGE detection.\n\n")
        f.write(f"AVIRIS master: {AVIRIS}\n")
        f.write(f"Field samples: {FIELD_SHP}\n")
        f.write(f"Output folder: {OUT}\n\n")
        f.write(f"Total field samples: {len(features)}\n")
        f.write(f"Inside AVIRIS: {int(features['inside_aviris'].sum())}\n")
        f.write(f"Unclassified: {int((features['lithology_family'] == 'UNCLASSIFIED').sum())}\n\n")
        f.write("Lithology family counts:\n")
        f.write(counts.to_string(index=False))
        f.write("\n\nSuggested merged classes:\n")
        f.write(merged_summary.to_string(index=False))
        f.write("\n\nRecommended next step:\n")
        f.write("Run Step14B to build lithology training ROIs / spectral reference sets and separability analysis.\n")

    print("\n" + "=" * 80)
    print("STEP14A COMPLETE")
    print("=" * 80)

    print("\nTotal field samples        :", len(features))
    print("Inside AVIRIS              :", int(features["inside_aviris"].sum()))
    print("Unclassified samples       :", int((features["lithology_family"] == "UNCLASSIFIED").sum()))

    print("\nLithology family counts:")
    print(counts[["lithology_family", "sample_count", "step14_readiness"]].to_string(index=False))

    print("\nSuggested merged classes:")
    print(merged_summary.to_string(index=False))

    print("\nMain ArcGIS GeoPackage:")
    print(gpkg)
    print("Layer:")
    print("Step14A_Field_Lithology_Training_Points")

    print("\nMain Excel summary:")
    print(xlsx)

    print("\nOutput folder:")
    print(OUT)

    print("\nNEXT STEP:")
    print("Step14B = spectral separability + training ROI preparation for full lithology classification")

    print("\nPASS")


if __name__ == "__main__":
    main()
