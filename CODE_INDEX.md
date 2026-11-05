# Code index

This repository contains the analysis scripts used for the
Sittampundi AVIRIS-NG hyperspectral and Random Forest study.

## 01_preprocessing
Scene registration, registration QC, radiometric seam analysis,
final wavelength-dependent normalization, 425-band mosaicking,
mosaic QC, and 326-band scientific-cube generation.

## 02_pca_mnf_ppi
PCA statistics, MNF noise estimation, MNF component generation,
Pixel Purity Index analysis, and candidate screening.

## 03_field_asd_roi
Local ASD quality control, field training-data preparation,
ROI separability assessment, hierarchical ROI QC,
white-reference exclusion, clean raw ASD-guided revalidation,
and AVIRIS-NG versus local ASD comparison.

## 04_random_forest
Five broad lithological-group Random Forest probability,
winner-class, maximum-probability, and uncertainty mapping.

## 05_endmembers_spectral
High-confidence reference-guided subtype support,
13 AVIRIS-NG image endmembers, spectral-library generation,
continuum-removal and absorption analysis,
pairwise spectral-angle/redundancy analysis,
and defensible reduced spectral grouping.

## 06_grouping_validation_synthesis
Reduced SG1/SG2 mapping, official field-site grouped
Random Forest accuracy assessment, and final
uncertainty-aware evidence synthesis.

## Important interpretation
The 13 reference-guided image spectra are not treated as
13 independently validated lithological classes.
SG1 and SG2 are data-driven spectral groups, not geological units.
Chromitite-associated outputs are not direct chromite or PGE detection.
