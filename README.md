# Uncertainty-aware AVIRIS-NG hyperspectral and Random Forest workflow for the Sittampundi Anorthosite Complex

## Overview

This repository contains the analysis code associated with the manuscript on AVIRIS-NG hyperspectral lithological probability interpretation and chromitite-associated spectral targeting in the Sittampundi Anorthosite Complex, southern India.

The workflow combines geometric and radiometric image harmonisation, dimensionality reduction, candidate spectral screening, field-centred Random Forest modelling, grouped field-site validation, local ASD spectral comparison, image-derived endmember analysis, spectral redundancy testing, hierarchical grouping, and uncertainty-aware interpretation.

## Repository scope

Only analysis scripts are included. Raw AVIRIS-NG imagery, laboratory ASD source files, field shapefiles, large raster outputs, and derived publication figures are not redistributed here.

## Scientific interpretation

The five broad Random Forest groups are probability-based lithological support classes. The thirteen image spectra are reference-guided endmember candidates. High spectral redundancy means they should not be interpreted as thirteen independently separable hard lithological classes. The reduced SG1 and SG2 outputs are data-driven spectral groups and should not be renamed as validated geological units.

Chromitite-related outputs are reported as chromitite-associated spectral/model support and are not direct detection of chromite or PGE mineralisation.

## Code organisation

- `scripts/01_preprocessing/` — registration, radiometric harmonisation, mosaicking, scientific-band subset
- `scripts/02_pca_mnf_ppi/` — PCA, MNF, PPI and dimensionality/spectral-purity screening
- `scripts/03_field_asd_roi/` — ASD quality control, field references and ROI preparation
- `scripts/04_random_forest/` — broad-group Random Forest probability/uncertainty mapping
- `scripts/05_endmembers_spectral/` — subtype/endmember extraction, continuum removal, ASD comparison, separability/redundancy
- `scripts/06_grouping_validation_synthesis/` — reduced spectral groups, grouped validation, evidence synthesis

## Environment

A suggested environment is provided in `environment.yml` and `requirements.txt`. Exact versions should be frozen from the analysis machine before archival release when possible.

## Data availability

AVIRIS-NG data are subject to the access conditions of the original data provider. This repository contains processing code only. Users should obtain source imagery and field/laboratory data from the appropriate providers or authors under the applicable access conditions.

## Citation

Please cite the associated manuscript and this repository when using the workflow.

## Quick start

Create the Conda environment:

    conda env create -f environment.yml
    conda activate aviris-sittampundi

For analyses requiring the original project data, define the project root:

    export SAC_AVIRIS_ROOT="/path/to/AVIRIS_NG_project"

For analyses requiring the local ASD spectral library:

    export SAC_ASD_ROOT="/path/to/SAC_ASD_spectra"

Example validation run:

    python scripts/06_grouping_validation_synthesis/step16a_official_rf_accuracy_assessment.py

The official field-site grouped validation uses 43 field sites and reproduces approximately:

- Overall accuracy: 0.604651
- Balanced accuracy: 0.574048
- Cohen's kappa: 0.496903
- Macro F1: 0.589032
- Weighted F1: 0.616619

Raw AVIRIS-NG imagery and field/laboratory source datasets are not redistributed in this repository and must be obtained separately under the applicable access conditions.
