# Functional verification

The repository was tested in a newly created Conda environment using `environment.yml`.

## Dependency test

The following required packages imported successfully:

- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- scikit-image
- rasterio
- affine
- geopandas
- shapely
- pyproj
- spectral
- openpyxl

## Syntax test

All 32 analysis scripts in `scripts/` passed Python syntax parsing.

## Scientific runtime test

The following script was executed using the original Sittampundi project data:

    python scripts/06_grouping_validation_synthesis/step16a_official_rf_accuracy_assessment.py

The run completed successfully and reproduced the official field-site grouped Random Forest validation:

- Validation field sites: 43
- Overall Accuracy: 0.604651 (60.47%)
- Balanced Accuracy: 0.574048 (57.40%)
- Cohen's Kappa: 0.496903
- Macro F1: 0.589032
- Weighted F1: 0.616619
- OA 95% bootstrap CI: 46.51%–74.42%
- BA 95% bootstrap CI: 41.78%–73.00%

The script returned:

    PASS_MATCHES_PREVIOUS_STEP14C2_V2_VALIDATION

and:

    READY_TO_FREEZE_FOR_CURRENT_WORKFLOW

This verifies that the declared software environment and the official validation workflow execute successfully when the required source datasets are available.
