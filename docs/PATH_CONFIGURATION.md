# Path configuration

The original analysis was performed on a local project directory.

For portability, scripts that used the original project ROOT path
have been changed to read the environment variable:

    SAC_AVIRIS_ROOT

Example on macOS/Linux:

    export SAC_AVIRIS_ROOT="/path/to/AVIRIS_NG_project"

For the local ASD spectral directory, if required:

    export SAC_ASD_ROOT="/path/to/SAC_ASD_spectra"

No scientific thresholds, model parameters, class definitions,
equations, or analysis logic were altered during this path conversion.

The large AVIRIS-NG imagery and laboratory/field source data are
not bundled with this code repository.
