# GitHub upload checklist

Before making the repository public:

- [ ] Run `tools/collect_original_analysis_codes.py` on the Mac used for the analysis.
- [ ] Confirm `MISSING_CORE_SCRIPTS.txt` is empty or explain every missing item.
- [ ] Inspect `MANIFEST.csv` and verify that only final analysis scripts are included.
- [ ] Remove passwords, API keys, tokens, private paths, personal identifiers, and unpublished confidential data.
- [ ] Confirm no AVIRIS-NG imagery, ASD raw files, shapefiles, GeoTIFFs, Excel outputs, or figures are committed unless redistribution is permitted.
- [ ] Freeze exact package versions if required (`conda env export --from-history` or `pip freeze`).
- [ ] Add the final manuscript DOI when available.
- [ ] Choose a software license before public release if desired.
- [ ] Create a tagged release (for example `v1.0.0`) matching the submitted manuscript.
- [ ] Optionally archive the GitHub release in Zenodo to obtain a DOI.
