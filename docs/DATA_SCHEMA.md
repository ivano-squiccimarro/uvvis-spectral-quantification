# Measured data schema

Each case directory contains four CSVs and `manifest.json`. CSVs preserve the
measured numeric values; stable sample IDs replace unwieldy export-column names.
The original column name remains in metadata for traceability. Empty numeric
references are missing data, not zero.

## Wide spectra

`training_spectra.csv` and `observation_spectra.csv` contain `wavelength_nm` and
one absorbance column per measured spectrum. Wavelengths are in nm, absorbance
is in AU. Input order may be increasing or decreasing. Duplicate/nonfinite axes,
nonfinite absorbances and incomplete/non-1-nm calibration grids are rejected.

## Training metadata

| Field | Meaning |
| --- | --- |
| sample_id | Unique column identifier in training_spectra.csv |
| role | calibration or background |
| reference_uM | Known calibration level; missing for background parents |
| original_column | Original measured export label, unchanged |
| observation_unit | measured_spectrum; not a guaranteed independent synthesis |

## Observation metadata

| Field | Meaning |
| --- | --- |
| sample_id | Unique column identifier in observation_spectra.csv |
| role | blank or supernatant |
| reference_uM | Zero for drug-free blanks; missing for nonblank residuals |
| nominal_input_uM_at_1mL | Original input nominal defined at 1 mL, not a residual reference |
| original_column | Original measured export label, unchanged |
| source_curated_exclusion | Existing source-producer curation flag; records remain included |
| observation_unit | measured_spectrum |

## Manifest and lineage

The manifest records original source paths relative to the private thesis
workspace and SHA256s before export, plus hashes of each exported table. It
contains no local absolute paths or credentials. `augmentation_lineage.csv`
is a generated output containing calibration_id, background_id, the scattering
amplitude and the concentration label for every augmented descendant.

Labels and group identity are data inputs. This package does not infer biological
replication or silently repair unexpected labels. The code uses explicit metadata
rather than requiring each future dataset to follow the thesis naming convention.
