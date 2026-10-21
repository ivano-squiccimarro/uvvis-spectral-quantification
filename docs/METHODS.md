# Method and provenance

## Extraction boundary

The measured case was exported read-only from two existing thesis tables: the
talazoparib/double-emulsion calibration-and-background matrix and the cleaned
supernatant spectral table. Exact relative paths and hashes are in the manifest.
The producer is `scripts/build_talazoparib_de_ee.py` in that private workspace.
No thesis manuscript, laboratory correspondence, unrelated assay, credentials
or repository history is imported.

Calibration/background columns follow the same explicit selector as the source
producer. All 68 columns in the observation table are retained. Four inherited
curation flags are documented, not applied as new exclusions. Nominal references
for calibration spectra are parsed once during export and then stored explicitly.

## Processing and calibration

Spectra are sorted by wavelength. The 780-800-nm median is subtracted per measured
parent, then the 270-380-nm band is selected. Calibration and scattering parents
are added at the five fixed scattering amplitudes. A nine-point second derivative
is applied before fitting a two-component, unscaled PLS regression. No additional
Gaussian noise is injected in this chromophore case; varying measured scattering
amplitudes is the inherited augmentation mechanism.

The spectral window and component count are fixed from the existing model. They
were developed using prior knowledge of these data: the repository does not
present its diagnostic folds as a prospective external assay validation.

The standalone package rejects prediction grids different from its fitted grid,
rather than applying the original producer's interpolation. The bundled measured
case has matching grids, and its predictions agree with the producer within
1e-8 uM after CSV export/reload. Signed outputs, including negative blanks, remain
unmodified. See `reports/producer_parity.csv` and the manifest's maximum error.

## Evaluation

Each concentration fold is split at the measured calibration-parent level before
augmentation. A second diagnostic also partitions the measured background library
into three stable folds, excluding both parent types. The same held-out parent
appears under several physically specified scattering amplitudes; these are
correlated descendants, not independent experimental replicates.

Metrics pool squared errors across the augmented prediction rows, overall and
separately for extrapolation (lowest or highest level held out) and
interpolation (any other level held out). The plotted
bars show the dispersion of descendant predictions, not confidence intervals.
No fold-based statistical superiority claim is made. The two baselines are the
training mean and an OLS regression on one derivative wavelength selected by
training-only correlation. They are transparent comparators, not an exhaustive
model search.

The nonblank supernatants are unknown residuals and have no positive-concentration
ground truth in this case. Blank mean and sample SD provide an observed matrix
bias diagnostic. A negative blank mean must not be reinterpreted as zero error
or used to force apparent encapsulation to 100%.

## Reproducibility boundary

Software tests cover axis order, nonfinite/duplicate values, preprocessing,
metadata mappings, split ancestry, signed predictions, serialization and real
producer parity. Test fixtures are generated mathematical examples and are never
reported as laboratory measurements. The current empirical results describe this
matrix and calibration range, not all drugs, solvents or nanoparticle formulations.
