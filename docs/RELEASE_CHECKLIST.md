# Preliminary public release and follow-up checklist

## Release decision — 21 October 2026

The repository owner explicitly confirmed permission to publish the included
code and experimental data and approved MIT for code/documentation and CC BY 4.0
for the bundled dataset. This records the owner's confirmation, not an
independent legal determination of institutional rights.

## Checks completed for this preliminary release

- [x] Standalone package without imports from the thesis workspace.
- [x] Measured spectra, source hashes, units and metadata retained.
- [x] Source-producer parity and measured-case execution checked.
- [x] 28 unit/integration tests passed on the reviewed source.
- [x] GitHub Actions passed on the D1-D5 audit commit.
- [x] Input correspondence, duplicate headers, required hashes and units checked.
- [x] Interpolation and extrapolation errors reported separately.
- [x] Signed matrix bias and non-independent development diagnostics disclosed.
- [x] Tracked egg-info build artifacts removed from the current version.
- [x] Initial isolated project history reviewed; no inherited thesis history.
- [x] Owner confirmed publication rights and approved the code/data licenses.
- [x] Explicit owner authorization to make this repository public.

## Scope of this release

This is preliminary research software, not a finished or independently validated
assay. Passing software tests does not establish clinical suitability or accuracy
on new matrices. Source data and numerical claims were not changed for release.

## Continuing work — not represented as completed

- [ ] Independently measured positive residual-concentration references.
- [ ] Evaluation on additional matrices and independent preparations.
- [ ] Broader dependency and serialization compatibility testing; load only
      trusted joblib files.
- [ ] Additional case studies, each with its own export, unit and parity checks.

Other matrices and dual-analyte models are outside this release. Any stronger
accuracy or generalization claim requires additional evidence.
