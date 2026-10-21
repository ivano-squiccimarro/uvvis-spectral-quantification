# Private development and public-release gate

The owner requested private development first. Passing software tests does not
authorize making this repository public. Public release requires a new explicit
owner decision after the items below have been reviewed.

## Completed locally

- [x] Standalone package without imports from the thesis package.
- [x] Measured spectra, explicit metadata, source hashes and units.
- [x] Agreement with the existing producer after export/reload.
- [x] Unit/integration tests and full measured-case execution.
- [x] Signed matrix-blank reporting without concentration clipping.
- [x] Concentration holdout and additional joint-parent holdout.
- [x] Simple baselines, reproducible metrics and diagnostic figures.
- [x] No thesis documents, credentials or inherited Git history in the package.

## Before public release

- [ ] Author and collaborator approval for every included dataset and code asset.
- [ ] Agree on code license, data terms and attribution.
- [ ] Verify a clean installation on the documented Python environments.
- [ ] Confirm the remote GitHub Actions run completes successfully.
- [ ] Review dependencies and serialization across supported versions.
- [ ] Review remaining matrix bias, calibration range and validation wording.
- [ ] Decide whether additional measured external reference samples are needed
      for any stronger accuracy claim. Do not manufacture such a benchmark.
- [ ] Inspect the entire proposed public Git history and files for private details.
- [ ] Explicit owner authorization to change visibility to public.

## Optional subsequent case studies

Nanoprecipitation single-agent and dual-analyte models can be added after their
own export, unit, metadata and parity checks. Their preprocessing differs and must
not be silently replaced with the first case's settings. A profile pin or personal
README change is not part of this repository's current scope.
