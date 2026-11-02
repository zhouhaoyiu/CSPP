# Release Notes

## 0.2.0 Submission Supplement - 2026-11-03

- Added exact archived public numerical inputs for Figures 5, S1, S2 and S5,
  completing the previously missing public figure-source inputs.
- Added one real NCSN P-wave demo, expected output, MiniSEED preparation check
  and provider-specific terms; no Chinese record-level inputs were added.
- Added three tests, giving 35 passing local tests, plus source-data verification.
- Measured a fresh environment installation with dependency downloads and three
  demo executions. See `docs/installation.md` for conditions and timings.
- No runtime API, model tensor, threshold or saved scientific result changed.
  The software version remains 0.2.0; the supplement is identified by date.

## 0.2.0 - 2026-11-02

- Added the CPU `StreamingPicker` packet API and installed inference CLI.
- Added an offline synthetic example, interface documentation and API tests.
- Retained the three C0 models, initializer, frozen thresholds and historical results.
- Added explicit Chinese-data access restrictions, including unavailable reviewer access.
- Prepared separate repository and submission deliverables; no remote publication.

## 0.1.0 - Inherited Scientific Payload

- Current standalone `cspp` runtime and three frozen C0 checkpoints.
- 21 historical training dependency files isolated under `reproduction/training_snapshot`.
- Original sources and external artifacts are not modified.
- Private filesystem prefixes removed from exported metadata. Numeric JSON values and tensor values preserved.
- Manuscript builds and their unrelated legacy tests are outside this release.
- No new model training, threshold selection, or scientific accuracy estimate was performed while packaging.
