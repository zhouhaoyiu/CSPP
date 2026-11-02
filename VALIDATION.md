# Release Validation

## Submission Supplement, 2026-11-03

All 35 tests passed locally with Python 3.12.14 on macOS 27.2 ARM64. This includes
the existing 32 tests, a real NCSN waveform regression check, rejection of an
incorrect demo count, and public figure-source verification. The runtime package,
checkpoints, original scientific evidence and frozen thresholds are unchanged.

The new public inputs reproduce Figure 5/S1/S2 cohort counts and means and Figure
S5 exposure aggregation. Training selections match the frozen indices exactly.
The supplied NCSN MiniSEED independently rebuilds all 72000 component samples of
the prepared demo. No new test-set accuracy estimate is inferred from that single
existing Figure 2 case. Full raw-dataset and training reproduction remain outside
this validation.

Fresh installation, including network downloads into an empty dedicated cache,
took 55.635 seconds with uv. The three real-demo executions took 1.544, 0.903 and
0.905 seconds after a separate first-import check. Details and scope are in
`docs/installation.md` and `provenance/submission_software_validation.json`.
The release allows only the two explicitly hash-approved public waveform files;
all other unapproved raw-data files still fail the distribution audit.

## Original 0.2.0 Validation, 2026-11-02

Version 0.2.0 was checked locally on 2026-11-02 using macOS ARM64 and
Python 3.12. Dependency versions and machine-readable outcomes are recorded in
`provenance/repository_validation.json`.

- All 32 tests passed, including packet/array output parity, timestamp gaps,
  invalid-packet resets, warmup, bounded state, CLI behavior and archive inventory.
- Saved-row statistics matched the retained event-window, STEAD and continuous
  replay results. This checks aggregation, not the correctness of reference labels
  or a new waveform inference run.
- All 172 distributed Python source files passed syntax checks. Relative links in
  the README and documentation resolved locally.
- The offline wheel contains 32 runtime Python modules. An installation outside
  the repository loaded the frozen checkpoint and processed synthetic packets.
  The 80-packet example had 80 valid packets and 61 eligible packets after warmup.
- All 280 files in the original release still match their original manifest.
  Scientific model weights, stored evidence, frozen protocols and upstream
  licenses were not changed.
- Chinese waveforms, record-level labels, station inventories and record-level
  results are excluded. Original scientific provenance and third-party attribution
  are retained. The release verifier checks hashes, exact inventory, symlinks,
  raw-data extensions, private paths and suspected credentials.

No training, raw-waveform dataset reacquisition, full baseline rerun, MATLAB or
figure rerun was performed for this release. GitHub Actions and cross-platform
numerical behavior were not tested. Passing these checks does not establish full
manuscript reproduction or editorial acceptance of the Chinese-data restriction.

The earlier validation record is preserved separately in
`provenance/validation_0.1.0.md`. It describes the earlier release, not additional
experiments performed during this packaging task.
