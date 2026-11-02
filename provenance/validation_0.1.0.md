# Release Validation

Validation date: 2026-09-07. Device: single-thread CPU, macOS ARM64, Python 3.12. The installed versions are recorded in `provenance/environment.json`.

- Both editions were extracted from ZIP into separate directories with spaces in their paths. Each passed the 15-test portable suite.
- All distributed file hashes were checked against the manifest. Archives were checked for CRC errors and unintended raw waveforms, caches, hidden macOS sidecar files, and private path prefixes.
- Three final checkpoints and one candidate-core initializer were exported without training metadata. Every tensor was checked for exact equality with the source checkpoint; file hashes differ because the serialized metadata changed.
- The named and anonymous scientific payloads are byte-identical. Only `LICENSE`, `CITATION.cff`, `EDITION.md`, and the resulting manifest differ.
- Saved-row calculations matched the recorded metrics for five event-window cohorts (three seeds), 18 selected STEAD comparisons, two STEAD diagnostic cohorts, and eight continuous-replay conditions. This validation does not independently reconstruct waveform labels, preprocessing, or station-day exposure.
- The portable CPU STEAD command was run on the first 30 records of the saved cohort. Candidate counts and hit/miss decisions matched the saved default-point results. The maximum absolute timing difference was approximately 1.7e-6 seconds. This small subset is an engineering regression check, not a new accuracy estimate.
- The offline wheel build contains exactly the 30 current `cspp` Python modules. The historical training CLI imports and displays help. No full training run was performed.

The GitHub Actions workflow is supplied for future runs; it was not executed on GitHub during this local release. Cross-platform numerical behavior has not been validated here. Complete waveform reruns, dataset reacquisition, training from scratch, and manuscript/submission checks remain outside this validation.
