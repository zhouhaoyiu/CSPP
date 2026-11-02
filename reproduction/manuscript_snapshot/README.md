# Experiment Source Snapshot

These are the archived manuscript-related experiment entrypoints and their local
dependencies. See `../../docs/manuscript_coverage.md` for scope and prerequisites,
and `../../provenance/manuscript_source_exports.json` for file hashes and entrypoints.

No raw waveforms, private labels, station inventories or Chinese record-level
results accompany these sources. Paths beginning with `workspace`, `external_data`,
`external_home` or `external_volume` are sanitized roles, not download locations.
Do not execute scripts against original experiment folders. Use isolated outputs,
review the CLI and provide only data you are authorized to access.

Python source compilation verifies syntax, not full experiment reproducibility.
MATLAB, GMT/PyGMT and their external data dependencies are not run by the Python CI.
