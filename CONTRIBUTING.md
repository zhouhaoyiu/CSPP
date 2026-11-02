# Contributing

Use a branch and include a focused test for behavior changes. Run:

```sh
python -m pytest -q tests
python tools/reproduce_metrics.py --output runs/recomputed_metrics.json
```

Do not overwrite frozen checkpoints, historical protocols or saved scientific
results in a maintenance change. New models and operating points need distinct
versions and a documented evaluation protocol. Keep unfavorable results and
denominators in evaluation exports.

For bug reports, include the package version, Python/PyTorch versions, operating
system, checkpoint SHA-256, settings and a minimal synthetic reproducer. Never
upload restricted data, credentials or private station information. The issue
template is usable after a remote repository is created; no remote is configured
by this local package.

The release manifest is immutable evidence for a distributed archive. Maintainers
regenerate it for a new release only after reviewing changes, using
`python tools/build_manifest.py`. A changed development tree is expected to differ
from the previous release's checksums.
