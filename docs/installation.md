# Installation

Install from the repository or extracted release archive:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python tools/verify_release.py
python -m pytest -q tests
```

The package requires Python >=3.10, NumPy and PyTorch. Local validation uses Python
3.12 on macOS ARM64; the included Linux CI configuration is not a report of a
completed GitHub run. `requirements-tested.txt` records the historical validation
versions, not a portable lockfile. The current validation environment is recorded
in `provenance/repository_validation.json` after validation.

Optional dependencies:

```sh
python -m pip install -e '.[waveforms]'
python -m pip install -e '.[baselines,datasets,training]'
```

These install libraries, not data or baseline checkpoints. The packet API is CPU
only and sets PyTorch's process-wide intra-op thread count to one by default.
Do not change this setting concurrently in separate station workers.

The Python wheel contains the runtime package only. Model weights, protocols,
tests, historical scripts and evidence remain in the repository/release ZIP.
Pass an explicit checkpoint path; loading never downloads a model automatically.
Do not run `pip install cspp` expecting this local release to be on PyPI.

For offline use, install dependencies first and retain a local wheel cache. Only
load checkpoints from a trusted source.

## Measured Installation and Demo Time

The 3 November 2026 validation used macOS 27.2 ARM64 on an Apple M4 Max with
36 GiB RAM, Python 3.12.14, NumPy 2.2.6, PyTorch 2.13.0 and pytest 9.1.1.
The CPU demo requires no GPU or non-standard hardware. Linux and Windows were
not independently tested in this supplement.

A new internal-disk virtual environment and an empty dedicated uv dependency
cache were used. The command equivalent to
`uv pip install -c requirements-tested.txt -e '.[test]'` took 55.635 seconds,
including dependency downloads, package build and installation. Environment
creation took 0.011 seconds. Allow about a minute under comparable conditions;
network speed, package cache, operating system and hardware affect the total.
This is an observed uv installation time, not a measured pip installation time.

`python examples/public_waveform.py --verify` took 1.544, 0.903 and 0.905 seconds
in three fresh processes, including process startup and model loading. A separate
first-import check before these runs took 6.518 seconds. Each run processed the
same 240-second example and matched its expected output. First-use initialization
can therefore take longer. These timings are not the manuscript's per-update
benchmark or an end-to-end field-warning latency. The sanitized receipt is
`provenance/submission_software_validation.json`.
