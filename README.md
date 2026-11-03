# CSPP

**Causal streaming P-wave detection and arrival-time picking for earthquake early warning**

CSPP processes three-component strong-motion data as non-overlapping 0.5-second
packets. Each packet is encoded once. Physical descriptors and a recurrent state
retain information from received packets, and separate detection and confirmation
heads produce station-level P-wave candidates and arrival estimates.

This repository contains the frozen C0 models, a Python package, packet and array
inference examples, training-stage sources, evaluation protocols, and saved results
for metric recalculation. Version 0.2.0 adds a public packet API without changing
the released model tensors or historical results. The 3 November 2026 submission
supplement adds verified public figure inputs and one public NCSN waveform example.
Full raw waveform datasets and third-party baseline weights are not included.

## Installation

Use Python 3.12 for the tested environment. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python tools/verify_release.py
```

On Windows, activate with `.venv\Scripts\activate` instead. Installation may need
internet access for dependencies. The examples below run offline after installation.
No PyPI publication or public repository URL is claimed by this local release.
See [installation](docs/installation.md) for optional waveform and training dependencies.

## Quick Example

```sh
cspp-inspect --checkpoint models/cspp_seed20261101.pt
python examples/quickstart.py --checkpoint models/cspp_seed20261101.pt
```

The quickstart generates synthetic noise to exercise the interface. For an actual
P-wave candidate from the existing Figure 2 NCSN record, run:

```sh
python examples/public_waveform.py --verify --output runs/public_demo.json
python tools/verify_source_data.py
```

The real-waveform example produces one candidate with a 0.980 s output delay and
approximately +0.238 s picking error. This selected illustration is not a new
accuracy estimate. [Waveform preparation, expected output and data terms](docs/public_demo.md).

To process already prepared 100 Hz ZNE acceleration data:

```sh
cspp-predict --input /path/to/prepared.npy \
  --checkpoint models/cspp_seed20261101.pt --output runs/picks.json
```

The array must have shape `(3, N)`, with positive `N` divisible by 50. No component
rotation, resampling or instrument correction is performed by this command.

## Real-Time Packets

```python
import numpy as np
from cspp import StreamingPicker

picker = StreamingPicker("models/cspp_seed20261101.pt")
rng = np.random.default_rng(42)
for index in range(40):
    packet = rng.normal(0, 0.01, (3, 50)).astype(np.float32)
    result = picker.process_packet(packet, timestamp_seconds=index * 0.5)
    if result.pick is not None:
        print(result.pick)
```

Use one picker per station. Timestamps identify each packet's first sample; actual
data availability is its end. The default thresholds are `q=0.1`, `e=0.975`.
The release requires 20 consecutive valid packets at startup and after invalid
data, equivalent to 10 seconds at 100 Hz. This is initialization, not a wait
after every P arrival. [Streaming API and output fields](docs/streaming.md).

## Models and Evaluation

| Checkpoint | Role |
|---|---|
| `models/cspp_seed20261101.pt` | Primary frozen C0 model |
| `models/cspp_seed20261102.pt` | Replication seed |
| `models/cspp_seed20261103.pt` | Replication seed |

All three models have 329,423 parameters and a 320-scalar neural state. Checkpoint
seeds and settings are in [the model inventory](models/model_manifest.json). The later expanded/full
models are not interchangeable with these manuscript checkpoints.

Saved primary-model event-window results:

| Network | Records | Precision | Recall | F1 | Matched median output delay |
|---|---:|---:|---:|---:|---:|
| NCSN | 578 | 80.74% | 86.33% | 83.44% | 0.610 s |
| TexNet | 633 | 93.52% | 88.94% | 91.17% | 0.745 s |

Sources: [NCSN](evidence/summaries/nc.json), [TexNet](evidence/summaries/tx.json).
Event-window precision is not a continuous false-alarm rate. Delays summarize
matched records and do not include telemetry or computation. The release also
retains misses, threshold sensitivity and results unfavorable to CSPP.

```sh
python -m pytest -q tests
python tools/reproduce_metrics.py --output runs/recomputed_metrics.json
python tools/train_stage.py
```

The last command reports training prerequisites without starting training.
[Reproduction scope](docs/reproduction.md) and [manuscript coverage](docs/manuscript_coverage.md)
distinguish saved-result recalculation, waveform replay, and training. Full raw-data
reproduction is not claimed by passing the software tests.

## Data Access

Except for the included single NCSN example, public-source waveforms must be
obtained from their providers under the original terms. [Data access](DATA_ACCESS.md)
lists source services and preparation contracts. Numerical public figure inputs
are in `evidence/source_data/`; its coverage file explicitly identifies restricted
Chinese inputs and non-data schematics.

**The Chinese evaluation dataset is not distributed.** Its provider prohibits
redistribution and institutional management requirements prevent external access,
including access by editors and peer reviewers. No request-based access route is
available. Consequently, the Chinese-data results cannot be independently reproduced
from this repository. No Chinese waveforms, record-level labels, station lists or
derived record-level caches are included.

## Repository Layout

| Location | Contents |
|---|---|
| `cspp/` | Model, streaming API, preprocessing helpers and evaluation primitives |
| `models/` | Frozen weights, seeds and settings |
| `examples/` | Synthetic API example and real public NCSN P-wave example |
| `docs/` | Installation, API, reproduction and release guidance |
| `tests/` | Causality, parity, reset, checkpoint and metric tests |
| `tools/` | Prediction, verification, metric recalculation and training preflight |
| `evidence/` | Public-source saved decisions, figure source data and historical summaries |
| `protocols/` | Frozen evaluation definitions and data provenance |
| `reproduction/` | Training configurations, selected indices and historical code |
| `provenance/` | Source inventories and validation records |

## Citation and Contact

Manuscript authors: Haoyu Zhou, Qiang Ma, Shiliang Zhang and Jiang Wang.
Correspondence: Shiliang Zhang, [zsl@iem.ac.cn](mailto:zsl@iem.ac.cn),
[ORCID 0000-0002-3151-1909](https://orcid.org/0000-0002-3151-1909).
The manuscript has no publication DOI in this repository. Software citation
metadata are in [CITATION.cff](CITATION.cff); the inherited software-author list is
distinct from the manuscript-author list.

Repository organization follows the installation, examples, documentation, tests,
model distribution and citation sections of
[EQTransformer](https://github.com/smousavi05/EQTransformer). No EQTransformer
implementation or pretrained weights were copied into CSPP.

## License and Contributions

Self-owned code and the included self-trained weights retain the existing
[MIT license](LICENSE). See [model licensing](MODEL_LICENSE.md) and
[third-party notices](THIRD_PARTY_NOTICES.md) for exclusions. Dataset metadata do
not acquire an MIT license through inclusion in this repository.

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests and bug reports. Do not attach
restricted waveforms or private station metadata to issues or pull requests.
