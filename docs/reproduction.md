# Archived Reproduction Scope (0.1.0)

This document preserves the scientific scope of the inherited reproduction
payload. For the current interface and validation see the root README and
VALIDATION.md. Paths below are relative to the repository root. The statements
about prior validation are historical, not new waveform or training runs.

The 3 November 2026 supplement adds one public NCSN waveform demonstration and
verified public figure inputs. Its scope and commands are in `docs/public_demo.md`
and `docs/manuscript_coverage.md`; the historical no-waveform statement below
describes the inherited 0.1.0 payload, not this added public example.

Current CSPP C0 (`verifier_feature_mode=none`), release 0.1.0, 2026-09-07.
The primary checkpoint is seed 20261101. Seeds 20261102 and 20261103 are replication checkpoints.
The frozen default operating point is q=0.1, e=0.975. The secondary higher-recall point is q=0.1, e=0.9.
The archive does not redefine either point using STEAD labels.

## Quick Start

Use Python 3.12 in a fresh environment. Run these commands in the extracted package directory:

```sh
python -m venv .venv
# Activate .venv using the command appropriate for your shell.
python -m pip install -e '.[test]'
python tools/verify_release.py
python -m cspp --checkpoint models/cspp_seed20261101.pt
python -m pytest -q tests
python tools/reproduce_metrics.py --output runs/recomputed_metrics.json
```

Installation may require internet access to obtain third-party dependencies. After installation, all commands above work offline.
`requirements-tested.txt` records the directly used dependency versions in the validation environment; it is not a cross-platform lockfile.

## Contents

| Directory | Contents |
|---|---|
| `cspp/` | Current model, packet processing, evaluation, and association modules |
| `models/` | Three tensor-identical exports of the frozen C0 checkpoints |
| `tests/` | Independent CPU contract tests and a synthetic regression fixture |
| `tools/` | Integrity check, row-level reaggregation, array inference, and optional STEAD replay |
| `evidence/event_windows/` | TexNet, Puerto Rico, original/extended NCSN, and GeoNet decisions |
| `evidence/continuous/` | Eight original/expanded-model conditions, including misses and unmatched groups |
| `evidence/stead/` | Full-test diagnostics and the 5,981-record exploratory comparison |
| `evidence/summaries/` | 32 source summaries, including negative results and ablations |
| `protocols/` | Frozen selection, baseline, evaluation, and data provenance records |
| `reproduction/` | Training configurations, selected indices, initializer, and historical dependency snapshot |
| `provenance/` | Source/export hashes, tensor checks, environment, and source reorganization audit |

## What Can Be Reproduced

1. File integrity, checkpoint loading, packet causality, reset behavior, and regression outputs can be checked offline.
2. Saved-row TP/FP/FN, precision/recall/F1, timing metrics, and continuous-stream association counts can be recalculated offline. The script compares them with the recorded summaries and fails on disagreement.
3. Waveform inference requires separately obtained data. A CPU STEAD replay command is provided in `DATA_ACCESS.md`; its validation covers a small real-waveform subset, not a fresh rerun of every experiment.
4. Training provenance and a dependency-complete historical trainer are included. End-to-end data preparation and retraining were not rerun for this release. Prepared training stores and the original base-core training lineage remain prerequisites.

The synthetic fixture is an engineering regression test, not an accuracy experiment. Reaggregating saved decisions does not independently validate their reference labels, preprocessing, waveform availability, or model predictions. Continuous station-day exposure is read from the saved audit; network-day exposure is recalculated from per-date rows.

## Scientific Scope

Event-window precision is conditional on the supplied event windows; it is not a continuous-noise false-alarm rate. Each matched event-window record contributes at most one TP. Additional candidate episodes contribute FP. Timing distributions use matched records; availability-limited recall uses all eligible records.

STEAD's 5,981-record comparison contains original train/dev/test splits, fixed P labels at 10 s, raw counts, and cold starts. Full-cohort threshold selection is exploratory. Dev-selected thresholds are also evaluated on the original 600 test records; event/station isolation has not been established. Original EQTransformer used STEAD for training. Window-fill latency is separate from compute time.

The expanded-training continuous replay is retained even when it performs worse. `evidence/summaries/` also contains historical diagnostic results whose complete waveform reruns are outside this release's validation scope. A hash or passing software test does not establish scientific generalization or journal readiness.

No raw waveforms, third-party pretrained weights, manuscript, private notes, local environment, or submission metadata are distributed. The package is a code/model/evidence supplement; manuscript consistency and venue-specific submission requirements need a separate check.

## Licensing

Self-owned code and the distributed self-trained model weights are provided under MIT, as explicitly authorized for this release. See `LICENSE`, `MODEL_LICENSE.md`, and `THIRD_PARTY_NOTICES.md`. Third-party data and software keep their own terms.
