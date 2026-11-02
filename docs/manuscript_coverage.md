# Manuscript Reproduction Coverage

The repository targets the current CSPP C0 manuscript, not the later 5,000-record
NCSN study or expanded/full checkpoints. It contains the code and preserved
evidence that can be distributed, with the following boundaries.

| Manuscript component | Included material | External prerequisites or limits |
|---|---|---|
| Architecture and streaming inference | `cspp/`, three frozen C0 checkpoints, packet API, regression tests | Prepared three-component input; no telemetry adapter |
| Training | Configurations and selected indices for three seeds, candidate-core initializer, final-stage trainer and dependencies | Prepared training stores; initializer-from-scratch lineage not fully reconstructed |
| GeoNet threshold calibration | Frozen protocols, saved decisions and historical evaluation sources | Original provider waveforms and preprocessing metadata |
| NCSN and TexNet event records | Saved per-record decisions, summaries, selection/scoring protocols, evaluation and baseline code | Provider waveforms and independently obtained baseline weights |
| Continuous event/background replay | All eight preserved replay conditions, candidate and association rows, exposure audits, scoring and baseline sources | Provider station-day data; station-day exposure not rebuilt from raw data in this package |
| Computational efficiency | Original runtime summary, historical benchmark scripts | Original input record, hardware and dependency environment for matching reported timing |
| Ablations and normalization | Original summary/protocol files, training and diagnostic sources | Respective prepared stores and checkpoint lineage |
| Chinese instrument evaluation | Conversion, scoring, threshold-scan and baseline source code | Data, labels, station lists and record-level results excluded; no reviewer access |
| Figures | Historical plotting and MATLAB aggregation sources, public numerical figure inputs and a verifier in `evidence/source_data/` | Restricted Chinese figure inputs excluded; full waveform datasets remain external |

The 3 November 2026 supplement recovered the exact Figure 5 training records,
three GeoNet Figure S1 cohorts, TexNet Figure S2 coordinates/depths, and Figure S5
station-day exposure inputs. Membership, displayed means and exposure aggregation
were checked against the archived cohorts. `python tools/verify_source_data.py`
repeats the public-input checks. `python examples/public_waveform.py --verify`
replays the existing Figure 2 NCSN example and verifies a real P-wave output.

`python tools/reproduce_metrics.py` recalculates saved-row statistics. It does not
re-run neural inference, check reference-label accuracy, reacquire raw datasets,
or verify a full training trajectory. The new packet API is checked against the
frozen array implementation using synthetic inputs, including resets.

## Historical Source Snapshot

`reproduction/manuscript_snapshot/` preserves the experiment scripts and recursively
resolved local Python imports. Its inventory is
`provenance/manuscript_source_exports.json`. Original and exported source hashes
identify each file; private absolute path strings are sanitized. Scientific
calculations and attribution are not intentionally changed.

These scripts retain historical names, directory layouts and some original device
defaults. They document the experiments; they are not advertised as a one-command
end-to-end runner. Before running one, map the sanitized paths to authorized local
inputs, use a new output directory and check its frozen protocol. Do not bypass
input-hash checks merely to make a command run. The current `cspp` package should
be installed separately so imports resolve consistently.

Historical plotting defaults are preserved as source provenance. For new figures,
use Times New Roman for English/numbers and SimSun for Chinese, checking font
availability first. No historical plot was regenerated during repository assembly.

Some older diagnostic sources describe automatically reviewed reference intervals.
Those provenance statements are retained; they must not be relabeled as independent
human annotations. The historical diagnostic evidence is distinct from the
manuscript's NCSN/TexNet catalog references and the withheld Chinese dataset.

## Limits at Submission

The code repository does not remove the need to disclose data-access restrictions.
The Chinese evaluation cannot be independently rerun under the current prohibition
on access by readers, editors and reviewers. The authors must discuss this with the
editor at submission. A successful code test does not establish acceptance of that
restriction by Nature Communications.

There is no public repository URL or registered software DOI yet. Reviewer access
to the code can be provided by a release ZIP or a real repository/archive link once
created. Do not cite a placeholder or a reserved, non-resolving DOI as accessible.
