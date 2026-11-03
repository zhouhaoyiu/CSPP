# Streaming Interface

`StreamingPicker(checkpoint, candidate_threshold=0.1, event_threshold=0.975,
warmup_packets=20, torch_threads=1)` wraps the frozen C0 model and existing
normalization, quality-gate and candidate-episode rules. The default model is
selected by an explicit path, not by downloading the newest checkpoint.

## Inputs

- One instance represents one station and one uninterrupted sequence of calls.
- Supply complete real numeric arrays of shape `(3, 50)`, ordered Z, N, E, at 100 Hz.
- Use acceleration with consistent component units and orientation. Do not label
  instrument counts as acceleration. Apply the experiment's response and orientation
  treatment before calling the API.
- `timestamp_seconds` is the first-sample time, either relative seconds or Unix
  epoch seconds. Do not switch clocks within a stream.
- The API reads only the supplied packet and previous state. A packet containing
  samples after the P onset is usable once those samples have actually arrived.

Bad shape, non-real input or a nonfinite timestamp raises `ValueError` without
advancing state. Nonfinite samples, constant components, duplicate packets or
timestamp discontinuities are handled by the frozen quality gate: the packet is
discarded and normalization, recurrent state and candidate history reset. The
timestamp tolerance is 0.05 s. Gap packets are not interpolated or carried into
the next segment. After a reset, 20 consecutive valid packets are required before
candidate output is eligible; probabilities are still calculated during warmup.

## Output

`process_packet` returns a `PacketResult`. Call `.to_dict()` for JSON-compatible
fields. Invalid packets have `None` probabilities, not NaN values.

| Field | Meaning |
|---|---|
| `packet_index` | Zero-based count of submitted packets since explicit reset |
| `packet_start_seconds` | Supplied first-sample timestamp |
| `available_seconds` | Packet-end time on the input clock |
| `quality_valid`, `reason_codes`, `quality_flags` | Quality decision and diagnostics |
| `candidate_allowed`, `warmup_remaining` | Startup/reset output eligibility |
| `candidate_probability`, `event_probability` | Detection and confirmation scores |
| `arrival_age_fraction` | Predicted look-back fraction, in [0, 1] |
| `pick` | New candidate episode, or `None` |

A pick has `episode_index`, `episode_start_packet`, `available_epoch`, and
`estimated_p_epoch`. The legacy `epoch` names use the same clock as the supplied
timestamps; they are Unix times only when the input uses Unix times. The P estimate
is at most 2 s before availability. A continuing candidate episode is not emitted
again on every packet. Scores are model outputs, not calibrated probabilities of
an earthquake or network alerts.

`reset()` starts an independent stream, including packet and episode numbering.
`iter_array(wave, start_seconds=0)` replays an array without resetting an existing
stream; when chaining arrays, supply the actual start time of each array. Partial
packets are rejected rather than padded. The API retains fixed-size neural,
quality, normalization and candidate states, not all preceding waveforms.

## Nondefault Settings

`warmup_packets=0` explicitly disables startup gating. It changes output eligibility,
not the model, and is not the manuscript's default setting. Threshold overrides
must be reported with evaluation results. The CLI writes thresholds, warmup,
checkpoint filename and clock convention into its output.

Packet-end availability is not wall-clock delivery time. Telemetry, preprocessing,
queueing, CPU execution and network confirmation are outside that timestamp.
