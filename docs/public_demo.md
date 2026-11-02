# Public P-wave Demonstration

From the repository root after installation:

```sh
python examples/public_waveform.py --verify --output runs/public_demo.json
```

No network connection or optional waveform library is needed for this command.
It uses the primary frozen C0 model and replays 240 seconds from NC.KSM.03 during
NCSN event 75079016 (26 October 2024), as 480 non-overlapping 0.5-second packets.
The record is the existing high-SNR Figure 2 illustration. It was not selected by
running a new performance search, and this demonstration is not a test-set score.

## Expected Output

The complete JSON is `examples/data/ncsn_75079016_KSM/expected_output.json`.
The expected counts are 480 valid packets, 461 candidate-allowed packets after
startup gating, and one candidate. Times are seconds from the supplied waveform's
start, not Unix timestamps:

| Quantity | Seconds |
|---|---:|
| Catalogue reference P arrival | 80.020000 |
| Estimated P arrival | 80.257998 |
| Result availability | 81.000000 |
| Signed picking error | +0.237998 |
| Output delay after reference P | 0.980000 |

Verification uses exact counts and a 0.00001-second absolute tolerance for numeric
pick fields. The output delay excludes telemetry and execution time; it is not
the demo's wall-clock runtime. See `docs/installation.md` for measured execution
and installation times.

## Input Preparation

The included `native.mseed` contains only this public three-component station
record. `metadata.json` records the original archive hash, public provider,
request URL, channel orientations and gains, reference pick, array hash, and
frozen preprocessing-code hash. `waveform.npy` has shape `(3, 24000)`, in ZNE order
at 100 Hz. It uses the archived adapter's gain-equalized count-like acceleration:
native counts divided by each channel gain and multiplied by their geometric
mean gain. It is not SI acceleration or a display-normalized Figure 2 trace.
All three channels in this example share the same gain. The model performs its
own causal normalization. Do not silently substitute another amplitude convention
when reproducing this example.

To independently rebuild the prepared array from the included MiniSEED:

```sh
python -m pip install -e '.[waveforms]'
python examples/rebuild_public_fixture.py --output runs/rebuilt_waveform.npy
```

The rebuild checks the raw and preprocessing-code hashes, recomputes orientation
and preparation using the unchanged adapter, and verifies every array sample.
It does not overwrite the distributed fixture or access restricted data.
For other data, obtain the correct orientation, response/gain and timing metadata
from the provider and follow `DATA_ACCESS.md` and `docs/streaming.md`. The inference
CLI itself does not rotate, resample or calibrate input arrays.

## Data Terms

This example retains the [NCEDC public-data terms](https://registry.opendata.aws/northern-california-earthquakes/),
not the repository's MIT software licence. Cite [NCEDC, doi:10.7932/NCEDC](https://ncedc.org/ncedc_doi_metadata.html)
and the [NC network, doi:10.7914/SN/NC](https://www.fdsn.org/networks/detail/NC/).
See the fixture's `DATA_LICENSE.md`. No Chinese evaluation data are included.
