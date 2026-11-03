# Data Access and Waveform Reproduction

## Restricted Chinese Evaluation Data

The Chinese evaluation dataset is not publicly available because the data provider
prohibits redistribution and institutional management requirements restrict
external access. These restrictions also prevent providing the data to editors
or peer reviewers. No request-based access route is available. Chinese waveforms,
record-level labels, station inventories and derived record-level caches are not
included. Any code retained for that evaluation does not grant access to its data.
The corresponding Chinese-data results cannot be independently reproduced from
this repository. These restrictions must be disclosed at journal submission.

## Public-Source Data

Raw data are not bundled. `protocols/` and per-experiment evidence retain event/station IDs, windows, split definitions, source identities, preprocessing settings, and exclusions when present in the original artifacts. Sanitized `workspace/`, `work_artifacts/`, and `external_data/` strings identify original artifact roles; they are not installed paths. Original experiment protocols retain the input identities needed by the historical code.

Official source entry points:

- STEAD: https://github.com/smousavi05/STEAD . Retain dataset attribution and original data terms.
- SeisBench: https://github.com/seisbench/seisbench . The packaged cohort uses its bucketed STEAD representation, ZNE, 100 Hz.
- NCSN/NCEDC: https://service.ncedc.org/ . Event, station, and waveform FDSN services.
- GeoNet: https://www.geonet.org.nz/data/access/FDSN . Event and waveform identifiers are in the GeoNet records.
- K-NET/KiK-net: https://www.kyoshin.bosai.go.jp/ . Obtain access under NIED's current terms.
- InstanceGM: follow the source references and input hashes in the training protocols; obtain the same prepared acceleration subset.
- CWA and TexNet/Puerto Rico: use the provider/FDSN service metadata in the archived protocols. Service availability and current access conditions may change.

## Array Inference

Supply a NumPy float32 array of shape `(3, N)` in ZNE order, at 100 Hz, with N divisible by 50. Preserve physical units consistently within each stream. The model was trained for strong-motion acceleration; the STEAD raw-count test is an out-of-domain diagnostic. Do not silently relabel counts as acceleration. Resampling, instrument correction, and component rotation must match the relevant protocol and are not performed by this command.

```sh
python tools/predict.py --input /path/to/prepared_zne_100hz.npy --output runs/picks.json
```

## STEAD Waveform Replay

Obtain the SeisBench STEAD `waveforms.hdf5` corresponding to `cohort.csv`; original trace names are also retained for cross-checking. Bucket layout depends on the dataset preparation version. A differently repacked HDF5 must be mapped to the original trace identities before use.

```sh
python -m pip install -e '.[datasets]'
python tools/replay_stead.py --waveforms /path/to/stead/waveforms.hdf5 --output-dir runs/stead_smoke --limit 3
python tools/replay_stead.py --waveforms /path/to/stead/waveforms.hdf5 --output-dir runs/stead_default
```

The default replay uses q=0.1/e=0.975 and the saved 5,981-record cohort. Match tolerance is +/-1.5 s; choose the smallest absolute pick error, break ties by earliest availability, and count at most one TP per record. Use `--split test` for the original test subset. Overrides are explicit CLI arguments and must be reported. PhaseNet/EQTransformer weights are not redistributed; their original protocols identify the models for separate retrieval. Their waveform reruns and the full threshold-grid scan are not covered by this portable command.

## Training

`reproduction/training/seed*/` contains each final training configuration, history, protocol, and selected row indices. The shared pretrained candidate-core initializer is in `reproduction/initializer.pt`. Its tensors are unchanged by metadata sanitization. This enables reconstruction of the final verifier-training stage once all prepared stores are supplied; it does not reproduce the initializer from scratch.

`reproduction/training_snapshot/` is the current on-disk historical trainer and its recursively resolved local imports. It retains historical version names to preserve the training implementation; new inference uses `cspp/`. This snapshot was not asserted to be the exact July training-time source because no complete immutable source snapshot is established here. The source inventories identify what is actually supplied.

Install the `training` optional dependencies before using the historical trainer. `python tools/train_stage.py` lists required external data keys without opening datasets or starting training. Put their local paths in a JSON object, then run:

```sh
python -m pip install -e '.[training]'
python tools/train_stage.py --paths /path/to/data_paths.json --output runs/training_seed20261101 --device cpu
python tools/train_stage.py --paths /path/to/data_paths.json --output runs/training_seed20261101 --device cpu --run
```

The first command after installation is a read-only preflight. Only `--run` starts training. SeisBench InstanceGM must already be available in its configured cache. The launcher restores the recorded configuration, remaps external paths, uses the exported initializer and selected indices, and retains the current historical trainer's defaults only for arguments absent from the recorded configuration. These defaults and the lack of an immutable original source snapshot prevent a claim of exact training reconstruction. The recorded runs used MPS; another backend need not yield bitwise-identical retraining.
