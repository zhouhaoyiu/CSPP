"""Rebuild the public example from its MiniSEED without changing the fixture."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from obspy import UTCDateTime

from cspp.io.orientation import ChannelOrientation, stream_to_oriented_gappy_zne_array

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/data/ncsn_75079016_KSM"


def rebuild():
    metadata = json.loads((DATA / "metadata.json").read_text())
    native = DATA / "native.mseed"
    assert hashlib.sha256(native.read_bytes()).hexdigest() == metadata["native_mseed_sha256"]
    assert hashlib.sha256((ROOT / "cspp/io/orientation.py").read_bytes()).hexdigest() == metadata["preprocessing_code_sha256"]
    preparation = metadata["preprocessing"]
    orientations = [ChannelOrientation(**row) for row in preparation["channel_orientations"]]
    station = {key: preparation[key] for key in ("network", "station", "location")}
    wave, _ = stream_to_oriented_gappy_zne_array(native, station, orientations,
        UTCDateTime(metadata["start_utc"]), UTCDateTime(metadata["end_utc_exclusive"]))
    np.testing.assert_array_equal(wave, np.load(DATA / "waveform.npy", allow_pickle=False))
    return wave


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/rebuilt_waveform.npy")
    args = parser.parse_args()
    if args.output.resolve() in {p.resolve() for p in DATA.iterdir()}:
        parser.error("Output must not overwrite a distributed fixture.")
    wave = rebuild()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, wave, allow_pickle=False)
    print(json.dumps({"array_equal": True, "shape": list(wave.shape), "output": str(args.output)}))
