import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from cspp import StreamingPicker
from tools.predict import load_model, predict


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models/cspp_seed20261101.pt"


@pytest.fixture
def wave():
    values = np.random.default_rng(38).normal(size=(3, 6000)).astype(np.float32)
    values[:, 2050:2100] = np.nan
    values[:, 4200:4250] = 0
    return values


@pytest.mark.parametrize("q,e", [(0.1, 0.975), (0.0, 0.0)])
def test_streaming_matches_frozen_array_replay(wave, q, e):
    reference, reference_picks = predict(wave, load_model(CHECKPOINT), q, e)
    picker = StreamingPicker(CHECKPOINT, candidate_threshold=q, event_threshold=e)
    results = list(picker.iter_array(wave))
    for key, expected in reference.arrays().items():
        attribute = "event_probability" if key == "event_probability__primary" else key
        actual = np.array([getattr(result, attribute) for result in results], dtype=expected.dtype)
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=2e-6, equal_nan=True)
        else:
            np.testing.assert_array_equal(actual, expected)
    picks = [result.pick for result in results if result.pick]
    assert len(picks) == len(reference_picks)
    for actual, expected in zip(picks, reference_picks, strict=True):
        assert actual.keys() == expected.keys()
        for name in actual:
            assert actual[name] == pytest.approx(expected[name], abs=2e-6)


def test_gap_drops_packet_resets_state_and_preserves_clock():
    picker = StreamingPicker(CHECKPOINT, candidate_threshold=0, event_threshold=0, warmup_packets=0)
    packets = np.random.default_rng(4).normal(size=(5, 3, 50)).astype(np.float32)
    assert picker.process_packet(packets[0], timestamp_seconds=100).pick is not None
    gap = picker.process_packet(packets[1], timestamp_seconds=105)
    assert gap.reason_codes == ("timestamp_gap",)
    assert gap.pick is None and picker._model_state is None
    after = picker.process_packet(packets[2], timestamp_seconds=105.5)
    assert after.pick["available_epoch"] == 106.0
    fresh = StreamingPicker(CHECKPOINT, candidate_threshold=0, event_threshold=0, warmup_packets=0)
    expected = fresh.process_packet(packets[2], timestamp_seconds=105.5)
    assert after.event_probability == expected.event_probability
    assert after.pick["estimated_p_epoch"] == expected.pick["estimated_p_epoch"]


def test_warmup_reset_duplicate_and_bounded_state():
    picker = StreamingPicker(CHECKPOINT)
    rng = np.random.default_rng(2)
    for index in range(40):
        packet = rng.normal(size=(3, 50)).astype(np.float32)
        result = picker.process_packet(packet, timestamp_seconds=index * 0.5)
        assert result.candidate_allowed == (index >= 19)
        assert sum(x.numel() for x in picker._model_state.values()) == 320
        assert all(not x.requires_grad for x in picker._model_state.values())
    duplicate = picker.process_packet(packet, timestamp_seconds=20)
    assert duplicate.reason_codes == ("duplicate_packet",)
    assert duplicate.warmup_remaining == 20
    assert picker._model_state is None and picker._normalization is None
    assert duplicate.to_dict()["event_probability"] is None
    json.dumps(duplicate.to_dict(), allow_nan=False)
    picker.reset()
    result = picker.process_packet(rng.normal(size=(3, 50)), timestamp_seconds=0)
    assert result.packet_index == 0 and result.warmup_remaining == 19


@pytest.mark.parametrize("kwargs", [
    {"candidate_threshold": -0.1}, {"event_threshold": float("nan")},
    {"event_threshold": 1.1}, {"warmup_packets": -1}, {"warmup_packets": 0.5},
    {"torch_threads": 0}, {"warmup_packets": True},
])
def test_reject_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        StreamingPicker(CHECKPOINT, **kwargs)


def test_reject_bad_shape_and_clock_without_advancing():
    picker = StreamingPicker(CHECKPOINT)
    with pytest.raises(ValueError, match="shape"):
        picker.process_packet(np.ones((3, 49)), timestamp_seconds=0)
    with pytest.raises(ValueError, match="finite"):
        picker.process_packet(np.ones((3, 50)), timestamp_seconds=float("inf"))
    with pytest.raises(ValueError):
        list(picker.iter_array(np.zeros((3, 0))))
    assert picker._packet_index == 0


def test_cli_records_settings_and_refuses_overwrite(tmp_path, wave):
    source = tmp_path / "prepared.npy"
    destination = tmp_path / "picks.json"
    np.save(source, wave)
    command = [sys.executable, "-m", "cspp.cli", "--input", str(source),
               "--checkpoint", str(CHECKPOINT), "--output", str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    payload = json.loads(destination.read_text())
    assert payload["checkpoint"] == CHECKPOINT.name
    assert set(payload) == {"candidates", "diagnostics", "checkpoint", "sample_rate_hz",
                            "component_order", "packet_seconds", "start_seconds",
                            "candidate_threshold", "event_threshold", "warmup_packets", "time_basis"}
    assert payload["warmup_packets"] == 20
    assert payload["event_threshold"] == 0.975
    assert payload["diagnostics"]["packets"] == 120
    original = destination.read_bytes()
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert original == destination.read_bytes()
