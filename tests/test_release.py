from pathlib import Path
import sys
import subprocess

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from cspp import CSPP, load_checkpoint_model, model_from_checkpoint
from cspp.streaming.candidate import PacketCandidateState
from cspp.streaming.quality import PacketQualityConfig, PacketQualityGate
from cspp.streaming.normalization import normalize_packet_ewma_causal
from predict import load_model, predict
from reproduce_metrics import point_metrics, reproduce
from verify_release import verify


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize("seed", [20261101, 20261102, 20261103])
def test_released_checkpoint(seed):
    model, stored = load_checkpoint_model(ROOT / f"models/cspp_seed{seed}.pt")
    assert sum(p.numel() for p in model.parameters()) == 329423
    assert model.state_size_scalars == 320
    assert model.config.verifier_feature_mode == "none"
    assert stored["candidate_policy"]["warmup_packets"] == 20
    assert not model.training


@pytest.mark.parametrize("mode", ["none", "current", "current_delta"])
def test_batch_packet_equivalence_and_future_independence(mode):
    torch.manual_seed(42)
    model = CSPP(
        packet_width=16,
        embedding_dim=16,
        sequence_width=16,
        gru_layers=1,
        temporal_bins=2,
        dropout=0,
        packet_kernels=(3,),
        verifier_hidden_dim=16,
        verifier_feature_mode=mode,
    ).eval()
    x = torch.randn(1, 8, 3, 50)
    with torch.inference_mode():
        outputs = model.forward_with_aux(x)
        changed = x.clone()
        changed[:, 5:] = torch.randn_like(changed[:, 5:]) * 100
        future = model.forward_with_aux(changed)
        for before, after in zip(outputs, future):
            torch.testing.assert_close(before[:, :5], after[:, :5], rtol=0, atol=0)
        state = None
        streamed = []
        for packet in x[0]:
            result = model.forward_streaming_packet_with_aux(packet, state)
            state = result[-1]
            streamed.append(result[:-1])
        for index, expected in enumerate(outputs):
            actual = torch.stack([part[index].reshape(-1) for part in streamed], dim=0)
            torch.testing.assert_close(
                actual, expected[0].reshape(8, -1), rtol=1e-5, atol=1e-6
            )


def test_checkpoint_rejects_wrong_family_and_missing_tensors():
    with pytest.raises(ValueError):
        model_from_checkpoint({"model_class": "unsupported"})
    _, stored = load_checkpoint_model(ROOT / "models/cspp_seed20261101.pt")
    stored["model"].pop(next(iter(stored["model"])))
    with pytest.raises(RuntimeError):
        model_from_checkpoint(stored)


def test_quality_and_normalization_reset():
    gate = PacketQualityGate(PacketQualityConfig(warmup_packets=2))
    rng = np.random.default_rng(5)
    state = None
    for i in range(5):
        packet = rng.normal(size=(3, 50)).astype(np.float32)
        assert gate.evaluate(packet, i * 0.5).valid
        normalized, state = normalize_packet_ewma_causal(packet, state)
        assert np.isfinite(normalized).all()
    packet[0, 0] = np.nan
    decision = gate.evaluate(packet, 2.5)
    assert not decision.valid and decision.reset_required


def test_candidate_episode_closure_reset_and_timing():
    state = PacketCandidateState(4)
    kwargs = dict(
        valid=True,
        candidate_allowed=True,
        candidate_probability=0.8,
        event_probability=0.99,
        arrival_age_fraction=0.25,
        candidate_threshold=0.1,
        event_threshold=0.975,
        start_epoch=100.0,
        packet_seconds=0.5,
        max_arrival_age_packets=4.0,
    )
    first = state.step(packet_index=20, **kwargs)
    assert first["available_epoch"] == 110.5
    assert first["estimated_p_epoch"] == 110.0
    assert state.step(packet_index=21, **kwargs) is None
    assert state.step(packet_index=22, **dict(kwargs, valid=False)) is None
    assert state.step(packet_index=23, **kwargs)["episode_index"] == 2
    with pytest.raises(ValueError):
        state.step(packet_index=24, **dict(kwargs, event_probability=float("nan")))


def test_saved_synthetic_regression_fixture():
    data = np.load(ROOT / "tests/fixtures/synthetic_regression.npz", allow_pickle=False)
    ensemble = load_model(ROOT / "models/cspp_seed20261101.pt")
    cache, _ = predict(data["wave"], ensemble)
    for name, actual in cache.arrays().items():
        expected = data[name]
        if expected.dtype.kind == "f":
            np.testing.assert_allclose(
                actual, expected, rtol=1e-5, atol=2e-6, equal_nan=True
            )
        else:
            np.testing.assert_array_equal(actual, expected)


def test_invalid_input_layout():
    with pytest.raises(ValueError, match="shape"):
        predict(np.ones((100, 3)), None)


def test_denominators_include_misses_and_extra_candidates():
    result = point_metrics(
        [
            dict(
                candidate_count="3",
                matched="True",
                pick_error_seconds=".2",
                availability_delay_seconds=".5",
            ),
            dict(
                candidate_count="0",
                matched="False",
                pick_error_seconds="",
                availability_delay_seconds="",
            ),
        ]
    )
    assert (result["tp"], result["fp"], result["fn"]) == (1, 2, 1)
    assert result["recall_available_within_0.5s"] == 0.5


def test_manifest():
    assert verify() > 100


def test_primary_row_level_metric_reaggregation():
    assert reproduce()["all_checked_metrics_match"]


def test_training_preflight_does_not_start_training():
    result = subprocess.run(
        [sys.executable, str(ROOT / "tools/train_stage.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    report = __import__("json").loads(result.stdout)
    assert report["training_started"] is False
    assert len(report["missing_data_keys"]) == 10
