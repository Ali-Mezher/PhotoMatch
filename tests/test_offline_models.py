from pathlib import Path

from src.detection.device import ComputeDeviceStatus
from src.detection.offline_models import prepare_offline_models


class _PreparedModel:
    def __init__(self, calls: list[str], name: str):
        self._calls = calls
        self._name = name

    def prepare_model(self):
        self._calls.append(self._name)


def test_prepare_offline_models_loads_both_models_and_reports_persisted_weights(
    tmp_path,
):
    calls: list[str] = []
    weights_dir = tmp_path / ".deepface" / "weights"
    weights_dir.mkdir(parents=True)
    weight_file = weights_dir / "facenet512_weights.h5"
    weight_file.write_bytes(b"test-weights")
    device = ComputeDeviceStatus("cpu", "CPU mode", "/CPU:0")

    result = prepare_offline_models(
        detector_factory=lambda: _PreparedModel(calls, "detector"),
        extractor_factory=lambda: _PreparedModel(calls, "extractor"),
        device_status_provider=lambda: device,
        weights_dir=weights_dir,
    )

    assert calls == ["detector", "extractor"]
    assert result.device is device
    assert result.weight_files == (weight_file.resolve(),)


def test_prepare_offline_models_rejects_a_missing_persistent_weight(tmp_path):
    calls: list[str] = []

    try:
        prepare_offline_models(
            detector_factory=lambda: _PreparedModel(calls, "detector"),
            extractor_factory=lambda: _PreparedModel(calls, "extractor"),
            device_status_provider=lambda: ComputeDeviceStatus(
                "cpu", "CPU mode", "/CPU:0"
            ),
            weights_dir=Path(tmp_path / "missing"),
        )
    except RuntimeError as exc:
        assert "no persistent DeepFace weight file" in str(exc)
    else:
        raise AssertionError("Expected missing weights to fail offline preparation")
