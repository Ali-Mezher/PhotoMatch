from contextlib import nullcontext

import pytest

from src.detection import device


class _FakeConfig:
    def __init__(self, gpus):
        self._gpus = gpus
        self.experimental = self
        self.hidden = None

    def list_physical_devices(self, kind):
        assert kind == "GPU"
        return self._gpus

    def set_memory_growth(self, gpu, enabled):
        assert enabled is True

    def set_visible_devices(self, devices, kind):
        self.hidden = (devices, kind)


class _FakeTensorFlow:
    def __init__(self, gpus=()):
        self.config = _FakeConfig(list(gpus))
        self.scopes = []

    def device(self, name):
        self.scopes.append(name)
        return nullcontext()


def test_detects_tensorflow_gpu_without_physical_hardware():
    status = device.detect_compute_device(_FakeTensorFlow(["gpu0"]), "auto")
    assert status.kind == "gpu"
    assert status.accelerated is True
    assert status.tensorflow_device == "/GPU:0"


def test_uses_cpu_when_gpu_is_unavailable_or_explicitly_disabled():
    assert device.detect_compute_device(_FakeTensorFlow(), "auto").kind == "cpu"
    tensorflow = _FakeTensorFlow(["gpu0"])
    assert device.detect_compute_device(tensorflow, "cpu").kind == "cpu"
    assert tensorflow.config.hidden == ([], "GPU")


def test_rejects_unknown_device_preference():
    with pytest.raises(ValueError, match="auto, gpu, or cpu"):
        device.detect_compute_device(_FakeTensorFlow(), "quantum")


def test_gpu_operation_retries_once_on_cpu(monkeypatch):
    tensorflow = _FakeTensorFlow(["gpu0"])
    monkeypatch.setattr(
        device,
        "_status",
        device.ComputeDeviceStatus("gpu", "GPU acceleration", "/GPU:0", 1),
    )
    monkeypatch.setattr(device, "_tensorflow", tensorflow)
    attempts = []

    def operation():
        attempts.append(True)
        if len(attempts) == 1:
            raise RuntimeError("simulated CUDA failure")
        return "complete"

    assert device.run_tensorflow_operation(operation) == "complete"
    assert tensorflow.scopes == ["/GPU:0", "/CPU:0"]
    assert device.compute_device_status().kind == "cpu"
