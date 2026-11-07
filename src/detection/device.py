"""TensorFlow compute-device selection with a safe CPU fallback.

The persisted PhotoMatch vectors and FAISS indexes stay CPU-portable.  This
module only controls where TensorFlow-backed detection and embedding work is
executed.
"""

from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable


LOGGER = logging.getLogger(__name__)
VALID_PREFERENCES = {"auto", "gpu", "cpu"}


@dataclass(frozen=True)
class ComputeDeviceStatus:
    kind: str
    label: str
    tensorflow_device: str
    gpu_count: int = 0
    detail: str = ""

    @property
    def accelerated(self) -> bool:
        return self.kind == "gpu"


_status: ComputeDeviceStatus | None = None
_tensorflow: Any | None = None


def detect_compute_device(
    tensorflow_module: Any | None = None,
    preference: str | None = None,
) -> ComputeDeviceStatus:
    """Return the usable TensorFlow device without requiring a GPU.

    ``tensorflow_module`` is injectable so unit tests never need TensorFlow or
    physical CUDA hardware.
    """
    selected = (preference or os.environ.get("PHOTOMATCH_COMPUTE_DEVICE", "auto")).lower()
    if selected not in VALID_PREFERENCES:
        raise ValueError("PHOTOMATCH_COMPUTE_DEVICE must be auto, gpu, or cpu")
    if selected == "cpu":
        if tensorflow_module is not None:
            try:
                tensorflow_module.config.set_visible_devices([], "GPU")
            except (AttributeError, RuntimeError, ValueError):
                pass
        return ComputeDeviceStatus("cpu", "CPU mode", "/CPU:0", detail="CPU requested")

    try:
        tf = tensorflow_module
        if tf is None:
            import tensorflow as tf  # type: ignore[no-redef]

        gpus = list(tf.config.list_physical_devices("GPU"))
        if not gpus:
            detail = "No compatible TensorFlow GPU was detected"
            if selected == "gpu":
                detail = "GPU requested but unavailable; using CPU"
            return ComputeDeviceStatus("cpu", "CPU mode", "/CPU:0", detail=detail)

        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except (AttributeError, RuntimeError, ValueError):
                # TensorFlow may already be initialized. The device is still
                # usable; memory growth is an optimization, not a requirement.
                pass
        return ComputeDeviceStatus(
            "gpu",
            "GPU acceleration",
            "/GPU:0",
            gpu_count=len(gpus),
            detail=f"TensorFlow detected {len(gpus)} GPU(s)",
        )
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        return ComputeDeviceStatus(
            "cpu",
            "CPU mode",
            "/CPU:0",
            detail=f"GPU initialization unavailable: {exc}",
        )


def configure_compute_device(*, refresh: bool = False) -> ComputeDeviceStatus:
    """Detect once, cache the result, and log the selected execution mode."""
    global _status, _tensorflow
    if _status is not None and not refresh:
        return _status
    if os.environ.get("PHOTOMATCH_COMPUTE_DEVICE", "auto").lower() == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import tensorflow as tf
    except (ImportError, OSError, RuntimeError, ValueError):
        tf = None
    _tensorflow = tf
    _status = detect_compute_device(tf)
    log = LOGGER.info if _status.accelerated else LOGGER.warning
    log("PhotoMatch compute device: %s (%s)", _status.label, _status.detail)
    return _status


def compute_device_status() -> ComputeDeviceStatus:
    return configure_compute_device()


def run_tensorflow_operation(operation: Callable[[], Any]) -> Any:
    """Run on the chosen device and retry once on CPU after a GPU failure."""
    global _status
    status = configure_compute_device()
    scope = _tensorflow.device(status.tensorflow_device) if _tensorflow else nullcontext()
    try:
        with scope:
            return operation()
    except Exception as exc:  # noqa: BLE001 - TensorFlow has backend-specific errors
        if not status.accelerated or isinstance(exc, (TypeError, ValueError)):
            raise
        LOGGER.exception("GPU inference failed; retrying this operation on CPU")
        _status = ComputeDeviceStatus(
            "cpu",
            "CPU mode",
            "/CPU:0",
            detail="GPU inference failed; automatic CPU fallback is active",
        )
        cpu_scope = _tensorflow.device("/CPU:0") if _tensorflow else nullcontext()
        with cpu_scope:
            return operation()


def _reset_device_cache_for_tests() -> None:
    global _status, _tensorflow
    _status = None
    _tensorflow = None
