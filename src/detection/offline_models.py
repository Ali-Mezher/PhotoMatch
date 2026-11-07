"""Prepare and verify the model assets required for offline operation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from config import EMBEDDING_MODEL, MODELS_DIR

from .device import ComputeDeviceStatus, compute_device_status
from .embeddings import EmbeddingExtractor
from .face_detector import FaceDetector

_REQUIRED_DEEPFACE_WEIGHTS = {
    "Facenet512": ("facenet512_weights.h5",),
}


@dataclass(frozen=True)
class OfflineModelStatus:
    """Summary returned after both inference models load successfully."""

    embedding_model: str
    weights_dir: Path
    weight_files: tuple[Path, ...]
    device: ComputeDeviceStatus


def deepface_weights_dir() -> Path:
    """Return the persistent DeepFace weight directory used by PhotoMatch."""
    home = Path(os.environ.get("DEEPFACE_HOME", str(MODELS_DIR.resolve())))
    return home / ".deepface" / "weights"


def prepare_offline_models(
    *,
    detector_factory: Callable[[], FaceDetector] = FaceDetector,
    extractor_factory: Callable[[], EmbeddingExtractor] = EmbeddingExtractor,
    device_status_provider: Callable[[], ComputeDeviceStatus] = compute_device_status,
    weights_dir: Path | None = None,
) -> OfflineModelStatus:
    """Load detector and embedding assets, downloading missing weights if online.

    Run this once during installation while dependency/model downloads are
    permitted. Running the same function with sockets blocked in a fresh process
    proves that the persisted assets are sufficient for normal model startup.
    """
    detector = detector_factory()
    detector.prepare_model()

    extractor = extractor_factory()
    extractor.prepare_model()

    resolved_weights = (weights_dir or deepface_weights_dir()).resolve()
    required_names = _REQUIRED_DEEPFACE_WEIGHTS.get(EMBEDDING_MODEL)
    if required_names:
        weight_files = tuple(resolved_weights / name for name in required_names)
        weight_files = tuple(
            path for path in weight_files if path.is_file() and path.stat().st_size > 0
        )
    else:
        weight_files = tuple(
            sorted(
                path
                for path in resolved_weights.rglob("*")
                if path.is_file() and path.stat().st_size > 0
            )
        )
    if not weight_files:
        raise RuntimeError(
            "The face models loaded, but no persistent DeepFace weight file was "
            f"found in {resolved_weights}. Do not begin the offline pilot."
        )

    return OfflineModelStatus(
        embedding_model=EMBEDDING_MODEL,
        weights_dir=resolved_weights,
        weight_files=weight_files,
        device=device_status_provider(),
    )
