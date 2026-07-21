"""Unit tests for attendee-only liveness challenges and verification."""

from __future__ import annotations

import numpy as np
import pytest

from src.detection import Detection
from src.web.liveness import (
    RAW_LEFT,
    RAW_RIGHT,
    LivenessChallengeStore,
    LivenessVerificationError,
    LivenessVerifier,
)


def _detection(nose_ratio: float, identity: int = 0) -> Detection:
    eye_left = 30.0
    eye_right = 70.0
    return Detection(
        bbox=(identity, 10, 80, 80),
        confidence=0.99,
        keypoints={
            "left_eye": (eye_left, 35),
            "right_eye": (eye_right, 35),
            "nose": (eye_left + (eye_right - eye_left) * nose_ratio, 52),
        },
    )


def _verifier(detections: list[list[Detection]]) -> LivenessVerifier:
    pending = iter(detections)
    embeddings = {
        0: np.array([1.0, 0.0], dtype=np.float32),
        1: np.array([0.0, 1.0], dtype=np.float32),
    }
    return LivenessVerifier(
        detector=lambda _image: next(pending),
        embedder=lambda _image, detection: embeddings[detection.bbox[0]],
    )


def _frames():
    return [np.full((100, 120, 3), value, dtype=np.uint8) for value in (80, 100, 120)]


@pytest.mark.parametrize(
    ("direction", "turn_ratio"),
    [(RAW_LEFT, 0.25), (RAW_RIGHT, 0.75)],
)
def test_valid_random_turn_returns_final_frame(direction, turn_ratio):
    verifier = _verifier(
        [[_detection(0.50)], [_detection(turn_ratio)], [_detection(0.53)]]
    )
    front, turn, returned = _frames()

    result = verifier.verify(front, turn, returned, direction)

    assert result is returned


def test_static_or_wrong_direction_sequence_is_rejected():
    verifier = _verifier(
        [[_detection(0.50)], [_detection(0.52)], [_detection(0.51)]]
    )

    with pytest.raises(LivenessVerificationError, match="head turn"):
        verifier.verify(*_frames(), RAW_RIGHT)


def test_sequence_rejects_multiple_faces_and_identity_changes():
    multiple = _verifier(
        [[_detection(0.50), _detection(0.51)], [_detection(0.75)], [_detection(0.50)]]
    )
    with pytest.raises(LivenessVerificationError, match="Only 1 person"):
        multiple.verify(*_frames(), RAW_RIGHT)

    changed = _verifier(
        [[_detection(0.50)], [_detection(0.75, identity=1)], [_detection(0.50)]]
    )
    with pytest.raises(LivenessVerificationError, match="same person"):
        changed.verify(*_frames(), RAW_RIGHT)


def test_sequence_rejects_no_face_unclear_landmarks_and_bad_return_pose():
    no_face = _verifier([[], [_detection(0.75)], [_detection(0.50)]])
    with pytest.raises(LivenessVerificationError, match="No face"):
        no_face.verify(*_frames(), RAW_RIGHT)

    unclear = _detection(0.50)
    unclear.keypoints = {}
    bad_landmarks = _verifier([[unclear], [_detection(0.75)], [_detection(0.50)]])
    with pytest.raises(LivenessVerificationError, match="landmarks"):
        bad_landmarks.verify(*_frames(), RAW_RIGHT)

    bad_return = _verifier(
        [[_detection(0.50)], [_detection(0.75)], [_detection(0.66)]]
    )
    with pytest.raises(LivenessVerificationError, match="Return to the starting"):
        bad_return.verify(*_frames(), RAW_RIGHT)


def test_challenges_are_scoped_single_use_and_expire():
    now = {"value": 100.0}
    store = LivenessChallengeStore(
        ttl_seconds=10,
        clock=lambda: now["value"],
        direction_provider=lambda: RAW_LEFT,
    )
    challenge = store.issue("event-a", "access-a")

    assert store.consume(challenge.token, "event-b", "access-a") is None
    assert store.consume(challenge.token, "event-a", "access-a") is None

    fresh = store.issue("event-a", "access-a")
    assert store.consume(fresh.token, "event-a", "access-a") == fresh
    assert store.consume(fresh.token, "event-a", "access-a") is None

    expired = store.issue("event-a", "access-a")
    now["value"] = 111.0
    assert store.consume(expired.token, "event-a", "access-a") is None
