"""Short-lived attendee liveness challenges and in-memory verification."""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from random import SystemRandom
from threading import Lock
from typing import Callable

import numpy as np

from src.detection import Detection, detect_faces, embed_detection
from src.preprocessing import preprocess_image


RAW_LEFT = "raw_left"
RAW_RIGHT = "raw_right"
CHALLENGE_DIRECTIONS = (RAW_LEFT, RAW_RIGHT)


@dataclass(frozen=True)
class LivenessChallenge:
    token: str
    event_id: str
    access_token: str
    direction: str
    expires_at: float


class LivenessChallengeStore:
    """Issue event-bound, access-bound, single-use challenges in memory."""

    def __init__(
        self,
        ttl_seconds: int = 120,
        *,
        clock: Callable[[], float] = time.monotonic,
        direction_provider: Callable[[], str] | None = None,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._direction_provider = direction_provider or (
            lambda: SystemRandom().choice(CHALLENGE_DIRECTIONS)
        )
        self._lock = Lock()
        self._challenges: dict[str, LivenessChallenge] = {}

    def issue(self, event_id: str, access_token: str) -> LivenessChallenge:
        now = self._clock()
        direction = self._direction_provider()
        if direction not in CHALLENGE_DIRECTIONS:
            raise ValueError("direction_provider returned an invalid direction")
        challenge = LivenessChallenge(
            token=secrets.token_urlsafe(24),
            event_id=event_id,
            access_token=access_token,
            direction=direction,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            self._purge(now)
            self._challenges[challenge.token] = challenge
        return challenge

    def consume(
        self, token: str, event_id: str, access_token: str
    ) -> LivenessChallenge | None:
        """Consume a challenge once, returning it only when its scope matches."""
        now = self._clock()
        with self._lock:
            self._purge(now)
            challenge = self._challenges.pop(token, None)
        if challenge is None:
            return None
        if challenge.event_id != event_id or challenge.access_token != access_token:
            return None
        return challenge

    def _purge(self, now: float) -> None:
        expired = [
            token
            for token, challenge in self._challenges.items()
            if challenge.expires_at <= now
        ]
        for token in expired:
            del self._challenges[token]


class LivenessVerificationError(ValueError):
    """Raised when a live-selfie sequence cannot be verified."""


class LivenessVerifier:
    """Validate a front -> random turn -> front sequence without persistence."""

    def __init__(
        self,
        *,
        detector: Callable[[np.ndarray], list[Detection]] = detect_faces,
        embedder: Callable[[np.ndarray, Detection], np.ndarray] = embed_detection,
        center_tolerance: float = 0.18,
        turn_delta: float = 0.12,
        return_tolerance: float = 0.10,
        identity_similarity: float = 0.65,
    ):
        if min(center_tolerance, turn_delta, return_tolerance) <= 0:
            raise ValueError("pose thresholds must be positive")
        if not 0 < identity_similarity <= 1:
            raise ValueError("identity_similarity must be between zero and one")
        self._detector = detector
        self._embedder = embedder
        self.center_tolerance = center_tolerance
        self.turn_delta = turn_delta
        self.return_tolerance = return_tolerance
        self.identity_similarity = identity_similarity

    def verify(
        self,
        front_frame: np.ndarray,
        turn_frame: np.ndarray,
        return_frame: np.ndarray,
        direction: str,
    ) -> np.ndarray:
        """Return the final frame only after the complete sequence passes."""
        if direction not in CHALLENGE_DIRECTIONS:
            raise LivenessVerificationError(
                "The live-selfie challenge was invalid. Restart the check."
            )

        original_frames = (front_frame, turn_frame, return_frame)
        analyzed = [self._analyze_frame(frame) for frame in original_frames]
        ratios = [item[2] for item in analyzed]

        if abs(ratios[0] - 0.5) > self.center_tolerance:
            raise LivenessVerificationError(
                "Start with your face centered and looking straight at the camera."
            )
        if abs(ratios[2] - 0.5) > self.center_tolerance:
            raise LivenessVerificationError(
                "Finish with your face centered and looking straight at the camera."
            )

        movement = ratios[1] - ratios[0]
        expected_movement = (
            movement <= -self.turn_delta
            if direction == RAW_LEFT
            else movement >= self.turn_delta
        )
        if not expected_movement:
            raise LivenessVerificationError(
                "We could not confirm the requested head turn. Restart and follow the arrow."
            )
        if abs(ratios[2] - ratios[0]) > self.return_tolerance:
            raise LivenessVerificationError(
                "Return to the starting position before taking the final frame."
            )

        embeddings = [item[1] for item in analyzed]
        if any(
            float(np.dot(embeddings[left], embeddings[right]))
            < self.identity_similarity
            for left, right in ((0, 1), (0, 2), (1, 2))
        ):
            raise LivenessVerificationError(
                "Keep the same person in frame for the complete live-selfie check."
            )

        return return_frame

    def _analyze_frame(
        self, frame: np.ndarray
    ) -> tuple[Detection, np.ndarray, float]:
        cleaned = preprocess_image(frame)
        detections = self._detector(cleaned)
        if not detections:
            raise LivenessVerificationError(
                "No face was found. Use even lighting and keep your full face in frame."
            )
        if len(detections) != 1:
            raise LivenessVerificationError(
                "Only 1 person can appear during the live-selfie check."
            )

        detection = detections[0]
        ratio = self._nose_ratio(detection)
        try:
            embedding = self._embedder(cleaned, detection)
        except (RuntimeError, ValueError) as exc:
            raise LivenessVerificationError(
                "Your face could not be verified. Improve the lighting and try again."
            ) from exc
        return detection, embedding, ratio

    @staticmethod
    def _nose_ratio(detection: Detection) -> float:
        try:
            left_eye = detection.keypoints["left_eye"]
            right_eye = detection.keypoints["right_eye"]
            nose = detection.keypoints["nose"]
            eye_left_x, eye_right_x = sorted(
                (float(left_eye[0]), float(right_eye[0]))
            )
            eye_span = eye_right_x - eye_left_x
            if eye_span <= 1:
                raise ValueError
            return (float(nose[0]) - eye_left_x) / eye_span
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise LivenessVerificationError(
                "Face landmarks were unclear. Face the camera in even lighting and retry."
            ) from exc
