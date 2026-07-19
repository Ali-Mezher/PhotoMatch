"""
Matching package — turns a student's selfie into a tiered list of their
photos from one event. This is what src/interface calls when a student
hits "Search".
"""

from .similarity import cosine_similarity, classify_tier
from .matcher import (
    PhotoMatch,
    match_selfie,
    NoFaceDetectedError,
    EventNotIndexedError,
)

__all__ = [
    "cosine_similarity",
    "classify_tier",
    "PhotoMatch",
    "match_selfie",
    "NoFaceDetectedError",
    "EventNotIndexedError",
]
