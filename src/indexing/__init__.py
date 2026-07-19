"""
Indexing package — builds and queries per-event FAISS vector indexes.
This is what src/matching calls to load an event's index before
searching it against a student's selfie.
"""

from .faiss_index import EventIndex, IndexedFace
from .build_index import build_event_index, load_event_index

__all__ = ["EventIndex", "IndexedFace", "build_event_index", "load_event_index"]
