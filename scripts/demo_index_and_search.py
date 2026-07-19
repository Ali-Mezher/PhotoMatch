"""
Manual smoke-test for the full index -> match pipeline, no GUI needed.
Useful for Malek/Hazem to verify indexing + matching work together
before Nourallah's interface is ready to test against, and for anyone
debugging a specific event without launching the kiosk app.

Usage:
    # Build (or rebuild) the index for an event:
    python scripts/demo_index_and_search.py index <event_id>

    # Search an already-indexed event with a selfie:
    python scripts/demo_index_and_search.py search <event_id> path/to/selfie.jpg

Expects photos at: data/events/<event_id>/raw/*.jpg
"""

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.indexing import build_event_index
from src.matching import match_selfie, NoFaceDetectedError, EventNotIndexedError


def run_index(event_id: str):
    build_event_index(event_id)


def run_search(event_id: str, selfie_path: str):
    selfie = cv2.imread(selfie_path)
    if selfie is None:
        print(f"Could not read image: {selfie_path}")
        sys.exit(1)

    try:
        results = match_selfie(selfie, event_id)
    except NoFaceDetectedError:
        print("No face detected in the selfie — try a clearer photo.")
        sys.exit(1)
    except EventNotIndexedError:
        print(f"Event '{event_id}' hasn't been indexed yet. Run:")
        print(f"  python scripts/demo_index_and_search.py index {event_id}")
        sys.exit(1)

    print(f"\nConfident matches ({len(results['confident'])}):")
    for m in results["confident"]:
        print(f"  {m.score:.3f}  {m.photo_path}")

    print(f"\nPossible matches ({len(results['possible'])}):")
    for m in results["possible"]:
        print(f"  {m.score:.3f}  {m.photo_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    command, event_id = sys.argv[1], sys.argv[2]

    if command == "index":
        run_index(event_id)
    elif command == "search":
        if len(sys.argv) != 4:
            print("Usage: python scripts/demo_index_and_search.py search <event_id> <selfie_path>")
            sys.exit(1)
        run_search(event_id, sys.argv[3])
    else:
        print(f"Unknown command: {command}. Use 'index' or 'search'.")
        sys.exit(1)
