"""
Manual smoke-test for the full index -> match pipeline, no GUI needed.
Useful for Malek/Hazem to verify indexing + matching work together
before Nourallah's interface is ready to test against, and for anyone
debugging a specific event without launching the kiosk app.

Usage:
    # First incremental run (event date is stored in SQLite):
    python scripts/demo_index_and_search.py index <event_id> --date YYYY-MM-DD

    # Later runs process only new/changed work:
    python scripts/demo_index_and_search.py index <event_id>

    # Explicit recovery rebuild:
    python scripts/demo_index_and_search.py index <event_id> --force

    # Search an already-indexed event with a selfie:
    python scripts/demo_index_and_search.py search <event_id> path/to/selfie.jpg

Expects photos at: data/events/<event_id>/raw/*.jpg
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.matching import match_selfie, NoFaceDetectedError, EventNotIndexedError
from src.services import IndexStatus, IndexingService


def run_index(
    event_id: str, event_date: str | None = None, force: bool = False
) -> int:
    service = IndexingService()
    try:
        service.get_event(event_id)
    except KeyError:
        if event_date is None:
            print(
                f"Event '{event_id}' is not registered yet. Its first incremental "
                "run needs an explicit event date:"
            )
            print(
                "  python scripts/demo_index_and_search.py index "
                f"{event_id} --date YYYY-MM-DD"
            )
            return 2
        service.register_event(event_id, event_date)
    else:
        if event_date is not None:
            service.register_event(event_id, event_date)

    if force:
        service.force_rebuild(event_id)
    else:
        service.request_index(event_id)

    queued = service.get_event(event_id)
    if queued.status is not IndexStatus.QUEUED:
        print(f"Event '{event_id}' is already up to date; no photos were reprocessed.")
        return 0

    if force or queued.rebuild_required:
        mode = "full rebuild"
    elif queued.pending_images == queued.total_images:
        mode = "initial build"
    else:
        mode = "incremental update"
    print(
        f"Running {mode} for '{event_id}' "
        f"({queued.pending_images} pending image(s))..."
    )
    service.run_pending(show_progress=True)

    completed = service.get_event(event_id)
    if completed.status is IndexStatus.FAILED:
        print(f"Indexing failed for '{event_id}': {completed.error}")
        return 1

    print(
        f"Index ready: {completed.indexed_images} image(s) with faces, "
        f"{completed.no_face_images} with no face, "
        f"{completed.failed_images} failed."
    )
    return 0


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    index_parser = commands.add_parser("index", help="incrementally index an event")
    index_parser.add_argument("event_id")
    index_parser.add_argument(
        "--date", dest="event_date", help="event date (YYYY-MM-DD), required once"
    )
    index_parser.add_argument(
        "--force", action="store_true", help="rebuild the complete event index"
    )

    search_parser = commands.add_parser("search", help="search with a selfie")
    search_parser.add_argument("event_id")
    search_parser.add_argument("selfie_path")

    args = parser.parse_args(argv)
    if args.command == "index":
        return run_index(args.event_id, args.event_date, args.force)
    run_search(args.event_id, args.selfie_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
