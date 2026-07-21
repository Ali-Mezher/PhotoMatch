"""Create conservative candidate identity clusters for one indexed event.

Usage:
    python scripts/cluster_event.py <event_id>
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.clustering import cluster_event


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("event_id", help="Event folder to cluster after it has been indexed")
    args = parser.parse_args()

    try:
        result, output_path = cluster_event(args.event_id)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Could not cluster event '{args.event_id}': {exc}")
        return 1

    print(f"Created {len(result.clusters)} candidate cluster(s).")
    for cluster in result.clusters:
        print(f"  {cluster.cluster_id}: {len(cluster.members)} face(s) across "
              f"{len({member.photo_path for member in cluster.members})} photo(s)")
    print(f"Unclustered faces: {len(result.unclustered)}")
    print(f"Saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
