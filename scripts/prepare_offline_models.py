"""Prepare PhotoMatch models and prove that they reopen without network access."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detection.offline_models import OfflineModelStatus, prepare_offline_models
from src.evaluation.scalability import NetworkAccessError, verify_offline_operation


def _print_status(status: OfflineModelStatus, *, verified: bool) -> None:
    action = "Offline verification passed" if verified else "Model preparation passed"
    print(f"{action}.")
    print(f"Embedding model: {status.embedding_model}")
    print(f"Processing device: {status.device.label}")
    print(f"Persistent weights: {status.weights_dir}")
    print(f"Weight files found: {len(status.weight_files)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download/load PhotoMatch model assets, or verify in a fresh process "
            "that the cached assets require no network socket."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="load the models and download missing weights while online",
    )
    mode.add_argument(
        "--verify-offline",
        action="store_true",
        help="block socket creation while loading the persisted models",
    )
    args = parser.parse_args(argv)

    try:
        if args.verify_offline:
            status = verify_offline_operation(prepare_offline_models)
            _print_status(status, verified=True)
        else:
            status = prepare_offline_models()
            _print_status(status, verified=False)
    except NetworkAccessError as exc:
        print(f"Offline verification failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"Model preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
