"""
Issue #13/#16 support — loads a hand-built evaluation set from disk.

This is the piece that makes evaluation "just drop photos in a folder,
no code or CSV editing" — everything below reads a fixed folder
convention that the team builds by hand:

    data/events/<event_id>/raw/                  (already exists — the
                                                    real event photos,
                                                    same folder src.indexing
                                                    uses)

    data/evaluation/<event_id>/identities/
        <person_name>/
            selfie/      exactly ONE image — the selfie used as the
                          search query for this person.
            matches/     copies of every raw/ photo that actually shows
                          this person. Filenames MUST match the
                          corresponding file in data/events/<event_id>/raw/
                          exactly — that's how a photo in matches/ gets
                          linked back to the indexed photo it stands for.

To evaluate a new event: index it normally (src.indexing.build_event_index),
then create one identities/<name>/ folder per person with their selfie
and copies of their known photos. Add as many people and events as you
have real (or realistic stand-in) photos for — nothing else needs to
change.
"""

from dataclasses import dataclass
from pathlib import Path

from config import (
    EVAL_MATCHES_SUBDIR,
    EVAL_SELFIE_SUBDIR,
    EVALUATION_DIR,
    EVENT_RAW_SUBDIR,
    EVENTS_DIR,
)
from src.indexing.build_index import IMAGE_EXTENSIONS


@dataclass
class Identity:
    """One person in an evaluation set."""

    name: str
    selfie_path: Path
    # Filenames (not full paths) — the raw/ photos that actually show
    # this person, used as ground truth for precision/recall.
    ground_truth_filenames: set[str]


@dataclass
class EvaluationSet:
    """A full evaluation set for one event: where its photos are, and
    who should be found in them."""

    event_id: str
    raw_dir: Path
    identities: list[Identity]


def _find_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix in IMAGE_EXTENSIONS)


def load_evaluation_set(event_id: str) -> EvaluationSet:
    """
    Load the evaluation set for one event.

    Raises:
        FileNotFoundError: if the event's raw/ photos or identities/
            folder don't exist yet — the team hasn't set this event up
            for evaluation.
        ValueError: if an identity folder is malformed (no selfie,
            more than one selfie, or no ground-truth photos) — this is
            almost always a mistake in how photos were dropped in, so
            it's raised loudly rather than silently skipped.
    """
    raw_dir = EVENTS_DIR / event_id / EVENT_RAW_SUBDIR
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"No raw photos found for event '{event_id}' at {raw_dir}. "
            "Index this event first (see src.indexing.build_event_index)."
        )

    identities_dir = EVALUATION_DIR / event_id / "identities"
    if not identities_dir.exists() or not any(identities_dir.iterdir()):
        raise FileNotFoundError(
            f"No evaluation identities found at {identities_dir}. "
            "Create one folder per person with a selfie/ and matches/ "
            "subfolder — see the module docstring for the exact layout."
        )

    identities = []
    for person_dir in sorted(p for p in identities_dir.iterdir() if p.is_dir()):
        selfie_images = _find_images(person_dir / EVAL_SELFIE_SUBDIR)
        if len(selfie_images) != 1:
            raise ValueError(
                f"'{person_dir.name}': expected exactly one image in "
                f"{person_dir / EVAL_SELFIE_SUBDIR}, found {len(selfie_images)}."
            )

        match_images = _find_images(person_dir / EVAL_MATCHES_SUBDIR)
        if not match_images:
            raise ValueError(
                f"'{person_dir.name}': no ground-truth photos found in "
                f"{person_dir / EVAL_MATCHES_SUBDIR}. Add copies of this "
                f"person's real event photos there (same filenames as in "
                f"{raw_dir})."
            )

        unmatched = [
            m.name for m in match_images if not (raw_dir / m.name).exists()
        ]
        if unmatched:
            raise ValueError(
                f"'{person_dir.name}': these files in matches/ don't have "
                f"a same-named file in {raw_dir}: {unmatched}. Ground-truth "
                f"filenames must exactly match the raw event photo they "
                f"refer to."
            )

        identities.append(
            Identity(
                name=person_dir.name,
                selfie_path=selfie_images[0],
                ground_truth_filenames={m.name for m in match_images},
            )
        )

    if not identities:
        raise ValueError(f"No identity folders found under {identities_dir}.")

    return EvaluationSet(event_id=event_id, raw_dir=raw_dir, identities=identities)
