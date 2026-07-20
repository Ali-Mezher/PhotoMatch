"""
Tests for src/evaluation/ground_truth.py. Builds the real folder
convention under pytest's tmp_path and points the module at it via
monkeypatch — no real photos needed, just files with the right names in
the right places.
"""

from pathlib import Path

import pytest

from src.evaluation.ground_truth import load_evaluation_set


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake image bytes")


@pytest.fixture
def wired_dirs(tmp_path, monkeypatch):
    """Point the module at a fresh tmp_path 'data/' layout."""
    events_dir = tmp_path / "events"
    evaluation_dir = tmp_path / "evaluation"
    monkeypatch.setattr("src.evaluation.ground_truth.EVENTS_DIR", events_dir)
    monkeypatch.setattr("src.evaluation.ground_truth.EVALUATION_DIR", evaluation_dir)
    return events_dir, evaluation_dir


class TestLoadEvaluationSet:
    def test_raises_when_event_not_indexed(self, wired_dirs):
        with pytest.raises(FileNotFoundError, match="raw photos"):
            load_evaluation_set("nonexistent_event")

    def test_raises_when_no_identities_folder(self, wired_dirs):
        events_dir, _ = wired_dirs
        _touch(events_dir / "grad2026" / "raw" / "photo1.jpg")

        with pytest.raises(FileNotFoundError, match="identities"):
            load_evaluation_set("grad2026")

    def test_raises_when_identity_has_no_selfie(self, wired_dirs):
        events_dir, evaluation_dir = wired_dirs
        _touch(events_dir / "grad2026" / "raw" / "photo1.jpg")
        _touch(evaluation_dir / "grad2026" / "identities" / "alice" / "matches" / "photo1.jpg")
        # no selfie/ folder for alice

        with pytest.raises(ValueError, match="exactly one image"):
            load_evaluation_set("grad2026")

    def test_raises_when_identity_has_two_selfies(self, wired_dirs):
        events_dir, evaluation_dir = wired_dirs
        _touch(events_dir / "grad2026" / "raw" / "photo1.jpg")
        identity_dir = evaluation_dir / "grad2026" / "identities" / "alice"
        _touch(identity_dir / "selfie" / "selfie1.jpg")
        _touch(identity_dir / "selfie" / "selfie2.jpg")
        _touch(identity_dir / "matches" / "photo1.jpg")

        with pytest.raises(ValueError, match="exactly one image"):
            load_evaluation_set("grad2026")

    def test_raises_when_no_matches(self, wired_dirs):
        events_dir, evaluation_dir = wired_dirs
        _touch(events_dir / "grad2026" / "raw" / "photo1.jpg")
        identity_dir = evaluation_dir / "grad2026" / "identities" / "alice"
        _touch(identity_dir / "selfie" / "selfie.jpg")
        (identity_dir / "matches").mkdir(parents=True)

        with pytest.raises(ValueError, match="no ground-truth photos"):
            load_evaluation_set("grad2026")

    def test_raises_when_match_filename_not_in_raw(self, wired_dirs):
        events_dir, evaluation_dir = wired_dirs
        _touch(events_dir / "grad2026" / "raw" / "photo1.jpg")
        identity_dir = evaluation_dir / "grad2026" / "identities" / "alice"
        _touch(identity_dir / "selfie" / "selfie.jpg")
        _touch(identity_dir / "matches" / "photo_that_does_not_exist.jpg")

        with pytest.raises(ValueError, match="don't have a same-named file"):
            load_evaluation_set("grad2026")

    def test_loads_valid_evaluation_set(self, wired_dirs):
        events_dir, evaluation_dir = wired_dirs
        raw_dir = events_dir / "grad2026" / "raw"
        _touch(raw_dir / "photo1.jpg")
        _touch(raw_dir / "photo2.jpg")
        _touch(raw_dir / "photo3.jpg")

        alice_dir = evaluation_dir / "grad2026" / "identities" / "alice"
        _touch(alice_dir / "selfie" / "alice_selfie.jpg")
        _touch(alice_dir / "matches" / "photo1.jpg")
        _touch(alice_dir / "matches" / "photo2.jpg")

        bob_dir = evaluation_dir / "grad2026" / "identities" / "bob"
        _touch(bob_dir / "selfie" / "bob_selfie.jpg")
        _touch(bob_dir / "matches" / "photo3.jpg")

        eval_set = load_evaluation_set("grad2026")

        assert eval_set.event_id == "grad2026"
        assert eval_set.raw_dir == raw_dir
        assert len(eval_set.identities) == 2

        alice = next(i for i in eval_set.identities if i.name == "alice")
        assert alice.selfie_path.name == "alice_selfie.jpg"
        assert alice.ground_truth_filenames == {"photo1.jpg", "photo2.jpg"}

        bob = next(i for i in eval_set.identities if i.name == "bob")
        assert bob.ground_truth_filenames == {"photo3.jpg"}
