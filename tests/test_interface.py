"""Focused checks for issue #11's UI-to-matching integration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from src.interface.app import PhotoMatchApp
from src.matching import EventNotIndexedError, NoFaceDetectedError


def _worker_app():
    app = PhotoMatchApp.__new__(PhotoMatchApp)
    app.root = Mock()
    app._show_results = Mock()
    app._show_error = Mock()
    return app


def test_search_worker_sends_matches_back_to_ui_thread():
    app = _worker_app()
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    results = {"confident": [], "possible": []}

    with patch("src.interface.app.cv2.imread", return_value=image), patch(
        "src.interface.app.match_selfie", return_value=results
    ) as match:
        app._search_worker("graduation", Path("selfie.jpg"))

    match.assert_called_once_with(image, "graduation")
    app.root.after.assert_called_once_with(0, app._show_results, results)


def test_search_worker_explains_missing_face():
    app = _worker_app()

    with patch(
        "src.interface.app.cv2.imread", return_value=np.zeros((8, 8, 3))
    ), patch("src.interface.app.match_selfie", side_effect=NoFaceDetectedError):
        app._search_worker("graduation", Path("selfie.jpg"))

    callback = app.root.after.call_args.args
    assert callback[:2] == (0, app._show_error)
    assert "No face" in callback[2]


def test_search_worker_explains_unindexed_event():
    app = _worker_app()

    with patch(
        "src.interface.app.cv2.imread", return_value=np.zeros((8, 8, 3))
    ), patch("src.interface.app.match_selfie", side_effect=EventNotIndexedError):
        app._search_worker("graduation", Path("selfie.jpg"))

    callback = app.root.after.call_args.args
    assert callback[:2] == (0, app._show_error)
    assert "graduation" in callback[2]
    assert "hasn't been indexed" in callback[2]


def test_cluster_worker_returns_groups_to_ui_thread():
    app = _worker_app()
    app._show_clusters = Mock()
    payload = {"clusters": [], "unclustered_count": 1}

    with patch("src.interface.app.cluster_event_if_needed", return_value=(payload, True)):
        app._cluster_worker("graduation")

    app.root.after.assert_called_once_with(0, app._show_clusters, payload, True)


def test_cluster_worker_explains_unindexed_event():
    app = _worker_app()
    app._show_cluster_error = Mock()

    with patch("src.interface.app.cluster_event_if_needed", side_effect=FileNotFoundError):
        app._cluster_worker("graduation")

    callback = app.root.after.call_args.args
    assert callback[:2] == (0, app._show_cluster_error)
    assert "must be indexed" in callback[2]


def test_show_results_keeps_confident_and_possible_tiers_separate():
    app = PhotoMatchApp.__new__(PhotoMatchApp)
    app.status_var = Mock()
    app.search_button = Mock()
    app._render_tier = Mock()
    confident = [SimpleNamespace(photo_path="one.jpg", score=0.9)]
    possible = [SimpleNamespace(photo_path="two.jpg", score=0.6)]

    app._show_results({"confident": confident, "possible": possible})

    assert app._render_tier.call_args_list[0].args == ("Confident Matches", confident)
    assert app._render_tier.call_args_list[1].args == ("Possible Matches", possible)
    app.search_button.config.assert_called_once_with(state="normal")


def test_load_photo_image_returns_none_for_missing_file():
    assert PhotoMatchApp._load_photo_image("missing-photo.jpg", (100, 100)) is None
