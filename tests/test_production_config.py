import pytest

from src.web import create_app


def _clear_production_environment(monkeypatch):
    for name in (
        "PHOTOMATCH_ADMIN_USERNAME",
        "PHOTOMATCH_ADMIN_PASSWORD",
        "PHOTOMATCH_SECRET_KEY",
        "PHOTOMATCH_SECURE_COOKIE",
        "PHOTOMATCH_ALLOWED_NETWORKS",
        "PHOTOMATCH_TRUST_PROXY",
    ):
        monkeypatch.delenv(name, raising=False)


def test_production_fails_fast_when_fixed_credentials_are_missing(tmp_path, monkeypatch):
    _clear_production_environment(monkeypatch)
    monkeypatch.setenv("PHOTOMATCH_ENV", "production")
    with pytest.raises(RuntimeError, match="PHOTOMATCH_ADMIN_USERNAME"):
        create_app(
            {
                "EVENTS_DIR": tmp_path / "events",
                "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
                "ADMIN_START_WORKER": False,
            }
        )


def test_valid_production_configuration_starts_with_secure_cookie(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATCH_ENV", "production")
    monkeypatch.setenv("PHOTOMATCH_ADMIN_USERNAME", "shop-worker")
    monkeypatch.setenv("PHOTOMATCH_ADMIN_PASSWORD", "test-only-long-password")
    monkeypatch.setenv("PHOTOMATCH_SECRET_KEY", "s" * 48)
    monkeypatch.setenv("PHOTOMATCH_SECURE_COOKIE", "1")
    monkeypatch.setenv("PHOTOMATCH_ALLOWED_NETWORKS", "127.0.0.1/32")
    app = create_app(
        {
            "EVENTS_DIR": tmp_path / "events",
            "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
            "ADMIN_START_WORKER": False,
        }
    )
    assert app.config["SESSION_COOKIE_SECURE"] is True
    assert app.test_client().get("/healthz").status_code == 200


def test_production_rejects_insecure_cookie_setting(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOMATCH_ENV", "production")
    monkeypatch.setenv("PHOTOMATCH_ADMIN_USERNAME", "shop-worker")
    monkeypatch.setenv("PHOTOMATCH_ADMIN_PASSWORD", "test-only-long-password")
    monkeypatch.setenv("PHOTOMATCH_SECRET_KEY", "s" * 48)
    monkeypatch.setenv("PHOTOMATCH_SECURE_COOKIE", "0")
    monkeypatch.setenv("PHOTOMATCH_ALLOWED_NETWORKS", "127.0.0.1/32")
    with pytest.raises(RuntimeError, match="SECURE_COOKIE"):
        create_app(
            {
                "EVENTS_DIR": tmp_path / "events",
                "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
                "ADMIN_START_WORKER": False,
            }
        )


def test_production_requires_an_explicit_company_network_allowlist(tmp_path, monkeypatch):
    _clear_production_environment(monkeypatch)
    monkeypatch.setenv("PHOTOMATCH_ENV", "production")
    monkeypatch.setenv("PHOTOMATCH_ADMIN_USERNAME", "shop-worker")
    monkeypatch.setenv("PHOTOMATCH_ADMIN_PASSWORD", "test-only-long-password")
    monkeypatch.setenv("PHOTOMATCH_SECRET_KEY", "s" * 48)
    monkeypatch.setenv("PHOTOMATCH_SECURE_COOKIE", "1")

    with pytest.raises(RuntimeError, match="PHOTOMATCH_ALLOWED_NETWORKS"):
        create_app(
            {
                "EVENTS_DIR": tmp_path / "events",
                "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
                "ADMIN_START_WORKER": False,
            }
        )


def test_network_allowlist_blocks_unapproved_computers(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "EVENTS_DIR": tmp_path / "events",
            "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
            "ADMIN_START_WORKER": False,
            "ALLOWED_NETWORKS": "10.20.30.40/32,192.168.50.0/24",
        }
    )
    client = app.test_client()

    exact = client.get("/healthz", environ_overrides={"REMOTE_ADDR": "10.20.30.40"})
    office = client.get("/healthz", environ_overrides={"REMOTE_ADDR": "192.168.50.25"})
    blocked = client.get("/", environ_overrides={"REMOTE_ADDR": "203.0.113.8"})

    assert exact.status_code == 200
    assert office.status_code == 200
    assert blocked.status_code == 403
    assert b"approved company computers" in blocked.data


def test_server_loopback_remains_available_for_health_checks(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "EVENTS_DIR": tmp_path / "events",
            "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
            "ADMIN_START_WORKER": False,
            "ALLOWED_NETWORKS": "10.20.30.40/32",
        }
    )

    assert app.test_client().get("/healthz").status_code == 200


def test_invalid_network_allowlist_fails_at_startup(tmp_path):
    with pytest.raises(RuntimeError, match="Invalid PHOTOMATCH_ALLOWED_NETWORKS"):
        create_app(
            {
                "TESTING": True,
                "EVENTS_DIR": tmp_path / "events",
                "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
                "ADMIN_START_WORKER": False,
                "ALLOWED_NETWORKS": "not-an-ip-address",
            }
        )


def test_forwarded_company_ip_is_used_only_when_proxy_is_trusted(tmp_path, monkeypatch):
    config = {
        "TESTING": True,
        "EVENTS_DIR": tmp_path / "events",
        "INDEX_STATUS_DB": tmp_path / "status.sqlite3",
        "ADMIN_START_WORKER": False,
        "ALLOWED_NETWORKS": "203.0.113.10/32",
    }
    untrusted_app = create_app(config)
    spoofed = untrusted_app.test_client().get(
        "/healthz",
        environ_overrides={"REMOTE_ADDR": "10.0.0.5"},
        headers={"X-Forwarded-For": "203.0.113.10"},
    )

    monkeypatch.setenv("PHOTOMATCH_TRUST_PROXY", "1")
    trusted_app = create_app(
        {**config, "INDEX_STATUS_DB": tmp_path / "trusted-status.sqlite3"}
    )
    forwarded = trusted_app.test_client().get(
        "/healthz",
        environ_overrides={"REMOTE_ADDR": "10.0.0.5"},
        headers={"X-Forwarded-For": "203.0.113.10"},
    )

    assert spoofed.status_code == 403
    assert forwarded.status_code == 200
