# Web Application Configuration

PhotoMatch reads its web-server configuration from environment variables when
`create_app()` runs. Restart the Flask process after changing a value.

## Supported environment variables

| Variable | Default | Purpose |
|---|---:|---|
| `PHOTOMATCH_ADMIN_USERNAME` | Not set | Username for the local operator dashboard. Both this and the admin password must be set before an operator can sign in. |
| `PHOTOMATCH_ADMIN_PASSWORD` | Not set | Password for the local operator dashboard. Keep it out of source control, screenshots, and shared shell history. |
| `PHOTOMATCH_SECRET_KEY` | Random for each process | Signs Flask sessions and CSRF state. Set a stable, private value so login sessions remain valid after a restart. |
| `PHOTOMATCH_BACKGROUND_WORKER` | `1` | Set to `1` to start the interrupt-driven indexing and clustering worker with Flask. Set to `0` only when background jobs must remain paused or another process owns them. |
| `PHOTOMATCH_DEBUG` | `0` | Set to `1` to enable Flask debug mode and the development reloader. Use only during local development because debug pages can expose internal details. |
| `PHOTOMATCH_SECURE_COOKIE` | `0` in development, `1` in production | Set to `1` when serving the app over HTTPS. Production refuses to start without it. |
| `PHOTOMATCH_ENV` | `development` | Set to `production` to require credentials, a stable 32+ character secret, and secure cookies at startup. |
| `PHOTOMATCH_COMPUTE_DEVICE` | `auto` | `auto` prefers a TensorFlow GPU; `gpu` requests one with CPU fallback; `cpu` disables GPU selection. |
| `PHOTOMATCH_EVENTS_DIR` | `data/events` | Persistent event-photo and index directory. |
| `PHOTOMATCH_INDEX_STATUS_DB` | `data/indexing_status.sqlite3` | Persistent SQLite database path. |
| `PHOTOMATCH_MODELS_DIR` | `models` | Persistent, gitignored model directory. DeepFace stores downloaded weights beneath `<value>/.deepface/weights`. |
| `PHOTOMATCH_ALLOWED_NETWORKS` | Loopback only in development; required in production | Comma-separated company computer IP addresses or CIDR ranges allowed to access any page. |
| `PHOTOMATCH_MAX_UPLOAD_MB` | `12` | Maximum customer lookup photograph size. |
| `PHOTOMATCH_EVENT_UPLOAD_MB` | `500` | Maximum HTTP request size for a batch of event photographs. |
| `PHOTOMATCH_TRUST_PROXY` | `0` | Trust one reverse-proxy hop. Enable only for a correctly configured trusted proxy. |
| `PHOTOMATCH_WEB_THREADS` | `4` | Gunicorn thread count. The production process count remains fixed at one. |

Boolean variables use `1` for enabled and `0` for disabled.

## Standard local launch

PowerShell environment variables apply to the current terminal session and its
child processes. They are not written to the repository.

```powershell
$env:PHOTOMATCH_ADMIN_USERNAME = 'admin'
$env:PHOTOMATCH_ADMIN_PASSWORD = 'choose-a-password'
$env:PHOTOMATCH_SECRET_KEY = 'choose-a-long-private-secret'

python -m src.web
```

Open `http://127.0.0.1:5000/`; it redirects to the staff login. Public attendee
search routes are disabled by default.

The local entry point always binds to `127.0.0.1:5000`; it does not expose the
server to other devices on the network.

For the owner-only Windows pilot, use `Setup-Owner-Pilot.cmd` once and then
`Start-PhotoMatch.cmd`. See [`OWNER_PILOT.md`](OWNER_PILOT.md) for the offline
model check, private credential storage, data locations, and trial checklist.

## Common variants

Pause automatic background work while keeping the web interface available:

```powershell
$env:PHOTOMATCH_BACKGROUND_WORKER = '0'
python -m src.web
```

Enable local debug mode:

```powershell
$env:PHOTOMATCH_DEBUG = '1'
python -m src.web
```

Use secure session cookies behind a local HTTPS deployment:

```powershell
$env:PHOTOMATCH_SECURE_COOKIE = '1'
python -m src.web
```

## Configuration stored in SQLite

Match thresholds, search candidate count, cluster similarity, and minimum
cluster size are edited from the operator dashboard. These values are stored in
SQLite and are not controlled by environment variables.

Event folders, model names, and pipeline constants continue to come from
`config.py`. They are application-level constants rather than deployment
environment settings.

For Docker, HTTPS, persistent volumes, backup, GPU setup, and rollback, see
[`DEPLOYMENT.md`](DEPLOYMENT.md).
