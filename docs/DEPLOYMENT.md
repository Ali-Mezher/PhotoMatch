# Production deployment

PhotoMatch is a staff-only Flask service. Run it behind HTTPS, keep its data on
an encrypted local disk, and do not expose the data volume or SQLite file over a
public file share.

## Required configuration

Copy `.env.example` to `.env` and replace every placeholder. Production startup
fails if the fixed staff username, password, stable secret key, or secure-cookie
setting is missing.

Generate a secret without saving it in shell history:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The application has one fixed staff account. To change it, update
`PHOTOMATCH_ADMIN_USERNAME` or `PHOTOMATCH_ADMIN_PASSWORD` in the deployment
secret store and restart the container. There is no registration or password
recovery page. Never commit `.env`.

Important settings:

- `PHOTOMATCH_COMPUTE_DEVICE=auto` prefers a TensorFlow GPU and falls back to CPU.
- `PHOTOMATCH_BACKGROUND_WORKER=1` starts the single indexing coordinator.
- `PHOTOMATCH_TRUST_PROXY=1` is required only behind a trusted reverse proxy that
  sets `X-Forwarded-For`, `X-Forwarded-Proto`, and `Host` correctly.
- `PHOTOMATCH_ALLOWED_NETWORKS` is required in production. It accepts a
  comma-separated list of approved computer IP addresses or CIDR ranges. Use a
  `/32` entry for each company computer when device-specific restriction is
  required, for example `192.168.50.20/32,192.168.50.21/32`.
- `PHOTOMATCH_SECURE_COOKIE=1` is mandatory in production, so the browser-facing
  URL must use HTTPS.

## CPU deployment

```bash
cp .env.example .env
# Edit .env and put TLS/reverse-proxy secrets in place.
docker compose build
docker compose up -d
curl --fail http://127.0.0.1:8000/healthz
```

Publish port 8000 only to a trusted HTTPS reverse proxy. The application uses
one Gunicorn process with multiple threads because one process must own the
SQLite database and background coordinator.

## Company-computer access only

The application denies every request whose source address is outside
`PHOTOMATCH_ALLOWED_NETWORKS`, including requests to the login and health pages.
Local development defaults to loopback access, so only the computer running the
application can connect until an allowlist is supplied. The server's own loopback
address remains allowed for health checks and local administration.

For computers on one shop network, assign each company computer a stable IP in
the router and list each address with `/32`. Allowing a whole subnet such as
`192.168.50.0/24` also permits personal devices connected to that subnet, so use
it only when the network itself accepts managed company devices exclusively.

For a cloud deployment, use the company's fixed public egress IP or, preferably,
a company VPN and allow only its address range. When an HTTPS reverse proxy is
present, enable `PHOTOMATCH_TRUST_PROXY=1` only when the application port cannot
be reached except through that trusted proxy. Otherwise a client could forge a
forwarded address. Keep the server firewall restricted as an independent layer;
the application allowlist complements rather than replaces it.

## NVIDIA GPU deployment

Use Linux, a supported NVIDIA driver, Docker, and NVIDIA Container Toolkit.
Confirm that `nvidia-smi` works on the host and that Docker can access a GPU,
then build the optional TensorFlow CUDA dependency set:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Open **Advanced settings** in the staff dashboard. System status displays
`GPU acceleration` when TensorFlow accepted a GPU or `CPU mode` when no
compatible GPU is available. The same selection and fallback reason are written
to the container log:

```bash
docker compose logs photomatch
```

FAISS indexes are intentionally saved in CPU-portable form. GPU acceleration is
used for TensorFlow-backed face detection and embedding inference; a runtime GPU
inference error is logged and retried once on CPU.

## Persistent data

Compose creates `photomatch-data` for event photographs, indexes, and SQLite,
plus `photomatch-models` for downloaded model assets. These volumes survive
container replacement. Restrict host and backup access because event photos and
face embeddings are biometric data. `PHOTOMATCH_MODELS_DIR=/app/models` also
directs DeepFace's cache into that persistent model volume instead of the
container user's temporary home directory.

## Backup and restore

For a consistent backup, stop the service before copying its data:

```bash
docker compose stop photomatch
docker compose cp photomatch:/app/data ./backup-data
docker compose cp photomatch:/app/models ./backup-models
docker compose start photomatch
```

To restore, stop the service, keep the current volume as a rollback copy, copy
the saved directories back into `/app`, and start the service. Confirm the
health endpoint and one known event before deleting the rollback copy.

## Upgrade and rollback

1. Back up data and record the current image/tag or Git commit.
2. Build the new image and run the automated test suite.
3. Restart with `docker compose up -d --build`.
4. Check `/healthz`, logs, sign-in, and a test event search.
5. To roll back, deploy the recorded image/tag and restore the pre-upgrade data
   only if the new version changed persistent data incompatibly.

Gunicorn allows 30 seconds for graceful worker shutdown. Large active indexing
jobs should be paused in the dashboard before a planned upgrade.

## Operational checks

- Health: `GET /healthz` returns only service name and status.
- Logs: `docker compose logs --tail=200 photomatch`.
- Sign-in: `/` redirects to the fixed staff login; public event-code and selfie
  routes are not registered in staff-only mode.
- Storage: monitor the Docker volume and retain encrypted backups according to
  `docs/PRIVACY_AND_RETENTION.md`.

Horizontal web scaling is not supported with the in-process worker and SQLite.
Moving to multiple application processes requires a separate job queue and a
database designed for multi-process deployment.
