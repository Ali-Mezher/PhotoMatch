# syntax=docker/dockerfile:1
FROM python:3.11-slim

ARG REQUIREMENTS_FILE=requirements.txt
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PHOTOMATCH_ENV=production \
    PHOTOMATCH_EVENTS_DIR=/app/data/events \
    PHOTOMATCH_INDEX_STATUS_DB=/app/data/indexing_status.sqlite3 \
    PHOTOMATCH_MODELS_DIR=/app/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-gpu.txt ./
RUN python -m pip install --no-cache-dir -r "$REQUIREMENTS_FILE"

COPY config.py wsgi.py gunicorn.conf.py ./
COPY src ./src
COPY models/.gitkeep ./models/.gitkeep

RUN useradd --create-home --uid 10001 photomatch \
    && mkdir -p /app/data/events /app/models \
    && chown -R photomatch:photomatch /app

USER photomatch
EXPOSE 8000
VOLUME ["/app/data", "/app/models"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl --fail --silent http://127.0.0.1:8000/healthz >/dev/null || exit 1

CMD ["gunicorn", "--config", "gunicorn.conf.py", "wsgi:app"]
