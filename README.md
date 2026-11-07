# PhotoMatch

Local face recognition for instant event-photo retrieval — built for
Golden Camera, a photography business shooting weddings, corporate
events, and school/university graduations.

Customers currently wait while staff manually search large event collections.
PhotoMatch lets an authorized shop worker take or upload a customer photo and
retrieve matching event photographs in seconds—entirely on controlled hardware.

CPEG 418 — Computer Vision — American University of Kuwait — Summer 2026.
Full write-up: [`PhotoMatch_Proposal.pdf`](PhotoMatch_Proposal.pdf).
Final deliverables: [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md) and
[`docs/PRESENTATION_OUTLINE.md`](docs/PRESENTATION_OUTLINE.md).

## Status

The core pipeline, staff dashboard, central event storage, GPU-to-CPU fallback,
threshold experiments, clustering review, reindex controls, deployment
packaging, and company-network allowlist are implemented. The next milestone is
a private offline pilot on the owner's work PC. Worker accounts, Admin/Worker
roles, and private LAN sharing follow only if that pilot is approved. See
[`ROADMAP.md`](ROADMAP.md) for completed work, rollout gates, and remaining
tasks.

For the first private Windows trial, follow
[`docs/OWNER_PILOT.md`](docs/OWNER_PILOT.md). The provided setup prepares the
face models, verifies they reopen with network sockets blocked, encrypts the
owner credentials for the current Windows account, and creates a loopback-only
launcher.

## Pipeline

```
Ingest → Preprocess → Detect & Embed → Index → Match → Deliver
```

Indexed events can also be grouped into conservative, staff-review candidate
identity clusters. This is separate from selfie matching: it never assigns a
person's name automatically and leaves uncertain faces unclustered.

| Stage | Module | Owner |
|---|---|---|
| Preprocessing | `src/preprocessing/` | Ali Mezher |
| Detection & embedding | `src/detection/` | Mahmood Tendail |
| Indexing | `src/indexing/` | Malek AlKashat |
| Matching | `src/matching/` | Hazem Ahmed |
| Interface | `src/interface/` | Nourallah Mourad |
| Evaluation | `src/evaluation/` | Ahmad AlAli |

Shared constants (image size, thresholds, paths) live in [`config.py`](config.py)
at the repo root — import from there rather than hardcoding values, so
everyone's modules stay compatible.

## Application services

[`src/services/`](src/services/) is the framework-independent application
boundary for both interfaces. Tkinter uses `PhotoMatchService`, a compatibility
facade over the same incremental `IndexingService` used by Flask, so the two
interfaces do not maintain separate schedulers or SQLite schemas.

On startup, the Flask coordinator reconciles registered events once and queues
only missing, changed, failed, or interrupted work. The Tkinter facade also
adopts legacy event folders before delegating to that service. Operational state
is persisted in a gitignored SQLite file. One interrupt-driven worker serializes
indexing and clustering; it does not poll on a timer or reprocess unchanged
photos when a new photo is added.

## Evaluation (Week 4)

`src/evaluation/` measures the system against real data: precision/recall@k,
query time, FAR/FRR thresholds, FAISS scalability up to 100,000 faces, an
offline-operation check, and manual-vs-automatic time savings — all tied
together into one report.

**To evaluate a real (or realistic stand-in) event, you only need to drop
photos into a folder — no code changes.** See
[`data/evaluation/README.md`](data/evaluation/README.md) for the exact
layout, then run:

```bash
python scripts/run_evaluation.py <event_id>            # core metrics
python scripts/run_evaluation.py <event_id> --scalability  # + FAISS scaling suite
```

This saves a full markdown report to `data/evaluation/<event_id>/report.md`.

## Setup

```bash
git clone https://github.com/Ali-Mezher/PhotoMatch.git
cd PhotoMatch
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The staff-only application uses Flask. Launch it locally with:

```bash
python -m src.web
```

Then open `http://127.0.0.1:5000`. The root redirects to the authenticated staff
dashboard. A worker selects an event, adds its photographs, waits for processing,
and takes or uploads a customer photo to retrieve matches. Public event-code and
selfie routes are disabled. Lookup photos and embeddings stay in memory and are
not saved. The existing desktop interface
(`src/interface/`) remains available during web-interface review and uses Tkinter, which
ships with Python on Windows and macOS. On Linux you may need to install it separately:
```bash
sudo apt install python3-tk
```

Full contributor workflow (branching, PRs, review): see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

### Operator dashboard

The authenticated dashboard is available at `http://127.0.0.1:5000/admin/`.
Before starting the app, configure one operator account and a stable Flask secret for
the current PowerShell session:

```powershell
$env:PHOTOMATCH_ADMIN_USERNAME = "operator"
$env:PHOTOMATCH_ADMIN_PASSWORD = 'choose-a-password'
$env:PHOTOMATCH_SECRET_KEY = '<a long random local secret>'
python -m src.web
```

The dashboard creates events, imports JPG/PNG photos, queues incremental indexing,
shows live per-event processing progress, controls validated search settings, and runs
review-only identity clustering. Each event page runs an in-memory staff-assisted
customer search. The event inventory supports filtering with configurable page sizes for larger
catalogs. The overview shows the 5 latest audit entries, with a separate 25/50/100-row
activity history. The worker is interrupt-driven and serializes indexing and clustering; it
never scans on a timer. Set
`PHOTOMATCH_BACKGROUND_WORKER=0` only when a separate process owns background work.
Debug mode is off by default and can be explicitly enabled for local development with
`PHOTOMATCH_DEBUG=1`.

See [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) for every supported environment
variable, its default, and standard local, paused-worker, debug, and HTTPS examples.
Production Docker, GPU, backup, upgrade, and rollback instructions are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

Tests across preprocessing, detection, indexing, matching, the Flask interfaces,
clustering, threshold tuning, and
evaluation — all run on synthetic data and need no model downloads or sample photos.
Full detection/embedding inference needs `mtcnn` and `deepface`
installed (already in `requirements.txt`) — see the integration-check
snippet at the bottom of `tests/test_detection.py`.

## Quick manual checks

Once you have real (or stand-in) photos locally under
`data/events/<event_id>/raw/`:

```bash
# Run preprocess -> detect -> embed on a single photo:
python scripts/demo_pipeline.py path/to/photo.jpg

# First index run (registers the event date and prepares its photographs):
python scripts/demo_goated_index_and_search.py index my_event --date 2026-07-01

# Later runs process only newly added photos:
python scripts/demo_goated_index_and_search.py index my_event

# Force a complete rebuild after manual recovery work:
python scripts/demo_goated_index_and_search.py index my_event --force

# Search that event with a selfie, from the command line:
python scripts/demo_goated_index_and_search.py search my_event path/to/selfie.jpg

# Tune thresholds from labeled genuine/impostor similarity scores:
python scripts/tune_thresholds.py path/to/scores.csv

# Compare matching and clustering threshold grids and write CSV/Markdown reports:
python scripts/experiment_thresholds.py --matching-csv scores.csv --possible-range 0.50:0.70:0.05 --confident-range 0.65:0.85:0.05

# Or launch the full kiosk app:
python -m src.interface.app

# Create candidate identity clusters for an already indexed event:
python scripts/cluster_event.py my_event
```

## Incremental indexing service

`src.services.IndexingService` keeps event and per-image indexing state in a
local SQLite database. Once an event has been indexed, adding a photo processes
only that photo. Changing or removing an existing photo safely rebuilds that
event so stale faces cannot remain searchable.

The worker is interrupt-driven: it sleeps until `request_index()` is called and
does not scan on a timer. Application layers should register/import an image,
then signal its event:

```python
from src.services import IndexingService

indexing = IndexingService()
indexing.register_event("graduation_2026", "2026-07-01")
indexing.start()
indexing.request_index("graduation_2026")

# During application shutdown:
indexing.shutdown()
```

Queued events run one at a time, ordered by their explicit event date (oldest
first). `demo_goated_index_and_search.py index` uses this incremental path; pass
`--force` only when a complete recovery rebuild is required. Starting and supervising
the shared worker is now handled by the Flask operator dashboard.

The threshold-tuning CSV needs `score` and `label` columns. Use
`genuine` when both faces belong to the same person and `impostor` when
they belong to different people:

```csv
score,label
0.86,genuine
0.42,impostor
```

The tool reports FAR and FRR and recommends values for `config.py`; it
does not change production thresholds automatically.

For configurable matching and identity-group grids, CSV/Markdown output, false
merge ranking, and input formats, see
[`docs/THRESHOLD_EXPERIMENTS.md`](docs/THRESHOLD_EXPERIMENTS.md).

## Data & privacy

Real event photos are never committed — `data/events/` and all image
files under it are gitignored, along with generated embeddings/indexes
and model weights. See [`.gitignore`](.gitignore). Photos contain
biometric data of real people, including minors — handle accordingly.

Before collecting or loading any real data, follow the project's
[privacy, consent, access-control, and deletion policy](docs/PRIVACY_AND_RETENTION.md).
It defines the required notice and consent records, separate guardian and
evaluation consent, local-storage protections, retention deadlines, and the
verified event-deletion procedure.

## Team

| Name | ID | Role |
|---|---|---|
| Mahmood Tendail | 66597 | Project Lead & Face Detection |
| Ali Mezher | 61392 | Preprocessing Engineer |
| Malek AlKashat | 61513 | Indexing & Search |
| Hazem Ahmed | 70493 | Matching Logic |
| Nourallah Mourad | 59762 | Interface & Clustering |
| Ahmad AlAli | 64024 | Evaluation & Report |
