# PhotoMatch

Local face recognition for instant event-photo retrieval — built for
Golden Camera, a photography business shooting weddings, corporate
events, and school/university graduations.

Students currently spend 5–6 hours manually searching 10,000+ photos per
event to find themselves. PhotoMatch lets them upload one selfie and get
their photos back in seconds — running entirely on local hardware, no
cloud upload, no per-event cost.

CPEG 418 — Computer Vision — American University of Kuwait — Summer 2026.
Full write-up: [`PhotoMatch_Proposal.pdf`](PhotoMatch_Proposal.pdf).

## Status

Week 4 of 4 — evaluation & validation tooling is built and tested;
running it against real event photos to produce final numbers is the
remaining work. Weeks 1–3 (proposal through interface) are done. See
[`ROADMAP.md`](ROADMAP.md) for the full task breakdown and owners.

## Pipeline

```
Ingest → Preprocess → Detect & Embed → Index → Match → Deliver
```

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

The public attendee interface uses Flask. Launch it locally with:

```bash
python -m src.web
```

Then open `http://127.0.0.1:5000`. The server binds to localhost in this
development entry point. Attendees enter the 8-character code printed by the event's
index command; events are not listed on the public page. The existing desktop interface
(`src/interface/`) remains available during web-interface review and uses Tkinter, which
ships with Python on Windows and macOS. On Linux you may need to install it separately:
```bash
sudo apt install python3-tk
```

Full contributor workflow (branching, PRs, review): see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Running the tests

```bash
pip install pytest
pytest tests/ -v
```

130 tests across preprocessing, detection, indexing, matching, threshold tuning, and
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

# First index run (registers the event date and prints its attendee code):
python scripts/demo_goated_index_and_search.py index my_event --date 2026-07-01

# Later runs process only newly added photos:
python scripts/demo_goated_index_and_search.py index my_event

# Force a complete rebuild after manual recovery work:
python scripts/demo_goated_index_and_search.py index my_event --force

# Search that event with a selfie, from the command line:
python scripts/demo_goated_index_and_search.py search my_event path/to/selfie.jpg

# Tune thresholds from labeled genuine/impostor similarity scores:
python scripts/tune_thresholds.py path/to/scores.csv

# Or launch the full kiosk app:
python -m src.interface.app
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
the worker from the future staff/admin Flask interface remains follow-up work.

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
