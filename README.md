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

Week 3 of 4 — indexing, matching & interface in progress; preprocessing
and detection (Week 2) are done. See [`ROADMAP.md`](ROADMAP.md) for the
full task breakdown and owners.

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

## Setup

```bash
git clone https://github.com/Ali-Mezher/PhotoMatch.git
cd PhotoMatch
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The interface (`src/interface/`) uses Tkinter, which ships with Python
on Windows and macOS. On Linux you may need to install it separately:
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

36 tests across preprocessing, detection, indexing, and matching — all
run on synthetic data and need no model downloads or sample photos.
Full detection/embedding inference needs `mtcnn` and `deepface`
installed (already in `requirements.txt`) — see the integration-check
snippet at the bottom of `tests/test_detection.py`.

## Quick manual checks

Once you have real (or stand-in) photos locally under
`data/events/<event_id>/raw/`:

```bash
# Run preprocess -> detect -> embed on a single photo:
python scripts/demo_pipeline.py path/to/photo.jpg

# Build the searchable index for an event:
python scripts/demo_index_and_search.py index my_event

# Search that event with a selfie, from the command line:
python scripts/demo_index_and_search.py search my_event path/to/selfie.jpg

# Or launch the full kiosk app:
python -m src.interface.app
```

## Data & privacy

Real event photos are never committed — `data/events/` and all image
files under it are gitignored, along with generated embeddings/indexes
and model weights. See [`.gitignore`](.gitignore). Photos contain
biometric data of real people, including minors — handle accordingly.

## Team

| Name | ID | Role |
|---|---|---|
| Mahmood Tendail | 66597 | Project Lead & Face Detection |
| Ali Mezher | 61392 | Preprocessing Engineer |
| Malek AlKashat | 61513 | Indexing & Search |
| Hazem Ahmed | 70493 | Matching Logic |
| Nourallah Mourad | 59762 | Interface & Clustering |
| Ahmad AlAli | 64024 | Evaluation & Report |
