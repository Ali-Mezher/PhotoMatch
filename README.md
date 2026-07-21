# PhotoMatch Testing Branch

This is the internal `testosterone` branch for repeatable, end-to-end testing of
PhotoMatch. It intentionally keeps local event photos, generated indexes, and the SQL
catalog together so the attendee interface, admin dashboard, incremental indexing,
clustering, search, and downloads can be exercised against a shared fixture snapshot.

This branch is not a release branch and should not be merged into `dev`. Rebase it onto
`dev` as the application moves forward. Event photos and face indexes may contain
sensitive data; do not copy them to other branches or replace the dummy credentials
below with real credentials.

## Setup

Create and activate a virtual environment, then install the project dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The application reads admin credentials from the current PowerShell session. Set the
dummy testing account and a local Flask session secret before every new terminal
session:

```powershell
$env:PHOTOMATCH_ADMIN_USERNAME = 'admin'
$env:PHOTOMATCH_ADMIN_PASSWORD = 'pass'
$env:PHOTOMATCH_SECRET_KEY = 'testing-only-local-secret'
python -m src.web
```

Open the attendee interface at `http://127.0.0.1:5000/` or the admin dashboard at
`http://127.0.0.1:5000/admin/`. The background indexing worker is enabled by default.
Use `$env:PHOTOMATCH_BACKGROUND_WORKER = '0'` only when testing without it.

## Working with fixtures

The event folders and `data/indexing_status.sqlite3` are deliberately tracked on this
branch. Running the app can modify the database and generated indexes, while adding or
deleting photos changes the fixture set. Review those changes before committing so the
shared snapshot stays intentional. `.env` and `.venv` remain ignored.

Run the complete automated suite before publishing a new snapshot:

```powershell
python -m pytest
```

## Keeping up with `dev`

Rebase this branch whenever `dev` advances:

```powershell
git switch testosterone
git fetch origin
git rebase origin/dev
python -m pytest
git push --force-with-lease origin testosterone
```

Resolve any conflicts in favor of the current `dev` application code while preserving
this branch-specific README, `.gitignore` fixture exceptions, and deliberate fixture
state. Never use a plain forced push.
