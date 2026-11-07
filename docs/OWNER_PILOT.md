# Private Owner Pilot (Windows)

This is the first deployment stage for Golden Camera. PhotoMatch runs only on
the owner's work PC at `127.0.0.1`; no worker computer and no internet client can
open it. Events, photographs, indexes, settings, and activity stay on that PC.

## Before setup

- Use Windows 10 or 11 with Python 3.11 installed.
- Keep at least 10 GB of free space initially, plus enough space for event
  photographs and backups.
- Internet is needed during the one-time setup to install Python packages and
  download the FaceNet model. Normal pilot operation works after disconnecting
  the internet.
- Run setup and later start PhotoMatch from the same Windows user account. The
  owner credential file is encrypted by Windows for that account on that PC.

## One-time setup

1. Double-click `Setup-Owner-Pilot.cmd`.
2. Choose the owner's username.
3. Enter and confirm a password containing at least 12 characters.
4. Wait for package installation, model preparation, and the offline check.
5. Setup is complete only when it prints `Offline verification passed`.

The setup process deliberately starts a fresh Python process with network socket
creation blocked. It loads both face models from the persistent `models` folder;
failure stops setup instead of allowing an unprepared offline pilot.

## Start and stop

1. Disconnect the PC from the internet for the offline test.
2. Double-click `Start-PhotoMatch.cmd`.
3. Open `http://127.0.0.1:5000` on that same PC.
4. Sign in with the owner credentials selected during setup.
5. To stop PhotoMatch, return to its terminal window and press `Ctrl+C`.

The launcher forces loopback-only access and does not trust proxy headers. Even
another computer on the same Wi-Fi cannot open this Phase 1 site.

## Where private data is stored

| Data | Location | Notes |
|------|----------|-------|
| Event photographs and generated indexes | `data/events/` | Persistent and ignored by Git. |
| Event/status/settings database | `data/indexing_status.sqlite3` | Persistent and ignored by Git. |
| Face model weights | `models/.deepface/weights/` | Prepared once, persistent, and ignored by Git. |
| Owner credentials | `.photomatch-owner-config.xml` | Windows-encrypted for the setup account and ignored by Git. |
| Customer lookup photo | Memory only | Not added to the event or retained after the temporary search expires. |

Do not copy the credential file to another computer: Windows cannot decrypt it
there. To change the username/password, run setup again from the same account.

## Pilot checklist

For each trial, record the event size, indexing time, GPU/CPU status, search
time, incorrect matches, incorrect face clusters, and any errors.

- Create a small event and a realistically large event.
- Upload event photos, then pause/resume and retry indexing.
- Search using several people under different lighting and angles.
- Inspect clustered groups for different people incorrectly merged together.
- Change clustering thresholds and use **Try indexing again** rather than
  creating another event.
- Restart Windows and confirm the events still exist.
- Delete a disposable test event and confirm it does not return.
- Keep the internet disconnected throughout a complete trial.

Do not begin storing irreplaceable real-event data until an encrypted external
backup and a successful restore test are in place.
