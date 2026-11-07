# PhotoMatch — Project Roadmap

**Last updated:** 7 November 2026

**Deployment goal:** Run PhotoMatch privately for Golden Camera. First, the
owner tests it only on his work PC. If the pilot succeeds, that PC (or a later
dedicated office PC) serves the same application and event library to 3–4
approved workers over the private office network. The application is not
published to the internet.

## Resume Here — Next Checkpoint

Phase 1 software preparation and evaluation tooling are implemented. Phase 1 is
still open: installation on the owner's work PC, real-photo accuracy results,
offline operation, restart/recovery, and backup/restore acceptance remain pending.
Phase 2 worker accounts and office sharing have not started.

The immediate next step is a small evaluation dry run with **5 people, 30–50
event photographs, and 2 independent selfies per person**. Identify every event
photo containing each participant, index the event, run the evaluation, and
confirm face labels before interpreting clustering recommendations.

After the dry run, prepare two calibration events and one separate final test
event. A practical initial target is 10–15 labeled people and 75–150 event photos
per event, with several appearances per person and separate shop-style selfies.
These are pilot targets, not evidence of production accuracy. Keep the final
event unused while choosing thresholds. No production thresholds have yet been
selected from real company data.

The last recorded full automated run passed 256 tests. This verifies software
behavior on synthetic/mocked inputs; target-PC model inference and business
acceptance are separate checks.

## Current Implementation Snapshot

| Area | Status | Current result |
|------|--------|----------------|
| Face detection, embeddings, indexing, and matching | ✅ Done | Each event has an isolated FAISS index and searchable photograph library. |
| Staff-only workflow | ✅ Done | The public customer page is disabled. A worker takes or uploads the customer's photo inside the shop. Lookup images and embeddings are kept in memory rather than saved as event data. |
| Operator dashboard | ✅ Done | Workers can create events, upload photographs, monitor indexing, search within an event, review activity, and inspect clustered face groups. |
| Central event storage | ✅ Done | Events, source photographs, generated indexes, cluster data, and operational state live on the computer running PhotoMatch. Everyone using that one server sees the same events. |
| Indexing controls | ✅ Done | Incremental processing, pause/resume/stop, retry, and an always-available full reindex action are implemented. A threshold change does not require creating a new event. |
| GPU support | ✅ Done | Compatible GPU acceleration is selected automatically, with a CPU fallback when a GPU is unavailable or fails. |
| Threshold evaluation | ✅ Enhanced tooling ready | One command supports multiple selfies, exports labeled matching scores, creates human-review face crops, exports confirmed embeddings, ranks matching/clustering grids, and warns about cross-person merges. Final values still require representative shop/event testing. |
| Conservative identity clustering | ✅ Done | Mutual-neighbor and cohesion checks reduce false merges; staff can inspect groups, label them, and leave uncertain faces unclustered. |
| Fixed operator login | ✅ Done for owner pilot | One configured username and password protects the current dashboard. Separate users and roles are not implemented yet. |
| Company-computer network allowlist | ✅ Implemented | Unapproved source IPs receive a 403 before login. Production requires an explicit allowlist; the company's real device IPs still need to be configured. |
| Deployment packaging | ✅ Ready for testing | Docker, CPU/GPU configuration, persistent volumes, health checks, and production configuration validation are documented. The application has not yet been installed on the owner's work PC. |
| Runtime photos in Git | ✅ Current tree protected; history audit pending | Runtime event data is ignored and absent from the current project tree. The 7 November check found three local-only commits containing old event data; the publication branch is based on remote `dev` and excludes those commits. Audit remaining local/remote historical references separately before declaring the purge complete. |
| Automated verification | ✅ Passing | The latest full run completed with 256 passing tests. |
| Real-shop validation | 🔧 Next | Reliability, accuracy, thresholds, performance, storage growth, and recovery must be proven with representative events on the target PC. |
| Worker accounts and permissions | ⏳ Phase 2 | Needed only after the owner approves the private single-PC pilot. |

---

## Week 1 — Requirements & Proposal ✅
- Confirmed scope with client (Golden Camera)
- Gathered initial sample of event photos
- Submitted project proposal

---

## Week 2 — Detection & Preprocessing
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| [#1](https://github.com/Ali-Mezher/PhotoMatch/issues/1) | Setup: configure dev environment & install dependencies | Ali Mezher | ✅ Done | |
| [#2](https://github.com/Ali-Mezher/PhotoMatch/issues/2) | Preprocessing: color & geometry correction | Ali Mezher | ✅ Done | |
| [#3](https://github.com/Ali-Mezher/PhotoMatch/issues/3) | Preprocessing: intensity transformation (brightness & contrast) | Ali Mezher | ✅ Done | |
| [#4](https://github.com/Ali-Mezher/PhotoMatch/issues/4) | Preprocessing: convolution filtering (denoise & sharpen) | Ali Mezher | ✅ Done | |
| [#5](https://github.com/Ali-Mezher/PhotoMatch/issues/5) | Preprocessing: morphological operations & segmentation | Ali Mezher | ✅ Done | |
| [#6](https://github.com/Ali-Mezher/PhotoMatch/issues/6) | Detection: implement face detector (MTCNN / RetinaFace) | Ali Mezher & Mahmood Tendail | ✅ Done | |
| [#7](https://github.com/Ali-Mezher/PhotoMatch/issues/7) | Detection: generate face embeddings (FaceNet / ArcFace) | Ali Mezher & Mahmood Tendail | ✅ Done | |

---

## Week 3 — Indexing, Matching & Interface
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| [#8](https://github.com/Ali-Mezher/PhotoMatch/issues/8) | Indexing: build per-event FAISS vector index | Ali Mezher & Mahmood Tendail | ✅ Done | Builds, saves, loads, and queries an isolated FAISS index per event. SQLite-backed per-image status now enables incremental additions, ordered background jobs, and safe rebuilds for changed/deleted sources. Use `scripts/demo_goated_index_and_search.py`; pass `--force` only for a complete rebuild. |
| [#9](https://github.com/Ali-Mezher/PhotoMatch/issues/9) | Matching: cosine similarity search & confidence-tiered results | Ali Mezher & Mahmood Tendail | ✅ Done | Selfie embeddings are searched against one event's FAISS index, deduplicated by photo, and returned in confidence tiers; final threshold tuning remains in issue #10. |
| [#10](https://github.com/Ali-Mezher/PhotoMatch/issues/10) | Matching: threshold tuning (FAR vs FRR trade-off) | Ali Mezher & Nourallah Mourad | 🔧 Tooling complete; calibration pending | Tested FAR/FRR tuner recommends confident and possible thresholds from labeled genuine/impostor scores; collect a representative dataset before changing production values. |
| [#11](https://github.com/Ali-Mezher/PhotoMatch/issues/11) | Interface: selfie upload & results display UI | Nourallah Mourad & Mahmood Tendail | ✅ Done | Selfie validation and preview, background matching, confidence-tiered results, responsive thumbnails, and pipeline error handling are implemented and covered by integration-focused tests. |
---

## Week 4 — Evaluation & Validation
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| [#13](https://github.com/Ali-Mezher/PhotoMatch/issues/13) | Evaluation: precision/recall @ top-k & query time | Ahmad AlAli & Mahmood Tendail | 🔧 Tooling ready | `src/evaluation/benchmark.py` runs real selfies through the full pipeline and computes precision/recall@k + timing; needs real (or realistic stand-in) photos dropped into `data/evaluation/` to produce real numbers — see `data/evaluation/README.md`. |
| [#14](https://github.com/Ali-Mezher/PhotoMatch/issues/14) | Evaluation: FAR & FRR benchmarks | Ahmad AlAli & Mahmood Tendail | 🔧 Tooling ready | `threshold_tuning.py` (tested) + `run_evaluation.py` now feed it real genuine/impostor scores collected from `benchmark.py`, not just the toy `scores.csv`; run against real data to get production threshold values for `config.py`. |
| #15 | Validation: scalability & fully offline operation | Mahmood Tendail | 🔧 Tooling ready | `src/evaluation/scalability.py` benchmarks FAISS at up to 100,000 synthetic faces and includes an automated check that indexing/matching make no network calls. Run `python scripts/run_evaluation.py <event_id> --scalability`. |
| #16 | Validation: real-event robustness testing | — | 🔧 Tooling ready | The evaluation report flags the lowest-recall identities per run as a starting point for manual review — actually reviewing *why* (lighting, angle, occlusion, blur, group photos) needs real event photos and a human look. |
| #17 | Evaluation: manual vs automatic time savings | Mahmood Tendail | 🔧 Tooling ready | `time_savings.py` compares a real measured query time against the proposal's 5-6 hour manual baseline; `run_evaluation.py` prints and reports this automatically once a benchmark has run. |

---

## Week 5 — Deployment Readiness & Final Report
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| #18 | Privacy: consent, secure biometric storage & retention/deletion policy | Nourallah Mourad & Malek Alkashat | ✅ Done | `docs/PRIVACY_AND_RETENTION.md` defines affirmative and guardian consent, separate evaluation consent, local encrypted storage and least-privilege access, exact retention limits, verified deletion, and incident response. |
| #19 | Report & presentation: final write-up | Nourallah Mourad | ✅ Draft complete | `docs/FINAL_REPORT.md` and `docs/PRESENTATION_OUTLINE.md` integrate the implemented pipeline, verification, evaluation protocol, deployment, privacy, limitations, demo flow, and speaking notes. Real-event metric fields remain explicitly pending consented data. |

---

## Post-Core — Stretch Goals
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| #20 | Interface: auto-clustering of similar faces *(stretch goal)* | Malek AlKashat | ✅ Implemented; real-event validation pending | Desktop and operator workflows use conservative mutual-neighbor/cohesion grouping. The dashboard persists reviewable groups, unclustered faces, optional labels, and stale-generation state. |
| #21 | Security: liveness / anti-spoofing *(stretch goal)* | Malek AlKashat | 🔧 Implementation ready | Attendee search now requires a consent-gated, randomized front/turn/front camera sequence verified server-side with MTCNN pose landmarks and same-person embeddings. Challenges are short-lived, single-use, and in-memory; admin lookup remains unrestricted. Complete the documented live-camera and replay evaluation before marking done. |
| #22 | Interface: watermarked previews & admin dashboard *(stretch goal)* | — | ✅ Done | The attendee web flow uses protected previews and original-only exports. The authenticated local operator dashboard creates events, safely imports photos, exposes access codes and per-image status, coordinates indexing, supports retries, controls validated runtime settings, searches people directly within an event, provides searchable paginated event inventory, and exposes a paginated activity log. |
| #23 | Indexing: automatic event queue & service foundation | Malek AlKashat | ✅ Done | A shared service facade now routes desktop and Flask workflows through the per-image SQLite index state, immutable generations, one interrupt-driven worker, startup reconciliation, oldest-first ordering, and incremental additions without full-event reprocessing. |

---

## Operational Rollout Plan

### Phase 1 — Private Owner Pilot on One PC

**Goal:** The owner tests the complete workflow repeatedly before any employee
receives access.

| Step | Status | Work |
|------|--------|------|
| 1. Prepare an offline installation | 🔧 Installer ready; target-PC run pending | `Setup-Owner-Pilot.cmd` creates the private environment, installs packages, persists the face model, and runs a fresh-process offline verification with sockets blocked. It still must be run successfully on the owner's work PC. |
| 2. Keep access local to that PC | ✅ Supported | Start the local application on `127.0.0.1`; do not expose a LAN port, create router port forwarding, or publish a public URL. |
| 3. Configure the owner login | 🔧 Setup ready; target-PC run pending | Setup prompts for the owner's fixed username and 12+ character password, generates a private session secret, and stores both using Windows account-bound encryption outside Git. |
| 4. Run representative event trials | ⏳ Next | Create several realistic events, upload small and large batches, wait for indexing, search using staff-taken customer photos, inspect cluster groups, delete test events, and restart the application between trials. |
| 5. Tune and reindex | ⏳ Next | Compare threshold combinations, prioritizing prevention of different people being grouped together. Use full reindex after changing clustering thresholds and record the chosen values. |
| 6. Validate target hardware | ⏳ Next | Record GPU/CPU mode, indexing duration, search time, disk usage, errors, and behavior after interruption or reboot. |
| 7. Establish offline backup | ⏳ Before real use | Back up the central data directory and database to an encrypted external drive, then perform one restore test. Event photographs must not be backed up to Git. |
| 8. Owner acceptance | ⏳ Decision gate | The owner decides whether accuracy, speed, usability, privacy, and reliability are good enough to proceed to worker access. |

#### Phase 1A — Recognition Calibration and Holdout Testing

This work is part of Phase 1 and happens before the owner makes the acceptance
decision. PhotoMatch uses pretrained MTCNN and FaceNet512 models; the shop data
is used to build event indexes and calibrate decision thresholds, not to train a
new neural network from scratch.

| Step | Status | Work |
|------|--------|------|
| 1. Prepare consented data | ⏳ Needs real photos | Collect representative event photos for at least 10–20 people, one or more separate shop-style selfies per person, exact identity-to-photo ground truth, difficult lighting/angles, similar-looking people, and realistic non-matches. |
| 2. Split calibration and holdout events | ⏳ Needs real photos | Use at least two events to choose thresholds and reserve a different event for final testing. Never select values and claim final accuracy on the same photographs. |
| 3. Index each event | ⏳ Needs real photos | Create the event through the owner dashboard, upload its photographs, complete indexing, and record indexing time, failures, GPU/CPU mode, and face count. |
| 4. Run multi-selfie retrieval evaluation | ✅ Tooling ready | `scripts/run_evaluation.py` evaluates every selfie independently and reports precision@k, recall@k, query time, FAR, FRR, and the weakest identity/selfie combinations. |
| 5. Export matching calibration data | ✅ Tooling ready | The same command writes `matching-scores.csv` and evaluates a grid of possible/confident thresholds with false accepts ranked before recall. |
| 6. Confirm per-face identities | ✅ Workspace generator ready; human labels pending | The command creates local face crops and `face-labels.csv`. A human fills the identity column; unknown/background faces remain blank. Filename hints are never trusted as face identity. |
| 7. Evaluate clustering thresholds | ✅ Tooling ready; labels pending | Confirmed face labels are joined to indexed embeddings automatically. Edge/cohesion combinations are ranked first by cross-person false merges, then false splits, purity, and unassigned faces. |
| 8. Select calibration candidates | ⏳ Needs calibration results | Review `threshold-report.md` across all calibration events. Do not accept a clustering candidate with an observed cross-person merge; expand the grid or improve data when warnings remain. |
| 9. Lock settings and run holdout test | ⏳ Needs untouched event | Apply the candidates without further tuning and run the full workflow on the reserved event. Record search errors, cluster merges/splits, latency, and detection failures. |
| 10. Approve or repeat | ⏳ Decision gate | Keep the values only if the holdout results meet the owner's accuracy expectations. Otherwise return to calibration, document the failure type, and test another candidate without altering the holdout result afterward. |

**Phase 1 exit criteria:** The owner can complete the normal event workflow
without technical help; no unacceptable cross-person matches remain in the
agreed test set; data survives restarts; an event can be restored from backup;
and the application works while the internet/WAN connection is disconnected.

### Phase 2 — Private Office Sharing with 3–4 Workers

**Goal:** Keep one central event library while allowing only approved company
computers on the office LAN to use it. No employee device stores a separate
copy of the repository or event database.

| Step | Status | Work |
|------|--------|------|
| 1. Confirm the host computer | ⏳ Planned | Continue using the owner's PC only if it can remain powered on, awake, and available throughout working hours. Otherwise move the same central installation to a dedicated office PC. |
| 2. Configure the private LAN host | ⏳ Planned | Give the host a reserved/static private IP and listen only on the required private interface. Keep router port forwarding disabled so the service has no public internet route. |
| 3. Approve exact company devices | ⏳ Planned | Give each worker PC a reserved IP, add each address as a `/32` entry in `PHOTOMATCH_ALLOWED_NETWORKS`, and enforce the same allowlist in Windows Firewall. Do not allow the entire Wi-Fi subnet if personal devices can join it. |
| 4. Add separate accounts and roles | ⏳ Planned development | Replace the single shared operator login with individual password-hashed accounts. The owner receives the **Admin** role; each employee receives a **Worker** role. Record the acting account in the activity log. |
| 5. Enforce role permissions | ⏳ Planned development | Workers may create events, upload photos, monitor/retry indexing, search, and review clusters. Only the Admin may delete events, change global thresholds/settings, manage accounts, view security administration, or change retention policy. |
| 6. Share events centrally | ✅ Architecture supported | All browsers use the host's one application, events directory, SQLite database, and indexes. An event created by one authorized worker therefore appears automatically for the owner and all other authorized workers. |
| 7. Test concurrent work | ⏳ Planned validation | Test simultaneous logins, uploads, event creation, searches, indexing queue behavior, account attribution, and safe handling when two workers open the same event. |
| 8. Prove isolation | ⏳ Planned validation | Disconnect the office WAN/internet connection and repeat the workflow. Confirm approved company PCs still work over the LAN while an unapproved PC and a personal phone receive no access. |
| 9. Train workers | ⏳ After validation | Provide a short operating guide covering sign-in, event creation, uploads, reindexing, customer lookup, cluster review, logout, and who to contact for errors. |

**Phase 2 exit criteria:** Only named accounts on approved company computers can
connect; worker permissions cannot reach Admin-only actions; all users see the
same event state; simultaneous use is reliable; the system operates without
internet; and backup/restore remains verified.

### Phase 3 — Reliability Improvements After Adoption

These items are intentionally deferred until the owner confirms that PhotoMatch
is effective for the business:

- Run PhotoMatch automatically as a Windows service after reboot, without an
  interactive terminal.
- Schedule encrypted external-drive backups and periodic restore drills.
- Add disk-capacity warnings, backup-age warnings, and clearer operational
  diagnostics for the Admin.
- Apply the documented retention/deletion policy and audit account, event,
  settings, and deletion actions.
- Move to a dedicated office server if relying on the owner's daily-use PC
  causes availability or performance problems.
- Revisit VPN, cloud backup, or remote access only as a separate future decision.
  None is part of the current offline plan.

### Decisions to Confirm Before Phase 2

1. Whether every worker receives an individual account (recommended) or workers
   share one account.
2. Whether the proposed Worker permissions are correct, especially event
   deletion and threshold changes.
3. Whether the owner's PC can stay powered on and awake during all working hours.
4. Which encrypted external drive and retention schedule will be used for
   offline backups.

## Remaining Work Items

These are local roadmap identifiers, not newly created GitHub issues. Implement
and validate Phase 1 items before enabling employee access.

| ID | Priority / stage | Work item | Completion evidence |
|----|------------------|-----------|---------------------|
| PM-01 | Next / Phase 1A | Run the 5-person dataset dry run | Saved retrieval report, reviewed face labels, threshold grids, and a list of errors to investigate. |
| PM-02 | High / Phase 1A | Add a final-test mode that evaluates frozen settings | The untouched event reports accuracy at explicitly supplied search and clustering values without selecting new values from its results. The current command is a calibration tool. |
| PM-03 | High / Phase 1A | Validate labels and evaluation coverage | Reject ambiguous multi-face query photos; detect query/event duplicates; verify crop coordinates against preprocessing; detect stale labels after reindexing; report missing faces and unlabeled coverage. |
| PM-04 | High / Phase 1A | Evaluate clustering in the full event context | Preserve all indexed faces and their original-photo constraints when measuring confirmed identities, so unlabeled neighbors cannot silently disappear from evaluation. |
| PM-05 | High / Phase 1A | Compare calibration events and document chosen settings | One recorded setting pair per task, false-accept/merge counts, false-split/missed-match trade-offs, dataset size, and an untouched final-test report. |
| PM-06 | High / Phase 1 | Validate Windows installation on the owner's PC | Fresh setup, real detector/embedding inference, stable model cache, credential loading, GPU/CPU status, and full operation with internet disconnected. |
| PM-07 | Before real use / Phase 1 | Implement and verify external-drive backup/restore | Consistent database/photos/index backup while writes are stopped, encrypted destination, successful restore, and a verified event search. |
| PM-08 | High / Phase 1 | Measure restart and interruption recovery | Events survive Windows restart; interrupted indexing recovers; storage requirements and event-size performance are recorded. |
| PM-09 | Pending repository audit | Finish removing historical private event data | Audit all remaining local/remote references, then clean affected histories with coordinated repository permissions. The publication branch excludes the known local photo commits; deleting current files alone does not purge history. |
| PM-10 | After owner approval / Phase 2 | Add individual Admin/Worker accounts | Password hashes, enforced route permissions, account-attributed activity, and tests proving Workers cannot perform Admin-only actions. |
| PM-11 | After owner approval / Phase 2 | Validate private LAN sharing | Reserved host/device addresses, firewall rules, approved-device access, simultaneous work on shared events, and successful operation with WAN disconnected. |
| PM-12 | After adoption / Phase 3 | Improve unattended operation | Automatic service startup, scheduled backups, disk/backup warnings, and an owner operating guide. |

---

## Team
| Name | ID | Role |
|------|----|------|
| Mahmood Tendail | 66597 | Project Lead & Face Detection |
| Ali Mezher | 61392 | Preprocessing Engineer |
| Malek AlKashat | 61513 | Indexing & Search |
| Hazem Ahmed | 70493 | Matching Logic |
| Nourallah Mourad | 59762 | Interface & Clustering |
| Ahmad AlAli | 64024 | Evaluation & Report |
