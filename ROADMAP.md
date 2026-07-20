# PhotoMatch — Project Roadmap

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
| [#8](https://github.com/Ali-Mezher/PhotoMatch/issues/8) | Indexing: build per-event FAISS vector index | Ali Mezher & Mahmood Tendail | ✅ Done | Builds, saves, loads, and queries an isolated FAISS index per event; covered by an end-to-end integration test. |
| [#9](https://github.com/Ali-Mezher/PhotoMatch/issues/9) | Matching: cosine similarity search & confidence-tiered results | Ali Mezher & Mahmood Tendail | ✅ Done | Selfie embeddings are searched against one event's FAISS index, deduplicated by photo, and returned in confidence tiers; final threshold tuning remains in issue #10. |
| [#10](https://github.com/Ali-Mezher/PhotoMatch/issues/10) | Matching: threshold tuning (FAR vs FRR trade-off) | Ali Mezher & Nourallah Mourad | ✅ Done | Tested FAR/FRR tuner recommends confident and possible thresholds from labeled genuine/impostor scores; collect a representative dataset before changing production values. |
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
| #18 | Privacy: consent, secure biometric storage & retention/deletion policy | Nourallah Mourad | ✅ Done | `docs/PRIVACY_AND_RETENTION.md` defines affirmative and guardian consent, separate evaluation consent, local encrypted storage and least-privilege access, exact retention limits, verified deletion, and incident response. |
| #19 | Report & presentation: final write-up | — | ⬜ Not started | Incorporate the completed evaluation, validation, privacy, and deployment-readiness results. |

---

## Post-Core — Stretch Goals
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| #20 | Interface: auto-clustering of similar faces *(stretch goal)* | — | ⬜ Not started | Cluster detected faces into candidate identities for staff review. |
| #21 | Security: liveness / anti-spoofing *(stretch goal)* | — | ⬜ Not started | Add and evaluate a basic blink, motion, or texture-based check to reduce searches using another person's photograph. |
| #22 | Interface: watermarked previews & admin dashboard *(stretch goal)* | — | ⬜ Not started | Show protected previews and give staff indexing status, failed-job retry, and manual review controls. |
| #23 | Indexing: automatic event queue & service foundation | Malek AlKashat | 🟨 Implemented — pending review/merge | Adds a framework-independent `PhotoMatchService`, local per-event/per-image SQLite status tracking, stale-event auto-queueing, and a bounded indexing scheduler (default 1, tunable up to 3 workers). Tkinter now uses the service layer; the future Flask interface will reuse it. 133 tests pass. |

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
