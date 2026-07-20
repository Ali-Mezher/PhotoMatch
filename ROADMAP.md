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
| [#13](https://github.com/Ali-Mezher/PhotoMatch/issues/13) | Evaluation: precision/recall @ top-k & query time | — | ⬜ Not started | |
| [#14](https://github.com/Ali-Mezher/PhotoMatch/issues/14) | Evaluation: FAR & FRR benchmarks | — | ⬜ Not started | |
| #17 | Validation: scalability & fully offline operation | — | ⬜ Not started | Verify that indexing and search require no cloud services, then benchmark representative indexes up to 50,000–100,000 photos. |
| #18 | Validation: real-event robustness testing | — | ⬜ Not started | Test varied lighting, angles, motion blur, group photos, caps/gowns, partial occlusion, and detection failures using representative event photos. |
| #19 | Evaluation: manual vs automatic time savings | — | ⬜ Not started | Compare end-to-end selfie search time with the current 5–6 hour manual-search baseline and document the improvement. |

---

## Week 5 — Deployment Readiness & Final Report
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| #16 | Privacy: consent, secure biometric storage & retention/deletion policy | — | ⬜ Not started | Document consent procedures, local storage protections, access controls, and when selfies, embeddings, and event data must be deleted. |
| [#15](https://github.com/Ali-Mezher/PhotoMatch/issues/15) | Report & presentation: final write-up | — | ⬜ Not started | Incorporate the completed evaluation, validation, privacy, and deployment-readiness results. |

---

## Post-Core — Stretch Goals
| Issue | Task | Ownership | Status | Notes |
|-------|------|-----------|--------|-------|
| [#12](https://github.com/Ali-Mezher/PhotoMatch/issues/12) | Interface: auto-clustering of similar faces *(stretch goal)* | — | ⬜ Not started | Cluster detected faces into candidate identities for staff review. |
| #20 | Security: liveness / anti-spoofing *(stretch goal)* | — | ⬜ Not started | Add and evaluate a basic blink, motion, or texture-based check to reduce searches using another person's photograph. |
| #21 | Interface: watermarked previews & admin dashboard *(stretch goal)* | — | ⬜ Not started | Show protected previews and give staff indexing status, failed-job retry, and manual review controls. |

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
