# PhotoMatch Final Presentation Outline

Target length: 10–12 minutes plus demonstration and questions.

## Slide 1 — Title and client problem (45 seconds)

- PhotoMatch: local face recognition for instant event-photo retrieval
- Golden Camera handles 10,000+ photos at a graduation event
- Manual search baseline: five to six hours per participant

**Visual:** one event folder narrowing to one participant's photos.

## Slide 2 — Objectives and scope (45 seconds)

- One selfie, one selected event, two result tiers
- Local/offline processing; no cloud upload
- Desktop kiosk workflow
- Out of scope: payment, mobile app, automatic identity labelling

## Slide 3 — Architecture (60 seconds)

```text
Preprocess -> detect -> embed -> per-event FAISS index -> match -> display
```

- Explain why event isolation protects relevance and privacy
- Explain why event photos can contain multiple indexed faces

## Slide 4 — Computer-vision pipeline (60 seconds)

- OpenCV correction, low-light improvement, denoising, and sharpening
- Face detection with confidence and size filtering
- Normalized 512-dimensional Facenet512 embeddings
- Exact cosine-similarity search using FAISS inner product

## Slide 5 — Matching and interface (60 seconds)

- Best face per source photograph removes duplicates
- Confident and possible thresholds balance false accepts and rejects
- Background search keeps the Tkinter interface responsive
- Selfie preview, clear recovery messages, responsive result thumbnails

## Slide 6 — Verification (60 seconds)

- 130 automated tests pass
- Coverage areas: preprocessing through UI integration and evaluation tooling
- Clearly distinguish software verification from real-event accuracy

**Command:** `python -m pytest tests -q`

## Slide 7 — Evaluation protocol and results (90 seconds)

- Precision@k, recall@k, FAR, FRR, end-to-end query time
- Scalability from 10,000 to 100,000 synthetic faces
- Manual robustness review for lighting, pose, occlusion, blur, and group size
- Time savings against the five-to-six-hour baseline

**Before presenting:** paste only aggregate results from a consented evaluation
run. Include event size, number of identities, k, thresholds, and hardware. If
the run has not happened, state “evaluation tooling complete; real-event
measurement pending” rather than showing synthetic accuracy as client data.

## Slide 8 — Privacy and deployment (75 seconds)

- Affirmative consent; guardian consent for minors
- Separate evaluation consent and manual alternative
- Encrypted local storage, least privilege, offline operation
- In-memory selfie/query embedding
- 30-day evaluation deletion and event/index deletion deadline
- Verified removal from backups and a documented incident process

## Slide 9 — Demonstration (2 minutes)

1. Select an indexed event.
2. Choose a consented stand-in selfie and show its preview.
3. Search without freezing the interface.
4. Explain confident and possible result tiers.
5. Show the no-face or unindexed-event guidance if time allows.

Do not use a real participant's face in a recorded presentation without
explicit presentation consent. A synthetic or team-owned stand-in is safer.

## Slide 10 — Limitations and next steps (60 seconds)

- Real-event and demographic evaluation is still required
- Difficult pose, blur, occlusion, and small faces remain risks
- No production liveness/anti-spoofing in v1
- Next: threshold validation, protected previews, staff audit controls

## Slide 11 — Conclusion (30 seconds)

- Complete local retrieval pipeline and reproducible evaluation tooling
- Major potential reduction in manual search time
- Privacy and deletion are part of the operating procedure

## Likely questions

**Why FAISS inner product?**  Embeddings are normalized, so inner product equals
cosine similarity and `IndexFlatIP` provides exact search.

**Why two thresholds?**  A strict confident tier limits false accepts, while a
possible tier lets a participant review borderline images instead of silently
missing them.

**Does the system identify everyone automatically?**  No. It searches one
selected event using a user-provided selfie and does not attach permanent names
to the event index.

**Are selfies stored?**  The implemented workflow reads the selfie for the
search and does not intentionally persist it or its query embedding.

**What proves accuracy?**  Automated tests prove software behavior. Accuracy
must be reported from the provided benchmark using representative, consented
event data; the presentation must identify the dataset size and conditions.
