# PhotoMatch: Local Face Recognition for Event-Photo Retrieval

## Final Technical Report

**Course:** CPEG 418 — Computer Vision  
**Institution:** American University of Kuwait  
**Client:** Golden Camera  
**Term:** Summer 2026

### Team

| Member | ID | Primary responsibility |
|---|---:|---|
| Mahmood Tendail | 66597 | Project lead and face detection |
| Ali Mezher | 61392 | Image preprocessing |
| Malek AlKashat | 61513 | Indexing and search |
| Hazem Ahmed | 70493 | Matching logic |
| Nourallah Mourad | 59762 | Interface and privacy documentation |
| Ahmad AlAli | 64024 | Evaluation and reporting |

## Abstract

Golden Camera may produce more than 10,000 photographs at one graduation
event. Finding every photograph of one participant can require five to six
hours of manual review. PhotoMatch is a local desktop system that accepts one
selfie, extracts a face embedding, searches only the selected event's FAISS
index, and displays confident and possible matches. The implemented pipeline
combines OpenCV preprocessing, MTCNN-compatible face detection, Facenet512
embeddings, cosine-similarity search, configurable confidence thresholds, and
a Tkinter kiosk interface. The repository includes automated evaluation for
precision and recall at k, query latency, false acceptance and rejection rates,
scalability, offline operation, and estimated time savings. The complete
automated suite currently contains 130 passing tests. Real-event accuracy
values are intentionally not claimed in this report because no consented
evaluation dataset is stored in the repository; the included evaluation runner
generates those values when authorized data is supplied. A dedicated privacy
policy defines consent, encrypted local storage, least-privilege access,
retention limits, and verified deletion of biometric data.

## 1. Introduction

Photo delivery, rather than image capture or editing, is a bottleneck for the
client. Event guests currently inspect a large shared collection manually,
which is slow and can miss relevant photographs. Face retrieval is well suited
to this problem because one query face can be compared with each face detected
across the event, while the original photograph remains the delivered result.

The design goals were to:

1. retrieve a participant's photographs from one selfie;
2. keep all biometric processing on local hardware;
3. isolate every search to a selected event;
4. support group photographs containing multiple faces;
5. separate high-confidence from review-worthy possible matches; and
6. measure accuracy, latency, scalability, and time savings.

The first version intentionally excludes cloud synchronization, mobile access,
payment processing, automatic identity labelling, and production-grade
liveness detection.

## 2. Requirements and system architecture

The operational pipeline is:

```text
Event photos
    -> preprocess
    -> detect every face
    -> generate normalized embeddings
    -> save one FAISS index per event

Selfie
    -> preprocess
    -> detect the highest-confidence face
    -> generate a query embedding
    -> search the selected event index
    -> deduplicate by source photo
    -> classify confident / possible
    -> display thumbnails
```

Per-event indexes prevent accidental cross-event results and bound search size.
Indexing is an explicit staff operation so unfinished event folders are not
made searchable. Search runs on a background thread so model inference and
FAISS lookup do not freeze the kiosk interface.

## 3. Image preprocessing

The preprocessing package provides independently tested operations that can be
composed before face detection:

- color correction and white-balance normalization;
- aspect-ratio-preserving resizing and geometric correction;
- brightness and contrast adjustment;
- CLAHE local-contrast enhancement in LAB color space;
- low-light detection and automatic brightness improvement;
- convolution filters for denoising and sharpening; and
- morphology and segmentation helpers for mask cleanup.

Shared values such as target image size live in `config.py`, preventing the
indexing and query paths from silently applying incompatible settings. Both raw
event photographs and selfies pass through the same preprocessing entry point.

## 4. Face detection and embeddings

The detection layer finds all usable faces rather than assuming one face per
event photograph. Detections below the configured confidence or minimum-size
limits are rejected. Each accepted face records a bounding box, confidence,
and a 512-dimensional Facenet512 embedding. Embeddings are L2-normalized before
indexing and again defensively at the FAISS boundary.

For a selfie, the matcher selects the highest-confidence detection. If no face
is found, it raises a domain-specific error that the interface converts into a
retake instruction. This is preferable to returning an empty or misleading
result set.

## 5. Event indexing and matching

Each event uses a FAISS `IndexFlatIP`. For normalized vectors, inner product is
cosine similarity, so this gives exact nearest-neighbour search without a
separate distance conversion. Metadata maps each vector back to its source
photo, bounding box, and detection confidence.

The matcher retrieves candidate faces, classifies their similarity using the
thresholds in `config.py`, and retains only the best-scoring face for each
source photograph. A photograph therefore appears at most once even when it
contains several indexed faces. Results are sorted by decreasing score into:

- **confident matches**, intended for direct review; and
- **possible matches**, intended to reduce silent false rejections.

The current threshold values are starting points. Production values must be
adopted only after evaluation across representative, consented events.

## 6. User interface

The local Tkinter interface lists available event folders, accepts common image
formats, validates and previews the selected selfie, and starts matching away
from the UI thread. Results appear as responsive, scrollable thumbnail grids
with scores and separate confidence tiers. EXIF orientation is applied before
display, and a missing or unreadable result image produces a placeholder rather
than terminating the search.

Expected operational errors receive specific guidance: the user is asked to
retake a selfie when no face is detected, while an unindexed event directs
staff to run indexing. The interface intentionally has no upload server or
remote account system.

## 7. Evaluation methodology and current evidence

### 7.1 Automated verification

The repository's 130 automated tests cover preprocessing, detection,
embeddings, indexing, matching, threshold tuning, interface integration,
retrieval metrics, ground-truth loading, reporting, scalability, offline
operation, and time-savings calculations. The final verification command is:

```powershell
..\.venv\Scripts\python.exe -m pytest tests -q
```

All 130 tests pass on the development environment. Synthetic unit and
integration tests establish software behavior, edge-case handling, numerical
validation, and module compatibility; they do not establish real-world face
recognition accuracy.

### 7.2 Real-event protocol

The evaluation runner expects one consented selfie and a ground-truth set of
known matching filenames for each identity. For each identity it runs the full
search pipeline and records precision@k, recall@k, top similarity score, and
end-to-end query time. Genuine and impostor scores feed the FAR/FRR threshold
tuner. The generated report also identifies the lowest-recall identities for
manual robustness review.

Run an authorized event with:

```powershell
..\.venv\Scripts\python.exe scripts\run_evaluation.py <event_id> --scalability
```

The evaluator saves its report inside the ignored event evaluation directory.
Staff must transfer only aggregate, de-identified values into the table below.

| Metric | Final value | Status |
|---|---:|---|
| Identities evaluated | Not yet measured | Requires consented event data |
| Precision@k | Not yet measured | Requires consented event data |
| Recall@k | Not yet measured | Requires consented event data |
| False acceptance rate | Not yet measured | Requires genuine/impostor scores |
| False rejection rate | Not yet measured | Requires genuine/impostor scores |
| Mean query time | Not yet measured | Requires target deployment hardware |
| 10k–100k-face search time | Tooling ready | Run scalability option on target hardware |

These fields must not be replaced with toy or synthetic accuracy values and
presented as client results. The final presentation should label preliminary
measurements with event size, identity count, hardware, k, and thresholds.

### 7.3 Robustness review

After each real-event run, reviewers inspect failed and low-scoring cases and
classify likely causes: lighting, pose, occlusion, motion blur, small faces,
group density, or incorrect ground truth. This qualitative review complements
aggregate metrics and guides whether preprocessing, detector limits, or match
thresholds should change.

### 7.4 Time savings

The evaluator compares measured automatic query time with the client's
five-to-six-hour manual baseline. The expected improvement is substantial, but
the final percentage must use latency measured on the deployment workstation.
Human review and export time should be reported separately from model search
time.

## 8. Privacy, security, and retention

Faces and embeddings are sensitive biometric data. PhotoMatch keeps search
local and its `.gitignore` excludes event photographs, evaluation identities,
embeddings, FAISS indexes, model files, and generated reports. These technical
defaults do not replace operational controls.

The project's privacy policy requires:

- affirmative notice and consent before selfie processing;
- verified guardian consent for minors;
- separate consent for identity-labelled evaluation data;
- an available non-biometric manual-search alternative;
- encrypted local storage and named least-privilege staff accounts;
- no cloud synchronization or copying to personal devices;
- deletion of in-memory selfies and query embeddings after the search;
- deletion of evaluation identity data within 30 days of the run;
- deletion of event photos and indexes 30 days after delivery and no later than
  90 days after the event without a new written agreement; and
- documented deletion, backup removal, and second-person verification.

The complete operating procedure is in
[`PRIVACY_AND_RETENTION.md`](PRIVACY_AND_RETENTION.md).

## 9. Deployment procedure

1. Prepare a dedicated encrypted Windows workstation and a non-administrator
   kiosk account.
2. Install Python 3.11 and the pinned project requirements in `.venv` while no
   event data is present.
3. Confirm model availability, disable cloud-sync software, and take the kiosk
   offline.
4. Record organizer approval, the consent notice version, responsible staff,
   and the event deletion date.
5. Copy edited photographs to `data/events/<event_id>/raw/` and build the index.
6. Perform staff acceptance searches and confirm that only the chosen event is
   returned.
7. Run the kiosk, deliver selected originals through the approved workflow, and
   keep filesystem access unavailable to guests.
8. Complete evaluation only with separately consented data, then delete it by
   the policy deadline.
9. Delete the event, index, logs, exports, and backups at the recorded deadline;
   record independent verification.

## 10. Limitations and future work

- Real-event accuracy and latency remain to be measured with a sufficiently
  diverse, consented dataset.
- Face recognition performance may degrade with extreme pose, occlusion, blur,
  masks, small group-photo faces, or demographic imbalance.
- Thresholds may not transfer unchanged between camera conditions and event
  types.
- A photograph of another person could be used as a query because v1 has no
  production liveness or anti-spoofing protection.
- The interface is a local kiosk and does not include multi-user administration,
  watermark release, payment, or remote delivery.
- Selective removal from an unlabelled shared face index requires rebuilding it
  from the remaining authorized source photographs.

Future work should prioritize representative evaluation, demographic and event
condition analysis, liveness detection, protected previews, staff audit tools,
and a reviewed deployment threat model before expanding beyond a supervised
local kiosk.

## 11. Conclusion

PhotoMatch demonstrates a complete local computer-vision retrieval pipeline:
preprocessing, multi-face detection, embedding extraction, per-event vector
indexing, confidence-tiered matching, and desktop result review. Automated
testing supports the correctness and integration of the implementation, while
the evaluation package provides a reproducible route to client-specific
metrics. The privacy policy makes consent, access, retention, and deletion part
of operation rather than an afterthought. The remaining evidence gap is not a
software component but an authorized real-event evaluation; the report leaves
those values explicit instead of overstating readiness.

## References

1. F. Schroff, D. Kalenichenko, and J. Philbin, “FaceNet: A Unified Embedding
   for Face Recognition and Clustering,” *CVPR*, 2015.
2. J. Deng et al., “ArcFace: Additive Angular Margin Loss for Deep Face
   Recognition,” *CVPR*, 2019.
3. K. Zhang et al., “Joint Face Detection and Alignment Using Multitask
   Cascaded Convolutional Networks,” *IEEE Signal Processing Letters*, 2016.
4. J. Johnson, M. Douze, and H. Jégou, “Billion-scale Similarity Search with
   GPUs,” *IEEE Transactions on Big Data*, 2019.
5. OpenCV documentation, image processing and geometric transformations.
6. PhotoMatch project proposal, source code, automated tests, evaluation
   protocol, and privacy policy, 2026.
