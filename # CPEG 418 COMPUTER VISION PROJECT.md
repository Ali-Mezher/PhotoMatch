# CPEG 418 COMPUTER VISION PROJECT PROPOSAL[cite: 1]
## PhotoMatch[cite: 1]
**Local Face Recognition for Instant Event Photo Retrieval**[cite: 1]
*Department of Computer Engineering, American University of Kuwait*[cite: 1]

---

## 01 INTRODUCTION & BACKGROUND[cite: 1]
**A photography business outgrowing manual delivery**[cite: 1]

*   **The client**: Golden Camera, a freelance photography business shooting weddings, corporate events, school graduations, and university graduations (even our university) often thousands of photos per event.[cite: 1]
*   **The workflow today**: Once an event is shot and edited, all photos are dumped into a shared folder.[cite: 1] Students and guests then scroll through the entire set by eye to find themselves.[cite: 1]
*   **10,000+**: photos per graduation event.[cite: 1]
*   **5-6 hrs**: spent per student searching manually.[cite: 1]
*   **Peak season**: many students need photos at once.[cite: 1]

---

## 02 PROBLEM STATEMENT[cite: 1]
**Manual photo retrieval is the bottleneck**[cite: 1]

Editing is no longer the slowest step in the business finding the right photos for the right person is.[cite: 1] Every event ends with hours of unpaid, error-prone manual sorting that delays delivery and caps how many events the business can serve.[cite: 1]

*   **Slow**: 5-6 hours of manual searching per student across 10,000+ unsorted photos.[cite: 1]
*   **Error-prone**: Easy to miss photos or hand a student the wrong set entirely.[cite: 1]
*   **Unscalable**: Bottleneck grows worse as event count and photo volume increase.[cite: 1]

---

## 03 OBJECTIVES[cite: 1]
**What the system needs to achieve**[cite: 1]

*   **Automate retrieval**: Let a student find their own photos in seconds by uploading a single selfie.[cite: 1]
*   **Keep it local**: Run entirely on the business's own hardware no cloud upload, strong privacy story.[cite: 1]
*   **Scope by event**: Tag every photo with an event ID so searches stay fast and accurate.[cite: 1]
*   **Handle real conditions**: Work reliably across angles, lighting, group shots, and partial occlusion (caps/gowns).[cite: 1]
*   **Reduce false negatives**: Surface "possible matches" separately so a real photo is never silently missed.[cite: 1]
*   **Grow with the business**: Design the index to scale from 10,000 to 100,000+ photos across events.[cite: 1]

---

## 04 SCOPE OF THE PROJECT[cite: 1]
**Where the project draws its boundaries**[cite: 1]

### In Scope[cite: 1]
*   Face detection & recognition pipeline on local hardware[cite: 1]
*   Event-scoped photo indexing (per event_id)[cite: 1]
*   Selfie-based student search & retrieval[cite: 1]
*   Manual trigger to index only fully-edited events[cite: 1]
*   Confidence-tiered results (confident vs. possible matches)[cite: 1]
*   Basic evaluation: precision/recall, FAR/FRR, query time[cite: 1]

### Out of Scope (v1)[cite: 1]
*   Cloud sync or remote/multi-branch access[cite: 1]
*   Payment processing & watermark release automation[cite: 1]
*   Mobile app / desktop/local kiosk workflow only[cite: 1]
*   Full liveness/anti-spoofing hardening (basic check only)[cite: 1]
*   Automatic identity tagging without any student selfie[cite: 1]

---

## 05 LITERATURE REVIEW & RELATED WORK[cite: 1]
**Building on established CV building blocks**[cite: 1]

*   **Face detection**: Haar cascades / MTCNN / RetinaFace.[cite: 1] Locating faces within a full event photo before recognition.[cite: 1]
*   **Face embeddings**: FaceNet / ArcFace-style embedding models.[cite: 1] Mapping each detected face to a compact vector for comparison.[cite: 1]
*   **Similarity search**: Cosine similarity, nearest-neighbor indexing (e.g., FAISS).[cite: 1] Matching a student's selfie embedding to stored event photo embeddings.[cite: 1]
*   **Classical CV toolkit**: OpenCV (transforms, morphology, segmentation).[cite: 1] Preprocessing: normalization, denoising, and image cleanup pre-detection.[cite: 1]

---

## 06 PROPOSED METHODOLOGY[cite: 1]
**From event folder to student selfie match**[cite: 1]

1.  **Ingest**: Editor confirms event is done; photos are pulled in with an event_id tag.[cite: 1]
2.  **Detect & Embed**: Detect every face per photo; generate an embedding for each.[cite: 1]
3.  **Index**: Store embeddings in a per-event, searchable local index.[cite: 1] Manual/hybrid trigger keeps unfinished or soon-to-be-replaced photos out of the index.[cite: 1]
4.  **Match**: Student uploads a selfie; embedding compared within that event only.[cite: 1]
5.  **Deliver**: Confident and possible matches returned as separate result tiers.[cite: 1]

---

## 07 TOOLS & TECHNOLOGIES[cite: 1]
**The stack behind PhotoMatch**[cite: 1]

*   **Python + OpenCV**: Core image I/O, preprocessing, and pipeline glue - built on CPEG 418 lab work.[cite: 1]
*   **Face detection / embedding model**: Pretrained local model (e.g., MTCNN + FaceNet/ArcFace-style embeddings).[cite: 1]
*   **Local vector index**: FAISS or equivalent for fast per-event nearest-neighbor similarity search.[cite: 1]
*   **Local storage only**: All photos, embeddings, and search run on-premise no external upload.[cite: 1]
*   **matplotlib**: Visualizing detections, matches, and evaluation results for the report.[cite: 1]
*   **Simple local UI**: Lightweight desktop/kiosk interface for selfie upload and results review.[cite: 1]

---

## 08 DATASET DESCRIPTION[cite: 1]
**Real event photos, not clean benchmarks**[cite: 1]

*   **Source**: Real photos from the business's own weddings, events, and school/university graduations the same set staff currently sort by hand.[cite: 1]
*   **Why real data matters**: Clean benchmark datasets don't capture this business's real challenges: varied angles, mixed lighting, group shots, motion blur, and caps/gowns partially covering faces.[cite: 1] Testing on an early real sample keeps the design honest.[cite: 1]
*   **Scale**: 10,000+ photos per graduation event today.[cite: 1] 50,000-100,000+ target scale as the business and event count grow.[cite: 1]
*   **Sensitivity**: Photos contain biometric data of real people, often minors handled under the consent and privacy plan on the following slides.[cite: 1]
*   **Multiple event types**: weddings, corporate events, school & university graduations.[cite: 1]

---

## 09 PREPROCESSING TECHNIQUES[cite: 1]
**Cleaning images before detection runs**[cite: 1]

*   **01 Color & geometry correction**: Normalize color balance and correct geometric distortion across mixed-lighting event shots.[cite: 1]
*   **02 Intensity transformation**: Adjust brightness/contrast so faces are detectable in both bright outdoor and dim indoor shots.[cite: 1]
*   **03 Convolution-based filtering**: Denoise and sharpen images affected by motion blur or low light before face detection.[cite: 1]
*   **04 Morphological operations**: Clean up detection masks remove noise, fill small gaps around detected face regions.[cite: 1]
*   **05 Segmentation**: Isolate face regions from cluttered group-photo backgrounds prior to embedding extraction.[cite: 1]

---

## 10 MODEL / ALGORITHM DESIGN[cite: 1]
**Detection, embedding, and matching logic**[cite: 1]

*   **Face detector**: Locates every face in a photo, including partially obscured faces in group / cap-and-gown shots.[cite: 1]
*   **Embedding model**: Converts each detected face into a fixed-length vector capturing identity, robust to angle/lighting.[cite: 1]
*   **Multi-face handling**: Each face in a group photo gets its own embedding; the photo matches if any face matches.[cite: 1]
*   **Threshold-tuned matching**: Cosine similarity vs. selfie embedding, with confident and possible-match tiers.[cite: 1]
*   **Auto-clustering (stretch)**: Group similar faces per event into identity clusters staff can quickly label.[cite: 1]

---

## 11 EVALUATION METRICS[cite: 1]
**Proving the system works not just demoing it**[cite: 1]

*   **Precision / Recall @ top-k**: How many of the top retrieved photos are correct, and how many correct photos are found.[cite: 1]
*   **False Acceptance Rate (FAR)**: How often the system wrongly matches a photo to the wrong person.[cite: 1]
*   **False Rejection Rate (FRR)**: How often the system misses a photo that actually belongs to the student.[cite: 1]
*   **Avg. query time per selfie**: End-to-end latency from selfie upload to returned results.[cite: 1]
*   **Time saved vs. baseline**: Measured against the current 5-6 hour manual search per student.[cite: 1]

---

## 12 EXPECTED OUTCOMES[cite: 1]
**What success looks like**[cite: 1]

*   **Minutes-to-seconds retrieval**: Student photo search drops from a 5-6 hour manual task to a seconds-long selfie search.[cite: 1]
*   **Privacy-first deployment**: Fully local pipeline with a defined consent and data-retention policy for biometric data.[cite: 1]
*   **Measured accuracy**: Documented precision/recall, FAR, and FRR on the business's real event photos.[cite: 1]
*   **A tool staff will actually use**: Simple enough for non-technical workers to trigger indexing and review matches.[cite: 1]

---

## 13 TIMELINE / PROJECT PLAN[cite: 1]
**4-week summer course plan**[cite: 1]

*   **Week 1 (Complete)**: Requirements & proposal.[cite: 1] Confirmed scope with client and gathered an initial sample of event photos.[cite: 1]
*   **Week 2**: Detection & preprocessing.[cite: 1] Build the face detection pipeline, preprocessing steps, and embedding extraction.[cite: 1]
*   **Week 3**: Indexing, matching & interface.[cite: 1] Build per-event similarity index, selfie matching logic, and a simple review UI.[cite: 1]
*   **Week 4**: Evaluation, report & presentation.[cite: 1] Run precision/recall, FAR/FRR, and query-time benchmarks; finalize report and present.[cite: 1]

---

## 14 BUDGET[cite: 1]
**Minimal cost built on existing infrastructure**[cite: 1]

*   **No dedicated budget required**: The system runs entirely on the shop's and workers' existing computers.[cite: 1] Software components (OpenCV, pretrained detection/embedding models, vector search library) are open-source with no licensing cost.[cite: 1]
*   **Hardware**: Existing shop / workstation computers no new purchase planned.[cite: 1]
*   **Software & models**: Open-source (OpenCV, detection/embedding models, local vector index).[cite: 1]
*   **Optional future spend**: Faster storage or a dedicated indexing machine if volume grows well past 100,000 photos.[cite: 1]

---

## 15 TEAM MEMBERS & ROLES[cite: 1]
**Who's building PhotoMatch**[cite: 1]

*   **Mahmood Tendail (ID 66597)**: Project Lead & Face Detection.[cite: 1] Owns the detection/embedding pipeline; coordinates integration.[cite: 1]
*   **Ali Mezher (ID 61392)**: Preprocessing Engineer.[cite: 1] Handles color, intensity, and convolution-based cleanup.[cite: 1]
*   **Malek AlKashat (ID 61513)**: Indexing & Search.[cite: 1] Builds the per-event embedding index and similarity search.[cite: 1]
*   **Hazem Ahmed (ID 70493)**: Matching Logic.[cite: 1] Tunes thresholds for confidence-tiered match results.[cite: 1]
*   **Nourallah Mourad (ID 59762)**: Interface & Clustering.[cite: 1] Builds the selfie UI and auto-clustering feature.[cite: 1]
*   **Ahmad AlAli (ID 64024)**: Evaluation & Report.[cite: 1] Runs evaluation metrics and writes the report.[cite: 1]

*Work is divided across the pipeline so every member owns one stage end-to-end all six collaborate closely at integration points.*[cite: 1]

---

## 16 REFERENCES[cite: 1]
**Grounding the approach**[cite: 1]

*   Viola, P. & Jones, M. Rapid Object Detection using a Boosted Cascade of Simple Features (foundational face detection).[cite: 1]
*   Schroff, F., Kalenichenko, D., & Philbin, J. FaceNet: A Unified Embedding for Face Recognition and Clustering.[cite: 1]
*   Deng, J. et al. ArcFace: Additive Angular Margin Loss for Deep Face Recognition.[cite: 1]
*   Zhang, K. et al. MTCNN: Joint Face Detection and Alignment using Multi-task Cascaded CNNs.[cite: 1]
*   OpenCV documentation image processing, transforms, and morphology reference used throughout CPEG 418.[cite: 1]
*   Johnson, J., Douze, M., & Jégou, H. FAISS: a library for efficient similarity search.[cite: 1]

*Thank you. Questions & discussion welcome. PhotoMatch CPEG 418-*[cite: 1]