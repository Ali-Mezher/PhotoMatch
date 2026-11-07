# Evaluation data — photos plus human-confirmed face labels

This folder is empty on purpose. Retrieval evaluation starts by adding photos
here—no code or configuration editing is needed. Clustering evaluation then
generates a CSV worksheet; a human must fill its `identity` column while looking
at the generated face crops so the system never grades its own identity guesses.

Evaluation data links a person's identity to biometric photos. Obtain separate,
explicit evaluation consent (and verified guardian consent for minors) before
adding it. Follow the full [privacy and retention policy](../../docs/PRIVACY_AND_RETENTION.md),
and delete each identity set immediately after its results are accepted and no
later than 30 days after the evaluation run.

## Folder layout to create

For an event you've already indexed (e.g. `grad2026`, matching
`data/events/grad2026/raw/`):

```
data/evaluation/grad2026/identities/
    alice/
        selfie/
            alice_front.jpg           <- one independent selfie of Alice
            alice_side.jpg            <- optional additional test selfie
        matches/
            IMG_0231.jpg              <- copy of an event photo that shows Alice
            IMG_0489.jpg              <- another one — same filename as in
                                          data/events/grad2026/raw/
    bob/
        selfie/
            bob_selfie.jpg
        matches/
            IMG_0102.jpg
```

## Rules

1. **One event = one folder**, named to match the event's `event_id`
   under `data/events/<event_id>/`. That event must already be indexed
   (`python scripts/demo_goated_index_and_search.py index <event_id>`) before
   you can evaluate it.
2. **One folder per person** under `identities/`, named however you
   like (a first name is fine).
3. **`selfie/` gets one or more independent photos** — normal shop-style
   photos that were not copied from the event. Use different angles and lighting
   when possible; every file is evaluated as a separate query.
4. **`matches/` gets copies of their real photos from the event** —
   and the filename must be *identical* to the file already sitting in
   `data/events/<event_id>/raw/`. That's how the evaluation code knows
   "this copy in `matches/` stands for that indexed photo." If you
   rename it, it won't be recognized as a match.
5. Add as many people and as many events as you have real (or
   realistic stand-in) photos for — more identities means a more
   trustworthy precision/recall/FAR/FRR number.

## Running it

Once at least one event has at least one identity set up:

```bash
python scripts/run_evaluation.py grad2026
```

This runs the full Week 4 evaluation (precision/recall, query time,
FAR/FRR thresholds, manual-vs-automatic time savings) and saves a
report to `data/evaluation/grad2026/report.md`.

The same command also creates:

- `matching-scores.csv` — every genuine/impostor score used for tuning.
- `face-label-crops/` — local crops of indexed faces from the ground-truth
  photographs for human review.
- `face-labels.csv` — a worksheet whose `identity` column must be filled by a
  person. `possible_identities` is only a filename-based hint; never copy it
  blindly when a photograph contains more than one face.
- `labeled-embeddings.csv` — generated automatically after at least two people
  have two confirmed face labels each.
- `threshold-results.csv` — all tested matching and clustering combinations.
- `threshold-report.md` — conservative candidate settings and explicit false-
  merge warnings.

The first run normally stops clustering evaluation at the labeling step. Open
each image under `face-label-crops/`, enter the correct folder/identity name in
the corresponding `face-labels.csv` row, leave unknown/background faces blank,
save the CSV, and run the same command again.

## Calibration and final testing

Do not tune thresholds and claim final accuracy on the same event. Use at least
two representative events for calibration, choose candidate settings across
them, and then test those unchanged settings on a separate holdout event. If the
holdout event shows a cross-person merge, the clustering values are not ready.

Nothing in this folder is committed to git except this README and the
`.gitkeep` placeholder — real people's photos, face crops, embeddings, names,
and reports stay local, same as `data/events/`.
