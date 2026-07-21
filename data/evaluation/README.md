# Evaluation data — just drop photos in

This folder is empty on purpose. To evaluate PhotoMatch against real
data, add photos here — no code, no CSV, no config editing needed.

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
            alice_selfie.jpg          <- one selfie of Alice
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
   (`python scripts/demo_index_and_search.py index <event_id>`) before
   you can evaluate it.
2. **One folder per person** under `identities/`, named however you
   like (a first name is fine).
3. **`selfie/` gets exactly one photo** — a normal selfie of that
   person, the kind a student would actually upload.
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

Nothing in this folder is committed to git except this README and the
`.gitkeep` placeholder — real people's photos and names stay local,
same as `data/events/`.
