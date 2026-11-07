# Threshold experiments

Search matching and identity grouping fail in different ways, so PhotoMatch
measures them separately. Never choose production values from a single event.

## Recommended real-event workflow

For an indexed event with the evaluation folders described in
`data/evaluation/README.md`, run one command:

```bash
python scripts/run_evaluation.py <event_id> --k 10
```

It accepts multiple selfies per identity, exports genuine/impostor matching
scores, tests the matching grid, and prepares a crop worksheet for safe human
face labeling. Fill the `identity` column in `face-labels.csv` and run the same
command again to export labeled embeddings and test the clustering grid. The
generated `threshold-report.md` prioritizes false accepts and cross-person false
merges and warns when none of the tested values are safe enough.

Use the sections below when rerunning a saved CSV with custom grid ranges.

## Matching input

Create a CSV with genuine (same person) and impostor (different person) scores:

```csv
score,label
0.86,genuine
0.42,impostor
```

Run a grid of possible and strong-match thresholds:

```bash
python scripts/experiment_thresholds.py \
  --matching-csv scores.csv \
  --possible-range 0.50:0.70:0.02 \
  --confident-range 0.64:0.86:0.02 \
  --output-dir threshold_experiments/run-01
```

The report includes FAR, FRR, precision, recall, F1, true positives, true
negatives, false positives, and false negatives for both tiers. Matching rows
are ranked by confident false accepts, possible false accepts, then recall.

## Identity-group input

Export one labeled embedding per row. The `face_id` is informational; embedding
columns can be named `e0,e1,...` or `embedding_0,embedding_1,...`:

```csv
face_id,identity,e0,e1,e2
a-1,person-a,0.91,0.10,0.02
a-2,person-a,0.89,0.12,0.01
b-1,person-b,0.05,0.94,0.03
```

Run edge/cohesion combinations:

```bash
python scripts/experiment_thresholds.py \
  --clustering-csv labeled_embeddings.csv \
  --edge-range 0.66:0.86:0.02 \
  --cohesion-range 0.64:0.86:0.02 \
  --neighbors 50 \
  --min-cluster-size 2 \
  --output-dir threshold_experiments/run-01
```

Clustering rows report false merges, false splits, weighted cluster purity,
unassigned faces, and group count. They are ranked first by false merges because
putting two different people in one group is the stated safety problem, then by
false splits, unassigned faces, and purity.

Use both inputs in one command to produce one CSV and one Markdown summary. The
script does not modify runtime settings. After validating results across several
events, enter the chosen matching and group similarity values in **Advanced
settings**. Increasing thresholds normally reduces false matches and merges but
can increase missed matches, false splits, and unassigned faces.
