# Day 1 integrated handoff

## Definition of done

The complete 50,000-entity development pipeline runs from subset files through candidates, features, grouped LightGBM training, validation inference, threshold search, macro F0.5 scoring and validation-format outputs without manual edits.

## Verified full-run result

Run date: 2026-09-26

| Check | Result |
| --- | ---: |
| Development Source 1 rows | 50,000 |
| Development Source 2 rows | 183,231 |
| Development Source 3 rows | 189,407 |
| Raw candidate rows | 1,127,024 |
| Training candidate rows | 1,150,117 |
| Feature rows | 1,150,117 |
| Training rows | 923,057 |
| Validation Source 1 entities | 10,081 |
| Validation candidate rows | 227,060 |
| Validation candidate link recall | 0.838985 |
| Validation full-entity candidate recall | 0.637651 |
| Selected threshold | 0.75 |
| Macro F0.5 | 0.912370 |
| Micro precision | 0.988071 |
| Micro recall | 0.823176 |
| Singleton accuracy | 0.957875 |
| Missing feature values | 0 |
| Maximum observed RSS after a stage | about 1.82 GB |
| Official validator, including ID-existence check | PASS |

This is a grouped local validation baseline, not a leaderboard score. Candidate recall is the main Day 2 limitation.

## Reproduce

```bash
python scripts/run_day1_pipeline.py --config configs/base.json
```

Generated evidence is intentionally excluded from Git because it is large:

- `data/dev/manifest.json` - selected subset and row counts.
- `data/work/day1/folds.tsv` - deterministic Source 1 grouped folds.
- `data/work/day1/candidate_pairs_raw.tsv` - heuristic candidates before training-only augmentation.
- `data/work/day1/candidate_pairs_training.tsv` - training candidates with missing positives added only outside the validation fold.
- `data/work/day1/features.tsv` - stable feature handoff with labels and folds.
- `data/work/day1/validation_scores.tsv` - model-score contract output.
- `data/work/day1/validation_matching_results.tsv` - validation predictions.
- `data/work/day1/validation_candidate_pairs.tsv` - exact validation candidates scored by the model.
- `artifacts/day1/day1_lightgbm.joblib` - trained pooled Day 1 model.
- `artifacts/day1/training_metrics.json` - row counts, missing-value count and feature importance.
- `artifacts/day1/threshold_metrics.json` - threshold sweep and exact entity-level metrics.
- `artifacts/day1/run_summary.json` - complete machine-readable run report.
- `logs/day1_pipeline.log` - stage timings, row counts and memory observations.
- `data/work/day1/validation_test/` - temporary validation source files used for the official validator check.

## Frozen Day 1 interfaces

- Candidate batches use the six columns in `docs/interfaces.md` and contain one row per unique S1-secondary pair.
- When several routes find one pair, route names are sorted and joined with `+`; rank and retrieval score are deterministic.
- Feature files use three ID columns, `label`, grouped `fold`, and the stable numeric columns defined in `features.py`.
- Training and inference use the same feature function and saved feature order.
- The split is grouped by Source 1 ID using seed `2026`; fold `0` is validation.
- The exact macro F0.5 implementation includes singleton entities.

## Teammate replacement points

- Person 2 replaces `candidate_pairs_raw.tsv` while retaining the candidate-pair contract.
- Person 3 replaces or extends the numeric feature columns while retaining ID, label and fold columns.
- Person 4 independently reruns the grouped split, macro F0.5 tests and threshold selection.
- Person 1 reruns the same integration command after each approved handoff.
