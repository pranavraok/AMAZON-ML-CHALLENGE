# Day 2 Person 1 integration handoff

Run date: 2026-09-26

## Outcome

Person 1's Day 2 work is complete on the shared 50,000-entity development subset. The pipeline now trains and compares a pooled LightGBM model against separate Source 2 and Source 3 LightGBM models, tunes the global decision threshold, performs restartable chunked inference, produces error slices for Person 4, reports country slices, and writes a provisional frozen runtime configuration.

The separate architecture is the provisional winner, but the advantage over pooled is only `0.000069` macro F0.5. Keep both models until Person 4 independently confirms the choice.

## Like-for-like model comparison

Both alternatives use the same candidate rows, feature columns, grouped Source 1 split, seed, 350 boosting rounds, and validation entities.

| Result | Pooled model | Separate S2/S3 models |
| --- | ---: | ---: |
| Best threshold | 0.74 | **0.70** |
| Macro F0.5 | 0.913244 | **0.913314** |
| Micro precision | **0.988357** | 0.986637 |
| Micro recall | 0.824178 | **0.826813** |
| False positives | **339** | 391 |
| False negatives | 6,139 | **6,047** |
| Singleton accuracy | **0.963370** | 0.959707 |

Decision: provisionally select separate S2/S3 models at threshold `0.70`. The pooled model at `0.74` remains the lower-risk fallback because its precision and singleton accuracy are slightly better.

## Most important finding

Candidate generation is still the main score ceiling. Of the 6,047 false-negative links from the selected system:

- 5,622, or about 93%, were absent from the candidate set.
- 425 were candidates but scored below the selected threshold.

This means Person 2's recall improvements are likely to matter much more than additional model tuning. Do not interpret a model-only change as solving missing candidates.

## Country robustness signal

These are ordinary country slices of the fixed validation fold, not a leave-one-country-out experiment.

| Country | Entities | Macro F0.5 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| India | 3,997 | 0.851103 | 0.982083 | 0.726108 |
| US | 6,084 | 0.954184 | 0.989059 | 0.892147 |

India is the clearest current weakness. Person 4 must still run the independent leave-country-out experiment, and the full pipeline must be smoke-tested on France as an unseen open-set country. No country name is hard-coded into normalization, feature generation, model training, or inference.

## Medium-scale benchmark

| Check | Result |
| --- | ---: |
| Source 1 development entities | 50,000 |
| Feature rows | 1,150,117 |
| Feature file size | 276,466,472 bytes |
| Validation score rows | 227,060 |
| Inference chunk size | 100,000 rows |
| Pooled inference throughput | about 18,253 rows/second |
| Separate inference throughput | about 16,864 rows/second |
| Maximum observed RSS after a stage | about 1,053 MB |
| Day 1 normal vs chunked probability difference | exactly 0.0 over 227,060 rows |
| Restart checkpoint after successful completion | removed as expected |

The checkpoint remains beside the partial score file if a run is interrupted. Rerunning the same command resumes after the last fully written chunk.

## Files to hand to teammates

Generated files are intentionally ignored by Git because they are large.

- Person 4 receives `data/work/day2/person4_error_slices.tsv`, `artifacts/day2/person4_error_summary.json`, `artifacts/day2/model_comparison.json`, `artifacts/day2/country_slice_metrics.json`, and `artifacts/day2/threshold_plan.json`.
- Person 2 uses the error rows marked `false_negative_candidate_miss` to improve candidate recall.
- Person 3 uses rows marked `false_negative_below_threshold` and `false_positive*` for feature ablations and hard-negative work.
- Person 1 keeps `artifacts/day2/pooled_350.joblib` and `artifacts/day2/separate_s2_s3_350.joblib` until independent validation decides the final architecture.

Tracked freeze and reproducibility files:

- `configs/day2_frozen.json` — provisional selected architecture and runtime settings.
- `scripts/run_day2_pipeline.py` — complete reproducible Person 1 Day 2 run.
- `docs/day2_final_run_checklist.md` — the Day 3 full-data and packaging gate.

## Reproduce

```bash
python scripts/run_day2_pipeline.py --config configs/base.json
```

Use the provisional frozen choice explicitly:

```bash
python scripts/run_day2_pipeline.py --config configs/day2_frozen.json
```

The complete Day 2 run took about 77 seconds on this machine after Day 1 artifacts existed.

## Validation evidence

- 10 automated tests pass.
- Chunked inference reproduced all 227,060 Day 1 probabilities with zero numerical difference.
- The challenge-supplied validator passed both Day 2 output files with `--check-ids`.
- Both output files contain exactly 10,081 validation Source 1 rows.
- Every selected match is a subset of the submitted candidates.

## Team merge gate before the architecture is truly frozen

Do not merge a teammate change just because it runs.

1. Person 2: report candidate link recall, full-entity recall, row count, runtime, and examples of recovered misses on the identical validation IDs.
2. Person 3: report ablation results on the identical folds, feature runtime, missing values, and memory usage. Features used only by positive guards are rejected.
3. Person 4: independently reproduce macro F0.5, threshold behavior, singleton behavior, US/India slices, and leave-country-out results.
4. Person 1: integrate one handoff at a time, rerun tests plus Day 2 comparison, and record the before/after metric. If several changes arrive together, keep a rollback point for each one.

The current freeze status is therefore `provisional_pending_teammate_handoffs`, not the final submission freeze.
