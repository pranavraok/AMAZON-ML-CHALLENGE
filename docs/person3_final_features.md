# Person 3 – Final features (deliverable checklist)

Feature version **`p3-v2.1`** · 79 features · frozen in `configs/p3_features_frozen.json`. This document supersedes the NaN-related parts of `docs/person3_day2_handoff.md`: v2.1 contains no NaN values.

The evidence files referenced below live in `docs/evidence/` (tracked in git). The large generated data sits under `data/work/` (git-ignored).

| # | Required item | Status | Evidence |
|---|---|---|---|
| 1 | Frozen feature list | ✅ | `configs/p3_features_frozen.json`, `FROZEN_MODEL_FEATURES`, `docs/feature_dictionary.md` |
| 2 | Feature code tested on Person 2's real candidates | ✅ | 9,052,389 pairs from `origin/candidate-generation` (commit `a9b18aa`) |
| 3 | Hard negatives re-mined from model mistakes | ✅ | 4,210 out-of-fold mistakes on Person 2's candidates |
| 4 | Confirmation that hard negatives use training folds only | ✅ | Asserted in code and checked independently |
| 5 | Feature ablation showing which features improve the score | ✅ | Like-for-like tables, bootstrap CIs, group ablation, LOCO |
| 6 | No NaN or infinite feature values | ✅ | Enforced by `validate_feature_frame`; 0 NaN/null/inf on 9.05M + 1.15M + France rows |
| 7 | Final runtime and peak-memory benchmark | ✅ | 64k pairs/s; 7.5 GB total peak in streaming mode |
| 8 | Code pushed to the Person 3 branch | ⏳ | Committed locally on `person3-similarity-features`; **push is done by the branch owner** |

---

## 1. Frozen feature list

- 79 columns in fixed order. Dtypes are `Int8`/`Int16`/`Float32` only.
- The list is `business_entity_resolution.similarity_features.FROZEN_MODEL_FEATURES`: 75 pair features plus 4 name-frequency features.
- Every column is described in `docs/feature_dictionary.md`.
- `configs/p3_features_frozen.json` stores the list, the dtypes and SHA-256 checksums of the feature source code and the two unsupervised artifacts (`artifacts/token_stats.json`, `artifacts/name_stats/`).

```bash
python scripts/freeze_p3_features.py --verify     # PASS = code + artifacts unchanged since freeze
```

Excluded on purpose:
- `ctx_*` context features: neutral in ablation, and they depend on whole-table state.
- `retr_*` retrieval features: they depend on candidate-generation internals. Person 1 excluded retrieval scores for the same reason.

## 2. Tested on Person 2's real candidates

Person 2's `CandidateGenerator` was run **unchanged** from their branch with their dev configuration (`name_top_k=100, address_top_k=20, name_min_score=0.30`). The runner is `scripts/run_person2_candidates.py`, and the input was Person 1's 50k development subset.

| Candidates (validation fold, 10,081 S1) | Pairs per S1 | Link recall |
|---|---|---|
| Person 1 heuristic | 22.5 | 0.8390 |
| **Person 2** | 180.5 | **0.9926** |

Model: Person 1's `_build_classifier` with 350 rounds, trained on folds 1–4 and scored on fold 0 with Person 1's `threshold_search`. Results are in `docs/evidence/evaluation_person2.json` and `bootstrap_person2.txt`.

| Features on Person 2 candidates | Arch | Thr | Macro F0.5 | Precision | Recall | FP | FN | Singleton acc |
|---|---|---|---|---|---|---|---|---|
| Person 1 baseline (22) | pooled | 0.75 | 0.97101 | 0.9896 | 0.9406 | 346 | 2073 | 0.9615 |
| Person 1 baseline (22) | separate | 0.74 | 0.97088 | 0.9883 | 0.9422 | 389 | 2017 | 0.9615 |
| **Person 3 frozen (79)** | **pooled** | 0.72 | **0.98777** | 0.9951 | 0.9755 | 169 | 857 | 0.9762 |
| **Person 3 frozen (79)** | separate | 0.75 | 0.98776 | 0.9949 | 0.9749 | 173 | 877 | 0.9762 |

The paired bootstrap (2,000 resamples over validation entities) gives **+0.0168** macro F0.5 for frozen over baseline:

| Architecture | 95% CI | Entities better / worse |
|---|---|---|
| Pooled | [+0.0149, +0.0185] | 1,346 / 162 |
| Separate | [+0.0151, +0.0186] | 1,320 / 174 |

**Bug found in Person 2's output (needs their fix).** 1,501,728 rows (17%) have an empty `target_source`. The cause is that `CandidateGenerator.generate_for_one` calls `self.blocking_index.generate(source1)` without passing `target_source=self.target_source`, and `BlockingIndex.generate` defaults it to `""`. Every blocking-route candidate (rare-name, transliterated, numeric) is affected.

- My runner derives `target_source` from the ID prefix and counts the repairs.
- `scripts/build_p3_feature_file.py` now **rejects** any candidate file whose `target_source` is not the ID's `S2`/`S3` prefix, so the defect cannot reach the submission run silently.
- **Person 2:** the fix is one line (`self.blocking_index.generate(source1, target_source=self.target_source)`).

Also for the team: Person 2's branch commits `src/amazon_ml_entity_resolution.egg-info/` and ships a modified `normalization.py` that differs from Person 1's shared contract. Person 1 should reconcile both when merging.

## 3. Hard negatives re-mined from model mistakes

`scripts/mine_hard_negatives.py --features data/work/p3d3/p3_features_person2.parquet --output-dir data/work/p3d3` does the following:

1. Cross-fits Person 1's LightGBM over the **training folds 1–4 only**. Each training pair is scored by a model that never saw its S1 group.
2. Selects label-0 pairs with out-of-fold probability ≥ 0.1, keeping the top 5 per S1.

Result on Person 2's candidates: **4,210 hard negatives over 3,631 S1** (`data/work/p3d3/hard_negatives_model.parquet`).

| Family | Count |
|---|---|
| other_retrieved | 1,781 |
| high_name_low_address | 1,601 |
| high_address_low_name | 549 |
| near_duplicate | 258 |
| common_name_collision | 21 |

Other counts: 1,146 training negatives had OOF probability ≥ 0.5, and 1,679 training positives had OOF probability < 0.5.

Up-weighting them (w=2) moves macro F0.5 from 0.98777 to 0.98788 (+0.0001, within noise). **Recommendation: no special weighting.** They are real retrieved candidates and already in training. The file is for error analysis.

The Day 2 set mined on Person 1's candidates is kept as well: 2,538 pairs in `data/work/p3d2/hard_negatives_model.parquet`.

## 4. Training-folds-only confirmation

The script asserts the following and writes it into the summary (`docs/evidence/hard_negatives_person2_candidates.summary.json`):

```json
"training_folds_only_check": {
  "validation_fold": 0,
  "folds_used_for_cross_fitting": [1, 2, 3, 4],
  "validation_fold_rows_used_for_mining": 0,
  "validation_s1_in_hard_negatives": 0,
  "oof_scores_from_models_that_never_saw_the_s1_group": true,
  "labels_used_only_to_select_label_0_rows": true
}
```

**Independent check** against Person 1's `data/work/day1/folds.tsv`:
- 4,210 hard negatives, split across folds 1/2/3/4 as 1,073 / 1,053 / 1,001 / 1,083.
- **0 rows in fold 0**, 0 without a fold.
- 0 hard negatives that are actually true matches.

The mining script raises an error if any validation-fold S1 appears.

## 5. Feature ablation: which features improve the score

**(a) Like-for-like on Person 1's candidates** (1,150,117 rows, identical folds, models and scorer). Source: `docs/evidence/ablation_person1_candidates.json`.

| Features | Arch | Macro F0.5 | FP | FN | Singleton acc |
|---|---|---|---|---|---|
| Person 1 baseline (22) | separate | 0.913314 | 391 | 6047 | 0.9597 |
| Person 1 baseline (22) | pooled | 0.913244 | 339 | 6139 | 0.9634 |
| Person 3 frozen v2.1 (79) | separate | 0.920806 | 102 | 5868 | 0.9835 |
| **Person 3 frozen v2.1 (79)** | **pooled** | **0.921357** | 102 | 5839 | 0.9853 |

Bootstrap, frozen vs baseline:

| Architecture | Δ | 95% CI |
|---|---|---|
| Separate | +0.0075 | [+0.0062, +0.0088] |
| Pooled | +0.0081 | [+0.0068, +0.0094] |

Adding the 22 baseline features on top of the P3 set gives no gain (±0.00001, Day 2).

**(b) Drop-one-group ablation** of the frozen set (pooled, full set 0.921357; `docs/evidence/group_ablation_pooled.json`):

| Group dropped | #cols | Macro F0.5 | Δ vs full | Reading |
|---|---|---|---|---|
| name_frequency | 4 | 0.920313 | **−0.00104** | improves the score |
| noise_ratio | 2 | 0.921015 | **−0.00034** | improves the score |
| idf_soft | 9 | 0.921169 | −0.00019 | improves the score |
| name_extra_fuzzy | 7 | 0.921238 | −0.00012 | small gain |
| address_numbers | 10 | 0.921337 | −0.00002 | redundant here |
| primary_number | 4 | 0.921393 | +0.00004 | redundant here |
| combo | 6 | 0.921440 | +0.00008 | redundant here |
| address_fuzzy | 5 | 0.921440 | +0.00008 | redundant here |
| skeleton_unicode | 7 | 0.921467 | +0.00011 | redundant here; kept for cross-script/France robustness |

"Redundant here" means the group adds nothing *on top of all the others* on this dev set; |Δ| ≤ 0.0001 is within noise. Earlier bootstraps put the noise band at about ±0.0007. No group hurts significantly. Groups are kept rather than tuned to noise.

Feature version history and its measured effect:

| Step | Pooled macro F0.5 | Change |
|---|---|---|
| Baseline | 0.913244 | |
| P3 v1 | 0.920695 | +0.0075 |
| + v2 features | 0.921715 | +0.0010, CI [+0.0002, +0.0018] |
| − context | 0.921784 | neutral |
| NaN → sentinel (v2.1) | 0.921357 | −0.0004, within the ±0.0007 noise band; FP 129 → 102 |

**(c) Leave-one-country-out** (`docs/evidence/loco_pooled.json`):

| Held out | Baseline in-country → LOCO | P3 in-country → LOCO |
|---|---|---|
| India | 0.8544 → 0.8344 (−0.020) | 0.8605 → 0.8559 (−0.005) |
| US | 0.9548 → 0.9446 (−0.010) | 0.9621 → 0.9598 (−0.002) |

France (unseen, 15% of test S1): `docs/evidence/france_smoke_report.json`, all checks pass.

## 6. No NaN or infinite values

- All "not applicable" values are out-of-range sentinels: **−1.0** for every feature whose valid range is ≥ 0, and **−5.0** for `name_{a,b}_unmatched_max_noise`, whose valid range is bounded below by log(N_S1/N_S2S3) ≈ −1.65.
- Not-applicable cases are a missing address, an address with no numbers, or all name tokens matched.
- Explicit flags (`addr_missing_a/b`, `addr_any_missing`, `addr_num_count_a/b`) accompany every sentinel, so a missing address never looks like a real low similarity.
- `validate_feature_frame` raises on any NaN, null, inf, wrong dtype or missing column. Every production path calls it: `build_p3_feature_file.py`, `compute_features.py`, `benchmark_p3_features.py`, `france_smoke_test.py`.
- Verified at **0 NaN / 0 null / 0 inf** on:
  - Person 2's 9,052,389 pairs
  - Person 1's 1,150,117 pairs
  - 24,469 France test pairs
  - my 975,086-pair dev set
- Unit tests: `test_no_nan_or_null_anywhere`, `test_validator_rejects_nan`, `test_missing_address_is_sentinel_with_flags_not_zero`.

## 7. Final runtime and peak-memory benchmark

Machine: Apple Silicon, 11 CPUs, 18 GB RAM, 10 worker processes. Peak RSS is sampled every 0.2 s over the main process **and all workers**. Script: `scripts/benchmark_p3_features.py`. Source: `docs/evidence/benchmark_*.json`.

| Run | Pairs | Time | Pairs/s | Peak RSS total | Main | Workers |
|---|---|---|---|---|---|---|
| Person 2 candidates, all at once | 9,052,389 | 143 s | 63,320 | 5.4 GB | 5.4 GB | 4.7 GB |
| **Person 2 candidates, streaming (`--s1-batch 10000`)** | 9,052,389 | 141 s | **64,052** | **7.5 GB** | **2.0 GB** | 5.7 GB |
| Person 1 candidates | 1,150,117 | 22 s | 52,826 | — | 1.2 GB | — |

The two all-at-once peaks don't happen at the same moment, so the total is below main + workers.

**Full test projection:** 1,732,544 test S1 × ~181 candidates ≈ 314M pairs, about **82 minutes**, about **22 GB** of parquet on disk (132 GB free).

Memory stays **bounded at about 7.5 GB** in streaming mode, because each worker holds the IDF tables regardless of how many candidates there are. The all-at-once mode would need about 74 GB for 314M pairs, so **the full test run must use `--s1-batch`**. Streaming output is bit-identical to the single-shot output (verified on 1,127,024 rows).

The projection assumes the full test file keeps the dev-subset candidate density. The dev subset's S2/S3 pools are smaller than the test pools, so re-measure after Person 2's first test batch.

## 8. Code and how to run it

```bash
export AMAZON_ML_DATASET_ROOT=../student_resource/dataset
python scripts/fit_token_stats.py --workers 10          # unsupervised, ~4 min
python scripts/fit_name_stats.py --workers 10           # unsupervised, ~2 min
python scripts/freeze_p3_features.py --verify           # must print PASS

# training features for Person 1 (any candidate TSV in the shared contract)
python scripts/build_p3_feature_file.py --candidates <candidates.tsv> \
    --source-dir data/dev/train --prefix train \
    --truth data/dev/train/train_ground_truth.tsv --folds data/work/day1/folds.tsv \
    --output <features.parquet> --workers 10

# full test inference, streaming
python scripts/build_p3_feature_file.py --candidates <test candidates.tsv> \
    --source-dir ../student_resource/dataset/test --prefix test \
    --output <features_dir> --s1-batch 10000 --workers 10
# read with: pl.scan_parquet("<features_dir>/*.parquet")

PYTHONPATH=src python -m unittest discover -s tests -v
```

Reproduce this document's numbers:
- `scripts/run_person2_candidates.py`, `scripts/evaluate_on_candidates.py`
- `scripts/mine_hard_negatives.py`, `scripts/ablate_features.py`
- `scripts/p3_experiments.py {bootstrap,groups,loco}`
- `scripts/benchmark_p3_features.py`, `scripts/france_smoke_test.py`

Each script's header shows its command.

## Asks for teammates

- **Person 1:** switch `MODEL_FEATURE_COLUMNS` to `FROZEN_MODEL_FEATURES`, re-run the threshold search (0.72 pooled on Person 2's candidates), and consider pooled, which is equal or better in every comparison. Your "missing feature values = 0" check now holds as-is.
- **Person 2:** fix the empty `target_source` (section 2) and remove the committed `egg-info`. With your candidates, 857 FN remain, compared with 5,839 using the heuristic candidates.
- **Person 4:** independently confirm the thresholds and pooled vs separate on the Person 2 candidate set.
