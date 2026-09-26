# Person 3 – Day 2 handoff: frozen features, ablation and hard negatives

> **Final state: see `docs/person3_final_features.md` (p3-v2.1).** v2.1 replaces NaN with explicit sentinels (no NaN/inf anywhere), was tested on Person 2's real candidates, and streams for the full test set. The Day 2 numbers below were measured on p3-v2.

Feature version **`p3-v2`**, frozen in `configs/p3_features_frozen.json`. The file lists the 79 features, their dtypes, and SHA-256 checksums of the code and artifacts. `python scripts/freeze_p3_features.py --verify` fails if anything has drifted.

This document supersedes the API section of `docs/person3_day1_handoff.md`.

## 1. Merge-gate evidence (Person 1's identical rows, folds, models and scorer)

All numbers come from Person 1's Day 1 development pipeline, reproduced locally. It gives exactly their published values: Day 1 0.912370 and Day 2 separate 0.913314.

Setup:
- Rows: `candidate_pairs_training.tsv`, 1,150,117 pairs.
- Folds: `folds.tsv`, with fold 0 as validation (10,081 S1 entities).
- Model: `modeling._build_classifier` with 350 rounds.
- Scorer: `evaluation.threshold_search` over thresholds 0.05–0.99.

Only the feature columns change between rows of the table. Reproduce with `python scripts/ablate_features.py`.

| Feature set | Arch | #feat | Thr | Macro F0.5 | Precision | Recall | FP | FN | Singleton acc |
|---|---|---|---|---|---|---|---|---|---|
| Person 1 baseline | separate | 22 | 0.70 | 0.913314 | 0.9866 | 0.8268 | 391 | 6047 | 0.9597 |
| Person 1 baseline | pooled | 22 | 0.74 | 0.913244 | 0.9884 | 0.8242 | 339 | 6139 | 0.9634 |
| **Person 3 frozen** | separate | 79 | 0.72 | **0.921008** | 0.9953 | 0.8337 | 137 | 5808 | 0.9835 |
| **Person 3 frozen** | **pooled** | 79 | 0.66 | **0.921784** | 0.9956 | 0.8347 | 129 | 5773 | 0.9872 |
| Baseline + frozen | separate | 101 | 0.76 | 0.921005 | 0.9959 | 0.8334 | 120 | 5817 | 0.9817 |
| Baseline + frozen | pooled | 101 | 0.75 | 0.921783 | 0.9962 | 0.8332 | 111 | 5825 | 0.9890 |

Paired entity-level bootstrap, 2,000 resamples (`scripts/p3_experiments.py bootstrap`):

| Comparison | Δ macro F0.5 | 95% CI | Entities better / worse |
|---|---|---|---|
| Baseline → frozen (separate) | +0.0077 | [+0.0064, +0.0090] | 572 / 118 |
| Baseline → frozen (pooled) | +0.0085 | [+0.0072, +0.0099] | 621 / 94 |

Recommendations for Person 1 and Person 4:

1. **Replace the 22 baseline features with the 79 frozen features.** Adding the baseline features on top brings no gain (±0.00001).
2. **Revisit pooled vs separate.** With the new features, pooled is better (0.92178 vs 0.92101). The earlier choice of separate rested on a 0.00007 margin with baseline features. Person 4 should confirm.
3. Thresholds move to about 0.66 (pooled) or 0.72 (separate). Re-run the threshold search after integrating; don't reuse 0.70.

## 2. Where the remaining errors are

| Error type | v1 features (separate, thr 0.61) | **Frozen v2 (pooled, thr 0.66)** | Owner |
|---|---|---|---|
| False negative, candidate never generated | 5,622 | **5,622 (97% of FNs)** | **Person 2** |
| False negative, scored below threshold | 178 | **151** | Person 3 / model |
| False positive | 153 | **129** | Person 3 / model |

The v1 analysis below is what motivated the v2 features.

* **FPs.** 90 of 153 have near-identical names. They are "sibling" records at the same street with a slightly different house number (`8-3-898/30/3` vs `/30/5`, `1788` vs `1793`, `Plot 69` vs `71`) or a missing address. The v2 primary-number features target the sibling cases.
* **Missing addresses** are involved in 67 FPs and 128 model FNs. The name alone is then genuinely ambiguous: identical names occur on both sides of the ledger. v2 adds exact-name frequency (how many businesses share the name) and a noise ratio for unmatched name words. Together they gave +0.0010 (CI [+0.0002, +0.0018]).

Error rows with all feature values: `data/work/p3d2/error_slices_frozen_pooled.parquet` (`scripts/error_slices_p3.py`).

## 3. What changed from v1 to v2

| Change | Why | Evidence |
|---|---|---|
| `name_core_freq_{s1,sec}_{a,b}` (4) | Ambiguity of name-only decisions | Dropping the group: −0.00037 |
| `name_{a,b}_unmatched_max_noise` (2) | An unmatched `services`/`holdings`/`www` is injected noise; an unmatched real word is a difference | −0.00033 |
| `addr_primary_equal`, `addr_primary_cross`, `addr_num_only_{a,b}` (4) | Sibling addresses share most numbers | −0.00026 |
| Context features `ctx_*` **removed from the model set** | Neutral: CI [−0.0010, +0.0004] separate, [−0.0007, +0.0009] pooled. They also need whole-table state that the training positive guards distort | Still available via `add_context_features` |
| Parallel driver `calculate_pair_features_parallel` | Runtime | Output bit-identical to serial (unit test) |

A hard noise-word vocabulary was tried and rejected. Its high-ratio tokens were mostly OCR/transliteration variants of real words (`c0nsulting`, `praivet`), which must not be removed.

Drop-one-group ablation (pooled, full set 0.921715): every group is worth ≤ 0.0006 alone, which shows useful redundancy and no group that is safe to remove except context. Details are in `data/work/p3d2/group_ablation_pooled.json`.

## 4. Country robustness (leave-one-country-out, pooled)

Train on the other country only, then evaluate on the held-out country's validation fold. `in_country` means trained on the same country's training folds.

| Held out | Baseline in-country | Baseline LOCO | P3 in-country | P3 LOCO |
|---|---|---|---|---|
| India (3,997 S1) | 0.8544 | 0.8344 (−0.020) | 0.8603 | **0.8576 (−0.003)** |
| US (6,084 S1) | 0.9548 | 0.9446 (−0.010) | 0.9619 | **0.9610 (−0.001)** |

The features do not collapse when the country is never seen in training. P3 trained on US only beats the baseline trained on India itself.

**France smoke test on real test data** (`scripts/france_smoke_test.py`): France is 259,452 of the test S1 records (15%). On 500 sampled France S1 records and 24,469 candidate pairs:
- All columns numeric, no infinities.
- Missing rates at or below the training levels.
- French abbreviations normalised (`Imp`→impasse, `Pl.`→place).
- Top pairs look right, e.g. `Union des Francophone, 19 Impasse le Bigot` ↔ `Union des Francophone EURL, 19 Imp Le Bigot`.

## 5. Hard negatives

| Set | File | Size | Notes |
|---|---|---|---|
| Similarity-ranked (Day 1) | `data/work/p3/hard_negatives.parquet` | 98,087 | Top 5 wrong candidates per S1 by `combo_mean` |
| Model-mined (Day 2) | `data/work/p3d2/hard_negatives_model.parquet` | 2,538 over 2,291 S1 | Label-0 pairs with out-of-fold probability ≥ 0.1 from 4-fold cross-fitting on training folds only |

Model-mined families: other_retrieved 1,192, high_name_low_address 1,011, near_duplicate 190, high_address_low_name 139, common_name_collision 6.

Up-weighting them does not help macro F0.5 (w=1: 0.92178, w=2: 0.92169, w=4: 0.92175). It only trades FP for FN (129 → 110 FP, 5,773 → 5,805 FN). **Recommendation: no special weighting.** They are already present in training because they are real retrieved candidates. Use the file for error analysis.

## 6. Runtime, memory and missing values

| Measurement | Value |
|---|---|
| 1.15M pairs, 10 processes, including record preparation | 21.8 s, **52.8k pairs/s** |
| Main-process RSS | 1.2 GB |
| Estimated full test run (~1.73M S1 × ~50 candidates ≈ 87M pairs) | ≈ 28 min |
| Small batches (100k pairs) | ~16k pairs/s, because each worker spends ~2 s loading the 16 MB IDF file |
| Infinite values | none (enforced by `validate_feature_frame`) |

**Missing values are intentional.** NaN means "not applicable": a missing address (with the `addr_missing_*` flags set), an address with no numbers, or all name tokens matched. LightGBM handles NaN natively.

**Person 1:** your checklist's "verify no missing values" should become "verify no *unexpected* missing values". `validate_feature_frame` plus the missing-rate report in `scripts/compute_features.py` cover this. If you ever write features to TSV, NaN is written as `NaN`, which your `_read_features` (`null_values`) and `float()` in chunked inference both accept.

**Train/inference parity:** `scripts/build_p3_feature_file.py`, run on your candidate TSV, reproduces the ablation features **bit-for-bit** on all 79 columns and 1,150,117 rows. Labels and folds match yours exactly.

## 7. How Person 1 integrates

```bash
export AMAZON_ML_DATASET_ROOT=../student_resource/dataset
# one-off artifacts (unsupervised, all six source files; ~6 min total)
python scripts/fit_token_stats.py --workers 10
python scripts/fit_name_stats.py --workers 10
python scripts/freeze_p3_features.py --verify        # must print PASS

# training rows
python scripts/build_p3_feature_file.py \
    --candidates data/work/day1/candidate_pairs_training.tsv \
    --source-dir data/dev/train --prefix train \
    --truth data/dev/train/train_ground_truth.tsv --folds data/work/day1/folds.tsv \
    --output data/work/p3d2/p3_features_training.parquet --workers 10

# test inference (same command, no truth/folds)
python scripts/build_p3_feature_file.py \
    --candidates <test candidate_pairs tsv> --source-dir ../student_resource/dataset/test \
    --prefix test --output <test features parquet> --workers 10
```

The model columns are `business_entity_resolution.similarity_features.FROZEN_MODEL_FEATURES`. Your `modeling.py` imports `MODEL_FEATURE_COLUMNS` from `features.py`, so point that import at the frozen tuple or pass the list through. I haven't edited your modules.

Python API:

```python
from business_entity_resolution.similarity_features import (
    calculate_pair_features_parallel, validate_feature_frame, FROZEN_MODEL_FEATURES)
features = calculate_pair_features_parallel(pairs, s1_raw, sec_raw,
    "artifacts/token_stats.json", "artifacts/name_stats", processes=10)
validate_feature_frame(features)
```

`artifacts/` is git-ignored: `token_stats.json` is 16 MB and `name_stats/` is 81 MB. Either regenerate them with the two commands above, where the checksums in the frozen config confirm the result, or share the files directly.

## 8. Reproduce every Day 2 number

```bash
python scripts/create_dev_subset.py --config configs/base.json      # Person 1
python scripts/run_day1_pipeline.py --config configs/base.json      # Person 1
python scripts/ablate_features.py --workers 10                      # table in section 1
python scripts/p3_experiments.py bootstrap --a data/work/p3d2/scores/baseline_pooled.tsv \
    --b data/work/p3d2/scores/frozen_pooled.tsv
python scripts/p3_experiments.py groups --architecture pooled
python scripts/p3_experiments.py loco --architecture pooled
python scripts/error_slices_p3.py --scores data/work/p3d2/scores/frozen_pooled.tsv --threshold 0.66 \
    --output data/work/p3d2/error_slices_frozen_pooled.parquet
python scripts/mine_hard_negatives.py --weights 1 2 4
python scripts/france_smoke_test.py --workers 10
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 9. Open items

* **Person 2:** 97% of the remaining false negatives are candidate misses. See the Day 1 notes on cross-script (skeleton-key) blocking and name-only routes for records with no address.
* **Person 4:** please confirm pooled vs separate and the new threshold independently, and rerun LOCO with your own split.
* **Day 3 (me):** parity check on the final model's feature list, monitoring the full test run, and an ambiguity report near the threshold.
