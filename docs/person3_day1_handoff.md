# Person 3 – Day 1 handoff: similarity features and hard negatives

Feature version: **`p3-v1.1`** (schema not frozen yet; freeze planned for the end of Day 2).

## 1. What is delivered

| Deliverable | Location |
| --- | --- |
| Record-level text views (Unicode + accent-folded + Indic romanisation + phonetic skeleton, legal/alias-aware names, normalised addresses) | `src/business_entity_resolution/feature_text.py` |
| Deterministic batch feature function `calculate_pair_features`, `add_context_features`, `add_retrieval_features`, schema + descriptions | `src/business_entity_resolution/similarity_features.py` |
| Unsupervised IDF statistics (fitted on all six source files, no labels) | `src/business_entity_resolution/token_stats.py`, artifact `artifacts/token_stats.json` |
| Interim blocker, labelling, hard-negative tagging/selection | `src/business_entity_resolution/pair_labels.py` |
| Feature dictionary with missing rate / AUC / importance | `docs/feature_dictionary.md` |
| 20-row labelled example matrix | `docs/examples/feature_matrix_sample.tsv` |
| Unit tests (determinism, chunk parity, missing-address contract, Unicode, unseen country) | `tests/test_similarity_features.py` |

Generated data (git-ignored, under `data/work/p3/`): `labelled_pairs.parquet`, `features.parquet`, `hard_negatives.parquet`, `missed_positives.parquet`, `dev_source1.parquet`, `dev_ground_truth.parquet`, plus `*.report.json` / `*.quality.json` / `*.summary.json`.

## 2. Reproduce (from the repository root)

```bash
export AMAZON_ML_DATASET_ROOT=../student_resource/dataset
python scripts/fit_token_stats.py --workers 10                                   # ~4 min
python scripts/build_training_pairs.py --source1-count 20000 --max-key-df 5000 --workers 10   # ~4 min
python scripts/compute_features.py --pairs data/work/p3/labelled_pairs.parquet \
    --output data/work/p3/features.parquet --workers 10                           # ~1 min
python scripts/feature_report.py --features data/work/p3/features.parquet \
    --ground-truth data/work/p3/dev_ground_truth.parquet
python scripts/build_hard_negatives.py --features data/work/p3/features.parquet \
    --output data/work/p3/hard_negatives.parquet
python scripts/write_feature_dictionary.py --report data/work/p3/features.report.json \
    --quality data/work/p3/features.quality.json
PYTHONPATH=src python -m unittest discover -s tests -v
```

S1 dev IDs come from Person 1's `_priority_sample_source1` with seed 2026 (the same selection logic as `create-dev-subset`).

## 3. API contract for Person 1

> **Superseded by `docs/person3_day2_handoff.md` (feature version p3-v2).** `calculate_pair_features` now also needs `name_noise=`, and the frozen model set excludes the context features.

```python
from business_entity_resolution.similarity_features import (
    prepare_records, calculate_pair_features, add_context_features,
    add_retrieval_features, model_feature_columns)
from business_entity_resolution.token_stats import TokenStats

stats = TokenStats.load("artifacts/token_stats.json")
s1 = prepare_records(s1_raw_df, workers=10)          # raw frame with SOURCE_COLUMNS
sec = prepare_records(sec_raw_df, workers=10)
feats = calculate_pair_features(pairs_df, s1, sec, stats)   # chunk freely by pairs
feats = add_retrieval_features(feats, candidates_long)      # optional, Person 2 long table
feats = add_context_features(feats)                         # ONCE on the full table
X = feats.select(model_feature_columns(include_context=True, include_retrieval=False))
```

* `pairs_df` needs `source1_entity_id`, `candidate_entity_id`; other columns (`label`, `fold`) pass through. Row order is preserved.
* Output features are `Int8`/`Int16`/`Float32` only. No strings or objects, and no infinities (checked on 975k pairs).
* **Missing-address contract:** blank addresses and placeholder addresses (`None`, `null`, `N/A`, …) set `addr_missing_*`. All address similarities become **NaN**, never 0. Tree models handle NaN natively.
* `add_context_features` needs whole S1 groups and whole candidate groups, so run it after concatenating all chunks. It is a cheap polars group-by (0.7 s for 975k rows).
* 85 model features with context, 89 with retrieval.

## 4. Day 1 measurements (dev: 20,000 S1, 975,086 retrieved pairs, 52,854 positives)

| Metric | Value |
| --- | --- |
| 100k-pair benchmark: pair features | 32.6k pairs/s (3.1 s) |
| 100k-pair benchmark: end-to-end incl. record prep | 11.1k pairs/s, peak RSS 2.2 GB |
| 975k pairs: pair features / end-to-end | 31.2k / 18.6k pairs/s, peak RSS 4.1 GB |
| NaN rate | 0.5% on address similarities (missing addresses) and 8.3% on `addr_num_jaccard` (no numbers) |
| Probe model (sklearn HGB, S1-grouped 80/20) pair AUC | 0.9999 |
| Indicative macro F0.5 (all dev S1 in holdout, incl. singletons and missed positives) | **0.852** at threshold 0.65 |
| Oracle macro F0.5 with these candidates (perfect classifier) | 0.870 |

The features recover about 98% of the macro F0.5 that the candidates allow. The remaining ceiling comes from the interim blocker's **76.5% pair recall**, and Person 2's candidate generation fixes that, not the features. The figures above are a feature-quality probe. The official metric and thresholds belong to Person 4.

Most useful features by permutation importance: `addr_num_jaccard`, `combo_mean`, `addr_token_containment`, `name_wratio_latin`, `name_unmatched_max_idf`, `name_legal_conflict`, `name_soft_idf_max`, `ctx_combo_rank`.

## 5. Data findings that matter for everyone

* **Each S2/S3 ID matches at most one S1** (0 re-used IDs among 7.64M positive pairs). Country always agrees between matched records. Person 1 can use one-to-one assignment in post-processing.
* **Deliberate sibling distractors.** The hardest negatives share name and street but differ in house number or legal form, e.g. `Pacific Star Bny LLC, 264 Ivydale Ridge Road` vs `Pacific Star Bny Partners, 269 Ivydale Ridge Rd`. That is why the number features rank first.
* **Cross-script names.** S2/S3 names can be in Devanagari, Bengali, Tamil, Gujarati and other scripts, while S1 is Latin. The skeleton key maps `ஆல் கன்சல்டன்ட்ஸ் பிரைவேட் லிமிடெட்` and `All Consultants Pvt Ltd` to the same key `al knsltnts`. **Person 2:** word-level Latin blocking will miss these; blocking on skeleton tokens (`pair_labels.record_keys`) catches them.
* **Name noise:** honorifics (`Mr`), suffix junk (`Center`, `Corp`), `DBA:` / `F/K/A` aliases, `@handles` and URLs, OCR swaps (`Diam0nd`, `SHlVAM`), and entirely unrelated trade names where only the address links the records.
* **Address noise:** component reordering, state code vs full name (normalised to codes), street-suffix abbreviations, house-number corruption (`5550` vs `550`, `001872`), and placeholder strings for missing values.
* In raw S2/S3 about 3.4% of addresses are missing, but only 0.5% of retrieved candidates have a missing address. Address-key blocking under-retrieves no-address records. **Person 2:** make sure a name-only route exists.

## 6. Hard negatives

`hard_negatives.parquet` contains 98,087 label-0 pairs: the top 5 per S1 by `combo_mean`, taken from actually retrieved candidates. Each row is tagged `hard_negative_type`:

| Type | Count |
| --- | --- |
| other_retrieved | 66,576 |
| high_address_low_name | 22,534 |
| near_duplicate (name and address ≥ 0.85) | 4,961 |
| high_name_low_address | 3,539 |
| common_name_collision (high name, only low-IDF tokens shared) | 477 |

On Day 2, rerun with `--score-column match_probability` on Person 1's V1 scores to mine the model's own confident mistakes.

## 7. Open items / requests

* **Person 1:** LightGBM 4.6 on macOS needs the OpenMP runtime. `brew install libomp` fixed it on this machine (verified that `import lightgbm` works); every macOS teammate needs the same step. Please also confirm the chunking plan for the full test run (see runtime below).
* **Person 2:** I accept the long candidate table (one row per route) and aggregate it in `add_retrieval_features`. Route names can be anything. Please send your first candidate batch; I can then report recall per route and hard negatives from your real candidates.
* **Person 4:** please send the S1-level fold file. `feature_report.py` currently uses a hash split. The feature dictionary is ready for error-analysis slicing.
* **Runtime estimate for the full test set** (~1.73M S1 × ~50 candidates ≈ 87M pairs): about 46 min single-process for pair features plus about 3 min record prep with 10 workers. On Day 2 I will parallelise chunks across processes and trim redundant features to cut this. Memory must be chunked by S1 ranges; each chunk prepares only the records it references.
