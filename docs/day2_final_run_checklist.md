# Final full-data run and submission checklist

Use this only after Person 2, Person 3, and Person 4 handoffs pass the Day 2 merge gate.

## Before the full run

- [ ] Record the exact Git commit and frozen configuration used for the run.
- [ ] Confirm candidate and feature headers still match `docs/interfaces.md`.
- [ ] Confirm the selected architecture with Person 4; current provisional choice is separate S2/S3 LightGBM models.
- [ ] Confirm the final threshold plan; current provisional global threshold is `0.70`.
- [ ] Confirm there is no external lookup, geocoding API, registry, web enrichment, or prohibited data.
- [ ] Confirm all libraries and any pretrained models have challenge-compatible MIT or Apache 2.0 licenses and any model is at most 8B parameters.
- [ ] Run all tests and stop on any failure.
- [ ] Estimate free disk space from the 50,000-entity benchmark and keep generous space for partial files.
- [ ] Keep restartable inference enabled with a 100,000-row chunk size unless a benchmark justifies changing it.

## Train and infer

- [ ] Generate full training candidates with the accepted Person 2 implementation.
- [ ] Measure candidate recall on held-out training data before continuing.
- [ ] Generate features with the accepted Person 3 implementation and verify no missing or non-finite values.
- [ ] Retrain the selected model architecture on all approved training rows only after model and threshold selection is finished.
- [ ] Generate test candidates for every Source 1 entity, including France.
- [ ] Confirm `candidate_pairs.tsv` represents the exact final candidate set actually scored by the model.
- [ ] Run chunked test inference and retain the checkpoint until scoring completes.
- [ ] Apply only the frozen threshold and post-processing rules; do not make last-minute hand edits.
- [ ] Sort Source 1 rows and comma-separated ID lists deterministically.

## Output correctness

- [ ] `output/matching_results.tsv` has exactly the headers `source1_entity_id` and `matched_entity_ids`.
- [ ] `output/candidate_pairs.tsv` has exactly the headers `source1_entity_id` and `candidate_entity_ids`.
- [ ] Both files are tab-separated, not comma-separated.
- [ ] Every test Source 1 ID appears exactly once in each file, including singletons with an empty list.
- [ ] Every listed candidate/match is an existing test Source 2 or Source 3 ID.
- [ ] No Source 1 ID appears inside a candidate or match list.
- [ ] No ID is duplicated inside a comma-separated list.
- [ ] Every final match is present in the corresponding final candidate list.
- [ ] Run the supplied validator with `--check-ids` and require `PASS`.

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

## Package structure

- [ ] The zip contains `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
- [ ] The zip contains a self-contained `code/business_entity_resolution/src/`.
- [ ] The code folder includes exact reproduction instructions and pinned dependencies.
- [ ] The filled `Documentation_template.md` is at the zip root.
- [ ] The methodology describes candidate generation, model architecture, features, thresholds, validation, runtime, and limitations.
- [ ] Re-extract the final zip into a clean folder and check its tree before uploading.
- [ ] Upload `matching_results.tsv` to the leaderboard portal and keep the exact same file in the final zip.

## Stop conditions

Do not submit when the validator fails, the candidate file is not the exact set scored, a teammate handoff lacks same-fold evidence, France rows are missing, output files were manually altered after validation, or the code cannot reproduce the outputs.
