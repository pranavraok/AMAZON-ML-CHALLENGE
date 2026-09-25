# Shared pipeline interfaces

These contracts allow all four workstreams to develop in parallel. Changes require an explicit team handoff because downstream modules rely on the exact names and meanings below.

## 1. Source-record schema

All source TSVs contain:

| Column | Meaning |
| --- | --- |
| `entity_id` | Unique source-prefixed record ID |
| `business_name` | Raw business name |
| `business_address` | Raw address; may be blank in S2/S3 |
| `country` | Open-set country string |

Ground truth contains:

| Column | Meaning |
| --- | --- |
| `source1_entity_id` | S1 reference ID |
| `matched_entity_ids` | Comma-separated S2/S3 IDs; blank for a singleton |

## 2. Normalized-record contract

`normalization.normalize_record()` returns a `NormalizedRecord` containing:

- Raw `entity_id` and `country`.
- Unicode-preserving normalized name and address.
- Accent-folded ASCII name and address.
- Token-sorted name and address forms.
- Name and address token tuples.
- Address numeric-token tuple.
- Explicit `address_missing` flag.

Person 2 and Person 3 must use this common base. They may add specialized derived values in their own modules, but they must not silently redefine these fields.

## 3. Candidate-pair contract

Person 2 returns one row for each S1-to-secondary candidate pair:

| Column | Type | Meaning |
| --- | --- | --- |
| `source1_entity_id` | string | S1 query ID |
| `candidate_entity_id` | string | S2 or S3 candidate ID |
| `target_source` | string | Exactly `S2` or `S3` |
| `retrieval_route` | string | Stable route name, such as `name_tfidf` |
| `retrieval_rank` | integer | One-based rank within the route |
| `retrieval_score` | float | Route-specific similarity score |

If the same pair is found by several routes, either preserve one row per route or aggregate deterministically. The chosen representation must be documented before feature work is frozen.

`candidate_pairs.tsv` is not this long pair table. It is the final submission view with exactly two columns:

```text
source1_entity_id<TAB>candidate_entity_ids
```

It must represent the exact final candidates fed to model inference.

## 4. Feature-table contract

Person 3 returns one row per unique candidate pair. The required identifier columns are:

| Column | Required |
| --- | --- |
| `source1_entity_id` | Yes |
| `candidate_entity_id` | Yes |
| `target_source` | Yes |
| `label` | Training only: 1 for a true link, otherwise 0 |
| `fold` | Added from Person 4's S1-level split |

All model features must be numeric or boolean with stable names and dtypes. Missing evidence must have explicit flags; it must not silently become a plausible zero similarity.

Training and inference must call the same feature implementation.

## 5. Model-score contract

Person 1 returns:

| Column | Meaning |
| --- | --- |
| `source1_entity_id` | S1 ID |
| `candidate_entity_id` | S2/S3 ID |
| `target_source` | `S2` or `S3` |
| `match_probability` | Model score used for thresholding |

The exact model version, feature version, candidate configuration, and threshold configuration must be saved together.

## 6. Evaluation contract

Person 4 evaluates predictions at the S1 entity level using the official macro F0.5 definition. Candidate recall and final matching F0.5 are separate reports.

Thresholds must be selected on grouped validation data, not on pair-level random splits.

## 7. Final output contracts

`matching_results.tsv`:

```text
source1_entity_id<TAB>matched_entity_ids
```

`candidate_pairs.tsv`:

```text
source1_entity_id<TAB>candidate_entity_ids
```

Rules:

- UTF-8 TSV with exact headers.
- One row for every test S1 ID, including empty lists.
- Comma-separated S2/S3 IDs without duplicates.
- No S1 self-matches.
- Every final match must be present in that S1 candidate list.

## 8. Handoff checklist

Every handoff includes:

1. File path and format.
2. Row count and unique S1 count.
3. Column names and dtypes.
4. Configuration and seed.
5. Runtime, peak memory, and chunk size.
6. Known limitations or failed checks.
7. One command that reproduces the output.
