# Day 2 - Person 2 candidate generation results

## Reproduce

```powershell
$env:PYTHONPATH = "src"
python -m business_entity_resolution.dev_candidate_generation --limit 1000
python -m pytest tests -q
```

Configuration is no longer overridden in the benchmark script. It reads
`CandidateGenerationConfig()` defaults so the benchmark always measures the
shipped configuration.

## Headline numbers (development subset, first 1,000 S1)

| Metric | Day 1 baseline | Day 2 | Change |
| --- | --- | --- | --- |
| Pair recall | 99.1056% | **99.7115%** | +0.606 pp |
| Missed true pairs | 31 | **10** | -21 |
| Entity / all-links coverage | 97.00% | **99.00%** | +2.00 pp |
| Average candidates / S1 | 181.94 | 352.27 | +170.3 |
| Median candidates / S1 | 227.50 | 358.00 | +130.5 |
| P95 candidates / S1 | 257.00 | 522.10 | +265.1 |
| P99 candidates / S1 | 304.00 | 629.01 | +325.0 |
| Maximum candidates / S1 | 346 | 844 | +498 |
| Generation time | - | 27.2 ms / S1 | |
| Index build time | - | 273 s (S2 + S3) | |
| Python peak memory | 2.52 GB | 3.26 GB | +0.74 GB |

### Confirmation at 10,000 S1

| Metric | Value |
| --- | --- |
| Total true pairs | 34,708 |
| Missed true pairs | 118 |
| Pair recall | 99.6600% |
| Entity / all-links coverage | 98.89% |
| Average candidates / S1 | 355.08 |
| Median / P95 / P99 / max | 369 / 514 / 585 / 859 |
| Generation time | 37.3 ms / S1 |
| Python peak memory | 4.63 GB |

The gain is not an artifact of the first-1,000 prefix.

## Root cause of the Day 1 misses

All 31 missed pairs had `country` equal, so no miss was a country-partition
problem. The dominant cause was a **cross-script name**:

- Dev S1 names are 100% ASCII. Dev S2/S3 names are 15.3% / 11.7% non-ASCII.
- 26 of 31 missed pairs had **exactly zero** shared name tokens and a true
  name TF-IDF score of `0.0` at rank 12,000-48,000.
- Example `S1-101689558` "Al Infotech Private Limited" vs
  `S3-934626016` "ಅಲ್ ಇನ್‌ಫೋಟೆಕ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್" (Kannada).
- The address was the only surviving evidence: 27/31 shared address tokens,
  21/31 shared address numbers, true address rank 22-1686 at score
  0.128-0.418.

The remaining 5 misses were misspelled Latin names whose true target has a
**blank address** (e.g. "Perfect Infrastructure" vs "Perfect Infrrsatrutcure"),
leaving name similarity as the only signal at rank 125-2116.

## Defects found and fixed

1. **`blocking.py` read `record.name_compact` and `record.address_compact`,
   which do not exist on `NormalizedRecord`.** Every read was wrapped in
   `getattr(..., "")`, so 7 of 8 routes and 6 of 10 indexes were silently
   dead. `exact_name`, `exact_combined`, `exact_address`, `address_tokens`
   and `numeric_address` never fired. The index now validates the record
   contract once in `__init__` and raises, so this cannot recur silently.
2. **Country-agnostic `global_*` routes** returned candidates from other
   countries, breaking the country-partition design. Removed. All 31 missed
   pairs were same-country, so no recall was lost.
3. **`blocking_index.generate()` was called without `target_source`**, so
   every blocking-only candidate carried `target_source=""` instead of
   `S2`/`S3`, violating `docs/interfaces.md` section 3.
4. **`diagnose_missed.py` used a different configuration than the benchmark**
   (20/20/0.0/0.0 vs 100/20/0.30/0.0) while claiming to match it. It now
   reads the same dataclass defaults.
5. **`generate_submission_candidates.py` read `AMAZON_ML_DATASET_ROOT` at
   import time** (`KeyError` on import) and hard-coded the stale Day 1
   configuration. It now resolves paths in `main()`, accepts CLI arguments,
   streams rows to disk, and uses the shipped defaults.
6. **Two tests were failing** before this work
   (`test_candidate_metadata_contains_routes`,
   `test_country_partition_is_preserved`). Both now pass.

## Route changes, and the evidence for each

| Change | Justification |
| --- | --- |
| Restore the 5 dead routes | Defect 1 above. Restoring `numeric_address` alone recovered 9 of the 31 pairs. |
| Drop `global_*` routes | Defect 2. Pure noise; no missed pair needed them. |
| `address_top_k` 20 -> 100 | True address rank for missed pairs is 22-1686. This is *not* a blind top-K increase: it is sized to the measured rank distribution. |
| `address_min_score` 0.0 -> 0.15 | `min_score=0.0` admits zero-similarity documents. The true missed-pair band is 0.128-0.418, so 0.15 keeps almost all of it and rejects the filler. |
| `name_min_score` stays 0.30 | Measured: `name_top_k=300` buys +2 pairs for +414 candidates; `k=1000` buys +4 for +1352. The true targets for misspelled names sit at rank 125-2116, so extra depth cannot reach them. |
| New `address_token_pair` route (cap 5) | S1-109093544 shares **five** address tokens with its true target, but each token individually exceeds the per-token cap of 50, so `address_tokens` rejected it. A *pair* of common tokens is far more selective. Measured: **+3 pairs for +6.9 candidates / S1** (P99 marginal 59, max 95) - the best recall-per-candidate ratio found. |
| Numeric tokens strip leading zeros | `S1-109093544` "F. No.-14-A" vs target "F. No.-0014-a". |

## Routes that were tested and REJECTED

Recorded so they are not retried:

| Route idea | Result | Why rejected |
| --- | --- | --- |
| Romanized (unidecode) name TF-IDF | 0/31 recovered, +67 cand/S1 | Transliteration is phonetic, not semantic. "infotech" vs "inphoottek" stays far apart in the romanized space. |
| `rare_name_tokens` cap 200 / 500 / 1000 | +1 / +4 / +4 for +77 / +485 / +852 | The distinctive token in a typo only exists on the *target* side, so query-side blocking cannot reach it. |
| Address token-pair keys with a per-token cap | 0/31 | Requiring *both* tokens to be rare means no pair ever qualifies, since real addresses are made of common tokens. The cap must apply to the combined bucket. |
| Adaptive deep address pass for "cross-script" S1 | 0/13 recovered | The weak-name-signal detector is invalid: the best name score is ~1.000 for these pairs, because name TF-IDF finds a *different* Latin-script business with a near-identical name. |
| Relax numeric tokens to >=1 digit | identical to >=2 digits | The numbers that matter already have 2+ digits. |
| Address TF-IDF `top_k` 200-1000 | +2 to +7 pairs for +55 to +680 cand/S1 | Straight recall/cost trade with worse marginal efficiency than `address_token_pair`. |

## Route contribution (1,000 S1)

| Route | Candidate pairs | S1 entities touched |
| --- | --- | --- |
| address_tfidf | 170,161 | 1,000 |
| name_tfidf | 130,305 | 1,000 |
| numeric_address | 35,986 | 630 |
| rare_name_tokens | 25,206 | 620 |
| address_token_pair | 17,442 | 999 |
| transliterated_token | 8,250 | 459 |
| address_tokens | 4,244 | 681 |
| token_sorted_name | 1,412 | 684 |
| transliterated_exact_name | 1,353 | 668 |
| exact_name | 1,169 | 615 |
| exact_address | 320 | 238 |
| exact_combined | 51 | 48 |

## Remaining 10 misses

- **6 are address-rank misses.** True address rank 102-1686. Reaching them
  means `address_top_k` 200-1000, which costs +55 to +680 candidates / S1.
  Deferred pending a candidate budget decision from the team.
- **4 have a blank target address** and a heavily misspelled name
  (e.g. "Perfect Infrrsatrutcure", "Om Sohlins", "Williams Agnc").
  These need name-side fuzzy matching, which the Day 2 measurements show is
  expensive and low-yield at this scale.

## OPEN: the `retrieval_route` contract needs a team decision

`docs/interfaces.md` section 3 leaves this open and says the representation
"must be documented before feature work is frozen". Measured on 200 S1
(70,432 unique candidate pairs, 352.16 candidates/S1):

| Question | Answer |
| --- | --- |
| One row per route, or one per unique pair? | **One per unique pair.** `CandidateGenerator.generate_for_one()` returns one `CandidateEvidence` per `candidate_entity_id`, with the routes merged into it. |
| How many pairs carry more than one route? | **8.05%** (5,673 of 70,432), up to 12 routes on a single pair. |

So `retrieval_route` is **multi-valued**, but section 3 declares it a
`string`. Any consumer that treats it as one name silently keeps only one
arbitrary route for those 8.05% of pairs.

### Two concrete hazards for Person 3

`CandidateEvidence` keeps `routes`, `scores` and `ranks` as three
independent lists, and `add()` appends a route with **no** score for the
blocking routes. Consequences, both measured:

1. **`zip(routes, scores)` is wrong for 23.78% of pairs** (16,750 of
   70,432). The lists are not index-parallel, so scores attach to the wrong
   route with no error raised.
2. **16.93% of pairs (11,922) have no score at all** - they came from
   blocking alone. `retrieval_score` then returns `0.0` via
   `max(self.scores) if self.scores else 0.0`.

Hazard 2 directly contradicts `docs/interfaces.md` section 4: *"Missing
evidence must have explicit flags; it must not silently become a plausible
zero similarity."* A Person 3 feature built on `retrieval_score` would read
"0.0 similarity" for roughly one candidate in six, when the truth is "this
route does not compute a similarity at all". Those are very different
features.

### Recommended resolution (needs Person 1 and Person 3 to ratify)

Keep one row per unique pair, but make the evidence explicit rather than
implicit:

- Replace the scalar `retrieval_score` / `retrieval_rank` properties with a
  separate nullable score/rank **per route**, so the three lists can never
  drift out of alignment.
- Add a boolean `retrieval_score_available` (or an explicit sentinel) so a
  blocking-only pair is distinguishable from a genuine zero similarity, per
  section 4.
- Add `route_count`, which is already computed and is a strong signal on its
  own: multi-route agreement is exactly the corroboration signal a
  classifier wants, and today it is computed but never exposed.

This was deliberately **not** changed in the Day 2 commit, because it
alters the Day 2 candidate-generation logic and the representation choice is
a team decision, not Person 2's alone. Person 3 should confirm the feature
schema against this before freezing features.

### Second decision still open: candidate budget

Unrelated to the representation question, and also unresolved: 6 of the 10
remaining misses are address-rank misses that would need
`address_top_k` 200-1000, at +55 to +680 candidates/S1. The configuration
ships at `address_top_k=100` and was deliberately left there. This needs a
team decision on the candidate budget before it can be revisited.

## Known limitations

- The benchmark's `--limit` takes a deterministic **prefix** of
  `train_source1.tsv`, which is sorted by `entity_id`. This is reproducible
  but not a random sample.
- `entity_coverage` counts an S1 with zero true matches as fully covered,
  because an empty set is a subset of every candidate set. The metric is
  therefore optimistic by roughly the singleton rate. Person 4 should report
  singletons separately.
- `Unidecode` is used by the transliteration routes. It was previously
  undeclared and has now been pinned in `requirements.txt`. Without it those
  two routes silently degrade into duplicates of the Unicode routes rather
  than failing, so a clean environment must keep the pin.
- Index build time (273 s for 372k dev targets) does not extrapolate to the
  full test set. The chunked architecture is still required.

## Handoff checklist (`docs/interfaces.md` section 8)

1. **Path / format** - `data/dev/train`, TSV. Reports printed by
   `dev_candidate_generation`; no artifact is written yet.
2. **Row count / unique S1** - 1,000 S1 evaluated, 3,466 true pairs;
   10,000 S1 / 34,708 true pairs for the confirmation run.
3. **Columns / dtypes** - unchanged. `NormalizedRecord` contract untouched.
4. **Configuration and seed** - `CandidateGenerationConfig()` defaults,
   printed by the benchmark. Dev subset seed 2026.
5. **Runtime / peak memory / chunk size** - 27.2 ms/S1, 273 s index build,
   3.26 GB peak. No chunking yet.
6. **Known limitations** - see above.
7. **Reproduce** - `python -m business_entity_resolution.dev_candidate_generation --limit 1000`
   with `PYTHONPATH=src`.
