# Day 3 candidate reduction results

Machine-readable source: `docs/day3_candidate_reduction_results.json`

Evaluated on the development subset (50,000 S1 against 183,231 S2 and 189,407 S3 targets) with ground truth. The frozen retrieval route set is identical in every configuration: only numeric budget knobs differ.

## Recommendation

**Keep baseline configuration.**

no tested reduction keeps pair recall within 0.02 pp and full-entity recall within 0.05 pp of the baseline while removing at least 10% of the candidates; the cheapest reductions are the smallest savings and the aggressive ones cost real recall, so the frozen baseline stands.

| Configuration | Rule | S1 count | Candidate rows | Total candidate pairs | Avg cand/S1 | Median cand/S1 | P95 cand/S1 | P99 cand/S1 | Max cand/S1 | Pair recall | Full-entity recall | Runtime (s) | Peak RAM (GB) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline | frozen CandidateGenerationConfig() defaults, unchanged | 50,000 | 50,000 | 17,762,767 | 355.26 | 368.00 | 515.00 | 585.00 | 1,150 | 99.629282% | 98.8640% | 1456.1 | 2.53 |
| B_name_top_k_50 | halve name_top_k | 50,000 | 50,000 | 15,195,931 | 303.92 | 306.00 | 441.00 | 504.00 | 1,051 | 99.581784% | 98.7100% | 1423.9 | 2.43 |
| C_address_top_k_50 | halve address_top_k | 50,000 | 50,000 | 13,933,016 | 278.66 | 299.00 | 427.00 | 493.00 | 1,104 | 99.559193% | 98.6860% | 784.0 | 2.38 |
| D_posting_caps_x0.6 | multiply every posting cap by 0.6, floor 1 | 50,000 | 50,000 | 16,416,139 | 328.32 | 335.00 | 457.00 | 500.00 | 775 | 99.597424% | 98.7700% | 1154.6 | 2.48 |
| E_combined_conservative | posting caps x0.8; name_top_k and address_top_k x0.75 | 50,000 | 50,000 | 14,045,238 | 280.90 | 297.00 | 396.00 | 448.00 | 784 | 99.567303% | 98.6800% | 893.8 | 2.39 |

## Cost of each reduction

| Configuration | Candidate reduction | Pair recall | Pair recall drop | Full-entity recall | Full-entity drop | Extra missed true pairs | Entities losing all links |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A_baseline | 0.00% | 99.629282% | 0.0000 pp | 98.8640% | 0.0000 pp | 0 | 0 |
| B_name_top_k_50 | 14.45% | 99.581784% | 0.0475 pp | 98.7100% | 0.1540 pp | 82 | 77 |
| C_address_top_k_50 | 21.56% | 99.559193% | 0.0701 pp | 98.6860% | 0.1780 pp | 121 | 89 |
| D_posting_caps_x0.6 | 7.58% | 99.597424% | 0.0319 pp | 98.7700% | 0.0940 pp | 55 | 47 |
| E_combined_conservative | 20.93% | 99.567303% | 0.0620 pp | 98.6800% | 0.1840 pp | 107 | 92 |

## Exact parameter values

- **A_baseline** - none (frozen baseline)
- **B_name_top_k_50** - `name_top_k`: 100 -> 50
- **C_address_top_k_50** - `address_top_k`: 100 -> 50
- **D_posting_caps_x0.6** - `address_pair_max_postings`: 5 -> 3, `address_token_max_postings`: 50 -> 30, `numeric_token_max_postings`: 100 -> 60, `rare_token_max_postings`: 50 -> 30, `translit_token_max_postings`: 100 -> 60
- **E_combined_conservative** - `address_pair_max_postings`: 5 -> 4, `address_token_max_postings`: 50 -> 40, `address_top_k`: 100 -> 75, `name_top_k`: 100 -> 75, `numeric_token_max_postings`: 100 -> 80, `rare_token_max_postings`: 50 -> 40, `translit_token_max_postings`: 100 -> 80

## Baseline cross-check

Re-measured A: 17,762,767 pairs, 99.629282% pair recall, 98.8640% full-entity recall.

Against the integrated baseline quoted in the brief (17,756,227 pairs, 99.6281% pair recall): +6,540 pairs (+0.037%), +0.0012 pp pair recall, +0.0040 pp full-entity recall.

the re-measured baseline on the frozen code is within 0.04% of the integrated baseline quoted in the brief; the difference is a re-measurement of the same frozen route set, not a route change.

## Documented fallback

documented fallback only; adopted if the full-test runtime estimate makes the baseline infeasible on the available machine.

Trigger: projected single-machine full-test runtime exceeds the available compute window.

Order of preference: `D_posting_caps_x0.6`, `B_name_top_k_50`, `E_combined_conservative`, `C_address_top_k_50`.

