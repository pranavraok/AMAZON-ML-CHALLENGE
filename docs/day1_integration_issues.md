# Day 2 integration issues and owners

| Priority | Evidence from Day 1 | Day 2 owner | Required action |
| --- | --- | --- | --- |
| P0 | Validation candidate link recall is 0.838985, below the desired competitive ceiling. | Person 2 | Add TF-IDF, rare-token, numeric-address and combined retrieval routes; report recall and candidate volume by source. |
| P0 | Full-entity candidate recall is 0.637651; many entities are missing at least one true link. | Person 2 | Analyse missed links and improve blocking before model tuning. |
| P0 | The current validation score is produced by the integrated exact scorer but has not been independently confirmed. | Person 4 | Rerun macro F0.5 and threshold search from the saved score file. |
| P1 | The model is a pooled S2/S3 baseline. | Person 1 | Train separate S1-to-S2 and S1-to-S3 models and compare against the pooled baseline. |
| P1 | The baseline uses a training-only positive guard to ensure all training positives are visible; it is excluded from validation. | Person 1 and Person 2 | Remove dependence on the guard by improving candidate recall; never use it during test inference. |
| P1 | Feature coverage is strong but limited to the Day 1 deterministic baseline. | Person 3 | Add hard negatives and stronger name/address features; preserve stable dtypes and benchmark speed. |
| P1 | Maximum observed RSS after a stage was about 1.82 GB during the 50,000-entity run. | Person 1 | Profile medium-scale execution and introduce chunked/disk-backed feature inference before the full test run. |
| P2 | France is absent from training by design. | All | Keep country handling open-set and run explicit France-path tests before full inference. |

No architecture change should bypass the shared schemas. Keep this Day 1 commit as the known-good fallback while Day 2 experiments are developed.
