# Person 1 - Day 1 foundation checklist

## Shared setup

- [x] Create repository structure.
- [x] Add safe `.gitignore` rules for datasets, outputs, models, caches, and large tabular files.
- [x] Define a repository-relative configuration with an environment-variable dataset override.
- [x] Define shared source, ground-truth, candidate, feature, score, and final-output schemas.
- [x] Create Unicode-safe normalization contract.
- [x] Create deterministic development-subset tooling.
- [x] Add foundational tests.

## Team handoff

- [x] Publish `docs/interfaces.md` for Persons 2, 3, and 4.
- [x] Freeze the Day 1 candidate representation: one unique pair row with deterministic aggregated route names.
- [x] Publish the first deterministic feature schema for Person 3 replacement work.
- [x] Produce a deterministic grouped S1 fold file and exact macro F0.5 tests compatible with Person 4's handoff.
- [x] Run the first end-to-end integration on the complete 50,000-entity subset.

## Evidence required before Day 1 ends

- [x] Development subset manifest and row counts.
- [x] Contract-compatible baseline candidate batch (temporary until Person 2 handoff).
- [x] Contract-compatible feature batch and speed benchmark (temporary until Person 3 handoff).
- [x] Exact baseline macro F0.5 and threshold report (pending independent Person 4 confirmation).
- [x] One reproducible command for the integrated subset pipeline.
- [x] Written Day 2 defect and owner list.
