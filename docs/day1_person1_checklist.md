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

- [ ] Share `docs/interfaces.md` with Persons 2, 3, and 4.
- [ ] Confirm whether Person 2 will preserve one candidate row per retrieval route or aggregate routes.
- [ ] Confirm the first feature schema with Person 3.
- [ ] Receive the grouped S1 fold file and metric tests from Person 4.
- [ ] Run a first end-to-end subset integration.

## Evidence required before Day 1 ends

- [ ] Development subset manifest and row counts.
- [ ] Candidate sample from Person 2.
- [ ] Feature sample and speed benchmark from Person 3.
- [ ] Baseline macro F0.5 from Person 4.
- [ ] One reproducible command for the integrated subset pipeline.
- [ ] Written Day 2 defect and owner list.
