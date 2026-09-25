# Amazon ML Challenge 2026 - Business Entity Resolution

Team repository for matching each Source 1 business to zero, one, or many records from Source 2 and Source 3.

The repository contains a complete Person 1 Day 1 development pipeline: shared configuration, schemas, text normalization, deterministic subset creation, baseline candidate generation, comparison features, grouped validation, LightGBM training, inference, exact macro F0.5 evaluation, logging, and teammate interface contracts. The baseline candidate and feature modules are intentionally replaceable by the stronger Person 2 and Person 3 implementations.

## Challenge constraints built into the project

- Source files and outputs are tab-separated.
- Every test Source 1 ID must appear exactly once in both output files.
- Final matches may contain only valid Source 2 or Source 3 IDs.
- `matching_results.tsv` contains the accepted matches.
- `candidate_pairs.tsv` contains the exact final candidate set scored by the model.
- The leaderboard metric is macro F0.5 per Source 1 entity, including singletons.
- Training contains US and India; test additionally contains France. Country values are treated as open-set strings.
- External business lookup, geocoding, registries, APIs, and web enrichment are prohibited.

## Repository layout

```text
.
|-- configs/
|   `-- base.json
|-- docs/
|   |-- day1_person1_checklist.md
|   `-- interfaces.md
|-- output/
|   `-- .gitkeep
|-- scripts/
|   |-- create_dev_subset.py
|   |-- infer.py
|   |-- run_day1_pipeline.py
|   `-- train.py
|-- src/business_entity_resolution/
|   |-- __init__.py
|   |-- __main__.py
|   |-- candidates.py
|   |-- cli.py
|   |-- config.py
|   |-- day1_pipeline.py
|   |-- evaluation.py
|   |-- features.py
|   |-- modeling.py
|   |-- normalization.py
|   |-- records.py
|   |-- schemas.py
|   `-- subset.py
|-- tests/
|   |-- test_config.py
|   |-- test_normalization.py
|   `-- test_subset.py
|-- pyproject.toml
`-- requirements.txt
```

Datasets, generated subsets, Parquet files, models, logs, and submission outputs are intentionally excluded from Git.

## Local setup

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Activate the environment, then install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The foundational subset and normalization tools use only the Python standard library, so their unit tests can run before the heavier ML dependencies are installed.

## Point the project to the dataset

The default local path is configured in `configs/base.json`. Override it without editing a tracked file:

PowerShell:

```powershell
$env:AMAZON_ML_DATASET_ROOT = "C:\path\to\student_resource\dataset"
```

Bash:

```bash
export AMAZON_ML_DATASET_ROOT=/path/to/student_resource/dataset
```

Confirm the resolved configuration and required input files:

```bash
python -m business_entity_resolution show-config --config configs/base.json
python -m business_entity_resolution validate-inputs --config configs/base.json
```

## Create the shared development subset

The command selects Source 1 records deterministically, copies all of their true S2/S3 matches, and adds reproducible same-country distractor records. It preserves the original TSV schemas.

```bash
python scripts/create_dev_subset.py \
  --config configs/base.json \
  --source1-count 50000 \
  --negative-multiplier 2.0
```

For a quick smoke test:

```bash
python scripts/create_dev_subset.py \
  --config configs/base.json \
  --source1-count 1000 \
  --negative-multiplier 1.0 \
  --output-dir data/dev-smoke
```

The generated directory contains:

```text
data/dev/
|-- train/
|   |-- train_source1.tsv
|   |-- train_source2.tsv
|   |-- train_source3.tsv
|   `-- train_ground_truth.tsv
`-- manifest.json
```

## Run tests

Without installing the package:

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

Bash:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Run the complete Day 1 baseline

Create the required 50,000-entity development subset:

```bash
python scripts/create_dev_subset.py --config configs/base.json --source1-count 50000 --negative-multiplier 2.0
```

Run candidates -> features -> grouped LightGBM -> inference -> threshold search -> macro F0.5:

```bash
python scripts/run_day1_pipeline.py --config configs/base.json
```

The command writes generated data under `data/work/day1/`, model and metric artifacts under `artifacts/day1/`, and the run log to `logs/day1_pipeline.log`. These large local artifacts are intentionally ignored by Git.

The standalone model entry points accept contract-compatible feature files:

```bash
python scripts/train.py --features data/work/day1/features.tsv --model artifacts/day1/day1_lightgbm.joblib --metrics artifacts/day1/training_metrics.json --validation-fold 0
python scripts/infer.py --features data/work/day1/features.tsv --model artifacts/day1/day1_lightgbm.joblib --scores data/work/day1/validation_scores.tsv --fold 0
```

See `docs/day1_handoff.md` for the measured run and `docs/day1_integration_issues.md` for Day 2 owners.

## Team integration

All teammate modules must follow [docs/interfaces.md](docs/interfaces.md). The interfaces define the candidate-pair columns, feature identifiers, output headers, ownership boundaries, and handoff rules. The Day 1 baseline aggregates duplicate pair routes deterministically into one candidate row with `+`-joined route names; Person 2 may replace this only through a documented schema decision.

Do not rename shared columns or change normalization behavior without coordinating with all module owners.

## Final submission target

The packaging stage will produce:

```text
<team_name>_submission.zip
|-- output/
|   |-- matching_results.tsv
|   `-- candidate_pairs.tsv
|-- code/business_entity_resolution/
|   |-- src/
|   |-- README.md
|   `-- requirements.txt
`-- Documentation_template.md
```

Before submission, run the official validator supplied with the challenge package.
