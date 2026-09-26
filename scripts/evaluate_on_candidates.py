"""Evaluate feature sets on an arbitrary candidate file (e.g. Person 2's real candidates).

Trains Person 1's LightGBM (``modeling._build_classifier``, config rounds) on
the training folds and scores the validation fold with Person 1's entity-level
scorer. Validation entities come from ``folds.tsv``, so S1 records without any
candidate still count (as singletons or misses).

Inputs are parquet feature tables with source1_entity_id, candidate_entity_id,
target_source, label, fold and the feature columns.

Example:
    python scripts/evaluate_on_candidates.py \
        --set frozen=data/work/p3d3/p3_features_person2.parquet \
        --set baseline=data/work/p3d3/baseline_features_person2.parquet \
        --output data/work/p3d3/evaluation_person2.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.evaluation import threshold_search
from business_entity_resolution.features import MODEL_FEATURE_COLUMNS as BASELINE_COLUMNS
from business_entity_resolution.modeling import TARGET_SOURCES, _build_classifier
from business_entity_resolution.records import read_ground_truth
from business_entity_resolution.similarity_features import FROZEN_MODEL_FEATURES

COLUMNS = {"frozen": list(FROZEN_MODEL_FEATURES), "baseline": list(BASELINE_COLUMNS)}
THRESHOLDS = [v / 100 for v in range(5, 100)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", action="append", required=True, help="name=path (name in frozen|baseline)")
    parser.add_argument("--architectures", nargs="+", default=["pooled", "separate"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scores-dir", type=Path, default=Path("data/work/p3d3/scores"))
    args = parser.parse_args()
    config = load_config("configs/base.json")
    vfold = config.day1_baseline.validation_fold
    truth = read_ground_truth(config.paths.development_dir / "train" / "train_ground_truth.tsv")
    folds = pl.read_csv(config.paths.work_dir / "day1" / "folds.tsv", separator="\t", infer_schema=False)
    valid_ids = set(folds.filter(pl.col("fold") == str(vfold))["source1_entity_id"])
    args.scores_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for spec in args.set:
        name, path = spec.split("=", 1)
        columns = COLUMNS[name]
        frame = pl.read_parquet(path, columns=["source1_entity_id", "candidate_entity_id",
                                               "target_source", "label", "fold", *columns])
        train = frame.filter(pl.col("fold") != vfold)
        valid = frame.filter(pl.col("fold") == vfold)
        for architecture in args.architectures:
            t0 = time.perf_counter()
            proba = np.empty(valid.height)
            groups = [(None, 0)] if architecture == "pooled" else [(s, i) for i, s in enumerate(TARGET_SOURCES)]
            for source, offset in groups:
                tr = train if source is None else train.filter(pl.col("target_source") == source)
                mask = (np.ones(valid.height, bool) if source is None
                        else (valid["target_source"] == source).to_numpy())
                model = _build_classifier(seed=config.project.seed + offset,
                                          num_boost_rounds=config.day2.num_boost_rounds)
                model.fit(tr.select(columns).to_numpy().astype(np.float32), tr["label"].to_numpy())
                proba[mask] = model.booster_.predict(
                    valid.filter(pl.Series(mask)).select(columns).to_numpy().astype(np.float32))
            path_scores = args.scores_dir / f"{name}_{architecture}.tsv"
            valid.select("source1_entity_id", "candidate_entity_id", "target_source").with_columns(
                match_probability=pl.Series(proba)).write_csv(path_scores, separator="\t")
            ev = threshold_search(path_scores, truth=truth, entity_ids=valid_ids, thresholds=THRESHOLDS)
            ev.pop("threshold_trials")
            ev.update(n_features=len(columns), training_rows=train.height,
                      validation_rows=valid.height, seconds=round(time.perf_counter() - t0, 1))
            results[f"{name}/{architecture}"] = ev
            print(f"{name}/{architecture}: macro_f0_5={ev['macro_f0_5']:.6f} thr={ev['best_threshold']:.2f} "
                  f"P={ev['micro_precision']:.4f} R={ev['micro_recall']:.4f} FP={ev['false_positives']} "
                  f"FN={ev['false_negatives']} singleton={ev['singleton_accuracy']:.4f}", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
