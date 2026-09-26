"""Like-for-like feature ablation on Person 1's identical rows, folds and models.

Only the feature columns change between runs. Rows, labels, grouped folds,
LightGBM hyperparameters (``modeling._build_classifier``), architectures and
the entity-level scorer (``evaluation.threshold_search``) are Person 1's.

Example:
    python scripts/ablate_features.py --config configs/base.json --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import numpy as np
import polars as pl
import psutil

from business_entity_resolution.config import load_config
from business_entity_resolution.evaluation import threshold_search
from business_entity_resolution.features import MODEL_FEATURE_COLUMNS as BASELINE_COLUMNS
from business_entity_resolution.modeling import TARGET_SOURCES, _build_classifier
from business_entity_resolution.records import read_ground_truth
from business_entity_resolution.similarity_features import (
    FEATURE_VERSION, FROZEN_MODEL_FEATURES, add_context_features,
    calculate_pair_features_parallel, model_feature_columns,
)

LOGGER = logging.getLogger("ablation")
KEYS = ["source1_entity_id", "candidate_entity_id"]
THRESHOLDS = [value / 100 for value in range(5, 100)]
V2_NEW = {
    "name_a_unmatched_max_noise", "name_b_unmatched_max_noise", "addr_primary_equal",
    "addr_primary_cross", "addr_num_only_a", "addr_num_only_b", "name_core_freq_s1_a",
    "name_core_freq_sec_a", "name_core_freq_s1_b", "name_core_freq_sec_b",
}


def build_combined_features(
    baseline_path: Path, subset_train: Path, token_stats_path: Path, name_stats_dir: Path,
    processes: int, chunk_pairs: int,
) -> tuple[pl.DataFrame, dict[str, float]]:
    baseline = pl.read_csv(baseline_path, separator="\t", infer_schema_length=10_000,
                           null_values=["", "null", "NaN"])
    read = dict(separator="\t", quote_char=None, infer_schema=False)
    s1_raw = pl.read_csv(subset_train / "train_source1.tsv", **read)
    sec_raw = pl.concat([pl.read_csv(subset_train / f"train_source{i}.tsv", **read)
                         for i in (2, 3)])
    pairs = baseline.select(*KEYS, "target_source")
    timings: dict[str, float] = {"pairs": pairs.height, "processes": processes}
    t0 = time.perf_counter()
    p3 = calculate_pair_features_parallel(
        pairs, s1_raw, sec_raw, token_stats_path, name_stats_dir,
        processes=processes, chunk_pairs=chunk_pairs)
    timings["pair_features_incl_prepare_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    p3 = add_context_features(p3)
    timings["context_s"] = time.perf_counter() - t0
    timings["pairs_per_second_end_to_end"] = pairs.height / (
        timings["pair_features_incl_prepare_s"] + timings["context_s"])
    baseline = baseline.rename({c: f"base__{c}" for c in BASELINE_COLUMNS})
    combined = pl.concat([baseline, p3.drop(*KEYS, "target_source")], how="horizontal")
    return combined, timings


def train_and_score(
    frame: pl.DataFrame, columns: list[str], architecture: str, *, seed: int,
    rounds: int, validation_fold: int, score_path: Path,
) -> dict[str, object]:
    train = frame.filter(pl.col("fold") != validation_fold)
    valid = frame.filter(pl.col("fold") == validation_fold)
    probabilities = np.empty(valid.height, dtype=np.float64)
    importance: dict[str, dict[str, float]] = {}
    groups = [("pooled", None)] if architecture == "pooled" else [
        (source, source) for source in TARGET_SOURCES]
    for offset, (name, source) in enumerate(groups):
        tr = train if source is None else train.filter(pl.col("target_source") == source)
        mask = (np.ones(valid.height, bool) if source is None
                else (valid["target_source"] == source).to_numpy())
        model = _build_classifier(seed=seed + offset, num_boost_rounds=rounds)
        model.fit(tr.select(columns).to_numpy().astype(np.float32),
                  tr["label"].to_numpy().astype(np.int8))
        probabilities[mask] = model.booster_.predict(
            valid.filter(pl.Series(mask)).select(columns).to_numpy().astype(np.float32))
        gains = model.booster_.feature_importance("gain")
        importance[name] = dict(sorted(zip(columns, (gains / gains.sum()).round(5).tolist()),
                                       key=lambda kv: -kv[1]))
    scores = valid.select(*KEYS, "target_source").with_columns(
        match_probability=pl.Series(probabilities))
    score_path.parent.mkdir(parents=True, exist_ok=True)
    scores.write_csv(score_path, separator="\t")
    return {"importance": importance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--token-stats", type=Path, default=Path("artifacts/token_stats.json"))
    parser.add_argument("--name-stats", type=Path, default=Path("artifacts/name_stats"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/p3d2"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-pairs", type=int, default=50_000)
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--architectures", nargs="+", default=["separate", "pooled"])
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--sets", nargs="+", default=["baseline", "frozen", "baseline_plus_frozen"])
    parser.add_argument("--drop", nargs="*", default=[],
                        help="extra ablation: p3 set minus these columns (named p3_minus)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    config = load_config(args.config)
    rounds = args.rounds or config.day2.num_boost_rounds
    subset_train = config.paths.development_dir / "train"
    day1 = config.paths.work_dir / "day1"
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    combined_path = out / "combined_features.parquet"

    if args.reuse_features and combined_path.exists():
        combined = pl.read_parquet(combined_path)
        timings = json.loads((out / "feature_timings.json").read_text())
    else:
        combined, timings = build_combined_features(
            day1 / "features.tsv", subset_train, args.token_stats, args.name_stats,
            args.workers, args.chunk_pairs)
        timings["main_process_rss_mb"] = psutil.Process().memory_info().rss / 2**20
        combined.write_parquet(combined_path)
        (out / "feature_timings.json").write_text(json.dumps(timings, indent=2))
    LOGGER.info("features ready: %s", timings)

    p3_columns = model_feature_columns(include_context=True)
    feature_sets = {
        "baseline": [f"base__{c}" for c in BASELINE_COLUMNS],
        "p3": p3_columns,
        "union": [f"base__{c}" for c in BASELINE_COLUMNS] + p3_columns,
        "p3_no_context": model_feature_columns(include_context=False),
        "p3_v1": [c for c in p3_columns if c not in V2_NEW],
        "frozen": list(FROZEN_MODEL_FEATURES),
        "baseline_plus_frozen": [f"base__{c}" for c in BASELINE_COLUMNS] + list(FROZEN_MODEL_FEATURES),
    }
    if args.drop:
        feature_sets["p3_minus"] = [c for c in p3_columns if c not in set(args.drop)]
        args.sets = list(dict.fromkeys([*args.sets, "p3_minus"]))

    truth = read_ground_truth(subset_train / "train_ground_truth.tsv")
    validation_ids = set(
        combined.filter(pl.col("fold") == config.day1_baseline.validation_fold)
        ["source1_entity_id"].unique().to_list())
    # Entities whose candidate list is empty still count (as singletons or misses).
    folds = pl.read_csv(day1 / "folds.tsv", separator="\t", infer_schema=False)
    validation_ids |= set(folds.filter(
        pl.col("fold") == str(config.day1_baseline.validation_fold))["source1_entity_id"])

    results: dict[str, dict[str, object]] = {}
    for set_name in args.sets:
        columns = feature_sets[set_name]
        for architecture in args.architectures:
            name = f"{set_name}/{architecture}"
            t0 = time.perf_counter()
            score_path = out / "scores" / f"{set_name}_{architecture}.tsv"
            trained = train_and_score(
                combined, columns, architecture, seed=config.project.seed, rounds=rounds,
                validation_fold=config.day1_baseline.validation_fold, score_path=score_path)
            evaluation = threshold_search(score_path, truth=truth, entity_ids=validation_ids,
                                          thresholds=THRESHOLDS)
            evaluation.pop("threshold_trials")
            results[name] = {**evaluation, "n_features": len(columns),
                             "train_seconds": round(time.perf_counter() - t0, 1),
                             "importance": trained["importance"]}
            LOGGER.info("%s macro_f0_5=%.6f thr=%.2f FP=%d FN=%d", name,
                        evaluation["macro_f0_5"], evaluation["best_threshold"],
                        evaluation["false_positives"], evaluation["false_negatives"])

    report = {"feature_version": FEATURE_VERSION, "rounds": rounds,
              "validation_entities": len(validation_ids), "feature_timings": timings,
              "results": results}
    (out / "ablation.json").write_text(json.dumps(report, indent=2))
    header = ("| Feature set | Arch | #feat | Thr | Macro F0.5 | Precision | Recall | FP | FN "
              "| Singleton acc |\n|---|---|---|---|---|---|---|---|---|---|")
    rows = [
        f"| {k.split('/')[0]} | {k.split('/')[1]} | {v['n_features']} | {v['best_threshold']:.2f} "
        f"| {v['macro_f0_5']:.6f} | {v['micro_precision']:.4f} | {v['micro_recall']:.4f} "
        f"| {v['false_positives']} | {v['false_negatives']} | {v['singleton_accuracy']:.4f} |"
        for k, v in results.items()
    ]
    table = "\n".join([header, *rows])
    (out / "ablation.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
