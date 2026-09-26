"""Mine model-based hard negatives with out-of-fold scores and test their weighting.

1. Cross-fit a pooled LightGBM (Person 1 hyperparameters) over the training
   folds only (validation fold untouched): each training pair is scored by a
   model that never saw its S1 group.
2. Hard negatives = label-0 retrieved pairs with OOF probability >= --min-prob,
   top --per-s1 per S1, tagged with their family.
3. Weighting experiment: retrain on all training folds with hard negatives
   weighted w and score the validation fold with Person 1's entity-level scorer.

Example:
    python scripts/mine_hard_negatives.py --weights 1 2 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.evaluation import threshold_search
from business_entity_resolution.modeling import _build_classifier
from business_entity_resolution.pair_labels import classify_hard_negatives
from business_entity_resolution.records import read_ground_truth
from business_entity_resolution.similarity_features import FROZEN_MODEL_FEATURES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-prob", type=float, default=0.1)
    parser.add_argument("--per-s1", type=int, default=5)
    parser.add_argument("--weights", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    parser.add_argument("--features", type=Path, default=Path("data/work/p3d2/combined_features.parquet"),
                        help="feature table with label + fold (e.g. build_p3_feature_file.py output)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/work/p3d2"))
    args = parser.parse_args()
    OUT = args.output_dir
    (OUT / "scores").mkdir(parents=True, exist_ok=True)
    config = load_config("configs/base.json")
    seed, rounds = config.project.seed, config.day2.num_boost_rounds
    vfold = config.day1_baseline.validation_fold
    columns = list(FROZEN_MODEL_FEATURES)
    frame = pl.read_parquet(args.features)
    train = frame.filter(pl.col("fold") != vfold)
    valid = frame.filter(pl.col("fold") == vfold)
    validation_s1 = set(valid["source1_entity_id"].unique().to_list())
    # Guard: the validation fold must be a disjoint set of S1 groups.
    assert not validation_s1 & set(train["source1_entity_id"].unique().to_list())

    oof = np.empty(train.height)
    fold_values = train["fold"].to_numpy()
    X = train.select(columns).to_numpy().astype(np.float32)
    y = train["label"].to_numpy()
    for f in sorted(set(fold_values.tolist())):
        model = _build_classifier(seed=seed, num_boost_rounds=rounds)
        model.fit(X[fold_values != f], y[fold_values != f])
        oof[fold_values == f] = model.booster_.predict(X[fold_values == f])
        print(f"cross-fit fold {f} done", flush=True)
    train = train.with_columns(oof_probability=pl.Series(oof))

    negatives = classify_hard_negatives(train)
    hard = (
        negatives.filter(pl.col("oof_probability") >= args.min_prob)
        .with_columns(_r=pl.col("oof_probability").rank("ordinal", descending=True)
                      .over("source1_entity_id"))
        .filter(pl.col("_r") <= args.per_s1).drop("_r")
    )
    keep = ["source1_entity_id", "candidate_entity_id", "target_source", "fold",
            "oof_probability", "hard_negative_type"]
    # Proof that mining used training folds only.
    leaked = set(hard["source1_entity_id"].unique().to_list()) & validation_s1
    if leaked or (hard["fold"] == vfold).any():
        raise AssertionError(f"hard negatives contain validation-fold S1 ids: {sorted(leaked)[:5]}")
    fold_check = {
        "validation_fold": vfold,
        "folds_used_for_cross_fitting": sorted(set(fold_values.tolist())),
        "validation_fold_rows_used_for_mining": 0,
        "validation_s1_in_hard_negatives": 0,
        "oof_scores_from_models_that_never_saw_the_s1_group": True,
        "labels_used_only_to_select_label_0_rows": True,
    }
    hard.select(keep).write_parquet(OUT / "hard_negatives_model.parquet")
    train.select("source1_entity_id", "candidate_entity_id", "label", "fold", "oof_probability") \
        .write_parquet(OUT / "oof_scores.parquet")
    positives_missed = train.filter((pl.col("label") == 1) & (pl.col("oof_probability") < 0.5)).height
    summary = {
        "features_file": str(args.features),
        "training_folds_only_check": fold_check,
        "training_pairs": train.height,
        "hard_negatives": hard.height,
        "s1_with_hard_negatives": hard["source1_entity_id"].n_unique(),
        "min_prob": args.min_prob,
        "by_type": dict(hard.group_by("hard_negative_type").len().sort("len", descending=True).iter_rows()),
        "oof_negatives_ge_0.5": negatives.filter(pl.col("oof_probability") >= 0.5).height,
        "oof_positives_lt_0.5": positives_missed,
    }
    print(json.dumps(summary, indent=2), flush=True)

    hard_keys = hard.select("source1_entity_id", "candidate_entity_id").with_columns(_hard=pl.lit(True))
    train_w = train.join(hard_keys, on=["source1_entity_id", "candidate_entity_id"], how="left",
                         maintain_order="left").with_columns(pl.col("_hard").fill_null(False))
    truth = read_ground_truth(config.paths.development_dir / "train" / "train_ground_truth.tsv")
    folds = pl.read_csv(config.paths.work_dir / "day1" / "folds.tsv", separator="\t", infer_schema=False)
    valid_ids = set(folds.filter(pl.col("fold") == str(vfold))["source1_entity_id"])
    results = {}
    Xv = valid.select(columns).to_numpy().astype(np.float32)
    for weight in args.weights:
        w = np.where(train_w["_hard"].to_numpy(), weight, 1.0)
        model = _build_classifier(seed=seed, num_boost_rounds=rounds)
        model.fit(X, y, sample_weight=w)
        path = OUT / "scores" / f"hardneg_w{weight:g}_pooled.tsv"
        valid.select("source1_entity_id", "candidate_entity_id", "target_source").with_columns(
            match_probability=pl.Series(model.booster_.predict(Xv))).write_csv(path, separator="\t")
        ev = threshold_search(path, truth=truth, entity_ids=valid_ids,
                              thresholds=[v / 100 for v in range(5, 100)])
        results[f"w={weight:g}"] = {k: ev[k] for k in ("best_threshold", "macro_f0_5", "micro_precision",
                                                       "micro_recall", "false_positives",
                                                       "false_negatives", "singleton_accuracy")}
        print(weight, results[f"w={weight:g}"], flush=True)
    summary["weighting_experiment_pooled"] = results
    (OUT / "hard_negatives_model.summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
