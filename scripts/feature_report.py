"""Feature-quality report: univariate AUC, permutation importance, indicative F0.5.

The gradient-boosting model here is only a probe of feature usefulness; the production
model belongs to Person 1 and the official metric/thresholds to Person 4.
Splits are grouped by Source 1 ID (never pair-level random).

Example:
    python scripts/feature_report.py --features data/work/p3/features.parquet \
        --ground-truth data/work/p3/dev_ground_truth.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from business_entity_resolution.similarity_features import (
    CONTEXT_FEATURE_COLUMNS, RETRIEVAL_FEATURE_COLUMNS, model_feature_columns,
)
from business_entity_resolution.pair_labels import explode_ground_truth


def f05(tp: int, n_pred: int, n_true: int) -> float:
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_true
    return 1.25 * p * r / (0.25 * p + r)


def macro_f05(scored: pl.DataFrame, truth_counts: pl.DataFrame, threshold: float) -> float:
    pred = (
        scored.filter(pl.col("score") >= threshold)
        .group_by("source1_entity_id")
        .agg(n_pred=pl.len(), tp=pl.col("label").sum())
    )
    joined = truth_counts.join(pred, on="source1_entity_id", how="left").fill_null(0)
    return float(np.mean([f05(tp, n, t) for tp, n, t in
                          joined.select("tp", "n_pred", "n_true").iter_rows()]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--drop", nargs="*", default=[], help="feature columns to ablate")
    args = parser.parse_args()

    df = pl.read_parquet(args.features)
    columns = [c for c in (*model_feature_columns(include_context=False), *CONTEXT_FEATURE_COLUMNS,
                           *RETRIEVAL_FEATURE_COLUMNS) if c in df.columns and c not in args.drop]
    y = df["label"].to_numpy()

    univariate = {}
    for column in columns:
        x = df[column].cast(pl.Float64).fill_nan(None).fill_null(-1.0).to_numpy()
        try:
            auc = roc_auc_score(y, x)
        except ValueError:
            auc = float("nan")
        univariate[column] = round(max(auc, 1 - auc), 4)

    fold = (df["source1_entity_id"].hash(seed=7) % 5).to_numpy()
    train, valid = fold != 0, fold == 0
    X = df.select(columns).cast(pl.Float32).fill_null(float("nan")).to_numpy()
    # sklearn's histogram GBM needs no system OpenMP library, unlike LightGBM on macOS.
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=50,
        l2_regularization=1.0, random_state=2026,
    )
    model.fit(X[train], y[train])
    proba = model.predict_proba(X[valid])[:, 1]

    # Permutation importance (drop in average precision) on a fixed validation sample.
    rng = np.random.default_rng(2026)
    valid_idx = np.flatnonzero(valid)
    sample = rng.choice(valid_idx, size=min(150_000, valid_idx.size), replace=False)
    Xs, ys = X[sample], y[sample]
    base_ap = average_precision_score(ys, model.predict_proba(Xs)[:, 1])
    importance = {}
    for j, column in enumerate(columns):
        shuffled = Xs.copy()
        shuffled[:, j] = rng.permutation(shuffled[:, j])
        importance[column] = round(base_ap - average_precision_score(
            ys, model.predict_proba(shuffled)[:, 1]), 5)
    importance = dict(sorted(importance.items(), key=lambda kv: -kv[1]))

    valid_ids = df.filter(pl.Series(valid))["source1_entity_id"].unique()
    gt = pl.read_parquet(args.ground_truth).filter(pl.col("source1_entity_id").is_in(valid_ids.implode()))
    truth_counts = gt.select("source1_entity_id").join(
        explode_ground_truth(gt).group_by("source1_entity_id").agg(n_true=pl.len()),
        on="source1_entity_id", how="left",
    ).fill_null(0)
    scored = df.filter(pl.Series(valid)).select("source1_entity_id", "label").with_columns(
        score=pl.Series(proba)
    )
    grid = {round(t, 2): round(macro_f05(scored, truth_counts, t), 4)
            for t in np.arange(0.3, 0.96, 0.05)}
    best_t = max(grid, key=grid.get)
    candidate_ceiling = macro_f05(
        scored.with_columns(score=pl.col("label").cast(pl.Float64)), truth_counts, 0.5
    )

    report = {
        "pairs": df.height,
        "valid_s1": gt.height,
        "valid_pair_auc": round(roc_auc_score(y[valid], proba), 5),
        "indicative_macro_f05_by_threshold": grid,
        "best_threshold": best_t,
        "best_macro_f05": grid[best_t],
        "oracle_macro_f05_given_candidates": round(candidate_ceiling, 4),
        "dropped": args.drop,
        "permutation_importance_ap_drop": importance,
        "univariate_auc": dict(sorted(univariate.items(), key=lambda kv: -kv[1])),
    }
    output = args.output or args.features.with_suffix(".quality.json")
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("permutation_importance_ap_drop", "univariate_auc")}, indent=2))
    print("top permutation importance:", list(importance.items())[:15])
    print(f"report -> {output}")


if __name__ == "__main__":
    main()
