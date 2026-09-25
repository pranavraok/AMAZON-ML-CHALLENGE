"""LightGBM training and inference for the Day 1 integration baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import polars as pl

from business_entity_resolution.features import MODEL_FEATURE_COLUMNS
from business_entity_resolution.schemas import MODEL_SCORE_COLUMNS


def _read_features(path: Path) -> pl.DataFrame:
    frame = pl.read_csv(
        path,
        separator="\t",
        infer_schema_length=10_000,
        null_values=["", "null", "NaN"],
    )
    required = {
        "source1_entity_id",
        "candidate_entity_id",
        "target_source",
        "label",
        "fold",
        *MODEL_FEATURE_COLUMNS,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing feature columns: {missing}")
    return frame


def train_model(
    feature_path: Path,
    model_path: Path,
    metrics_path: Path,
    *,
    validation_fold: int,
    seed: int,
    num_boost_rounds: int,
) -> dict[str, Any]:
    frame = _read_features(feature_path)
    train = frame.filter(pl.col("fold") != validation_fold)
    validation = frame.filter(pl.col("fold") == validation_fold)
    if train.height == 0:
        raise ValueError("training feature file has no non-validation rows")

    x_train = train.select(MODEL_FEATURE_COLUMNS).to_numpy().astype(np.float32)
    y_train = train.get_column("label").to_numpy().astype(np.int8)
    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError(
            f"training data needs both classes; positives={positives}, negatives={negatives}"
        )

    model = LGBMClassifier(
        objective="binary",
        n_estimators=num_boost_rounds,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=40,
        subsample=0.85,
        colsample_bytree=0.90,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "validation_fold": validation_fold,
        "seed": seed,
    }
    joblib.dump(artifact, model_path, compress=3)

    importances = sorted(
        zip(MODEL_FEATURE_COLUMNS, model.feature_importances_, strict=True),
        key=lambda item: (-int(item[1]), item[0]),
    )
    metrics: dict[str, Any] = {
        "feature_file": str(feature_path.resolve()),
        "model_file": str(model_path.resolve()),
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "training_rows": int(train.height),
        "validation_candidate_rows": int(validation.height),
        "training_positives": positives,
        "training_negatives": negatives,
        "missing_feature_values": int(np.isnan(x_train).sum()),
        "feature_importance": [
            {"feature": name, "split_importance": int(value)}
            for name, value in importances
        ],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def infer_scores(
    feature_path: Path,
    model_path: Path,
    score_path: Path,
    *,
    fold: int | None = None,
) -> dict[str, int | str]:
    frame = _read_features(feature_path)
    if fold is not None:
        frame = frame.filter(pl.col("fold") == fold)
    artifact = joblib.load(model_path)
    feature_names = tuple(artifact["feature_names"])
    if feature_names != MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "model feature contract differs from this code version; retrain the model"
        )
    x = frame.select(feature_names).to_numpy().astype(np.float32)
    # Use the native booster for inference so the saved feature-order contract is
    # authoritative and sklearn does not emit synthetic feature-name warnings.
    probabilities = artifact["model"].booster_.predict(x)
    scores = frame.select(
        "source1_entity_id", "candidate_entity_id", "target_source"
    ).with_columns(pl.Series("match_probability", probabilities))
    scores = scores.select(MODEL_SCORE_COLUMNS)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    scores.write_csv(score_path, separator="\t", include_header=True)
    return {
        "feature_file": str(feature_path.resolve()),
        "model_file": str(model_path.resolve()),
        "score_file": str(score_path.resolve()),
        "scored_rows": int(scores.height),
    }
