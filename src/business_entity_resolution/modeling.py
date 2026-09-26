"""LightGBM training and restartable inference for the ER pipeline."""

from __future__ import annotations

import json
import csv
from pathlib import Path
import time
from typing import Any

import joblib
from lightgbm import LGBMClassifier
import numpy as np
import polars as pl

from business_entity_resolution.features import MODEL_FEATURE_COLUMNS
from business_entity_resolution.schemas import MODEL_SCORE_COLUMNS


TARGET_SOURCES = ("S2", "S3")


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
        "kind": "pooled",
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


def _build_classifier(*, seed: int, num_boost_rounds: int) -> LGBMClassifier:
    return LGBMClassifier(
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


def train_separate_models(
    feature_path: Path,
    model_path: Path,
    metrics_path: Path,
    *,
    validation_fold: int,
    seed: int,
    num_boost_rounds: int,
) -> dict[str, Any]:
    """Train one classifier for Source 2 and another for Source 3."""

    frame = _read_features(feature_path)
    models: dict[str, LGBMClassifier] = {}
    source_metrics: dict[str, dict[str, Any]] = {}
    for source_offset, target_source in enumerate(TARGET_SOURCES):
        source_frame = frame.filter(pl.col("target_source") == target_source)
        train = source_frame.filter(pl.col("fold") != validation_fold)
        validation = source_frame.filter(pl.col("fold") == validation_fold)
        if train.height == 0:
            raise ValueError(f"no training rows for target source {target_source}")
        x_train = train.select(MODEL_FEATURE_COLUMNS).to_numpy().astype(np.float32)
        y_train = train.get_column("label").to_numpy().astype(np.int8)
        positives = int(y_train.sum())
        negatives = int(len(y_train) - positives)
        if positives == 0 or negatives == 0:
            raise ValueError(
                f"{target_source} needs both classes; "
                f"positives={positives}, negatives={negatives}"
            )
        model = _build_classifier(
            seed=seed + source_offset,
            num_boost_rounds=num_boost_rounds,
        )
        model.fit(x_train, y_train)
        models[target_source] = model
        importances = sorted(
            zip(MODEL_FEATURE_COLUMNS, model.feature_importances_, strict=True),
            key=lambda item: (-int(item[1]), item[0]),
        )
        source_metrics[target_source] = {
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

    artifact = {
        "kind": "separate",
        "models": models,
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "validation_fold": validation_fold,
        "seed": seed,
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path, compress=3)
    metrics: dict[str, Any] = {
        "architecture": "separate",
        "feature_file": str(feature_path.resolve()),
        "model_file": str(model_path.resolve()),
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "sources": source_metrics,
        "training_rows": sum(
            int(source["training_rows"]) for source in source_metrics.values()
        ),
        "validation_candidate_rows": sum(
            int(source["validation_candidate_rows"])
            for source in source_metrics.values()
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def _load_artifact(model_path: Path) -> dict[str, Any]:
    artifact = joblib.load(model_path)
    feature_names = tuple(artifact["feature_names"])
    if feature_names != MODEL_FEATURE_COLUMNS:
        raise ValueError(
            "model feature contract differs from this code version; retrain the model"
        )
    return artifact


def _predict_probabilities(
    artifact: dict[str, Any],
    x: np.ndarray,
    target_sources: np.ndarray,
) -> np.ndarray:
    kind = artifact.get("kind", "pooled")
    if kind == "pooled":
        return np.asarray(artifact["model"].booster_.predict(x), dtype=np.float64)
    if kind != "separate":
        raise ValueError(f"unsupported model architecture: {kind}")

    probabilities = np.empty(len(x), dtype=np.float64)
    assigned = np.zeros(len(x), dtype=bool)
    for target_source in TARGET_SOURCES:
        mask = target_sources == target_source
        if not np.any(mask):
            continue
        probabilities[mask] = artifact["models"][target_source].booster_.predict(
            x[mask]
        )
        assigned |= mask
    if not np.all(assigned):
        unknown = sorted(set(target_sources[~assigned].tolist()))
        raise ValueError(f"separate model received unknown target sources: {unknown}")
    return probabilities


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
    artifact = _load_artifact(model_path)
    feature_names = tuple(artifact["feature_names"])
    x = frame.select(feature_names).to_numpy().astype(np.float32)
    probabilities = _predict_probabilities(
        artifact,
        x,
        frame.get_column("target_source").to_numpy(),
    )
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


def infer_scores_chunked(
    feature_path: Path,
    model_path: Path,
    score_path: Path,
    checkpoint_path: Path,
    *,
    chunk_size: int,
    fold: int | None = None,
) -> dict[str, int | float | str | bool]:
    """Score a TSV in bounded chunks and resume from the last completed chunk."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    artifact = _load_artifact(model_path)
    part_path = score_path.with_suffix(score_path.suffix + ".part")
    processed_rows = 0
    resumed = False
    if checkpoint_path.is_file() and part_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("feature_file") == str(feature_path.resolve())
            and checkpoint.get("model_file") == str(model_path.resolve())
            and checkpoint.get("fold") == fold
            and checkpoint.get("chunk_size") == chunk_size
        ):
            processed_rows = int(checkpoint.get("processed_rows", 0))
            resumed = processed_rows > 0
        else:
            part_path.unlink()
    elif part_path.exists():
        part_path.unlink()

    score_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if processed_rows else "w"
    started = time.perf_counter()
    selected_seen = 0
    rows_written = processed_rows
    chunks_written = 0
    batch: list[dict[str, str]] = []

    def write_batch(
        writer: csv.DictWriter,
        feature_rows: list[dict[str, str]],
    ) -> int:
        if not feature_rows:
            return 0
        x = np.asarray(
            [
                [float(row[column]) for column in MODEL_FEATURE_COLUMNS]
                for row in feature_rows
            ],
            dtype=np.float32,
        )
        target_sources = np.asarray(
            [row["target_source"] for row in feature_rows], dtype=object
        )
        probabilities = _predict_probabilities(artifact, x, target_sources)
        for row, probability in zip(feature_rows, probabilities, strict=True):
            writer.writerow(
                {
                    "source1_entity_id": row["source1_entity_id"],
                    "candidate_entity_id": row["candidate_entity_id"],
                    "target_source": row["target_source"],
                    "match_probability": format(float(probability), ".17g"),
                }
            )
        return len(feature_rows)

    with feature_path.open("r", encoding="utf-8", newline="") as source_handle:
        reader = csv.DictReader(source_handle, delimiter="\t")
        required = {
            "source1_entity_id",
            "candidate_entity_id",
            "target_source",
            "fold",
            *MODEL_FEATURE_COLUMNS,
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{feature_path} is missing feature columns: {missing}")
        with part_path.open(write_mode, encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=list(MODEL_SCORE_COLUMNS),
                delimiter="\t",
                lineterminator="\n",
            )
            if not processed_rows:
                writer.writeheader()
            for row in reader:
                if fold is not None and int(row["fold"]) != fold:
                    continue
                if selected_seen < processed_rows:
                    selected_seen += 1
                    continue
                selected_seen += 1
                batch.append(row)
                if len(batch) < chunk_size:
                    continue
                rows_written += write_batch(writer, batch)
                chunks_written += 1
                output_handle.flush()
                checkpoint = {
                    "feature_file": str(feature_path.resolve()),
                    "model_file": str(model_path.resolve()),
                    "fold": fold,
                    "chunk_size": chunk_size,
                    "processed_rows": rows_written,
                }
                checkpoint_path.write_text(
                    json.dumps(checkpoint, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                batch.clear()
            if batch:
                rows_written += write_batch(writer, batch)
                chunks_written += 1
                output_handle.flush()
                batch.clear()

    part_path.replace(score_path)
    checkpoint_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    return {
        "feature_file": str(feature_path.resolve()),
        "model_file": str(model_path.resolve()),
        "score_file": str(score_path.resolve()),
        "scored_rows": rows_written,
        "chunks_written": chunks_written,
        "chunk_size": chunk_size,
        "resumed": resumed,
        "seconds": elapsed,
        "rows_per_second": rows_written / max(elapsed, 1e-9),
    }
