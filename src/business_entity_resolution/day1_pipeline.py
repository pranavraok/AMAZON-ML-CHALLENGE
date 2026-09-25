"""One-command Day 1 development pipeline and evidence generation."""

from __future__ import annotations

from contextlib import contextmanager
import json
import logging
from pathlib import Path
import time
from typing import Iterator

import psutil

from business_entity_resolution.candidates import (
    add_training_positive_guards,
    candidate_recall,
    write_candidate_rows,
)
from business_entity_resolution.config import AppConfig
from business_entity_resolution.evaluation import (
    threshold_search,
    write_validation_outputs,
)
from business_entity_resolution.features import write_feature_rows
from business_entity_resolution.modeling import infer_scores, train_model
from business_entity_resolution.records import (
    read_ground_truth,
    read_source_records,
    stable_fold,
    write_tsv,
)
from business_entity_resolution.schemas import SOURCE_COLUMNS


LOGGER = logging.getLogger(__name__)


class RunMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process()
        self.stages: list[dict[str, float | str]] = []

    def rss_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        LOGGER.info("START %s | rss_mb=%.1f", name, self.rss_mb())
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            rss = self.rss_mb()
            self.stages.append(
                {"stage": name, "seconds": round(elapsed, 3), "rss_mb_after": round(rss, 1)}
            )
            LOGGER.info("DONE %s | seconds=%.3f | rss_mb=%.1f", name, elapsed, rss)


def _configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path.resolve()
        for handler in root.handlers
    ):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    if not any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in root.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)


def run_day1_pipeline(
    config: AppConfig,
    *,
    subset_dir: Path | None = None,
    work_dir: Path | None = None,
    artifact_dir: Path | None = None,
    max_source1: int | None = None,
) -> dict[str, object]:
    subset_root = (subset_dir or config.paths.development_dir).resolve()
    subset_train = subset_root / "train"
    work_root = (work_dir or (config.paths.work_dir / "day1")).resolve()
    artifact_root = (artifact_dir or (config.paths.artifact_dir / "day1")).resolve()
    log_path = config.repository_root / "logs" / "day1_pipeline.log"
    _configure_logging(log_path, config.project.log_level)
    work_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    monitor = RunMonitor()

    required = (
        subset_train / "train_source1.tsv",
        subset_train / "train_source2.tsv",
        subset_train / "train_source3.tsv",
        subset_train / "train_ground_truth.tsv",
        subset_root / "manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "development subset is incomplete; missing: " + ", ".join(missing)
        )

    with monitor.stage("load_and_normalize_records"):
        source1 = read_source_records(subset_train / "train_source1.tsv")
        if max_source1 is not None:
            selected_ids = set(sorted(source1)[:max_source1])
            source1 = {entity_id: source1[entity_id] for entity_id in sorted(selected_ids)}
        source2 = read_source_records(subset_train / "train_source2.tsv")
        source3 = read_source_records(subset_train / "train_source3.tsv")
        truth_all = read_ground_truth(subset_train / "train_ground_truth.tsv")
        truth = {entity_id: truth_all[entity_id] for entity_id in source1}
        secondary = {**source2, **source3}
        LOGGER.info(
            "row_counts | source1=%d source2=%d source3=%d truth=%d",
            len(source1),
            len(source2),
            len(source3),
            len(truth),
        )

    folds = {
        entity_id: stable_fold(
            entity_id,
            seed=config.project.seed,
            n_folds=config.day1_baseline.n_folds,
        )
        for entity_id in source1
    }
    validation_ids = {
        entity_id
        for entity_id, fold in folds.items()
        if fold == config.day1_baseline.validation_fold
    }
    if not validation_ids:
        raise ValueError("grouped split produced no validation Source 1 entities")
    write_tsv(
        work_root / "folds.tsv",
        ("source1_entity_id", "fold"),
        (
            {"source1_entity_id": entity_id, "fold": folds[entity_id]}
            for entity_id in sorted(folds)
        ),
    )

    raw_candidates = work_root / "candidate_pairs_raw.tsv"
    with monitor.stage("candidate_generation"):
        raw_candidate_rows = write_candidate_rows(
            raw_candidates,
            source1,
            source2,
            source3,
            max_candidates_per_source=(
                config.day1_baseline.max_candidates_per_source
            ),
            fallback_candidates_per_source=(
                config.day1_baseline.fallback_candidates_per_source
            ),
            max_token_bucket=config.day1_baseline.max_token_bucket,
            seed=config.project.seed,
        )
        recall = candidate_recall(
            raw_candidates, truth=truth, entity_ids=validation_ids
        )
        LOGGER.info("candidate_rows=%d candidate_recall=%s", raw_candidate_rows, recall)

    training_candidates = work_root / "candidate_pairs_training.tsv"
    with monitor.stage("training_positive_guard"):
        training_candidate_rows, guard_rows = add_training_positive_guards(
            raw_candidates,
            training_candidates,
            truth=truth,
            folds=folds,
            validation_fold=config.day1_baseline.validation_fold,
            secondary_ids=set(secondary),
        )
        LOGGER.info(
            "training_candidate_rows=%d positive_guard_rows=%d",
            training_candidate_rows,
            guard_rows,
        )

    feature_path = work_root / "features.tsv"
    with monitor.stage("feature_generation"):
        feature_rows = write_feature_rows(
            training_candidates,
            feature_path,
            source1_records=source1,
            secondary_records=secondary,
            truth=truth,
            folds=folds,
        )
        LOGGER.info("feature_rows=%d", feature_rows)

    model_path = artifact_root / "day1_lightgbm.joblib"
    training_metrics_path = artifact_root / "training_metrics.json"
    with monitor.stage("lightgbm_training"):
        training_metrics = train_model(
            feature_path,
            model_path,
            training_metrics_path,
            validation_fold=config.day1_baseline.validation_fold,
            seed=config.project.seed,
            num_boost_rounds=config.day1_baseline.num_boost_rounds,
        )
        LOGGER.info(
            "training_rows=%d positives=%d missing_features=%d",
            training_metrics["training_rows"],
            training_metrics["training_positives"],
            training_metrics["missing_feature_values"],
        )

    score_path = work_root / "validation_scores.tsv"
    with monitor.stage("validation_inference"):
        inference_metrics = infer_scores(
            feature_path,
            model_path,
            score_path,
            fold=config.day1_baseline.validation_fold,
        )
        LOGGER.info("scored_rows=%d", inference_metrics["scored_rows"])

    with monitor.stage("threshold_search_and_entity_scoring"):
        evaluation = threshold_search(
            score_path,
            truth=truth,
            entity_ids=validation_ids,
        )
        with (artifact_root / "threshold_metrics.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(evaluation, handle, indent=2, sort_keys=True)
            handle.write("\n")
        matching_rows, candidate_output_rows = write_validation_outputs(
            raw_candidates,
            score_path,
            work_root / "validation_matching_results.tsv",
            work_root / "validation_candidate_pairs.tsv",
            entity_ids=validation_ids,
            threshold=float(evaluation["best_threshold"]),
        )
        validation_test_dir = work_root / "validation_test"
        write_tsv(
            validation_test_dir / "test_source1.tsv",
            SOURCE_COLUMNS,
            (
                {
                    "entity_id": record.entity_id,
                    "business_name": record.business_name,
                    "business_address": record.business_address,
                    "country": record.country,
                }
                for entity_id, record in sorted(source1.items())
                if entity_id in validation_ids
            ),
        )
        LOGGER.info(
            "macro_f0_5=%.6f threshold=%.2f matching_rows=%d candidate_rows=%d",
            evaluation["macro_f0_5"],
            evaluation["best_threshold"],
            matching_rows,
            candidate_output_rows,
        )

    summary: dict[str, object] = {
        "status": "PASS",
        "subset_dir": str(subset_root),
        "work_dir": str(work_root),
        "artifact_dir": str(artifact_root),
        "log_file": str(log_path.resolve()),
        "source1_rows": len(source1),
        "source2_rows": len(source2),
        "source3_rows": len(source3),
        "validation_entities": len(validation_ids),
        "raw_candidate_rows": raw_candidate_rows,
        "training_candidate_rows": training_candidate_rows,
        "positive_guard_rows": guard_rows,
        "feature_rows": feature_rows,
        "candidate_recall": recall,
        "training": training_metrics,
        "inference": inference_metrics,
        "evaluation": evaluation,
        "official_validator_test_dir": str(
            (work_root / "validation_test").resolve()
        ),
        "stage_metrics": monitor.stages,
    }
    with (artifact_root / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("DAY1 PIPELINE PASS | summary=%s", artifact_root / "run_summary.json")
    return summary
