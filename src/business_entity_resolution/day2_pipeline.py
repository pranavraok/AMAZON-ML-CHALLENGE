"""Person 1 Day 2 model comparison, profiling and freeze evidence."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

from business_entity_resolution.config import AppConfig
from business_entity_resolution.day1_pipeline import RunMonitor, _configure_logging
from business_entity_resolution.error_analysis import (
    country_robustness_metrics,
    write_error_slices,
)
from business_entity_resolution.evaluation import (
    threshold_search,
    write_validation_outputs,
)
from business_entity_resolution.modeling import (
    infer_scores_chunked,
    train_model,
    train_separate_models,
)
from business_entity_resolution.records import read_ground_truth, stable_fold
from business_entity_resolution.schemas import SOURCE_COLUMNS, require_exact_columns


LOGGER = logging.getLogger(__name__)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_countries(path: Path) -> dict[str, str]:
    countries: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        require_exact_columns(reader.fieldnames, SOURCE_COLUMNS, label=str(path))
        for row in reader:
            countries[row["entity_id"]] = row["country"]
    return countries


def _compare_score_files(left_path: Path, right_path: Path) -> dict[str, Any]:
    compared = 0
    max_absolute_difference = 0.0
    with left_path.open("r", encoding="utf-8", newline="") as left_handle:
        with right_path.open("r", encoding="utf-8", newline="") as right_handle:
            left_reader = csv.DictReader(left_handle, delimiter="\t")
            right_reader = csv.DictReader(right_handle, delimiter="\t")
            for left, right in zip(left_reader, right_reader, strict=True):
                left_key = (
                    left["source1_entity_id"],
                    left["candidate_entity_id"],
                    left["target_source"],
                )
                right_key = (
                    right["source1_entity_id"],
                    right["candidate_entity_id"],
                    right["target_source"],
                )
                if left_key != right_key:
                    raise ValueError(
                        f"score row order mismatch at row {compared + 1}: "
                        f"{left_key} != {right_key}"
                    )
                difference = abs(
                    float(left["match_probability"])
                    - float(right["match_probability"])
                )
                max_absolute_difference = max(max_absolute_difference, difference)
                compared += 1
    return {
        "left_file": str(left_path.resolve()),
        "right_file": str(right_path.resolve()),
        "compared_rows": compared,
        "max_absolute_probability_difference": max_absolute_difference,
        "pass": max_absolute_difference <= 1e-12,
    }


def run_day2_pipeline(
    config: AppConfig,
    *,
    subset_dir: Path | None = None,
    day1_work_dir: Path | None = None,
    day1_artifact_dir: Path | None = None,
    work_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> dict[str, object]:
    subset_root = (subset_dir or config.paths.development_dir).resolve()
    day1_work = (day1_work_dir or (config.paths.work_dir / "day1")).resolve()
    day1_artifacts = (
        day1_artifact_dir or (config.paths.artifact_dir / "day1")
    ).resolve()
    work_root = (work_dir or (config.paths.work_dir / "day2")).resolve()
    artifact_root = (
        artifact_dir or (config.paths.artifact_dir / "day2")
    ).resolve()
    log_path = config.repository_root / "logs" / "day2_pipeline.log"
    _configure_logging(log_path, config.project.log_level)
    work_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    monitor = RunMonitor()

    feature_path = day1_work / "features.tsv"
    raw_candidate_path = day1_work / "candidate_pairs_raw.tsv"
    day1_score_path = day1_work / "validation_scores.tsv"
    source1_path = subset_root / "train" / "train_source1.tsv"
    truth_path = subset_root / "train" / "train_ground_truth.tsv"
    required = (
        feature_path,
        raw_candidate_path,
        day1_score_path,
        day1_artifacts / "day1_lightgbm.joblib",
        source1_path,
        truth_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Day 2 inputs are missing: " + ", ".join(missing))

    with monitor.stage("load_validation_contract"):
        truth = read_ground_truth(truth_path)
        countries = _read_countries(source1_path)
        folds = {
            entity_id: stable_fold(
                entity_id,
                seed=config.project.seed,
                n_folds=config.day1_baseline.n_folds,
            )
            for entity_id in truth
        }
        validation_ids = {
            entity_id
            for entity_id, fold in folds.items()
            if fold == config.day1_baseline.validation_fold
        }

    pooled_model_path = artifact_root / "pooled_350.joblib"
    with monitor.stage("train_pooled_model"):
        pooled_training = train_model(
            feature_path,
            pooled_model_path,
            artifact_root / "pooled_training_metrics.json",
            validation_fold=config.day1_baseline.validation_fold,
            seed=config.project.seed,
            num_boost_rounds=config.day2.num_boost_rounds,
        )

    separate_model_path = artifact_root / "separate_s2_s3_350.joblib"
    with monitor.stage("train_separate_models"):
        separate_training = train_separate_models(
            feature_path,
            separate_model_path,
            artifact_root / "separate_training_metrics.json",
            validation_fold=config.day1_baseline.validation_fold,
            seed=config.project.seed,
            num_boost_rounds=config.day2.num_boost_rounds,
        )

    score_paths = {
        "pooled": work_root / "pooled_validation_scores.tsv",
        "separate": work_root / "separate_validation_scores.tsv",
    }
    model_paths = {
        "pooled": pooled_model_path,
        "separate": separate_model_path,
    }
    inference: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, object]] = {}
    thresholds = [value / 100 for value in range(5, 100)]
    for architecture in ("pooled", "separate"):
        with monitor.stage(f"chunked_validation_inference_{architecture}"):
            inference[architecture] = infer_scores_chunked(
                feature_path,
                model_paths[architecture],
                score_paths[architecture],
                work_root / f"{architecture}_inference_checkpoint.json",
                chunk_size=config.day2.inference_chunk_size,
                fold=config.day1_baseline.validation_fold,
            )
        with monitor.stage(f"threshold_search_{architecture}"):
            evaluations[architecture] = threshold_search(
                score_paths[architecture],
                truth=truth,
                entity_ids=validation_ids,
                thresholds=thresholds,
            )

    configured_choice = config.day2.selected_architecture
    if configured_choice == "best_validation":
        selected_architecture = max(
            ("pooled", "separate"),
            key=lambda name: (
                float(evaluations[name]["macro_f0_5"]),
                float(evaluations[name]["micro_precision"]),
                name == "pooled",
            ),
        )
    else:
        selected_architecture = configured_choice
    selected_evaluation = evaluations[selected_architecture]
    selected_threshold = float(selected_evaluation["best_threshold"])

    comparison: dict[str, object] = {
        "selection_rule": configured_choice,
        "selected_architecture": selected_architecture,
        "selected_model_file": str(model_paths[selected_architecture].resolve()),
        "selected_score_file": str(score_paths[selected_architecture].resolve()),
        "architectures": evaluations,
    }
    _write_json(artifact_root / "model_comparison.json", comparison)

    with monitor.stage("verify_chunked_inference_equivalence"):
        day1_equivalence = _compare_score_files(
            day1_score_path,
            work_root / "day1_model_chunked_scores.tsv",
        ) if (work_root / "day1_model_chunked_scores.tsv").is_file() else None
        if day1_equivalence is None:
            day1_chunked = infer_scores_chunked(
                feature_path,
                day1_artifacts / "day1_lightgbm.joblib",
                work_root / "day1_model_chunked_scores.tsv",
                work_root / "day1_model_inference_checkpoint.json",
                chunk_size=config.day2.inference_chunk_size,
                fold=config.day1_baseline.validation_fold,
            )
            day1_equivalence = _compare_score_files(
                day1_score_path,
                work_root / "day1_model_chunked_scores.tsv",
            )
            day1_equivalence["chunked_inference"] = day1_chunked
        if not day1_equivalence["pass"]:
            raise ValueError("chunked inference does not reproduce Day 1 probabilities")

    with monitor.stage("outputs_and_error_handoff"):
        matching_rows, candidate_rows = write_validation_outputs(
            raw_candidate_path,
            score_paths[selected_architecture],
            work_root / "validation_matching_results.tsv",
            work_root / "validation_candidate_pairs.tsv",
            entity_ids=validation_ids,
            threshold=selected_threshold,
        )
        error_summary = write_error_slices(
            score_paths[selected_architecture],
            work_root / "person4_error_slices.tsv",
            artifact_root / "person4_error_summary.json",
            truth=truth,
            countries=countries,
            entity_ids=validation_ids,
            threshold=selected_threshold,
        )
        country_metrics = country_robustness_metrics(
            score_paths[selected_architecture],
            truth=truth,
            countries=countries,
            entity_ids=validation_ids,
            threshold=selected_threshold,
        )
        _write_json(artifact_root / "country_slice_metrics.json", country_metrics)

    threshold_plan = {
        "status": "provisional_until_person4_independent_validation",
        "architecture": selected_architecture,
        "global_threshold": selected_threshold,
        "selection_metric": "entity-level macro F0.5",
        "beta": 0.5,
        "search_grid": {"minimum": 0.05, "maximum": 0.99, "step": 0.01},
        "post_processing": {
            "accept_if_probability_greater_than_or_equal_to_threshold": True,
            "empty_prediction_means_singleton": True,
            "predictions_must_be_subset_of_final_candidates": True,
            "sort_ids_lexicographically": True,
        },
    }
    _write_json(artifact_root / "threshold_plan.json", threshold_plan)

    benchmark: dict[str, object] = {
        "development_source1_entities": len(truth),
        "validation_entities": len(validation_ids),
        "feature_rows": int(pooled_training["training_rows"])
        + int(pooled_training["validation_candidate_rows"]),
        "input_feature_bytes": feature_path.stat().st_size,
        "pooled_model_bytes": pooled_model_path.stat().st_size,
        "separate_model_bytes": separate_model_path.stat().st_size,
        "pooled_score_bytes": score_paths["pooled"].stat().st_size,
        "separate_score_bytes": score_paths["separate"].stat().st_size,
        "inference": inference,
        "stage_metrics": monitor.stages,
        "maximum_observed_rss_mb_after_stage": max(
            float(stage["rss_mb_after"]) for stage in monitor.stages
        ),
        "restartable_chunking": {
            "enabled": True,
            "chunk_size": config.day2.inference_chunk_size,
            "checkpoint_removed_after_success": True,
            "day1_probability_equivalence": day1_equivalence,
        },
    }
    _write_json(artifact_root / "medium_scale_benchmark.json", benchmark)

    frozen_runtime_config = {
        "freeze_status": "provisional_pending_teammate_handoffs",
        "seed": config.project.seed,
        "n_folds": config.day1_baseline.n_folds,
        "validation_fold": config.day1_baseline.validation_fold,
        "feature_contract": "MODEL_FEATURE_COLUMNS in features.py",
        "candidate_contract": "CANDIDATE_PAIR_COLUMNS in schemas.py",
        "selected_architecture": selected_architecture,
        "selected_model_file": str(model_paths[selected_architecture].resolve()),
        "num_boost_rounds": config.day2.num_boost_rounds,
        "inference_chunk_size": config.day2.inference_chunk_size,
        "threshold": selected_threshold,
        "post_processing": threshold_plan["post_processing"],
        "external_handoffs": {
            "person2_candidate_recall_evidence": "pending",
            "person3_feature_ablation_evidence": "pending",
            "person4_independent_threshold_and_loco_evidence": "pending",
        },
    }
    _write_json(artifact_root / "frozen_runtime_config.json", frozen_runtime_config)

    summary: dict[str, object] = {
        "status": "PASS",
        "freeze_status": "PROVISIONAL_PENDING_TEAMMATE_HANDOFFS",
        "selected_architecture": selected_architecture,
        "selected_threshold": selected_threshold,
        "selected_evaluation": selected_evaluation,
        "pooled_training": pooled_training,
        "separate_training": separate_training,
        "model_comparison": comparison,
        "chunked_inference_equivalence": day1_equivalence,
        "validation_matching_rows": matching_rows,
        "validation_candidate_rows": candidate_rows,
        "error_handoff": error_summary,
        "country_slice_metrics": country_metrics,
        "benchmark_file": str(
            (artifact_root / "medium_scale_benchmark.json").resolve()
        ),
        "threshold_plan_file": str((artifact_root / "threshold_plan.json").resolve()),
        "frozen_runtime_config_file": str(
            (artifact_root / "frozen_runtime_config.json").resolve()
        ),
        "stage_metrics": monitor.stages,
    }
    _write_json(artifact_root / "run_summary.json", summary)
    LOGGER.info(
        "DAY2 PERSON1 PASS | architecture=%s threshold=%.2f macro_f0_5=%.6f",
        selected_architecture,
        selected_threshold,
        float(selected_evaluation["macro_f0_5"]),
    )
    return summary
