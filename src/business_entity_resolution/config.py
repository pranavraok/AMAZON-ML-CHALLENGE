"""Repository configuration loading and input-path validation."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


DATASET_ENV_VAR = "AMAZON_ML_DATASET_ROOT"


@dataclass(frozen=True)
class ProjectSettings:
    seed: int
    log_level: str


@dataclass(frozen=True)
class PathSettings:
    dataset_root: Path
    development_dir: Path
    work_dir: Path
    artifact_dir: Path
    output_dir: Path

    @property
    def train_dir(self) -> Path:
        return self.dataset_root / "train"

    @property
    def test_dir(self) -> Path:
        return self.dataset_root / "test"

    @property
    def train_source1(self) -> Path:
        return self.train_dir / "train_source1.tsv"

    @property
    def train_source2(self) -> Path:
        return self.train_dir / "train_source2.tsv"

    @property
    def train_source3(self) -> Path:
        return self.train_dir / "train_source3.tsv"

    @property
    def train_ground_truth(self) -> Path:
        return self.train_dir / "train_ground_truth.tsv"

    @property
    def test_source1(self) -> Path:
        return self.test_dir / "test_source1.tsv"

    @property
    def test_source2(self) -> Path:
        return self.test_dir / "test_source2.tsv"

    @property
    def test_source3(self) -> Path:
        return self.test_dir / "test_source3.tsv"

    def required_input_paths(self) -> tuple[Path, ...]:
        return (
            self.train_source1,
            self.train_source2,
            self.train_source3,
            self.train_ground_truth,
            self.test_source1,
            self.test_source2,
            self.test_source3,
        )


@dataclass(frozen=True)
class DevelopmentSubsetSettings:
    source1_count: int
    negative_multiplier_per_source: float


@dataclass(frozen=True)
class Day1BaselineSettings:
    n_folds: int
    validation_fold: int
    max_candidates_per_source: int
    fallback_candidates_per_source: int
    max_token_bucket: int
    num_boost_rounds: int


@dataclass(frozen=True)
class AppConfig:
    repository_root: Path
    project: ProjectSettings
    paths: PathSettings
    development_subset: DevelopmentSubsetSettings
    day1_baseline: Day1BaselineSettings

    def validate_inputs(self) -> list[Path]:
        return [path for path in self.paths.required_input_paths() if not path.is_file()]

    def as_dict(self) -> dict[str, Any]:
        return {
            "repository_root": str(self.repository_root),
            "project": {
                "seed": self.project.seed,
                "log_level": self.project.log_level,
            },
            "paths": {
                "dataset_root": str(self.paths.dataset_root),
                "development_dir": str(self.paths.development_dir),
                "work_dir": str(self.paths.work_dir),
                "artifact_dir": str(self.paths.artifact_dir),
                "output_dir": str(self.paths.output_dir),
            },
            "development_subset": {
                "source1_count": self.development_subset.source1_count,
                "negative_multiplier_per_source": (
                    self.development_subset.negative_multiplier_per_source
                ),
            },
            "day1_baseline": {
                "n_folds": self.day1_baseline.n_folds,
                "validation_fold": self.day1_baseline.validation_fold,
                "max_candidates_per_source": (
                    self.day1_baseline.max_candidates_per_source
                ),
                "fallback_candidates_per_source": (
                    self.day1_baseline.fallback_candidates_per_source
                ),
                "max_token_bucket": self.day1_baseline.max_token_bucket,
                "num_boost_rounds": self.day1_baseline.num_boost_rounds,
            },
        }


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_config(config_path: str | Path) -> AppConfig:
    """Load JSON configuration and apply the dataset-root environment override."""

    repository_root = _repository_root()
    path = Path(config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    project_raw = raw.get("project", {})
    paths_raw = raw.get("paths", {})
    subset_raw = raw.get("development_subset", {})
    baseline_raw = raw.get("day1_baseline", {})

    dataset_value = os.environ.get(
        DATASET_ENV_VAR,
        paths_raw.get("dataset_root", "dataset"),
    )

    source1_count = int(subset_raw.get("source1_count", 50_000))
    negative_multiplier = float(
        subset_raw.get("negative_multiplier_per_source", 2.0)
    )
    if source1_count <= 0:
        raise ValueError("development_subset.source1_count must be positive")
    if negative_multiplier < 0:
        raise ValueError(
            "development_subset.negative_multiplier_per_source cannot be negative"
        )

    n_folds = int(baseline_raw.get("n_folds", 5))
    validation_fold = int(baseline_raw.get("validation_fold", 0))
    max_candidates_per_source = int(
        baseline_raw.get("max_candidates_per_source", 12)
    )
    fallback_candidates_per_source = int(
        baseline_raw.get("fallback_candidates_per_source", 2)
    )
    max_token_bucket = int(baseline_raw.get("max_token_bucket", 200))
    num_boost_rounds = int(baseline_raw.get("num_boost_rounds", 250))
    if n_folds < 2:
        raise ValueError("day1_baseline.n_folds must be at least 2")
    if not 0 <= validation_fold < n_folds:
        raise ValueError("day1_baseline.validation_fold must be within n_folds")
    if max_candidates_per_source <= 0:
        raise ValueError("day1_baseline.max_candidates_per_source must be positive")
    if fallback_candidates_per_source < 0:
        raise ValueError(
            "day1_baseline.fallback_candidates_per_source cannot be negative"
        )
    if max_token_bucket <= 0:
        raise ValueError("day1_baseline.max_token_bucket must be positive")
    if num_boost_rounds <= 0:
        raise ValueError("day1_baseline.num_boost_rounds must be positive")

    return AppConfig(
        repository_root=repository_root,
        project=ProjectSettings(
            seed=int(project_raw.get("seed", 2026)),
            log_level=str(project_raw.get("log_level", "INFO")).upper(),
        ),
        paths=PathSettings(
            dataset_root=_resolve_path(dataset_value, repository_root),
            development_dir=_resolve_path(
                paths_raw.get("development_dir", "data/dev"), repository_root
            ),
            work_dir=_resolve_path(
                paths_raw.get("work_dir", "data/work"), repository_root
            ),
            artifact_dir=_resolve_path(
                paths_raw.get("artifact_dir", "artifacts"), repository_root
            ),
            output_dir=_resolve_path(
                paths_raw.get("output_dir", "output"), repository_root
            ),
        ),
        development_subset=DevelopmentSubsetSettings(
            source1_count=source1_count,
            negative_multiplier_per_source=negative_multiplier,
        ),
        day1_baseline=Day1BaselineSettings(
            n_folds=n_folds,
            validation_fold=validation_fold,
            max_candidates_per_source=max_candidates_per_source,
            fallback_candidates_per_source=fallback_candidates_per_source,
            max_token_bucket=max_token_bucket,
            num_boost_rounds=num_boost_rounds,
        ),
    )
