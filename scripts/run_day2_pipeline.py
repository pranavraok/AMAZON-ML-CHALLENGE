"""Run Person 1's complete Day 2 integration workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from business_entity_resolution.config import load_config  # noqa: E402
from business_entity_resolution.day2_pipeline import run_day2_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Person 1 Day 2 model comparison")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--subset-dir", type=Path)
    parser.add_argument("--day1-work-dir", type=Path)
    parser.add_argument("--day1-artifact-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    summary = run_day2_pipeline(
        load_config(args.config),
        subset_dir=args.subset_dir,
        day1_work_dir=args.day1_work_dir,
        day1_artifact_dir=args.day1_artifact_dir,
        work_dir=args.work_dir,
        artifact_dir=args.artifact_dir,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
