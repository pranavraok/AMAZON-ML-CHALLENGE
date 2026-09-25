"""Run the complete Day 1 development baseline without an editable install."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from business_entity_resolution.config import load_config  # noqa: E402
from business_entity_resolution.day1_pipeline import run_day1_pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Day 1 ER baseline pipeline")
    parser.add_argument("--config", default="configs/base.json")
    parser.add_argument("--subset-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument(
        "--max-source1",
        type=int,
        help="Optional smoke-test limit; omit for the complete development subset",
    )
    args = parser.parse_args()
    summary = run_day1_pipeline(
        load_config(args.config),
        subset_dir=args.subset_dir,
        work_dir=args.work_dir,
        artifact_dir=args.artifact_dir,
        max_source1=args.max_source1,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
