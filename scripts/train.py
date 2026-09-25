"""Train the Day 1 LightGBM model from a contract-compatible feature TSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from business_entity_resolution.modeling import train_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--validation-fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-boost-rounds", type=int, default=250)
    args = parser.parse_args()
    result = train_model(
        args.features,
        args.model,
        args.metrics,
        validation_fold=args.validation_fold,
        seed=args.seed,
        num_boost_rounds=args.num_boost_rounds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
