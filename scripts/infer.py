"""Score a contract-compatible feature TSV with a saved Day 1 model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from business_entity_resolution.modeling import infer_scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    result = infer_scores(
        args.features,
        args.model,
        args.scores,
        fold=args.fold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
