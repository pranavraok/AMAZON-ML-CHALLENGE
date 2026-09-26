"""Write/verify configs/p3_features_frozen.json (feature list, dtypes, artifact hashes).

Examples:
    python scripts/freeze_p3_features.py            # write
    python scripts/freeze_p3_features.py --verify   # fail if code or artifacts drifted
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from business_entity_resolution.similarity_features import (
    FEATURE_VERSION, FROZEN_FEATURE_DTYPES, FROZEN_MODEL_FEATURES,
)

CONFIG = Path("configs/p3_features_frozen.json")
ARTIFACTS = [
    Path("artifacts/token_stats.json"),
    Path("artifacts/name_stats/core_counts.parquet"),
    Path("artifacts/name_stats/token_counts.parquet"),
    Path("artifacts/name_stats/meta.parquet"),
]
SOURCE_FILES = [
    Path("src/business_entity_resolution/feature_text.py"),
    Path("src/business_entity_resolution/similarity_features.py"),
    Path("src/business_entity_resolution/name_stats.py"),
    Path("src/business_entity_resolution/token_stats.py"),
    Path("src/business_entity_resolution/normalization.py"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def current() -> dict:
    return {
        "feature_version": FEATURE_VERSION,
        "freeze_status": "final_person3_features",
        "n_features": len(FROZEN_MODEL_FEATURES),
        "features": [{"name": n, "dtype": str(FROZEN_FEATURE_DTYPES[n])} for n in FROZEN_MODEL_FEATURES],
        "excluded_by_decision": {
            "ctx_*": "neutral in like-for-like ablation (95% CI spans 0); depends on whole-table state",
            "retr_*": "depends on candidate-generation details; excluded like Person 1's retrieval scores",
        },
        "missing_value_contract": "no NaN/null/inf anywhere (enforced by validate_feature_frame); "
                                  "not-applicable values (missing address, no numbers, all tokens matched) use "
                                  "sentinel -1.0, or -5.0 for name_*_unmatched_max_noise; explicit "
                                  "addr_missing_* flags and number counts accompany every sentinel",
        "artifacts": {str(p): sha256(p) for p in ARTIFACTS},
        "artifact_commands": [
            "python scripts/fit_token_stats.py --workers 10",
            "python scripts/fit_name_stats.py --workers 10",
        ],
        "source_sha256": {str(p): sha256(p) for p in SOURCE_FILES},
        "feature_command": "python scripts/build_p3_feature_file.py --candidates <candidate tsv> "
                           "--source-dir <dataset split dir> --prefix <train|test> --output <parquet or dir> "
                           "[--s1-batch 10000 for streaming]",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    now = current()
    if not args.verify:
        CONFIG.write_text(json.dumps(now, indent=2) + "\n")
        print(f"wrote {CONFIG} ({now['n_features']} features, {FEATURE_VERSION})")
        return 0
    frozen = json.loads(CONFIG.read_text())
    problems = [key for key in ("feature_version", "features", "artifacts", "source_sha256")
                if frozen[key] != now[key]]
    if problems:
        for key in problems:
            print(f"DRIFT in {key}")
            if isinstance(now[key], dict):
                for k in now[key]:
                    if frozen[key].get(k) != now[key][k]:
                        print(f"  {k}")
        return 1
    print(f"PASS: code and artifacts match {CONFIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
