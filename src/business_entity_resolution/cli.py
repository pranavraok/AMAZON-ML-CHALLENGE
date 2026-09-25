"""Command-line entry points for the shared project foundation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

from business_entity_resolution.config import DATASET_ENV_VAR, load_config
from business_entity_resolution.subset import create_development_subset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="amazon-ml-er",
        description="Amazon ML Challenge entity-resolution utilities",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    show = subparsers.add_parser("show-config", help="Print resolved configuration")
    show.add_argument("--config", default="configs/base.json")

    validate = subparsers.add_parser(
        "validate-inputs", help="Check that all required challenge files exist"
    )
    validate.add_argument("--config", default="configs/base.json")

    subset = subparsers.add_parser(
        "create-dev-subset",
        help="Create a deterministic S1-grouped development dataset",
    )
    subset.add_argument("--config", default="configs/base.json")
    subset.add_argument("--output-dir", type=Path)
    subset.add_argument("--source1-count", type=int)
    subset.add_argument("--negative-multiplier", type=float)
    subset.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    logging.basicConfig(
        level=getattr(logging, config.project.log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if args.command == "show-config":
        print(json.dumps(config.as_dict(), indent=2, sort_keys=True))
        return 0

    if args.command == "validate-inputs":
        missing = config.validate_inputs()
        if missing:
            print("Missing required input files:")
            for path in missing:
                print(f"  - {path}")
            print(f"Set {DATASET_ENV_VAR} to the student_resource/dataset directory.")
            return 1
        print("PASS: all required train and test source files exist.")
        return 0

    if args.command == "create-dev-subset":
        summary = create_development_subset(
            config,
            output_dir=args.output_dir,
            source1_count=args.source1_count,
            negative_multiplier=args.negative_multiplier,
            overwrite=args.overwrite,
        )
        print(json.dumps(asdict(summary), indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
