"""Run Person 2's CandidateGenerator (unchanged, from their checkout) and write
the shared six-column candidate contract.

Person 2's branch ships its own ``business_entity_resolution`` package (with a
modified ``normalization.py``), so this script must run with *their* ``src``
first on ``sys.path``; it only uses the standard library plus their code and
writes a TSV that Person 3's feature pipeline then reads.

Example (Person 2 branch extracted to /tmp/p2):
    git archive origin/candidate-generation src | tar -x -C /tmp/p2
    python scripts/run_person2_candidates.py --person2-src /tmp/p2/src \
        --source-dir data/dev/train --prefix train \
        --output data/work/p3d3/person2_candidates.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--person2-src", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--prefix", choices=("train", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    # Person 2's dev_candidate_generation.py configuration.
    parser.add_argument("--name-top-k", type=int, default=100)
    parser.add_argument("--address-top-k", type=int, default=20)
    parser.add_argument("--name-min-score", type=float, default=0.30)
    parser.add_argument("--address-min-score", type=float, default=0.0)
    args = parser.parse_args()

    sys.path.insert(0, str(args.person2_src.resolve()))
    from business_entity_resolution.candidate_generation import (  # noqa: E402
        CandidateGenerationConfig, CandidateGenerator,
    )
    from business_entity_resolution.dev_candidate_generation import load_source  # noqa: E402

    start = time.perf_counter()
    source1 = load_source(args.source_dir / f"{args.prefix}_source1.tsv")
    if args.limit:
        source1 = source1[: args.limit]
    config = CandidateGenerationConfig(
        name_top_k=args.name_top_k, address_top_k=args.address_top_k,
        name_min_score=args.name_min_score, address_min_score=args.address_min_score,
    )
    rows = []
    repaired_target_source = 0
    for i in (2, 3):
        target = load_source(args.source_dir / f"{args.prefix}_source{i}.tsv")
        generator = CandidateGenerator(target_records=target, config=config)
        for source1_id, evidence in generator.generate(source1).items():
            for item in evidence:
                # Workaround for a Person 2 bug: CandidateGenerator calls
                # BlockingIndex.generate(source1) without target_source, so
                # blocking-route candidates carry "". Derive it from the ID.
                target_source = item.candidate_entity_id.split("-", 1)[0]
                if item.target_source != target_source:
                    repaired_target_source += 1
                rows.append((
                    source1_id, item.candidate_entity_id, target_source,
                    "+".join(sorted(set(item.routes))),
                    min(item.ranks) if item.ranks else 0,
                    max(item.scores) if item.scores else 0.0,
                ))
        print(f"S{i} done: {len(rows)} rows, {time.perf_counter() - start:.0f}s", flush=True)

    rows.sort(key=lambda r: (r[0], r[1]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "target_source",
                         "retrieval_route", "retrieval_rank", "retrieval_score"])
        writer.writerows(rows)
    summary = {"source1": len(source1), "rows": len(rows),
               "rows_with_repaired_empty_target_source": repaired_target_source,
               "seconds": round(time.perf_counter() - start, 1), "config": vars(config)}
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
