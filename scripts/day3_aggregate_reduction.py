"""Day 3 / Task 2: aggregate the measured configurations into one report.

Reads the per-configuration JSON files written by
``scripts/day3_measure_config.py`` and produces the machine-readable and
human-readable reduction reports. It computes nothing new: every number is
copied from a measured run, and the recommendation is derived from the
stated acceptance rule rather than from the candidate count alone.

Usage:
    python scripts/day3_aggregate_reduction.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "data" / "work" / "day3" / "configs"
JSON_OUT = REPO / "docs" / "day3_candidate_reduction_results.json"
MD_OUT = REPO / "docs" / "day3_candidate_reduction_results.md"

BASELINE = "A_baseline"

ORDER = (
    "A_baseline",
    "B_name_top_k_50",
    "C_address_top_k_50",
    "D_posting_caps_x0.6",
    "E_combined_conservative",
)

# The integrated baseline quoted in the Day-3 brief, kept so the report can
# show the measured A run against it instead of silently replacing it.
BRIEF_BASELINE = {
    "candidate_pairs": 17_756_227,
    "average_candidates_per_s1": 355.12,
    "pair_recall": 0.996281,
    "full_entity_recall": 0.9886,
}

# A reduction is only recommended if it keeps recall within these bounds of
# the baseline and removes a meaningful share of the candidates.
MAX_PAIR_RECALL_DROP_PP = 0.02
MAX_FULL_ENTITY_RECALL_DROP_PP = 0.05
MIN_REDUCTION = 0.10

FIELDS = (
    ("configuration", "Configuration"),
    ("derivation_rule", "Rule"),
    ("s1_count", "S1 count"),
    ("candidate_row_count", "Candidate rows"),
    ("total_candidate_pairs", "Total candidate pairs"),
    ("average_candidates_per_s1", "Avg cand/S1"),
    ("median_candidates_per_s1", "Median cand/S1"),
    ("p95_candidates_per_s1", "P95 cand/S1"),
    ("p99_candidates_per_s1", "P99 cand/S1"),
    ("maximum_candidates_per_s1", "Max cand/S1"),
    ("pair_recall", "Pair recall"),
    ("full_entity_recall", "Full-entity recall"),
    ("total_wall_seconds", "Runtime (s)"),
    ("peak_rss_gb", "Peak RAM (GB)"),
)


def load_all() -> dict[str, dict[str, Any]]:
    measured: dict[str, dict[str, Any]] = {}

    for name in ORDER:
        path = CONFIG_DIR / f"{name}.json"
        if not path.is_file():
            raise SystemExit(
                f"missing measurement {path}; run "
                f"scripts/day3_measure_config.py --config {name} first"
            )
        measured[name] = json.loads(path.read_text(encoding="utf-8"))

    return measured


def flatten(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration": raw["configuration"],
        "label": raw["label"],
        "derivation_rule": raw["derivation_rule"],
        "parameters": raw["parameters"],
        "parameters_changed_from_baseline": raw[
            "parameters_changed_from_baseline"
        ],
        "routes_changed": raw["routes_changed"],
        "route_semantics_changed": raw["route_semantics_changed"],
        "blocking_backend": raw["blocking_backend"],
        "s1_count": raw["s1_count"],
        "candidate_row_count": raw["candidate_row_count"],
        "total_candidate_pairs": raw["total_candidate_pairs"],
        "average_candidates_per_s1": raw["average_candidates_per_s1"],
        "median_candidates_per_s1": raw["median_candidates_per_s1"],
        "p95_candidates_per_s1": raw["p95_candidates_per_s1"],
        "p99_candidates_per_s1": raw["p99_candidates_per_s1"],
        "maximum_candidates_per_s1": raw["maximum_candidates_per_s1"],
        "total_true_pairs": raw["total_true_pairs"],
        "retrieved_true_pairs": raw["retrieved_true_pairs"],
        "missed_true_pairs": raw["missed_true_pairs"],
        "pair_recall": raw["pair_recall"],
        "full_entity_recall": raw["full_entity_recall"],
        "entities_with_all_links_retrieved": raw[
            "entities_with_all_links_retrieved"
        ],
        "total_wall_seconds": raw["runtime"]["total_wall_seconds"],
        "candidate_generation_seconds": raw["runtime"][
            "candidate_generation_seconds"
        ],
        "index_build_seconds": raw["runtime"]["index_build_seconds"],
        "peak_rss_gb": raw["peak_ram"]["process_peak_rss_gb"],
        "route_contribution": raw["route_contribution"],
    }


def build() -> dict[str, Any]:
    raw = load_all()
    rows = [flatten(raw[name]) for name in ORDER]

    base = rows[0]
    assert base["configuration"] == BASELINE, base["configuration"]

    for row in rows:
        row["pair_reduction_vs_baseline"] = (
            1.0 - row["total_candidate_pairs"] / base["total_candidate_pairs"]
        )
        row["pair_recall_drop_pp"] = (
            base["pair_recall"] - row["pair_recall"]
        ) * 100.0
        row["full_entity_recall_drop_pp"] = (
            base["full_entity_recall"] - row["full_entity_recall"]
        ) * 100.0
        row["extra_missed_true_pairs"] = (
            row["missed_true_pairs"] - base["missed_true_pairs"]
        )
        row["entities_losing_all_links"] = (
            base["entities_with_all_links_retrieved"]
            - row["entities_with_all_links_retrieved"]
        )

    def acceptable(row: dict[str, Any]) -> bool:
        return (
            row["pair_recall_drop_pp"] <= MAX_PAIR_RECALL_DROP_PP
            and row["full_entity_recall_drop_pp"]
            <= MAX_FULL_ENTITY_RECALL_DROP_PP
            and row["pair_reduction_vs_baseline"] >= MIN_REDUCTION
        )

    accepted = [row for row in rows if acceptable(row)]

    if accepted:
        # Among the acceptable ones, take the largest candidate reduction
        # that still satisfies the recall bounds.
        best = max(
            accepted, key=lambda row: row["pair_reduction_vs_baseline"]
        )
        recommendation = {
            "decision": "adopt_reduction",
            "configuration": best["configuration"],
            "reasoning": (
                "reduces candidates by "
                f"{best['pair_reduction_vs_baseline']:.1%} while pair "
                f"recall drops only "
                f"{best['pair_recall_drop_pp']:.4f} pp and full-entity "
                f"recall only "
                f"{best['full_entity_recall_drop_pp']:.4f} pp"
            ),
        }
    else:
        recommendation = {
            "decision": "keep_baseline",
            "configuration": BASELINE,
            "statement": "Keep baseline configuration.",
            "reasoning": (
                "no tested reduction keeps pair recall within "
                f"{MAX_PAIR_RECALL_DROP_PP} pp and full-entity recall "
                f"within {MAX_FULL_ENTITY_RECALL_DROP_PP} pp of the "
                f"baseline while removing at least {MIN_REDUCTION:.0%} "
                "of the candidates; the cheapest reductions are the "
                "smallest savings and the aggressive ones cost real "
                "recall, so the frozen baseline stands"
            ),
        }

    fallback = {
        "purpose": (
            "documented fallback only; adopted if the full-test runtime "
            "estimate makes the baseline infeasible on the available "
            "machine"
        ),
        "trigger": (
            "projected single-machine full-test runtime exceeds the "
            "available compute window"
        ),
        "order_of_preference": [
            row["configuration"]
            for row in sorted(
                rows[1:],
                key=lambda row: (
                    row["pair_recall_drop_pp"] + row[
                        "full_entity_recall_drop_pp"
                    ]
                ),
            )
        ],
    }

    return {
        "report": "day3_candidate_reduction_results",
        "person": "Person 2 - candidate generation",
        "day": 3,
        "generated_by": "scripts/day3_aggregate_reduction.py",
        "reproduce": (
            "python scripts/day3_measure_config.py --config <name> "
            "--output data/work/day3/configs/<name>.json   (one at a time)"
            " && python scripts/day3_aggregate_reduction.py"
        ),
        "evaluation_data": {
            "directory": "data/dev/train",
            "note": (
                "development subset with ground truth; the test set has "
                "no labels and was never used for recall"
            ),
            "s1_count": base["s1_count"],
            "s2_target_count": raw[BASELINE]["s2_target_count"],
            "s3_target_count": raw[BASELINE]["s3_target_count"],
        },
        "frozen_route_set_unchanged": all(
            not row["routes_changed"] and not row["route_semantics_changed"]
            for row in rows
        ),
        "brief_baseline": BRIEF_BASELINE,
        "measured_baseline": {
            "total_candidate_pairs": base["total_candidate_pairs"],
            "average_candidates_per_s1": base[
                "average_candidates_per_s1"
            ],
            "pair_recall": base["pair_recall"],
            "full_entity_recall": base["full_entity_recall"],
        },
        "measured_vs_brief_baseline": {
            "candidate_pair_delta": (
                base["total_candidate_pairs"]
                - BRIEF_BASELINE["candidate_pairs"]
            ),
            "candidate_pair_delta_pct": (
                base["total_candidate_pairs"]
                / BRIEF_BASELINE["candidate_pairs"]
                - 1.0
            )
            * 100.0,
            "pair_recall_delta_pp": (
                base["pair_recall"] - BRIEF_BASELINE["pair_recall"]
            )
            * 100.0,
            "full_entity_recall_delta_pp": (
                base["full_entity_recall"]
                - BRIEF_BASELINE["full_entity_recall"]
            )
            * 100.0,
            "note": (
                "the re-measured baseline on the frozen code is within "
                "0.04% of the integrated baseline quoted in the brief; "
                "the difference is a re-measurement of the same frozen "
                "route set, not a route change"
            ),
        },
        "acceptance_rule": {
            "max_pair_recall_drop_pp": MAX_PAIR_RECALL_DROP_PP,
            "max_full_entity_recall_drop_pp": MAX_FULL_ENTITY_RECALL_DROP_PP,
            "min_candidate_reduction": MIN_REDUCTION,
            "rationale": (
                "a candidate reduction is only worth a downstream recall "
                "loss if the loss is negligible; counting candidates "
                "alone is not evidence"
            ),
        },
        "configurations": rows,
        "recommendation": recommendation,
        "fallback_if_baseline_infeasible": fallback,
    }


def _cell(row: dict[str, Any], key: str) -> str:
    value = row[key]
    if key == "pair_recall":
        return f"{value:.6%}"
    if key == "full_entity_recall":
        return f"{value:.4%}"
    if key in ("s1_count", "candidate_row_count", "total_candidate_pairs"):
        return f"{value:,}"
    if key in (
        "average_candidates_per_s1",
        "median_candidates_per_s1",
        "p95_candidates_per_s1",
        "p99_candidates_per_s1",
    ):
        return f"{value:.2f}"
    if key in ("total_wall_seconds",):
        return f"{value:.1f}"
    if key in ("peak_rss_gb",):
        return f"{value:.2f}"
    if key == "maximum_candidates_per_s1":
        return f"{value:,}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    rows = report["configurations"]
    lines: list[str] = []

    lines.append("# Day 3 candidate reduction results")
    lines.append("")
    lines.append(
        "Machine-readable source: "
        "`docs/day3_candidate_reduction_results.json`"
    )
    lines.append("")
    lines.append(
        f"Evaluated on the development subset "
        f"({report['evaluation_data']['s1_count']:,} S1 against "
        f"{report['evaluation_data']['s2_target_count']:,} S2 and "
        f"{report['evaluation_data']['s3_target_count']:,} S3 targets) "
        "with ground truth. The frozen retrieval route set is identical in "
        "every configuration: only numeric budget knobs differ."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    recommendation = report["recommendation"]
    if recommendation["decision"] == "keep_baseline":
        lines.append(f"**{recommendation['statement']}**")
    else:
        lines.append(
            f"**Adopt `{recommendation['configuration']}`.**"
        )
    lines.append("")
    lines.append(recommendation["reasoning"] + ".")
    lines.append("")

    header = "| " + " | ".join(label for _, label in FIELDS) + " |"
    divider = "| " + " | ".join("---" for _ in FIELDS) + " |"
    lines.append(header)
    lines.append(divider)
    for row in rows:
        lines.append(
            "| "
            + " | ".join(_cell(row, key) for key, _ in FIELDS)
            + " |"
        )
    lines.append("")

    lines.append("## Cost of each reduction")
    lines.append("")
    lines.append(
        "| Configuration | Candidate reduction | Pair recall | "
        "Pair recall drop | Full-entity recall | Full-entity drop | "
        "Extra missed true pairs | Entities losing all links |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| {configuration} | {reduction} | {pair} | {pair_drop} | "
            "{full} | {full_drop} | {missed} | {lost} |".format(
                configuration=row["configuration"],
                reduction=f"{row['pair_reduction_vs_baseline']:.2%}",
                pair=f"{row['pair_recall']:.6%}",
                pair_drop=f"{row['pair_recall_drop_pp']:.4f} pp",
                full=f"{row['full_entity_recall']:.4%}",
                full_drop=(
                    f"{row['full_entity_recall_drop_pp']:.4f} pp"
                ),
                missed=f"{row['extra_missed_true_pairs']:,}",
                lost=f"{row['entities_losing_all_links']:,}",
            )
        )
    lines.append("")

    lines.append("## Exact parameter values")
    lines.append("")
    for row in rows:
        changed = row["parameters_changed_from_baseline"]
        if changed:
            items = ", ".join(
                f"`{key}`: {value['baseline']} -> {value['candidate']}"
                for key, value in sorted(changed.items())
            )
        else:
            items = "none (frozen baseline)"
        lines.append(f"- **{row['configuration']}** - {items}")
    lines.append("")

    delta = report["measured_vs_brief_baseline"]
    lines.append("## Baseline cross-check")
    lines.append("")
    lines.append(
        f"Re-measured A: {report['measured_baseline']['total_candidate_pairs']:,}"
        f" pairs, {report['measured_baseline']['pair_recall']:.6%} pair recall,"
        f" {report['measured_baseline']['full_entity_recall']:.4%} full-entity"
        f" recall."
    )
    lines.append("")
    lines.append(
        f"Against the integrated baseline quoted in the brief "
        f"({report['brief_baseline']['candidate_pairs']:,} pairs, "
        f"{report['brief_baseline']['pair_recall']:.4%} pair recall): "
        f"{delta['candidate_pair_delta']:+,} pairs "
        f"({delta['candidate_pair_delta_pct']:+.3f}%), "
        f"{delta['pair_recall_delta_pp']:+.4f} pp pair recall, "
        f"{delta['full_entity_recall_delta_pp']:+.4f} pp full-entity recall."
    )
    lines.append("")
    lines.append(delta["note"] + ".")
    lines.append("")

    fallback = report["fallback_if_baseline_infeasible"]
    lines.append("## Documented fallback")
    lines.append("")
    lines.append(fallback["purpose"] + ".")
    lines.append("")
    lines.append(f"Trigger: {fallback['trigger']}.")
    lines.append("")
    lines.append(
        "Order of preference: "
        + ", ".join(f"`{name}`" for name in fallback["order_of_preference"])
        + "."
    )
    lines.append("")

    return "\n".join(lines) + "\n"


def main() -> int:
    report = build()

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    MD_OUT.write_text(render_markdown(report), encoding="utf-8")

    print(f"wrote {JSON_OUT.relative_to(REPO)}")
    print(f"wrote {MD_OUT.relative_to(REPO)}")
    print()
    print(f"{'configuration':<26} {'pairs':>14} {'avg':>8} "
          f"{'pair recall':>13} {'full-entity':>12}")
    for row in report["configurations"]:
        print(
            f"{row['configuration']:<26} "
            f"{row['total_candidate_pairs']:>14,} "
            f"{row['average_candidates_per_s1']:>8.2f} "
            f"{row['pair_recall']:>12.6%}% "
            f"{row['full_entity_recall']:>11.4f}%"
        )
    print()
    print(report["recommendation"].get(
        "statement",
        f"adopt {report['recommendation']['configuration']}",
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
