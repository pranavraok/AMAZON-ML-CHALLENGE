"""Person 3 Day 2 experiments on the cached like-for-like feature table.

Subcommands (all use Person 1's rows, folds, hyperparameters and scorer):
  bootstrap  paired entity-level bootstrap of macro F0.5 between two score files
  groups     drop-one-group ablation of the Person 3 feature set
  loco       leave-one-country-out: train on one country, evaluate on the other

Examples:
    python scripts/p3_experiments.py bootstrap --a data/work/p3d2/scores/baseline_separate.tsv \
        --b data/work/p3d2/scores/p3_separate.tsv
    python scripts/p3_experiments.py groups --architecture pooled
    python scripts/p3_experiments.py loco
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from business_entity_resolution.config import load_config
from business_entity_resolution.evaluation import entity_fbeta, predictions_at_threshold, read_scores
from business_entity_resolution.features import MODEL_FEATURE_COLUMNS as BASELINE_COLUMNS
from business_entity_resolution.modeling import TARGET_SOURCES, _build_classifier
from business_entity_resolution.records import read_ground_truth
from business_entity_resolution.similarity_features import FROZEN_MODEL_FEATURES

THRESHOLDS = [v / 100 for v in range(5, 100)]
OUT = Path("data/work/p3d2")

FEATURE_GROUPS = {
    "context": lambda c: c.startswith("ctx_"),
    "name_frequency": lambda c: c.startswith("name_core_freq"),
    "noise_ratio": lambda c: c.endswith("unmatched_max_noise"),
    "primary_number": lambda c: c in {"addr_primary_equal", "addr_primary_cross",
                                      "addr_num_only_a", "addr_num_only_b"},
    "address_numbers": lambda c: c.startswith("addr_num_") or c == "addr_longest_num_equal",
    "skeleton_unicode": lambda c: "skeleton" in c or c in {"name_unicode_ratio", "name_script_mismatch",
                                                          "name_non_latin_a", "name_non_latin_b"},
    "idf_soft": lambda c: "idf" in c,
    "combo": lambda c: c.startswith("combo_"),
    "name_extra_fuzzy": lambda c: c in {"name_partial_ratio", "name_wratio_latin", "name_jaro_winkler",
                                        "name_levenshtein_sim", "name_nospace_ratio",
                                        "name_nospace_partial", "name_alias_best_ratio"},
    "address_fuzzy": lambda c: c in {"addr_ratio", "addr_token_sort_ratio", "addr_partial_ratio",
                                     "addr_token_jaccard", "addr_alpha_containment"},
}


def entity_scores(score_path: Path, truth, ids, threshold: float) -> dict[str, float]:
    preds = predictions_at_threshold(read_scores(score_path), threshold=threshold)
    return {e: entity_fbeta(preds.get(e, set()), truth.get(e, set()), beta=0.5) for e in ids}


def best_threshold(score_path: Path, truth, ids) -> tuple[float, float]:
    scores = read_scores(score_path)
    best = (0.0, -1.0)
    for t in THRESHOLDS:
        preds = predictions_at_threshold(scores, threshold=t)
        value = float(np.mean([entity_fbeta(preds.get(e, set()), truth.get(e, set()), beta=0.5)
                               for e in ids]))
        if value >= best[1]:
            best = (t, value)
    return best


def context():
    config = load_config("configs/base.json")
    truth = read_ground_truth(config.paths.development_dir / "train" / "train_ground_truth.tsv")
    folds = pl.read_csv(config.paths.work_dir / "day1" / "folds.tsv", separator="\t", infer_schema=False)
    vfold = str(config.day1_baseline.validation_fold)
    valid_ids = set(folds.filter(pl.col("fold") == vfold)["source1_entity_id"])
    return config, truth, valid_ids


def fit_predict(train: pl.DataFrame, valid: pl.DataFrame, columns, architecture, seed, rounds):
    proba = np.empty(valid.height)
    groups = [(None, 0)] if architecture == "pooled" else [(s, i) for i, s in enumerate(TARGET_SOURCES)]
    for source, offset in groups:
        tr = train if source is None else train.filter(pl.col("target_source") == source)
        mask = np.ones(valid.height, bool) if source is None else (valid["target_source"] == source).to_numpy()
        model = _build_classifier(seed=seed + offset, num_boost_rounds=rounds)
        model.fit(tr.select(columns).to_numpy().astype(np.float32), tr["label"].to_numpy())
        proba[mask] = model.booster_.predict(
            valid.filter(pl.Series(mask)).select(columns).to_numpy().astype(np.float32))
    return proba


def write_scores(valid: pl.DataFrame, proba, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid.select("source1_entity_id", "candidate_entity_id", "target_source").with_columns(
        match_probability=pl.Series(proba)).write_csv(path, separator="\t")
    return path


def cmd_bootstrap(args) -> None:
    _, truth, ids = context()
    ids = sorted(ids)
    ta, fa = best_threshold(args.a, truth, ids)
    tb, fb = best_threshold(args.b, truth, ids)
    ea = entity_scores(args.a, truth, ids, ta)
    eb = entity_scores(args.b, truth, ids, tb)
    diff = np.array([eb[e] - ea[e] for e in ids])
    rng = np.random.default_rng(2026)
    samples = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(args.n)])
    result = {"a": str(args.a), "b": str(args.b), "a_threshold": ta, "a_macro_f05": fa,
              "b_threshold": tb, "b_macro_f05": fb, "mean_diff": float(diff.mean()),
              "ci95": [float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))],
              "p_b_not_better": float((samples <= 0).mean()),
              "entities_improved": int((diff > 0).sum()), "entities_worse": int((diff < 0).sum())}
    print(json.dumps(result, indent=2))


def cmd_groups(args) -> None:
    config, truth, ids = context()
    frame = pl.read_parquet(OUT / "combined_features.parquet")
    vfold = config.day1_baseline.validation_fold
    train, valid = frame.filter(pl.col("fold") != vfold), frame.filter(pl.col("fold") == vfold)
    full = list(FROZEN_MODEL_FEATURES)
    results = {}
    groups = [g for g in FEATURE_GROUPS if any(FEATURE_GROUPS[g](c) for c in full)]
    for group in ["none", *groups]:
        columns = full if group == "none" else [c for c in full if not FEATURE_GROUPS[group](c)]
        proba = fit_predict(train, valid, columns, args.architecture, config.project.seed,
                            config.day2.num_boost_rounds)
        path = write_scores(valid, proba, OUT / "scores" / f"drop_{group}_{args.architecture}.tsv")
        t, f = best_threshold(path, truth, ids)
        results[group] = {"dropped": len(full) - len(columns), "threshold": t, "macro_f05": round(f, 6)}
        print(group, results[group], flush=True)
    (OUT / f"group_ablation_{args.architecture}.json").write_text(json.dumps(results, indent=2))


def cmd_loco(args) -> None:
    config, truth, _ = context()
    frame = pl.read_parquet(OUT / "combined_features.parquet")
    read = dict(separator="\t", quote_char=None, infer_schema=False)
    countries = pl.read_csv(config.paths.development_dir / "train" / "train_source1.tsv", **read) \
        .select(pl.col("entity_id").alias("source1_entity_id"), "country")
    frame = frame.join(countries, on="source1_entity_id", how="left", maintain_order="left")
    all_ids = dict(countries.iter_rows())
    feature_sets = {"baseline": [f"base__{c}" for c in BASELINE_COLUMNS],
                    "p3": list(FROZEN_MODEL_FEATURES)}
    results = {}
    for held_out in sorted(frame["country"].unique().to_list()):
        # Train on the other countries (all folds); evaluate on the held-out
        # country's validation fold so entity sets match the in-country numbers.
        vfold = config.day1_baseline.validation_fold
        train = frame.filter(pl.col("country") != held_out)
        valid = frame.filter((pl.col("country") == held_out) & (pl.col("fold") == vfold))
        ids = {e for e, c in all_ids.items() if c == held_out} & set(
            pl.read_csv(config.paths.work_dir / "day1" / "folds.tsv", separator="\t", infer_schema=False)
            .filter(pl.col("fold") == str(vfold))["source1_entity_id"])
        in_country_train = frame.filter((pl.col("country") == held_out) & (pl.col("fold") != vfold))
        for name, columns in feature_sets.items():
            for regime, tr in (("loco", train), ("in_country", in_country_train)):
                proba = fit_predict(tr, valid, columns, args.architecture, config.project.seed,
                                    config.day2.num_boost_rounds)
                path = write_scores(valid, proba, OUT / "scores" / f"{regime}_{held_out}_{name}.tsv")
                t, f = best_threshold(path, truth, ids)
                results[f"{held_out}/{name}/{regime}"] = {"threshold": t, "macro_f05": round(f, 6),
                                                          "entities": len(ids)}
                print(held_out, name, regime, results[f"{held_out}/{name}/{regime}"], flush=True)
    (OUT / f"loco_{args.architecture}.json").write_text(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bootstrap")
    b.add_argument("--a", type=Path, required=True)
    b.add_argument("--b", type=Path, required=True)
    b.add_argument("--n", type=int, default=2000)
    g = sub.add_parser("groups")
    g.add_argument("--architecture", default="pooled", choices=("pooled", "separate"))
    lo = sub.add_parser("loco")
    lo.add_argument("--architecture", default="pooled", choices=("pooled", "separate"))
    args = parser.parse_args()
    {"bootstrap": cmd_bootstrap, "groups": cmd_groups, "loco": cmd_loco}[args.cmd](args)


if __name__ == "__main__":
    main()
