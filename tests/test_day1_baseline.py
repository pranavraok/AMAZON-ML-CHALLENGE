import csv
from pathlib import Path
import tempfile
import unittest

from business_entity_resolution.candidates import iter_candidate_rows
from business_entity_resolution.evaluation import entity_fbeta, macro_fbeta
from business_entity_resolution.features import comparison_features
from business_entity_resolution.records import read_source_records, stable_fold


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def write_source(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SOURCE_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


class Day1BaselineTests(unittest.TestCase):
    def test_exact_candidate_and_features_follow_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source1_path = root / "s1.tsv"
            source2_path = root / "s2.tsv"
            source3_path = root / "s3.tsv"
            write_source(
                source1_path,
                [
                    {
                        "entity_id": "S1-1",
                        "business_name": "Cafe & Sons",
                        "business_address": "12 Main Road",
                        "country": "US",
                    }
                ],
            )
            write_source(
                source2_path,
                [
                    {
                        "entity_id": "S2-1",
                        "business_name": "Cafe and Sons",
                        "business_address": "12 Main Rd",
                        "country": "US",
                    }
                ],
            )
            write_source(
                source3_path,
                [
                    {
                        "entity_id": "S3-1",
                        "business_name": "Different Company",
                        "business_address": "99 Other Street",
                        "country": "US",
                    }
                ],
            )
            source1 = read_source_records(source1_path)
            source2 = read_source_records(source2_path)
            source3 = read_source_records(source3_path)
            rows = list(
                iter_candidate_rows(
                    source1,
                    source2,
                    source3,
                    max_candidates_per_source=3,
                    fallback_candidates_per_source=1,
                    max_token_bucket=10,
                    seed=2026,
                )
            )
            pairs = {
                (row["source1_entity_id"], row["candidate_entity_id"])
                for row in rows
            }
            self.assertIn(("S1-1", "S2-1"), pairs)
            values = comparison_features(
                source1["S1-1"],
                source2["S2-1"],
                retrieval_route="exact_name_unicode",
                retrieval_rank=1,
                retrieval_score=1.0,
            )
            self.assertEqual(values["name_unicode_exact"], 1)
            self.assertGreater(values["address_number_jaccard"], 0.0)

    def test_macro_f0_5_includes_singletons(self) -> None:
        truth = {"S1-1": {"S2-1"}, "S1-2": set()}
        perfect = {"S1-1": {"S2-1"}, "S1-2": set()}
        false_merge = {"S1-1": {"S2-1"}, "S1-2": {"S3-9"}}
        self.assertEqual(macro_fbeta(perfect, truth, truth), 1.0)
        self.assertEqual(macro_fbeta(false_merge, truth, truth), 0.5)
        self.assertAlmostEqual(
            entity_fbeta({"S2-1", "S3-1"}, {"S2-1"}),
            0.5555555555555556,
        )

    def test_fold_assignment_is_deterministic_and_grouped(self) -> None:
        first = stable_fold("S1-123", seed=2026, n_folds=5)
        second = stable_fold("S1-123", seed=2026, n_folds=5)
        self.assertEqual(first, second)
        self.assertIn(first, range(5))


if __name__ == "__main__":
    unittest.main()
