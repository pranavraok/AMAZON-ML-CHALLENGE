import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from business_entity_resolution.config import DATASET_ENV_VAR, load_config
from business_entity_resolution.subset import create_development_subset


SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


class DevelopmentSubsetTests(unittest.TestCase):
    def test_selected_truth_and_positive_records_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "dataset"
            train = dataset / "train"
            test = dataset / "test"

            source1_rows = [
                {"entity_id": "S1-1", "business_name": "A", "business_address": "1 Road", "country": "US"},
                {"entity_id": "S1-2", "business_name": "B", "business_address": "2 Road", "country": "US"},
                {"entity_id": "S1-3", "business_name": "C", "business_address": "3 Road", "country": "India"},
                {"entity_id": "S1-4", "business_name": "D", "business_address": "4 Road", "country": "India"},
            ]
            source2_rows = [
                {"entity_id": "S2-1", "business_name": "A1", "business_address": "1 Rd", "country": "US"},
                {"entity_id": "S2-2", "business_name": "B1", "business_address": "2 Rd", "country": "US"},
                {"entity_id": "S2-3", "business_name": "C1", "business_address": "3 Rd", "country": "India"},
                {"entity_id": "S2-4", "business_name": "D1", "business_address": "4 Rd", "country": "India"},
                {"entity_id": "S2-5", "business_name": "Noise", "business_address": "5 Rd", "country": "US"},
                {"entity_id": "S2-6", "business_name": "Noise", "business_address": "6 Rd", "country": "India"},
            ]
            source3_rows = [
                {"entity_id": "S3-1", "business_name": "A2", "business_address": "1 R", "country": "US"},
                {"entity_id": "S3-2", "business_name": "B2", "business_address": "2 R", "country": "US"},
                {"entity_id": "S3-3", "business_name": "C2", "business_address": "3 R", "country": "India"},
                {"entity_id": "S3-4", "business_name": "D2", "business_address": "4 R", "country": "India"},
                {"entity_id": "S3-5", "business_name": "Noise", "business_address": "5 R", "country": "US"},
                {"entity_id": "S3-6", "business_name": "Noise", "business_address": "6 R", "country": "India"},
            ]
            truth_rows = [
                {"source1_entity_id": "S1-1", "matched_entity_ids": "S2-1,S3-1"},
                {"source1_entity_id": "S1-2", "matched_entity_ids": "S2-2,S3-2"},
                {"source1_entity_id": "S1-3", "matched_entity_ids": "S2-3,S3-3"},
                {"source1_entity_id": "S1-4", "matched_entity_ids": "S2-4,S3-4"},
            ]

            write_tsv(train / "train_source1.tsv", SOURCE_COLUMNS, source1_rows)
            write_tsv(train / "train_source2.tsv", SOURCE_COLUMNS, source2_rows)
            write_tsv(train / "train_source3.tsv", SOURCE_COLUMNS, source3_rows)
            write_tsv(
                train / "train_ground_truth.tsv",
                ["source1_entity_id", "matched_entity_ids"],
                truth_rows,
            )
            for index in (1, 2, 3):
                write_tsv(test / f"test_source{index}.tsv", SOURCE_COLUMNS, [])

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project": {"seed": 2026, "log_level": "WARNING"},
                        "paths": {
                            "dataset_root": "ignored",
                            "development_dir": str(root / "dev"),
                        },
                        "development_subset": {
                            "source1_count": 2,
                            "negative_multiplier_per_source": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict(os.environ, {DATASET_ENV_VAR: str(dataset)}):
                config = load_config(config_path)
            summary = create_development_subset(config)

            dev_train = root / "dev" / "train"
            selected_s1 = read_tsv(dev_train / "train_source1.tsv")
            selected_truth = read_tsv(dev_train / "train_ground_truth.tsv")
            selected_s2_ids = {
                row["entity_id"] for row in read_tsv(dev_train / "train_source2.tsv")
            }
            selected_s3_ids = {
                row["entity_id"] for row in read_tsv(dev_train / "train_source3.tsv")
            }

            self.assertEqual(len(selected_s1), 2)
            self.assertEqual(len(selected_truth), 2)
            for truth in selected_truth:
                for match_id in truth["matched_entity_ids"].split(","):
                    if match_id.startswith("S2-"):
                        self.assertIn(match_id, selected_s2_ids)
                    else:
                        self.assertIn(match_id, selected_s3_ids)
            self.assertEqual(summary.selected_source1_count, 2)
            self.assertTrue((root / "dev" / "manifest.json").is_file())


if __name__ == "__main__":
    unittest.main()
