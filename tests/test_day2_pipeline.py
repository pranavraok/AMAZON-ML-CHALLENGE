import csv
import json
from pathlib import Path
import tempfile
import unittest

from business_entity_resolution.features import FEATURE_COLUMNS, FEATURE_FILE_COLUMNS
from business_entity_resolution.modeling import (
    infer_scores,
    infer_scores_chunked,
    train_separate_models,
)
from business_entity_resolution.records import write_tsv


class Day2PipelineTests(unittest.TestCase):
    def test_separate_models_have_restartable_chunked_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            feature_path = root / "features.tsv"
            rows = []
            for source_index, target_source in enumerate(("S2", "S3"), start=2):
                for index in range(6):
                    label = index % 2
                    feature_values = {column: 0.1 + 0.7 * label for column in FEATURE_COLUMNS}
                    feature_values["target_is_s3"] = int(target_source == "S3")
                    rows.append(
                        {
                            "source1_entity_id": f"S1-{source_index}-{index}",
                            "candidate_entity_id": f"{target_source}-{index}",
                            "target_source": target_source,
                            "label": label,
                            "fold": 0 if index >= 4 else 1,
                            **feature_values,
                        }
                    )
            write_tsv(feature_path, FEATURE_FILE_COLUMNS, rows)
            model_path = root / "separate.joblib"
            train_separate_models(
                feature_path,
                model_path,
                root / "metrics.json",
                validation_fold=0,
                seed=7,
                num_boost_rounds=5,
            )

            normal_path = root / "normal.tsv"
            chunked_path = root / "chunked.tsv"
            infer_scores(feature_path, model_path, normal_path, fold=0)
            metrics = infer_scores_chunked(
                feature_path,
                model_path,
                chunked_path,
                root / "checkpoint.json",
                chunk_size=3,
                fold=0,
            )
            self.assertEqual(metrics["scored_rows"], 4)
            self.assertEqual(metrics["chunks_written"], 2)
            self.assertFalse((root / "checkpoint.json").exists())

            with normal_path.open("r", encoding="utf-8", newline="") as normal_handle:
                normal = list(csv.DictReader(normal_handle, delimiter="\t"))
            with chunked_path.open("r", encoding="utf-8", newline="") as chunked_handle:
                chunked = list(csv.DictReader(chunked_handle, delimiter="\t"))
            self.assertEqual(len(normal), len(chunked))
            for expected, actual in zip(normal, chunked, strict=True):
                self.assertEqual(
                    (
                        expected["source1_entity_id"],
                        expected["candidate_entity_id"],
                        expected["target_source"],
                    ),
                    (
                        actual["source1_entity_id"],
                        actual["candidate_entity_id"],
                        actual["target_source"],
                    ),
                )
                self.assertAlmostEqual(
                    float(expected["match_probability"]),
                    float(actual["match_probability"]),
                    places=14,
                )

            resume_path = root / "resumed.tsv"
            resume_part_path = resume_path.with_suffix(".tsv.part")
            checkpoint_path = root / "resume_checkpoint.json"
            with resume_part_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=list(normal[0]),
                    delimiter="\t",
                    lineterminator="\n",
                )
                writer.writeheader()
                writer.writerows(normal[:2])
            checkpoint_path.write_text(
                json.dumps(
                    {
                        "feature_file": str(feature_path.resolve()),
                        "model_file": str(model_path.resolve()),
                        "fold": 0,
                        "chunk_size": 3,
                        "processed_rows": 2,
                    }
                ),
                encoding="utf-8",
            )
            resumed_metrics = infer_scores_chunked(
                feature_path,
                model_path,
                resume_path,
                checkpoint_path,
                chunk_size=3,
                fold=0,
            )
            self.assertTrue(resumed_metrics["resumed"])
            self.assertEqual(resumed_metrics["scored_rows"], 4)
            with resume_path.open("r", encoding="utf-8", newline="") as handle:
                resumed = list(csv.DictReader(handle, delimiter="\t"))
            self.assertEqual(normal, resumed)


if __name__ == "__main__":
    unittest.main()
