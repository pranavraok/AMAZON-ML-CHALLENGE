import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from business_entity_resolution.config import DATASET_ENV_VAR, load_config


class ConfigTests(unittest.TestCase):
    def test_dataset_environment_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config_path = temp_path / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "project": {"seed": 7, "log_level": "debug"},
                        "paths": {"dataset_root": "ignored"},
                        "development_subset": {
                            "source1_count": 12,
                            "negative_multiplier_per_source": 1.5,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {DATASET_ENV_VAR: str(temp_path / "dataset")}):
                config = load_config(config_path)

            self.assertEqual(config.project.seed, 7)
            self.assertEqual(config.project.log_level, "DEBUG")
            self.assertEqual(config.paths.dataset_root, (temp_path / "dataset").resolve())
            self.assertEqual(config.development_subset.source1_count, 12)
            self.assertEqual(config.day1_baseline.n_folds, 5)
            self.assertEqual(config.day1_baseline.validation_fold, 0)


if __name__ == "__main__":
    unittest.main()
