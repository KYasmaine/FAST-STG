import contextlib
import io
import sys
import unittest
from unittest.mock import patch

from train import default_config, get_params, normalize_split_ratio


class SplitRatioTest(unittest.TestCase):
    def test_notation_normalization(self):
        self.assertEqual(normalize_split_ratio("6:2:2"), "622")
        self.assertEqual(normalize_split_ratio("622"), "622")
        self.assertEqual(normalize_split_ratio("6-2-2"), "622")

    def test_omitted_split_defaults_to_622(self):
        for value in (None, "", "   "):
            with self.subTest(value=value):
                self.assertEqual(normalize_split_ratio(value), "622")

    def test_pems08_defaults_to_622(self):
        config = default_config("PEMS08", preferred_device="cpu")
        self.assertEqual(config.data.split_tag, "622")
        self.assertEqual(config.data.val_start_idx, 10713)
        self.assertEqual(config.data.test_start_idx, 14284)

    def test_all_datasets_default_to_622(self):
        for dataset in (
            "PEMS03", "PEMS04", "PEMS07", "PEMS08", "AIR_BJ", "AIR_GZ",
            "PEMS-BAY", "METR-LA", "KnowAir", "Electricity",
        ):
            with self.subTest(dataset=dataset):
                with patch("train.infer_dataset_meta", return_value={
                    "num_features": 1, "num_vertices": 2, "total_steps": 100,
                    "points_per_hour": 1, "steps_per_day": 24,
                }):
                    config = default_config(dataset, preferred_device="cpu")
                self.assertEqual(config.data.split_tag, "622")
                self.assertEqual(
                    (config.data.train_ratio, config.data.val_ratio, config.data.test_ratio),
                    (0.6, 0.2, 0.2),
                )
                self.assertEqual(config.data.val_start_idx, 60)
                self.assertEqual(config.data.test_start_idx, 80)

    def test_pems08_622_boundaries(self):
        config = default_config("PEMS08", preferred_device="cpu", split_ratio="6:2:2")
        self.assertEqual(config.data.split_tag, "622")
        self.assertEqual(config.data.val_start_idx, 10713)
        self.assertEqual(config.data.test_start_idx, 14284)

    def test_pems08_rejects_811(self):
        with self.assertRaisesRegex(ValueError, "Only 6:2:2"):
            default_config("PEMS08", preferred_device="cpu", split_ratio="8:1:1")

    def test_cli_defaults_to_622(self):
        with patch.object(sys, "argv", ["train.py"]):
            params = get_params()
        self.assertEqual(params.data, "PEMS08")
        self.assertEqual(params.split_ratio, "622")

    def test_cli_rejects_811(self):
        with patch.object(sys, "argv", ["train.py", "--split_ratio", "8:1:1"]):
            with contextlib.redirect_stderr(io.StringIO()) as stderr:
                with self.assertRaises(SystemExit) as error:
                    get_params()
        self.assertEqual(error.exception.code, 2)
        self.assertIn("Only 6:2:2", stderr.getvalue())

    def test_invalid_split_is_rejected(self):
        for value in ("8:1:1", "811", "8-1-1", "7:1:2"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "Only 6:2:2"):
                    normalize_split_ratio(value)


if __name__ == "__main__":
    unittest.main()
