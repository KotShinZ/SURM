from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pydantic
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eggroll_arc_finetune import flatten_metrics, parse_config  # noqa: E402


class EggrollFinetuneConfigTests(unittest.TestCase):
    def test_parse_config_uses_defaults_for_nested_sections(self) -> None:
        config = parse_config(["--checkpoint", "checkpoint.pt", "--output_dir", "outputs/run"])

        self.assertTrue(config.wandb.enabled)
        self.assertEqual(config.wandb.project, "eggroll-urm")
        self.assertTrue(config.short_eval.enabled)
        self.assertEqual(config.short_eval.every_generations, 5)
        self.assertEqual(config.pilot_sigmas, [5e-5, 1e-4, 2e-4])

    def test_parse_config_merges_yaml_and_cli_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "eggroll.yaml"
            config_path.write_text(
                yaml.safe_dump(
                    {
                        "checkpoint": "baseline.pt",
                        "output_dir": "outputs/from_yaml",
                        "lr": 0.002,
                        "wandb": {
                            "project": "yaml-project",
                            "mode": "disabled",
                        },
                        "short_eval": {
                            "enabled": True,
                            "every_generations": 7,
                            "max_problems": 128,
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            config = parse_config(
                [
                    "--config",
                    str(config_path),
                    "--lr",
                    "0.01",
                    "--wandb-project",
                    "cli-project",
                    "--no-short-eval-enabled",
                    "--short-eval-max-problems",
                    "256",
                ]
            )

        self.assertEqual(config.checkpoint, "baseline.pt")
        self.assertEqual(config.output_dir, "outputs/from_yaml")
        self.assertEqual(config.lr, 0.01)
        self.assertEqual(config.wandb.project, "cli-project")
        self.assertEqual(config.wandb.mode, "disabled")
        self.assertFalse(config.short_eval.enabled)
        self.assertEqual(config.short_eval.max_problems, 256)
        self.assertEqual(config.short_eval.every_generations, 7)

    def test_parse_config_rejects_invalid_population(self) -> None:
        with self.assertRaises(pydantic.ValidationError):
            parse_config(
                [
                    "--checkpoint",
                    "checkpoint.pt",
                    "--output_dir",
                    "outputs/run",
                    "--population",
                    "31",
                ]
            )

    def test_flatten_metrics_ignores_non_numeric_values(self) -> None:
        flattened = flatten_metrics(
            "search",
            {
                "generation": 5,
                "metrics": {
                    "exact_accuracy": 0.5,
                    "notes": "ignored",
                },
                "history": [1, 2, 3],
            },
        )

        self.assertEqual(
            flattened,
            {
                "search/generation": 5.0,
                "search/metrics/exact_accuracy": 0.5,
            },
        )


if __name__ == "__main__":
    unittest.main()
