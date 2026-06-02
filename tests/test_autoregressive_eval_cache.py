from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    try:
        import flash_attn  # noqa: F401
    except ImportError as exc:
        raise unittest.SkipTest("flash attention is required for the KV-cache decode test") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from autoregressive_eval import (  # noqa: E402
    _generate_batch_parallel_cached,
    _generate_sample_cached,
    _generate_sample_slow,
    _supports_prefix_lm_kv_cache,
)
from models.losses import IGNORE_LABEL_ID  # noqa: E402
from models.urm.urm import URM  # noqa: E402


def _urm_config(**overrides):
    config = dict(
        batch_size=1,
        seq_len=6,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=12,
        num_layers=1,
        hidden_size=32,
        expansion=1.0,
        num_heads=4,
        pos_encodings="rope",
        loops=2,
        L_cycles=1,
        H_cycles=1,
        grad_H_cycles=1,
        forward_dtype="bfloat16",
        profile=False,
        use_act=True,
        variable_seq_lengths=True,
        forward_mode="prefix_lm",
        input_injection_enabled=True,
        grad_logging_enabled=False,
    )
    config.update(overrides)
    return config


class AutoregressiveEvalCacheTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash-attention decode")
    def test_cached_prefix_lm_decode_matches_full_prefix_decode(self) -> None:
        torch.manual_seed(0)
        model = URM(_urm_config()).cuda().eval()
        batch = {
            "inputs": torch.tensor([2, 3, 4, 1, 9, 10], dtype=torch.int32, device="cuda"),
            "labels": torch.tensor([9, 10, 11], dtype=torch.int32, device="cuda"),
            "answer_mask": torch.tensor([False, False, False, True, True, True], dtype=torch.bool, device="cuda"),
            "source_inputs": torch.tensor([2, 3, 4, 9, 10, 11], dtype=torch.int32, device="cuda"),
            "position_ids": torch.stack(
                [
                    torch.zeros(6, dtype=torch.int32, device="cuda"),
                    torch.zeros(6, dtype=torch.int32, device="cuda"),
                    torch.arange(6, dtype=torch.int32, device="cuda"),
                    torch.zeros(6, dtype=torch.int32, device="cuda"),
                ],
                dim=1,
            ),
            "seq_lengths": torch.tensor([6], dtype=torch.int32, device="cuda"),
            "seq_offsets": torch.tensor([0, 6], dtype=torch.int32, device="cuda"),
            "label_seq_lengths": torch.tensor([3], dtype=torch.int32, device="cuda"),
            "label_seq_offsets": torch.tensor([0, 3], dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.int64, device="cuda"),
        }

        self.assertTrue(_supports_prefix_lm_kv_cache(model))
        with torch.inference_mode():
            _slow_batch, slow_preds, slow_metrics = _generate_sample_slow(
                model,
                batch,
                0,
                start_token_id=1,
            )
            _cached_batch, cached_preds, cached_metrics = _generate_sample_cached(
                model,
                batch,
                0,
                start_token_id=1,
            )

        self.assertEqual(slow_preds["preds"].tolist(), cached_preds["preds"].tolist())
        self.assertEqual(slow_metrics, cached_metrics)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash-attention decode")
    def test_cached_casual_decode_matches_full_causal_decode(self) -> None:
        torch.manual_seed(0)
        model = URM(_urm_config(forward_mode="casual")).cuda().eval()
        batch = {
            "inputs": torch.tensor([2, 3, 4, 1, 9, 10], dtype=torch.int32, device="cuda"),
            "labels": torch.tensor(
                [IGNORE_LABEL_ID, IGNORE_LABEL_ID, IGNORE_LABEL_ID, 9, 10, 11],
                dtype=torch.int32,
                device="cuda",
            ),
            "answer_mask": torch.tensor([False, False, False, True, True, True], dtype=torch.bool, device="cuda"),
            "source_inputs": torch.tensor([2, 3, 4, 9, 10, 11], dtype=torch.int32, device="cuda"),
            "position_ids": torch.stack(
                [
                    torch.zeros(6, dtype=torch.int32, device="cuda"),
                    torch.zeros(6, dtype=torch.int32, device="cuda"),
                    torch.arange(6, dtype=torch.int32, device="cuda"),
                    torch.zeros(6, dtype=torch.int32, device="cuda"),
                ],
                dim=1,
            ),
            "seq_lengths": torch.tensor([6], dtype=torch.int32, device="cuda"),
            "seq_offsets": torch.tensor([0, 6], dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.int64, device="cuda"),
        }

        self.assertTrue(_supports_prefix_lm_kv_cache(model))
        with torch.inference_mode():
            _slow_batch, slow_preds, slow_metrics = _generate_sample_slow(
                model,
                batch,
                0,
                start_token_id=1,
            )
            _cached_batch, cached_preds, cached_metrics = _generate_sample_cached(
                model,
                batch,
                0,
                start_token_id=1,
            )

        self.assertEqual(slow_preds["preds"].tolist(), cached_preds["preds"].tolist())
        self.assertEqual(slow_metrics, cached_metrics)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for flash-attention decode")
    def test_batched_cached_prefix_lm_decode_matches_per_sample_cached_decode(self) -> None:
        torch.manual_seed(0)
        model = URM(_urm_config(seq_len=10)).cuda().eval()
        batch = {
            "inputs": torch.tensor([2, 3, 4, 1, 9, 10, 5, 6, 1, 7], dtype=torch.int32, device="cuda"),
            "labels": torch.tensor([9, 10, 11, 7, 8], dtype=torch.int32, device="cuda"),
            "answer_mask": torch.tensor(
                [False, False, False, True, True, True, False, False, True, True],
                dtype=torch.bool,
                device="cuda",
            ),
            "source_inputs": torch.tensor([2, 3, 4, 9, 10, 11, 5, 6, 7, 8], dtype=torch.int32, device="cuda"),
            "position_ids": torch.cat(
                [
                    torch.stack(
                        [
                            torch.zeros(6, dtype=torch.int32, device="cuda"),
                            torch.zeros(6, dtype=torch.int32, device="cuda"),
                            torch.arange(6, dtype=torch.int32, device="cuda"),
                            torch.zeros(6, dtype=torch.int32, device="cuda"),
                        ],
                        dim=1,
                    ),
                    torch.stack(
                        [
                            torch.ones(4, dtype=torch.int32, device="cuda"),
                            torch.zeros(4, dtype=torch.int32, device="cuda"),
                            torch.arange(4, dtype=torch.int32, device="cuda"),
                            torch.zeros(4, dtype=torch.int32, device="cuda"),
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            ),
            "seq_lengths": torch.tensor([6, 4], dtype=torch.int32, device="cuda"),
            "seq_offsets": torch.tensor([0, 6, 10], dtype=torch.int32, device="cuda"),
            "label_seq_lengths": torch.tensor([3, 2], dtype=torch.int32, device="cuda"),
            "label_seq_offsets": torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda"),
            "puzzle_identifiers": torch.tensor([0, 0], dtype=torch.int64, device="cuda"),
        }

        self.assertTrue(_supports_prefix_lm_kv_cache(model))
        with torch.inference_mode():
            single_results = [
                _generate_sample_cached(model, batch, sample_idx, start_token_id=1)
                for sample_idx in range(2)
            ]
            parallel_results = _generate_batch_parallel_cached(model, batch, start_token_id=1)

        self.assertEqual(len(single_results), len(parallel_results))
        for (_single_batch, single_preds, single_metrics), (_parallel_batch, parallel_preds, parallel_metrics) in zip(
            single_results,
            parallel_results,
        ):
            self.assertEqual(single_preds["preds"].tolist(), parallel_preds["preds"].tolist())
            self.assertTrue(torch.allclose(single_preds["q_halt_logits"], parallel_preds["q_halt_logits"]))
            self.assertEqual(single_metrics, parallel_metrics)


if __name__ == "__main__":
    unittest.main()
