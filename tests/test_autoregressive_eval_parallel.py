from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


try:
    import flash_attn_interface  # noqa: F401
except ImportError:
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        flash_attn = types.ModuleType("flash_attn")

        def _missing_flash_attn(*_args, **_kwargs):
            raise RuntimeError("flash attention should not be used by this CPU-only test")

        flash_attn.flash_attn_func = _missing_flash_attn
        flash_attn.flash_attn_varlen_func = _missing_flash_attn
        sys.modules["flash_attn"] = flash_attn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from autoregressive_eval import (  # noqa: E402
    _generate_batch_parallel_slow,
    _generate_sample_slow,
    _iter_autoregressive_eval_batches,
    _resolve_autoregressive_eval_batch_size,
)


class _ToyAutoregressiveModel:
    def __init__(self, vocab_size: int = 64):
        self.config = SimpleNamespace(loops=1)
        self.vocab_size = vocab_size
        self.calls = 0

    def initial_carry(self, batch):
        batch_size = int(batch["puzzle_identifiers"].shape[0])
        return SimpleNamespace(
            halted=torch.zeros((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
        )

    def __call__(self, *, carry, batch):
        del carry
        self.calls += 1
        label_lengths = batch["label_seq_lengths"].to(torch.long)
        total_answer_len = int(label_lengths.sum().item())
        logits = torch.full(
            (total_answer_len, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=batch["inputs"].device,
        )

        label_offsets = batch["label_seq_offsets"].to(torch.long)
        for sample_idx, identifier in enumerate(batch["puzzle_identifiers"].to(torch.long).tolist()):
            start = int(label_offsets[sample_idx].item())
            end = int(label_offsets[sample_idx + 1].item())
            for answer_pos, logit_idx in enumerate(range(start, end)):
                logits[logit_idx, self._token_for(identifier, answer_pos)] = 1000.0

        halted = torch.ones_like(label_lengths, dtype=torch.bool)
        q_halt_logits = batch["puzzle_identifiers"].to(torch.float32) / 10.0
        return (
            SimpleNamespace(halted=halted, steps=label_lengths.to(torch.int32)),
            {"logits": logits, "q_halt_logits": q_halt_logits},
        )

    def _token_for(self, identifier: int, answer_pos: int) -> int:
        return int((identifier * 3 + answer_pos + 5) % self.vocab_size)


def _position_ids(length: int, sample_idx: int, is_answer: bool) -> torch.Tensor:
    positions = torch.zeros((length, 4), dtype=torch.int32)
    positions[:, 0] = sample_idx
    positions[:, 1] = 1 if is_answer else 0
    positions[:, 3] = torch.arange(length, dtype=torch.int32)
    return positions


def _build_packed_batch() -> dict:
    identifiers = torch.tensor([2, 4, 6], dtype=torch.int64)
    answer_lengths = [3, 2, 0]
    context_chunks = [
        torch.tensor([2, 3], dtype=torch.int32),
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([4, 4], dtype=torch.int32),
    ]
    answer_placeholders = [
        torch.tensor([1, 0, 0], dtype=torch.int32),
        torch.tensor([1, 0], dtype=torch.int32),
        torch.empty((0,), dtype=torch.int32),
    ]

    model = _ToyAutoregressiveModel()
    label_chunks = [
        torch.tensor([model._token_for(int(identifier), pos) for pos in range(answer_len)], dtype=torch.int32)
        for identifier, answer_len in zip(identifiers.tolist(), answer_lengths)
    ]

    input_chunks = []
    source_chunks = []
    answer_mask_chunks = []
    position_chunks = []
    for sample_idx, (context, answer) in enumerate(zip(context_chunks, answer_placeholders)):
        input_chunks.extend([context, answer])
        source_chunks.extend([context, label_chunks[sample_idx].to(torch.int32)])
        answer_mask_chunks.extend(
            [
                torch.zeros((context.numel(),), dtype=torch.bool),
                torch.ones((answer.numel(),), dtype=torch.bool),
            ]
        )
        position_chunks.extend(
            [
                _position_ids(context.numel(), sample_idx, is_answer=False),
                _position_ids(answer.numel(), sample_idx, is_answer=True),
            ]
        )

    seq_lengths = torch.tensor(
        [context.numel() + answer.numel() for context, answer in zip(context_chunks, answer_placeholders)],
        dtype=torch.int32,
    )
    label_lengths = torch.tensor(answer_lengths, dtype=torch.int32)
    return {
        "inputs": torch.cat(input_chunks, dim=0),
        "labels": torch.cat(label_chunks, dim=0),
        "answer_mask": torch.cat(answer_mask_chunks, dim=0),
        "source_inputs": torch.cat(source_chunks, dim=0),
        "position_ids": torch.cat(position_chunks, dim=0),
        "seq_lengths": seq_lengths,
        "seq_offsets": torch.nn.functional.pad(torch.cumsum(seq_lengths, dim=0), (1, 0)).to(torch.int32),
        "label_seq_lengths": label_lengths,
        "label_seq_offsets": torch.nn.functional.pad(torch.cumsum(label_lengths, dim=0), (1, 0)).to(torch.int32),
        "puzzle_identifiers": identifiers,
    }


class AutoregressiveEvalParallelTests(unittest.TestCase):
    def test_parallel_full_prefix_decode_matches_single_sample_decode(self) -> None:
        batch = _build_packed_batch()

        single_model = _ToyAutoregressiveModel()
        single_results = [
            _generate_sample_slow(single_model, batch, sample_idx, start_token_id=1)
            for sample_idx in range(int(batch["puzzle_identifiers"].shape[0]))
        ]

        parallel_model = _ToyAutoregressiveModel()
        parallel_results = _generate_batch_parallel_slow(parallel_model, batch, start_token_id=1)

        self.assertEqual(single_model.calls, 5)
        self.assertEqual(parallel_model.calls, 3)
        self.assertEqual(len(parallel_results), len(single_results))

        for (single_batch, single_preds, single_metrics), (parallel_batch, parallel_preds, parallel_metrics) in zip(
            single_results,
            parallel_results,
        ):
            self.assertEqual(single_metrics, parallel_metrics)
            for key in ("inputs", "source_inputs", "answer_mask", "position_ids", "seq_lengths", "seq_offsets"):
                self.assertTrue(torch.equal(single_batch[key], parallel_batch[key]), key)
            for key in ("preds", "q_halt_logits"):
                self.assertTrue(torch.equal(single_preds[key], parallel_preds[key]), key)

    def test_parallel_decode_can_be_microbatched(self) -> None:
        batch = _build_packed_batch()
        full_model = _ToyAutoregressiveModel()
        full_results = _generate_batch_parallel_slow(full_model, batch, start_token_id=1)

        micro_model = _ToyAutoregressiveModel()
        micro_results = []
        spans = []
        for sample_start, sample_end, micro_batch in _iter_autoregressive_eval_batches(batch, 2):
            spans.append((sample_start, sample_end))
            micro_results.extend(_generate_batch_parallel_slow(micro_model, micro_batch, start_token_id=1))

        self.assertEqual(spans, [(0, 2), (2, 3)])
        self.assertEqual(len(micro_results), len(full_results))
        for (_full_batch, full_preds, full_metrics), (_micro_batch, micro_preds, micro_metrics) in zip(
            full_results,
            micro_results,
        ):
            self.assertEqual(full_metrics, micro_metrics)
            self.assertTrue(torch.equal(full_preds["preds"], micro_preds["preds"]))

    def test_autoregressive_eval_batch_size_config_resolution(self) -> None:
        self.assertIsNone(
            _resolve_autoregressive_eval_batch_size(
                SimpleNamespace(arch=SimpleNamespace())
            )
        )
        self.assertEqual(
            _resolve_autoregressive_eval_batch_size(
                SimpleNamespace(autoregressive_eval_batch_size=2, arch=SimpleNamespace())
            ),
            2,
        )
        self.assertEqual(
            _resolve_autoregressive_eval_batch_size(
                SimpleNamespace(arch=SimpleNamespace(autoregressive_eval_batch_size=3))
            ),
            3,
        )
        with self.assertRaises(ValueError):
            _resolve_autoregressive_eval_batch_size(
                SimpleNamespace(autoregressive_eval_batch_size=0, arch=SimpleNamespace())
            )


if __name__ == "__main__":
    unittest.main()
