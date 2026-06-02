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
    _generate_casual_no_size_batch,
    _generate_batch_parallel_slow,
    _generate_sample_slow,
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


class _ToyCasualAutoregressiveModel(_ToyAutoregressiveModel):
    def __init__(self, vocab_size: int = 64):
        super().__init__(vocab_size=vocab_size)
        self.config = SimpleNamespace(loops=1, forward_mode="casual")

    def __call__(self, *, carry, batch):
        del carry
        self.calls += 1
        total_len = int(batch["inputs"].numel())
        logits = torch.full(
            (total_len, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=batch["inputs"].device,
        )

        seq_offsets = batch["seq_offsets"].to(torch.long)
        for sample_idx, identifier in enumerate(batch["puzzle_identifiers"].to(torch.long).tolist()):
            start = int(seq_offsets[sample_idx].item())
            end = int(seq_offsets[sample_idx + 1].item())
            answer_len = int(batch["answer_mask"][start:end].sum().item())
            if answer_len > 0:
                logits[end - 1, self._token_for(identifier, answer_len - 1)] = 1000.0

        halted = torch.ones((int(batch["puzzle_identifiers"].shape[0]),), dtype=torch.bool, device=batch["inputs"].device)
        q_halt_logits = batch["puzzle_identifiers"].to(torch.float32) / 10.0
        return (
            SimpleNamespace(halted=halted, steps=torch.ones_like(halted, dtype=torch.int32)),
            {"logits": logits, "q_halt_logits": q_halt_logits},
        )


class _ToyCasualEndTokenModel(_ToyAutoregressiveModel):
    def __init__(self, vocab_size: int = 16, end_token_id: int = 12):
        super().__init__(vocab_size=vocab_size)
        self.config = SimpleNamespace(loops=1, forward_mode="casual", vocab_size=vocab_size)
        self.end_token_id = end_token_id

    def __call__(self, *, carry, batch):
        del carry
        self.calls += 1
        total_len = int(batch["inputs"].numel())
        logits = torch.full(
            (total_len, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=batch["inputs"].device,
        )

        seq_offsets = batch["seq_offsets"].to(torch.long)
        for sample_idx, identifier in enumerate(batch["puzzle_identifiers"].to(torch.long).tolist()):
            end = int(seq_offsets[sample_idx + 1].item())
            last_token = int(batch["inputs"][end - 1].item())
            answer_token = int(identifier + 5)
            next_token = answer_token if last_token == 1 else self.end_token_id
            logits[end - 1, next_token] = 1000.0

        halted = torch.ones((int(batch["puzzle_identifiers"].shape[0]),), dtype=torch.bool, device=batch["inputs"].device)
        q_halt_logits = batch["puzzle_identifiers"].to(torch.float32) / 10.0
        return (
            SimpleNamespace(halted=halted, steps=torch.ones_like(halted, dtype=torch.int32)),
            {"logits": logits, "q_halt_logits": q_halt_logits},
        )


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

    def test_casual_parallel_full_prefix_decode_matches_single_sample_decode(self) -> None:
        batch = _build_packed_batch()

        single_model = _ToyCasualAutoregressiveModel()
        single_results = [
            _generate_sample_slow(single_model, batch, sample_idx, start_token_id=1)
            for sample_idx in range(int(batch["puzzle_identifiers"].shape[0]))
        ]

        parallel_model = _ToyCasualAutoregressiveModel()
        parallel_results = _generate_batch_parallel_slow(parallel_model, batch, start_token_id=1)

        self.assertEqual(len(parallel_results), len(single_results))
        for (_single_batch, single_preds, single_metrics), (_parallel_batch, parallel_preds, parallel_metrics) in zip(
            single_results,
            parallel_results,
        ):
            self.assertEqual(single_metrics, parallel_metrics)
            self.assertTrue(torch.equal(single_preds["preds"], parallel_preds["preds"]))

    def test_casual_no_size_decode_stops_on_end_token(self) -> None:
        prompt_lengths = torch.tensor([2, 3], dtype=torch.int32)
        prompt_positions = torch.cat(
            [
                _position_ids(2, 0, is_answer=False),
                _position_ids(3, 1, is_answer=False),
            ],
            dim=0,
        )
        batch = {
            "inputs": torch.tensor([2, 3, 4, 5, 6], dtype=torch.int32),
            "source_inputs": torch.tensor([2, 3, 4, 5, 6], dtype=torch.int32),
            "prompt_position_ids": prompt_positions,
            "seq_lengths": prompt_lengths,
            "seq_offsets": torch.nn.functional.pad(torch.cumsum(prompt_lengths, dim=0), (1, 0)).to(torch.int32),
            "puzzle_identifiers": torch.tensor([2, 4], dtype=torch.int64),
        }
        model = _ToyCasualEndTokenModel()

        results = _generate_casual_no_size_batch(
            model,
            batch,
            start_token_id=1,
            end_token_id=12,
            max_new_tokens=8,
        )

        self.assertEqual(model.calls, 2)
        self.assertEqual([preds["preds"].tolist() for _final_batch, preds, _metrics in results], [[0, 0, 7], [0, 0, 0, 9]])
        for final_batch, _preds, _metrics in results:
            self.assertIsNone(final_batch["labels"])
            self.assertIn("position_ids", final_batch)

if __name__ == "__main__":
    unittest.main()
