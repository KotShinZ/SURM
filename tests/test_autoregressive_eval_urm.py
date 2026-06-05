from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from autoregressive_eval_URM import _advance_positions, _generate_urm_batch


class _ToyCachedURM:
    def __init__(self, *, first_token_id: int = 5, end_token_id: int = 12, vocab_size: int = 16):
        self.config = SimpleNamespace(loops=1, grid_height=30, grid_width=30)
        self.first_token_id = int(first_token_id)
        self.end_token_id = int(end_token_id)
        self.vocab_size = int(vocab_size)
        self.decode_inputs = []
        self.decode_positions = []

    def initial_carry(self, batch):
        batch_size = int(batch["puzzle_identifiers"].shape[0])
        return SimpleNamespace(
            halted=torch.zeros((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            steps=torch.zeros((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
        )

    def __call__(
        self,
        *,
        carry,
        batch,
        cache=None,
        return_cache=False,
        max_cache_len=None,
        cache_chunk_size=None,
    ):
        del carry, max_cache_len, cache_chunk_size
        batch_size = int(batch["puzzle_identifiers"].shape[0])
        halted_carry = SimpleNamespace(
            halted=torch.ones((batch_size,), dtype=torch.bool, device=batch["inputs"].device),
            steps=torch.ones((batch_size,), dtype=torch.int32, device=batch["inputs"].device),
        )

        if return_cache:
            logits = torch.full(
                (int(batch["inputs"].numel()), self.vocab_size),
                -1000.0,
                dtype=torch.float32,
                device=batch["inputs"].device,
            )
            seq_offsets = batch["seq_offsets"].to(torch.long)
            logits[seq_offsets[1:] - 1, self.first_token_id] = 1000.0
            outputs = {
                "logits": logits,
                "q_halt_logits": torch.zeros((batch_size,), dtype=torch.float32, device=batch["inputs"].device),
            }
            return halted_carry, outputs, SimpleNamespace(layers=[])

        self.decode_inputs.append(batch["inputs"].detach().cpu().clone())
        self.decode_positions.append(batch["position_ids"].detach().cpu().clone())
        logits = torch.full(
            (batch_size, self.vocab_size),
            -1000.0,
            dtype=torch.float32,
            device=batch["inputs"].device,
        )
        logits[:, self.end_token_id] = 1000.0
        outputs = {
            "logits": logits,
            "q_halt_logits": torch.zeros((batch_size,), dtype=torch.float32, device=batch["inputs"].device),
        }
        return halted_carry, outputs, cache


class AutoregressiveEvalURMTests(unittest.TestCase):
    def test_advance_positions_uses_inferred_row_width_after_first_newline(self) -> None:
        tokens = torch.tensor([2, 2, 2, 1, 1, 1, 1, 0], dtype=torch.long)
        position = torch.tensor([[6, 1, 0, 0]], dtype=torch.int32)
        row_widths = torch.tensor([0], dtype=torch.int32)
        positions = []

        for token in tokens:
            positions.append(position[0].tolist())
            append_mask = token[None] != 12
            row_widths = torch.where(
                (row_widths <= 0) & append_mask & (token[None] == 1),
                position[:, -1].to(row_widths.dtype) + 1,
                row_widths,
            )
            position = _advance_positions(
                position,
                token[None],
                newline_token_id=1,
                end_token_id=12,
                grid_height=30,
                grid_width=30,
                row_widths=row_widths,
            )

        self.assertEqual(
            positions,
            [
                [6, 1, 0, 0],
                [6, 1, 0, 1],
                [6, 1, 0, 2],
                [6, 1, 0, 3],
                [6, 1, 1, 0],
                [6, 1, 1, 1],
                [6, 1, 1, 2],
                [6, 1, 1, 3],
            ],
        )

    def test_prompt_ending_with_start_uses_prefill_logits_for_first_token(self) -> None:
        model = _ToyCachedURM(first_token_id=5, end_token_id=12)
        batch = {
            "inputs": torch.tensor([2, 3, 13], dtype=torch.int32),
            "source_inputs": torch.tensor([2, 3, 13], dtype=torch.int32),
            "prompt_position_ids": torch.tensor(
                [
                    [6, 0, 0, 0],
                    [6, 0, 0, 1],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            "seq_lengths": torch.tensor([3], dtype=torch.int32),
            "seq_offsets": torch.tensor([0, 3], dtype=torch.int32),
            "puzzle_identifiers": torch.tensor([0], dtype=torch.int64),
        }

        final_batch, preds, _metrics = _generate_urm_batch(
            model,
            batch,
            start_token_id=13,
            end_token_id=12,
            newline_token_id=1,
            max_new_tokens=5,
            cache_chunk_size=4,
        )[0]

        generated_mask = final_batch["answer_mask"].to(torch.bool)
        self.assertEqual(preds["preds"][generated_mask].tolist(), [5])
        self.assertEqual(final_batch["position_ids"][generated_mask].tolist(), [[6, 1, 0, 0]])
        self.assertEqual([tokens.tolist() for tokens in model.decode_inputs], [[5]])
        self.assertEqual([positions.tolist() for positions in model.decode_positions], [[[6, 1, 0, 0]]])


if __name__ == "__main__":
    unittest.main()
