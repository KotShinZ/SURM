from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _install_test_stubs() -> None:
    if "flash_attn" not in sys.modules:
        flash_attn = types.ModuleType("flash_attn")

        def _unavailable(*args, **kwargs):
            raise RuntimeError("flash attention kernels are not needed in this unit test")

        flash_attn.flash_attn_func = _unavailable
        flash_attn.flash_attn_varlen_func = _unavailable
        sys.modules["flash_attn"] = flash_attn
        sys.modules["flash_attn_interface"] = flash_attn


_install_test_stubs()


from models.common import packed_norm_ratio_from_lengths  # noqa: E402
from models.trm.trm import TRM  # noqa: E402
from models.urm.urm import URM  # noqa: E402


def _manual_packed_norm_ratio(
    x1: torch.Tensor,
    x2: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    values = []
    offset = 0
    for length in lengths.tolist():
        next_offset = offset + int(length)
        left = x1[offset:next_offset]
        right = x2[offset:next_offset]
        values.append(torch.norm(left - right) / (1e-7 + torch.norm(left + right) / 2))
        offset = next_offset
    return torch.stack(values)


class PackedNormRatioTests(unittest.TestCase):
    def test_helper_matches_manual_packed_loop(self) -> None:
        lengths = torch.tensor([3, 1, 4], dtype=torch.int64)
        total_tokens = int(lengths.sum().item())
        x1 = torch.arange(total_tokens * 4, dtype=torch.float32).reshape(total_tokens, 4) / 10
        x2 = torch.flip(x1, dims=[0]) + 0.25

        actual = packed_norm_ratio_from_lengths(x1, x2, lengths)
        expected = _manual_packed_norm_ratio(x1, x2, lengths)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_helper_handles_empty_segments(self) -> None:
        lengths = torch.zeros((0,), dtype=torch.int64)
        x1 = torch.empty((0, 4), dtype=torch.float32)
        x2 = torch.empty((0, 4), dtype=torch.float32)

        actual = packed_norm_ratio_from_lengths(x1, x2, lengths)

        self.assertEqual(tuple(actual.shape), (0,))
        self.assertEqual(actual.dtype, torch.float32)

    def test_urm_norm_func_matches_manual_for_packed_hidden(self) -> None:
        model = URM(
            dict(
                batch_size=2,
                seq_len=8,
                puzzle_emb_ndim=6,
                num_puzzle_identifiers=8,
                vocab_size=12,
                num_layers=1,
                hidden_size=4,
                expansion=2.0,
                num_heads=1,
                pos_encodings="rope",
                loops=2,
                L_cycles=1,
                H_cycles=1,
                variable_seq_lengths=True,
            )
        )
        seq_lengths = torch.tensor([2, 3], dtype=torch.int32)
        packed_lengths = seq_lengths.to(torch.int64) + model.inner.puzzle_emb_len
        total_tokens = int(packed_lengths.sum().item())
        x1 = torch.arange(total_tokens * 4, dtype=torch.float32).reshape(total_tokens, 4) / 7
        x2 = x1 * 0.5 + 1.0

        actual = model.norm_func(x1, x2, seq_lengths)
        expected = _manual_packed_norm_ratio(x1, x2, packed_lengths)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_trm_norm_func_matches_manual_for_packed_hidden(self) -> None:
        model = TRM(
            dict(
                batch_size=2,
                seq_len=8,
                puzzle_emb_ndim=6,
                num_puzzle_identifiers=8,
                vocab_size=12,
                H_cycles=1,
                L_cycles=1,
                H_layers=1,
                L_layers=1,
                hidden_size=4,
                expansion=2.0,
                num_heads=1,
                pos_encodings="rope",
                halt_max_steps=2,
                halt_exploration_prob=0.0,
                puzzle_emb_len=0,
                variable_seq_lengths=True,
            )
        )
        seq_lengths = torch.tensor([1, 4], dtype=torch.int32)
        packed_lengths = seq_lengths.to(torch.int64) + model.inner.puzzle_emb_len
        total_tokens = int(packed_lengths.sum().item())
        x1 = torch.arange(total_tokens * 4, dtype=torch.float32).reshape(total_tokens, 4) / 5
        x2 = x1 + torch.linspace(0.1, 0.9, steps=total_tokens, dtype=torch.float32).unsqueeze(-1)

        actual = model.norm_func(x1, x2, seq_lengths)
        expected = _manual_packed_norm_ratio(x1, x2, packed_lengths)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
