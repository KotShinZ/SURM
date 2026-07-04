from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F


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


from models.layers import DepthwiseConv2DAttention  # noqa: E402
from models.urm.urm import URMBlock, URMConfig  # noqa: E402


def _urm_config(**overrides):
    config = dict(
        batch_size=2,
        seq_len=6,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=12,
        num_layers=1,
        hidden_size=4,
        expansion=1.0,
        num_heads=1,
        pos_encodings="rope",
        grid_height=2,
        grid_width=3,
        attention_type="conv",
        attention_window_size_2d=1,
        loops=1,
        L_cycles=1,
        H_cycles=1,
        forward_dtype="float32",
        profile=False,
        use_act=False,
    )
    config.update(overrides)
    return URMConfig(**config)


class URMConvAttentionTests(unittest.TestCase):
    def test_conv_attention_type_builds_depthwise_conv2d(self) -> None:
        config = _urm_config(hidden_size=4, num_memory_tokens=1)
        block = URMBlock(config, attention_window_size=-1)

        self.assertIsInstance(block.self_attn, DepthwiseConv2DAttention)
        self.assertEqual(block.self_attn.attention_type, "conv")
        self.assertEqual(block.self_attn.dwconv.groups, config.hidden_size)
        self.assertEqual(block.self_attn.dwconv.kernel_size, (3, 3))
        self.assertEqual(block.self_attn.dwconv.padding, (1, 1))

    def test_forward_applies_depthwise_conv_to_grid_tokens_only(self) -> None:
        module = DepthwiseConv2DAttention(
            hidden_size=2,
            grid_height=2,
            grid_width=2,
            prefix_seq_len=1,
            attention_window_size_2d=1,
        )
        with torch.no_grad():
            module.dwconv.weight.copy_(
                torch.arange(18, dtype=torch.float32).view(2, 1, 3, 3) / 10.0
            )
            module.dwconv.bias.copy_(torch.tensor([0.5, -0.25], dtype=torch.float32))

        hidden_states = torch.arange(10, dtype=torch.float32).view(1, 5, 2)
        output = module(cos_sin=None, hidden_states=hidden_states)

        grid_states = hidden_states[:, 1:].view(1, 2, 2, 2).permute(0, 3, 1, 2)
        expected_grid = F.conv2d(
            grid_states,
            module.dwconv.weight,
            bias=module.dwconv.bias,
            padding=module.dwconv.padding,
            groups=2,
        ).permute(0, 2, 3, 1).reshape(1, 4, 2)

        torch.testing.assert_close(output[:, :1], torch.zeros_like(output[:, :1]))
        torch.testing.assert_close(output[:, 1:], expected_grid)

    def test_forward_packed_matches_padded_forward(self) -> None:
        module = DepthwiseConv2DAttention(
            hidden_size=3,
            grid_height=2,
            grid_width=2,
            prefix_seq_len=1,
            attention_window_size_2d=1,
        )
        hidden_states = torch.arange(42, dtype=torch.float32).view(14, 3) / 10.0
        cu_seqlens = torch.tensor([0, 5, 14], dtype=torch.int32)
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = 9

        packed_output, key, value = module.forward_packed(
            cos_sin=None,
            hidden_states=hidden_states,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        padded = hidden_states.new_zeros((2, max_seqlen, 3))
        token_mask = torch.arange(max_seqlen).unsqueeze(0) < lengths.unsqueeze(1)
        padded[token_mask] = hidden_states
        expected = module(cos_sin=None, hidden_states=padded, sequence_lengths=lengths)

        torch.testing.assert_close(packed_output, expected[token_mask])
        self.assertEqual(tuple(key.shape), (14, 0, 0))
        self.assertEqual(tuple(value.shape), (14, 0, 0))


if __name__ == "__main__":
    unittest.main()
