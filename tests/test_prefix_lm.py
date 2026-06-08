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
        try:
            __import__("flash_attn")
            return
        except ImportError:
            try:
                __import__("flash_attn_interface")
                return
            except ImportError:
                pass

        flash_attn = types.ModuleType("flash_attn")

        def _unavailable(*args, **kwargs):
            raise RuntimeError("flash attention kernels are not needed in this unit test")

        flash_attn.flash_attn_func = _unavailable
        flash_attn.flash_attn_varlen_func = _unavailable
        sys.modules["flash_attn"] = flash_attn
        sys.modules["flash_attn_interface"] = flash_attn


_install_test_stubs()


from models.urm.urm import URM  # noqa: E402


def _urm_config(**overrides):
    config = dict(
        batch_size=2,
        seq_len=5,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=12,
        num_layers=1,
        hidden_size=4,
        expansion=1.0,
        num_heads=1,
        pos_encodings="rope",
        loops=1,
        L_cycles=1,
        H_cycles=1,
        forward_dtype="float32",
        profile=False,
        use_act=False,
        variable_seq_lengths=True,
        forward_mode="prefix_lm",
    )
    config.update(overrides)
    return config


class FakePrefixLayer:
    def __init__(self) -> None:
        self.causal_calls = []
        self.key_value_states = []

    def forward_packed(self, *, hidden_states, **_kwargs):
        return hidden_states + 1

    def forward_cross_packed(self, *, query_states, key_value_states, causal=False, **_kwargs):
        self.causal_calls.append(causal)
        self.key_value_states.append(key_value_states.clone())
        return query_states + 2


class PrefixLMTests(unittest.TestCase):
    def test_forward_mode_dispatches_packed_paths(self) -> None:
        calls = []
        model = URM(_urm_config(forward_mode="prefix_lm"))
        model.inner.forward_prefix_lm_packed = lambda carry, batch: calls.append("prefix_lm")
        model.inner.forward_answer_only_packed = lambda carry, batch: calls.append("answer_only")

        model.inner.forward_packed(None, {})
        self.assertEqual(calls, ["prefix_lm"])

        calls.clear()
        model = URM(_urm_config(forward_mode="answer_only"))
        model.inner.forward_prefix_lm_packed = lambda carry, batch: calls.append("prefix_lm")
        model.inner.forward_answer_only_packed = lambda carry, batch: calls.append("answer_only")

        model.inner.forward_packed(None, {})
        self.assertEqual(calls, ["answer_only"])

    def test_legacy_forward_flags_normalize_to_forward_mode(self) -> None:
        answer_model = URM(_urm_config(forward_mode="standard", answer_only=True))
        prefix_model = URM(_urm_config(forward_mode="standard", prefix_lm=True))

        self.assertEqual(answer_model.config.forward_mode, "answer_only")
        self.assertEqual(prefix_model.config.forward_mode, "prefix_lm")

    def test_casual_forward_mode_uses_causal_self_attention(self) -> None:
        model = URM(_urm_config(forward_mode="casual"))

        self.assertEqual(model.config.forward_mode, "casual")
        self.assertTrue(model.inner.layers[0].self_attn.causal)

    def test_causal_forward_mode_alias_normalizes_to_casual(self) -> None:
        model = URM(_urm_config(forward_mode="causal"))

        self.assertEqual(model.config.forward_mode, "casual")
        self.assertTrue(model.inner.layers[0].self_attn.causal)

    def test_prefix_lm_layers_use_context_then_causal_answer_attention(self) -> None:
        model = URM(_urm_config())
        layer = FakePrefixLayer()
        hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        cos_sin = (
            torch.ones((5, 4), dtype=torch.float32),
            torch.zeros((5, 4), dtype=torch.float32),
        )

        output = model.inner._run_prefix_lm_token_layers_packed(
            hidden_states=hidden,
            layers=[layer],
            cos_sin=cos_sin,
            context_indices=torch.tensor([0, 1, 3], dtype=torch.long),
            answer_indices=torch.tensor([2, 4], dtype=torch.long),
            cu_seqlens=torch.tensor([0, 3, 5], dtype=torch.int32),
            cu_context=torch.tensor([0, 2, 3], dtype=torch.int32),
            cu_answer=torch.tensor([0, 1, 2], dtype=torch.int32),
            max_seqlen=3,
            max_context_len=2,
            max_answer_len=1,
        )

        self.assertEqual(layer.causal_calls, [True])
        torch.testing.assert_close(output[[0, 1, 3]], hidden[[0, 1, 3]] + 1)
        torch.testing.assert_close(output[[2, 4]], hidden[[2, 4]] + 2)
        expected_kv = torch.cat(
            [
                hidden[[0, 1]] + 1,
                hidden[[2]],
                hidden[[3]] + 1,
                hidden[[4]],
            ],
            dim=0,
        )
        torch.testing.assert_close(layer.key_value_states[0], expected_kv)


if __name__ == "__main__":
    unittest.main()
