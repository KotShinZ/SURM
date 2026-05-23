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


from models.urm.urm import URM  # noqa: E402


def _urm_config(**overrides):
    config = dict(
        batch_size=2,
        seq_len=4,
        puzzle_emb_ndim=0,
        num_puzzle_identifiers=1,
        vocab_size=6,
        num_layers=1,
        hidden_size=6,
        expansion=1.0,
        num_heads=1,
        pos_encodings="rope",
        loops=2,
        L_cycles=1,
        H_cycles=1,
        forward_dtype="float32",
        profile=False,
        use_act=False,
    )
    config.update(overrides)
    return config


class URMAnswerInitialTests(unittest.TestCase):
    def test_fixed_default_initial_carry_uses_original_init_hidden(self) -> None:
        model = URM(_urm_config(answer_initial_mode="default"))
        with torch.no_grad():
            model.inner.init_hidden.copy_(torch.arange(6, dtype=torch.float32))
        batch = {
            "inputs": torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.int32),
            "labels": torch.tensor([[2, -100, 4, -100], [-100, 5, 1, -100]], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros((2,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        expected = model.inner.init_hidden.expand_as(carry.current_hidden)
        torch.testing.assert_close(carry.current_hidden, expected)

    def test_fixed_default_initial_carry_uses_original_low_init_hidden(self) -> None:
        model = URM(_urm_config(answer_initial_mode="default", H_layers=1, L_layers=1))
        with torch.no_grad():
            model.inner.init_hidden.copy_(torch.arange(6, dtype=torch.float32))
            model.inner.low_init_hidden.copy_(torch.arange(6, dtype=torch.float32) + 10)
        batch = {
            "inputs": torch.tensor([[2, 2, 2, 2]], dtype=torch.int32),
            "labels": torch.tensor([[2, -100, 4, -100]], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        self.assertIsNotNone(carry.current_low_hidden)
        torch.testing.assert_close(
            carry.current_hidden,
            model.inner.init_hidden.expand_as(carry.current_hidden),
        )
        torch.testing.assert_close(
            carry.current_low_hidden,
            model.inner.low_init_hidden.expand_as(carry.current_low_hidden),
        )

    def test_packed_default_initial_carry_uses_original_init_hidden(self) -> None:
        model = URM(
            _urm_config(
                seq_len=5,
                variable_seq_lengths=True,
                answer_initial_mode="default",
            )
        )
        with torch.no_grad():
            model.inner.init_hidden.copy_(torch.arange(6, dtype=torch.float32))
        batch = {
            "inputs": torch.tensor([2, 3, 4, 5, 2], dtype=torch.int32),
            "labels": torch.tensor([2, 3, 4, 5, 2], dtype=torch.int32),
            "seq_lengths": torch.tensor([3, 2], dtype=torch.int32),
            "seq_offsets": torch.tensor([0, 3, 5], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros((2,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        expected = model.inner.init_hidden.expand_as(carry.current_hidden)
        torch.testing.assert_close(carry.current_hidden, expected)

    def test_fixed_black_initial_carry_is_zero(self) -> None:
        model = URM(_urm_config(answer_initial_mode="black"))
        batch = {
            "inputs": torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.int32),
            "labels": torch.tensor([[2, -100, 4, -100], [-100, 5, 1, -100]], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros((2,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        self.assertTrue(torch.equal(carry.current_hidden, torch.zeros_like(carry.current_hidden)))

    def test_fixed_noised_label_c_mixes_label_embeddings_on_answer_tokens(self) -> None:
        model = URM(
            _urm_config(
                answer_initial_mode="noised_label_C",
                answer_initial_C_noise_distribution="uniform",
                answer_initial_C_noise_scale=0.0,
            )
        )
        labels = torch.tensor([[2, -100, 4, -100], [-100, 5, 1, -100]], dtype=torch.int32)
        batch = {
            "inputs": torch.tensor([[2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.int32),
            "labels": labels,
            "puzzle_identifiers": torch.zeros((2,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        safe_labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
        label_embeddings = model.inner.embed_scale * model.inner.embed_tokens(safe_labels)
        valid = labels != -100
        answer_hidden = carry.current_hidden[:, : labels.shape[1]][valid]
        answer_label_embeddings = label_embeddings[valid]
        coeff = (answer_hidden * answer_label_embeddings).sum(dim=-1)
        coeff = coeff / answer_label_embeddings.square().sum(dim=-1).clamp_min(1e-12)

        self.assertTrue(torch.all(coeff >= 0.0).item())
        self.assertTrue(torch.all(coeff <= 1.0).item())
        torch.testing.assert_close(answer_hidden, coeff.unsqueeze(-1) * answer_label_embeddings)
        non_answer_hidden = carry.current_hidden[:, : labels.shape[1]][~valid]
        self.assertTrue(torch.equal(non_answer_hidden, torch.zeros_like(non_answer_hidden)))

    def test_packed_noised_label_d_replaces_with_non_special_token_embeddings(self) -> None:
        model = URM(
            _urm_config(
                seq_len=5,
                variable_seq_lengths=True,
                answer_initial_mode="noised_label_D",
                answer_initial_D_ratio_distribution="constant",
                answer_initial_D_ratio_max=1.0,
                answer_initial_random_token_min=0,
                answer_initial_random_token_max=5,
                answer_initial_pad_token_id=0,
                answer_initial_eos_token_id=1,
            )
        )
        with torch.no_grad():
            model.inner.embed_tokens.embedding_weight.copy_(torch.eye(6))

        answer_mask = torch.tensor([True, False, True, True, False], dtype=torch.bool)
        batch = {
            "inputs": torch.tensor([2, 3, 4, 5, 2], dtype=torch.int32),
            "labels": torch.tensor([2, 3, 4, 5, 2], dtype=torch.int32),
            "answer_mask": answer_mask,
            "seq_lengths": torch.tensor([3, 2], dtype=torch.int32),
            "seq_offsets": torch.tensor([0, 3, 5], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros((2,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        token_ids = torch.argmax(carry.current_hidden[answer_mask], dim=-1)
        self.assertTrue(torch.all(token_ids >= 2).item())
        self.assertTrue(torch.all(token_ids <= 5).item())
        self.assertTrue(torch.equal(carry.current_hidden[~answer_mask], torch.zeros_like(carry.current_hidden[~answer_mask])))

    def test_packed_noised_label_c_supports_answer_only_labels(self) -> None:
        model = URM(
            _urm_config(
                seq_len=5,
                variable_seq_lengths=True,
                answer_initial_mode="noised_label_C",
                answer_initial_C_noise_distribution="uniform",
                answer_initial_C_noise_scale=0.0,
            )
        )
        answer_mask = torch.tensor([True, False, True, True, False], dtype=torch.bool)
        answer_labels = torch.tensor([2, 4, 5], dtype=torch.int32)
        batch = {
            "inputs": torch.tensor([2, 3, 4, 5, 2], dtype=torch.int32),
            "labels": answer_labels,
            "answer_mask": answer_mask,
            "seq_lengths": torch.tensor([3, 2], dtype=torch.int32),
            "seq_offsets": torch.tensor([0, 3, 5], dtype=torch.int32),
            "label_seq_lengths": torch.tensor([2, 1], dtype=torch.int32),
            "label_seq_offsets": torch.tensor([0, 2, 3], dtype=torch.int32),
            "puzzle_identifiers": torch.zeros((2,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        answer_hidden = carry.current_hidden[answer_mask]
        answer_label_embeddings = model.inner.embed_scale * model.inner.embed_tokens(answer_labels)
        coeff = (answer_hidden * answer_label_embeddings).sum(dim=-1)
        coeff = coeff / answer_label_embeddings.square().sum(dim=-1).clamp_min(1e-12)

        self.assertTrue(torch.all(coeff >= 0.0).item())
        self.assertTrue(torch.all(coeff <= 1.0).item())
        torch.testing.assert_close(answer_hidden, coeff.unsqueeze(-1) * answer_label_embeddings)
        non_answer_hidden = carry.current_hidden[~answer_mask]
        self.assertTrue(torch.equal(non_answer_hidden, torch.zeros_like(non_answer_hidden)))

    def test_eval_noised_label_c_uses_continuous_noise_without_labels(self) -> None:
        model = URM(
            _urm_config(
                answer_initial_mode="noised_label_C",
                answer_initial_C_noise_distribution="uniform",
                answer_initial_C_noise_scale=0.5,
            )
        )
        model.eval()
        answer_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)
        batch = {
            "inputs": torch.tensor([[2, 2, 2, 2]], dtype=torch.int32),
            "answer_mask": answer_mask,
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        self.assertTrue(torch.all(torch.isfinite(carry.current_hidden)).item())
        self.assertTrue(torch.any(carry.current_hidden[answer_mask] != 0).item())
        self.assertTrue(torch.equal(carry.current_hidden[~answer_mask], torch.zeros_like(carry.current_hidden[~answer_mask])))

    def test_eval_noised_label_d_uses_fully_random_embeddings_without_labels(self) -> None:
        model = URM(
            _urm_config(
                answer_initial_mode="noised_label_D",
                answer_initial_D_ratio_distribution="constant",
                answer_initial_D_ratio_max=0.0,
                answer_initial_random_token_min=0,
                answer_initial_random_token_max=5,
                answer_initial_pad_token_id=0,
                answer_initial_eos_token_id=1,
            )
        )
        model.eval()
        with torch.no_grad():
            model.inner.embed_tokens.embedding_weight.copy_(torch.eye(6))

        answer_mask = torch.tensor([[True, False, True, False]], dtype=torch.bool)
        batch = {
            "inputs": torch.tensor([[2, 2, 2, 2]], dtype=torch.int32),
            "answer_mask": answer_mask,
            "puzzle_identifiers": torch.zeros((1,), dtype=torch.int64),
        }

        carry = model.initial_carry(batch)

        token_ids = torch.argmax(carry.current_hidden[answer_mask], dim=-1)
        self.assertTrue(torch.all(token_ids >= 2).item())
        self.assertTrue(torch.all(token_ids <= 5).item())
        self.assertTrue(torch.equal(carry.current_hidden[~answer_mask], torch.zeros_like(carry.current_hidden[~answer_mask])))


if __name__ == "__main__":
    unittest.main()
