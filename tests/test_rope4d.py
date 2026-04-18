from __future__ import annotations

import importlib
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

    if "yaml" not in sys.modules:
        yaml = types.ModuleType("yaml")
        yaml.safe_load = lambda *args, **kwargs: {}
        yaml.safe_dump = lambda *args, **kwargs: None
        sys.modules["yaml"] = yaml

    if "coolname" not in sys.modules:
        coolname = types.ModuleType("coolname")
        coolname.generate_slug = lambda *args, **kwargs: "test-run"
        sys.modules["coolname"] = coolname

    if "hydra" not in sys.modules:
        hydra = types.ModuleType("hydra")
        hydra.main = lambda *args, **kwargs: (lambda fn: fn)
        sys.modules["hydra"] = hydra

    if "adam_atan2_pytorch" not in sys.modules:
        adam_mod = types.ModuleType("adam_atan2_pytorch")

        class AdamAtan2:
            def __init__(self, *args, **kwargs):
                pass

        adam_mod.AdamAtan2 = AdamAtan2
        sys.modules["adam_atan2_pytorch"] = adam_mod

    if "omegaconf" not in sys.modules:
        omegaconf = types.ModuleType("omegaconf")

        class DictConfig(dict):
            pass

        class OmegaConf:
            @staticmethod
            def load(*args, **kwargs):
                return {}

            @staticmethod
            def to_container(value, resolve: bool = True):
                if isinstance(value, dict):
                    return dict(value)
                return value

        omegaconf.DictConfig = DictConfig
        omegaconf.OmegaConf = OmegaConf
        sys.modules["omegaconf"] = omegaconf

    if "tqdm" not in sys.modules:
        tqdm = types.ModuleType("tqdm")
        tqdm.tqdm = lambda value=None, *args, **kwargs: value
        sys.modules["tqdm"] = tqdm

    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")

        class Settings:
            def __init__(self, *args, **kwargs):
                pass

        wandb.Settings = Settings
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb


_install_test_stubs()


from models.layers import RotaryEmbedding4D  # noqa: E402
from models.urm.urm import URMConfig, URM_Inner  # noqa: E402

pretrain = importlib.import_module("pretrain")


class Rope4DTests(unittest.TestCase):
    def test_pretrain_maps_4d_position_shape_into_model_config(self) -> None:
        model_cfg = {"grid_width": 99}
        resolved = pretrain._apply_position_id_shape_to_model_cfg(model_cfg, [13, 2, 30, 30])

        self.assertEqual(resolved["grid_depth"], 13)
        self.assertEqual(resolved["grid_io"], 2)
        self.assertEqual(resolved["grid_height"], 30)
        self.assertEqual(resolved["grid_width"], 99)

    def test_rotary_embedding_4d_prefix_and_lookup_shapes(self) -> None:
        rope = RotaryEmbedding4D(
            dim=16,
            grid_depth=13,
            grid_io=2,
            grid_height=30,
            grid_width=30,
            puzzle_emb_len=1,
            base=10_000.0,
        )
        position_ids = torch.tensor(
            [
                [
                    [0, 0, 0, 0],
                    [12, 1, 29, 29],
                ]
            ],
            dtype=torch.int32,
        )

        cos_sin = rope(position_ids, prefix_seq_len=1)

        self.assertEqual(len(cos_sin), 8)
        self.assertEqual(tuple(cos_sin[0].shape), (1, 3, rope.axis_dims[0]))
        self.assertEqual(tuple(cos_sin[2].shape), (1, 3, rope.axis_dims[1]))
        self.assertEqual(tuple(cos_sin[4].shape), (1, 3, rope.axis_dims[2]))
        self.assertEqual(tuple(cos_sin[6].shape), (1, 3, rope.axis_dims[3]))

        torch.testing.assert_close(cos_sin[0][0, 0], torch.ones_like(cos_sin[0][0, 0]))
        torch.testing.assert_close(cos_sin[1][0, 0], torch.zeros_like(cos_sin[1][0, 0]))
        torch.testing.assert_close(cos_sin[0][0, 1], rope.cos_depth[0])
        torch.testing.assert_close(cos_sin[2][0, 2], rope.cos_io[1])
        torch.testing.assert_close(cos_sin[4][0, 2], rope.cos_row[29])
        torch.testing.assert_close(cos_sin[6][0, 2], rope.cos_col[29])

    def test_urm_inner_selects_rope4d_from_4d_position_ids(self) -> None:
        config = URMConfig(
            batch_size=1,
            seq_len=13 * 2 * 4 * 5,
            puzzle_emb_ndim=0,
            num_puzzle_identifiers=2,
            vocab_size=12,
            num_layers=1,
            hidden_size=16,
            expansion=1.0,
            num_heads=1,
            pos_encodings="rope",
            grid_depth=13,
            grid_io=2,
            grid_height=4,
            grid_width=5,
            loops=1,
            L_cycles=1,
            H_cycles=1,
            use_act=False,
            profile=False,
        )
        inner = URM_Inner(config)
        self.assertIsInstance(inner.rotary_emb, RotaryEmbedding4D)

        position_ids = torch.tensor(
            [
                [
                    [0, 0, 0, 0],
                    [12, 1, 3, 4],
                ]
            ],
            dtype=torch.int32,
        )

        cos_sin = inner._rotary_cos_sin({"position_ids": position_ids})

        self.assertEqual(len(cos_sin), 8)
        torch.testing.assert_close(cos_sin[0][0, 0], inner.rotary_emb.cos_depth[0])
        torch.testing.assert_close(cos_sin[2][0, 1], inner.rotary_emb.cos_io[1])
        torch.testing.assert_close(cos_sin[4][0, 1], inner.rotary_emb.cos_row[3])
        torch.testing.assert_close(cos_sin[6][0, 1], inner.rotary_emb.cos_col[4])


if __name__ == "__main__":
    unittest.main()
