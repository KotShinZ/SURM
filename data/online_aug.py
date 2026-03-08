"""Online data augmentation applied per batch during training.

Supports ARC-AGI (seq_len=900) and Sudoku (seq_len=81).
Augmentation is applied independently to each sample in the batch.
"""

from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel

from data.common import dihedral_transform


# Token layout for ARC-AGI (30x30 grid flattened to 900 tokens):
#   0  = PAD  (outside grid region)
#   1  = EOS  (end-of-row/col marker)
#   2..11 = colors 0..9
ARC_MAX_GRID = 30
ARC_SEQ_LEN = ARC_MAX_GRID * ARC_MAX_GRID   # 900
ARC_NUM_COLORS = 10
ARC_TOKEN_OFFSET = 2   # color c  maps to token c+2

# Token layout for Sudoku (9x9 grid flattened to 81 tokens):
#   0  = PAD  (unused, no padding needed for 9x9)
#   1  = blank cell (0 in original grid)
#   2..10 = digits 1..9
SUDOKU_GRID = 9
SUDOKU_SEQ_LEN = SUDOKU_GRID * SUDOKU_GRID  # 81
SUDOKU_NUM_DIGITS = 9
SUDOKU_TOKEN_OFFSET = 2   # digit d maps to token d+1 (blank=1, digit 1=2, ..., digit 9=10)

# Value used to mask loss at padding positions (set by _collate_batch)
IGNORE_LABEL = -100


class OnlineAugConfig(BaseModel):
    """Configuration for online augmentation applied at training time."""

    enabled: bool = False

    # ── ARC-AGI ──────────────────────────────────────────────────────────────
    # Apply a uniformly random dihedral symmetry (8 rotations/reflections).
    arc_dihedral: bool = True
    # Randomly permute the 10 ARC color tokens (0-9 → some permutation of 0-9).
    arc_color_perm: bool = True

    # ── Sudoku ───────────────────────────────────────────────────────────────
    # Randomly permute digit tokens 1-9.
    sudoku_digit_perm: bool = True
    # Randomly permute rows and columns within their 3x3 bands/stacks.
    sudoku_row_col_perm: bool = True
    # Randomly transpose the board (swap rows ↔ columns).
    sudoku_transpose: bool = True


# ---------------------------------------------------------------------------
# Dataset-type detection
# ---------------------------------------------------------------------------

def _detect_dataset_type(seq_len: int) -> str:
    if seq_len == ARC_SEQ_LEN:
        return "arc"
    if seq_len == SUDOKU_SEQ_LEN:
        return "sudoku"
    return "unknown"


# ---------------------------------------------------------------------------
# ARC-AGI augmentation
# ---------------------------------------------------------------------------

def _aug_arc_sample(
    inp: np.ndarray, lbl: np.ndarray, config: OnlineAugConfig
):
    """Augment one ARC-AGI sample.

    Parameters
    ----------
    inp : (900,) int32  – token values in {0=PAD, 1=EOS, 2..11=colors}
    lbl : (900,) int32  – token values in {IGNORE_LABEL, 1=EOS, 2..11=colors}

    Returns flattened (inp, lbl) after the same random transform.
    """
    inp2d = inp.reshape(ARC_MAX_GRID, ARC_MAX_GRID)
    lbl2d = lbl.reshape(ARC_MAX_GRID, ARC_MAX_GRID)

    # 1) Dihedral symmetry (8 possibilities)
    if config.arc_dihedral:
        tid = np.random.randint(0, 8)
        inp2d = dihedral_transform(inp2d, tid)
        lbl2d = dihedral_transform(lbl2d, tid)

    # 2) Color permutation
    # Build a full-token permutation table:
    #   perm[0]=0 (PAD→PAD), perm[1]=1 (EOS→EOS), perm[2..11]=shuffled colors
    if config.arc_color_perm:
        color_shuffle = np.random.permutation(ARC_NUM_COLORS).astype(np.int32) + ARC_TOKEN_OFFSET
        perm = np.empty(ARC_TOKEN_OFFSET + ARC_NUM_COLORS, dtype=np.int32)
        perm[0] = 0
        perm[1] = 1
        perm[ARC_TOKEN_OFFSET:] = color_shuffle

        inp2d = perm[inp2d]  # inp has no IGNORE_LABEL values

        # For labels, guard against IGNORE_LABEL (-100) before indexing
        valid = lbl2d >= 0
        lbl2d = np.where(valid, perm[np.clip(lbl2d, 0, ARC_TOKEN_OFFSET + ARC_NUM_COLORS - 1)], lbl2d)

    return inp2d.flatten(), lbl2d.flatten()


# ---------------------------------------------------------------------------
# Sudoku augmentation
# ---------------------------------------------------------------------------

def _aug_sudoku_sample(
    inp: np.ndarray, lbl: np.ndarray, config: OnlineAugConfig
):
    """Augment one Sudoku sample.

    Parameters
    ----------
    inp : (81,) int32  – token values in {0=PAD, 1=blank, 2..10=digits 1..9}
    lbl : (81,) int32  – token values in {IGNORE_LABEL, 2..10=digits 1..9}

    Returns flattened (inp, lbl) after the same random transform.
    """
    inp2d = inp.reshape(SUDOKU_GRID, SUDOKU_GRID).copy()
    lbl2d = lbl.reshape(SUDOKU_GRID, SUDOKU_GRID).copy()

    # 1) Random transpose (swap rows ↔ columns)
    if config.sudoku_transpose and np.random.rand() < 0.5:
        inp2d = inp2d.T.copy()
        lbl2d = lbl2d.T.copy()

    # 2) Row/column band permutation
    # Shuffle the 3 bands (rows) and within each band shuffle its 3 rows;
    # similarly for stacks (column bands).
    if config.sudoku_row_col_perm:
        bands = np.random.permutation(3)
        row_perm = np.concatenate([b * 3 + np.random.permutation(3) for b in bands])
        stacks = np.random.permutation(3)
        col_perm = np.concatenate([s * 3 + np.random.permutation(3) for s in stacks])
        inp2d = inp2d[row_perm][:, col_perm]
        lbl2d = lbl2d[row_perm][:, col_perm]

    # 3) Digit permutation
    # Build a full-token permutation table:
    #   perm[0]=0 (PAD→PAD), perm[1]=1 (blank→blank),
    #   perm[2..10]=shuffled digits
    if config.sudoku_digit_perm:
        digit_shuffle = np.random.permutation(SUDOKU_NUM_DIGITS).astype(np.int32) + SUDOKU_TOKEN_OFFSET
        perm = np.empty(SUDOKU_TOKEN_OFFSET + SUDOKU_NUM_DIGITS, dtype=np.int32)
        perm[0] = 0
        perm[1] = 1
        perm[SUDOKU_TOKEN_OFFSET:] = digit_shuffle

        inp2d = perm[inp2d]  # inp values are 0..10, no negatives

        # Guard labels against IGNORE_LABEL
        valid = lbl2d >= 0
        lbl2d = np.where(valid, perm[np.clip(lbl2d, 0, SUDOKU_TOKEN_OFFSET + SUDOKU_NUM_DIGITS - 1)], lbl2d)

    return inp2d.flatten(), lbl2d.flatten()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def apply_online_aug(batch: dict, seq_len: int, config: OnlineAugConfig) -> dict:
    """Apply per-sample random augmentation to a training batch.

    Only modifies ``inputs`` and ``labels``; ``puzzle_identifiers`` and index
    tensors are left unchanged.

    Parameters
    ----------
    batch     : dict of CPU int32 tensors returned by PuzzleDataset
    seq_len   : sequence length from dataset metadata (used to infer task type)
    config    : augmentation settings

    Returns the (possibly modified) batch dict.
    """
    if not config.enabled:
        return batch

    dataset_type = _detect_dataset_type(seq_len)
    if dataset_type == "unknown":
        return batch

    aug_fn = _aug_arc_sample if dataset_type == "arc" else _aug_sudoku_sample

    inputs = batch["inputs"].numpy().copy()   # (B, seq_len) int32
    labels = batch["labels"].numpy().copy()   # (B, seq_len) int32

    for i in range(inputs.shape[0]):
        inputs[i], labels[i] = aug_fn(inputs[i], labels[i], config)

    batch = dict(batch)
    batch["inputs"] = torch.from_numpy(inputs)
    batch["labels"] = torch.from_numpy(labels)
    return batch
