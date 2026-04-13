from typing import Dict, List, Tuple
from dataclasses import dataclass
import os
import json
import hashlib

import numpy as np

from argdantic import ArgParser
from pydantic import BaseModel

from data.common import PuzzleDatasetMetadata, dihedral_transform, inverse_dihedral_transform


cli = ArgParser()


# ARCの元データをどのように学習用配列へ落とし込むかをまとめた設定。
# `input_file_prefix` には `..._training-challenges.json` のような
# ファイル群の共通プレフィックスを渡す想定。
class DataProcessConfig(BaseModel):
    # 入力JSONの共通プレフィックス。
    input_file_prefix: str
    # `.npy` や `dataset.json` を保存する出力先ディレクトリ。
    output_dir: str
    # 読み込むARCサブセット名の一覧。
    subsets: List[str]

    # `test` split に送るサブセット名。
    test_set_name: str

    # 再現性のための乱数シード。
    seed: int = 42
    # 1問あたりに作る拡張バリエーション数。
    num_aug: int = 1000
    # True の場合は 30x30 に固定せず、必要最小限の長さで保存する。
    no_padding: bool = True
    # 1つのターゲットを予測するのに最低限必要な文脈ペア数。
    min_context_pairs: int = 2


# ARCの各グリッドは最大 30x30 なので、その制約を定数化しておく。
ARCMaxGridSize = 30
# 拡張は重複が起きうるため、多少多めに試して十分な数を集める。
ARCAugmentRetriesFactor = 5
# 出力側の position id では、入力行と重ならないように行方向へオフセットを足す。
ARCOutputRowOffset = ARCMaxGridSize

# 拡張情報を puzzle id に埋め込むときの区切り文字。
PuzzleIdSeparator = "|||"
# 実際の puzzle id を埋め込まない代わりに、全問題共通で使う識別子。
DummyPuzzleIdentifier = 1


# 1つのARC問題を「複数の input/output ペア」と
# 「そのうち予測対象になりうるインデックス群」で表現する。
@dataclass
class ARCFullPuzzle:
    id: str
    pairs: List[Tuple[np.ndarray, np.ndarray]]
    target_indices: List[int]


def arc_grid_to_np(grid: List[List[int]]):
    # JSON 由来の二重リストを、扱いやすい uint8 の 2 次元配列へ変換する。
    arr = np.array(grid)

    # ARC は 2 次元グリッド問題なので、次元数が崩れていないか確認する。
    assert arr.ndim == 2
    # ARC の仕様上、各辺は 30 以下でなければならない。
    assert arr.shape[0] <= ARCMaxGridSize and arr.shape[1] <= ARCMaxGridSize
    # 色は 0〜9 の 10 色のみを想定する。
    assert np.all((arr >= 0) & (arr <= 9))
    return arr.astype(np.uint8)


def np_grid_to_fixed_seq_translational_augment(
    inp: np.ndarray,
    out: np.ndarray,
    do_translation: bool,
):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    # 固定長版では 30x30 キャンバス上の配置だけをランダムにずらして拡張する。
    # 入力と出力は同じ位置へ置かないと対応が壊れるため、同じオフセットを使う。
    if do_translation:
        pad_r = np.random.randint(0, ARCMaxGridSize - max(inp.shape[0], out.shape[0]) + 1)
        pad_c = np.random.randint(0, ARCMaxGridSize - max(inp.shape[1], out.shape[1]) + 1)
    else:
        pad_r = pad_c = 0

    # 入力グリッドと出力グリッドをそれぞれトークン列へ変換する。
    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        grid = np.pad(
            grid + 2,
            ((pad_r, ARCMaxGridSize - pad_r - nrow), (pad_c, ARCMaxGridSize - pad_c - ncol)),
            constant_values=0,
        )

        # 実データの右端と下端に EOS を置き、矩形の終わりを明示する。
        eos_row, eos_col = pad_r + nrow, pad_c + ncol
        if eos_row < ARCMaxGridSize:
            grid[eos_row, pad_c:eos_col] = 1
        if eos_col < ARCMaxGridSize:
            grid[pad_r:eos_row, eos_col] = 1

        # 2 次元を 1 次元へ潰して Transformer に入れやすい形へする。
        result.append(grid.flatten())

    return result


def np_grids_to_unpadded_seq(inp: np.ndarray, out: np.ndarray):
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    #
    # full-context 版では複数ペアを 1 本の列へ連結する。
    # そのため各ペアは「入力と出力を両方収められる最小キャンバス」に圧縮し、
    # 必要なときだけ EOS 用に 1 行 / 1 列を追加する。
    canvas_h = max(inp.shape[0], out.shape[0])
    canvas_w = max(inp.shape[1], out.shape[1])
    if canvas_h < ARCMaxGridSize:
        canvas_h += 1
    if canvas_w < ARCMaxGridSize:
        canvas_w += 1

    result = []
    for grid in [inp, out]:
        nrow, ncol = grid.shape
        # まずは必要最小限のキャンバスをゼロ埋めで確保する。
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        # 実際の色トークンは PAD / EOS と衝突しないように +2 している。
        canvas[:nrow, :ncol] = grid + 2

        # 実データの直後へ EOS を置いて、サイズ情報を暗黙的に伝える。
        if nrow < canvas_h:
            canvas[nrow, :ncol] = 1
        if ncol < canvas_w:
            canvas[:nrow, ncol] = 1

        result.append(canvas.flatten())

    return result, (canvas_h, canvas_w)


def grid_hash(grid: np.ndarray):
    # 拡張結果の重複判定に使うため、shape も含めてハッシュ化する。
    assert grid.ndim == 2
    assert grid.dtype == np.uint8

    buffer = [x.to_bytes(1, "big") for x in grid.shape]
    buffer.append(grid.tobytes())

    return hashlib.sha256(b"".join(buffer)).hexdigest()


def puzzle_hash(puzzle: ARCFullPuzzle):
    # ペア順序の違いでは別物とみなしたくないので、各ペアのハッシュをソートする。
    hashes = []
    for inp, out in puzzle.pairs:
        hashes.append(f"{grid_hash(inp)}|{grid_hash(out)}")

    hashes.sort()
    # どのペアがターゲット候補かもテンプレートの意味を変えるので含める。
    target_repr = ",".join(str(x) for x in puzzle.target_indices)
    return hashlib.sha256(f"{target_repr}|{'|'.join(hashes)}".encode()).hexdigest()


def aug(name: str):
    # 幾何変換と色置換をランダムに選び、同じ変換を input/output 両方へ適用する。
    trans_id = np.random.randint(0, 8)
    mapping = np.concatenate(
        [np.arange(0, 1, dtype=np.uint8), np.random.permutation(np.arange(1, 10, dtype=np.uint8))]
    )  # Permute colors, Excluding "0" (black)

    # 後から逆変換できるよう、拡張内容を id に文字列として埋め込む。
    name_with_aug_repr = (
        f"{name}{PuzzleIdSeparator}t{trans_id}{PuzzleIdSeparator}{''.join(str(x) for x in mapping)}"
    )

    def _map_grid(grid: np.ndarray):
        return dihedral_transform(mapping[grid], trans_id)

    return name_with_aug_repr, _map_grid


def inverse_aug(name: str):
    # `aug` で id に埋め込んだ変換情報を読み取り、元のグリッドへ戻す関数を返す。
    if PuzzleIdSeparator not in name:
        return name, lambda x: x

    trans_id, perm = name.split(PuzzleIdSeparator)[-2:]
    trans_id = int(trans_id[1:])  # Remove "t" letter
    inv_perm = np.argsort(list(perm)).astype(np.uint8)

    def _map_grid(grid: np.ndarray):
        return inv_perm[inverse_dihedral_transform(grid, trans_id)]

    return name.split(PuzzleIdSeparator)[0], _map_grid


def _build_train_template(name: str, train_pairs: List[Tuple[np.ndarray, np.ndarray]]):
    # train 用テンプレートでは、与えられた train ペアすべてがターゲット候補になる。
    return ARCFullPuzzle(
        id=name,
        pairs=list(train_pairs),
        target_indices=list(range(len(train_pairs))),
    )


def _build_joint_template(
    name: str,
    train_pairs: List[Tuple[np.ndarray, np.ndarray]],
    test_pairs: List[Tuple[np.ndarray, np.ndarray]],
    target_indices: List[int],
):
    # train と test を連結した full-context 用テンプレート。
    # 予測対象は通常 test 側のペアだけになる。
    return ARCFullPuzzle(
        id=name,
        pairs=[*train_pairs, *test_pairs],
        target_indices=target_indices,
    )


def convert_single_arc_puzzle(
    results: dict,
    name: str,
    puzzle: dict,
    aug_count: int,
    min_context_pairs: int,
    dest_mapping: Dict[str, Tuple[str, str]],
):
    # まず JSON の train/test を numpy 配列へ正規化する。
    train_pairs = [
        (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
        for example in puzzle.get("train", [])
    ]
    test_pairs = [
        (arc_grid_to_np(example["input"]), arc_grid_to_np(example["output"]))
        for example in puzzle.get("test", [])
    ]

    train_dest = dest_mapping["train"]
    test_dest = dest_mapping["test"]

    # 出力先 split / subset ごとに、どのテンプレートを作るか決める。
    # train と test の行き先が同じなら 1 つにまとめ、違うなら別テンプレートに分ける。
    converted: Dict[Tuple[str, str], ARCFullPuzzle] = {}
    if train_dest == test_dest:
        all_pairs = [*train_pairs, *test_pairs]
        converted[train_dest] = ARCFullPuzzle(
            id=name,
            pairs=all_pairs,
            target_indices=list(range(len(all_pairs))),
        )
    else:
        converted[train_dest] = _build_train_template(name, train_pairs)
        converted[test_dest] = _build_joint_template(
            name,
            train_pairs=train_pairs,
            test_pairs=test_pairs,
            target_indices=list(range(len(train_pairs), len(train_pairs) + len(test_pairs))),
        )

    # full-context では「文脈ペア + ターゲット 1 件」が最低限必要なので、
    # 条件を満たさないテンプレートはここで落とす。
    converted = {
        dest: template
        for dest, template in converted.items()
        if len(template.pairs) >= min_context_pairs + 1 and len(template.target_indices) > 0
    }
    if not converted:
        return False

    group = [converted]

    # 元問題 + 拡張問題群を 1 グループとして扱う。
    # 同じ本質の問題から出た派生であることを後段で保てるようにしている。
    if aug_count > 0:
        hashes = {"|".join(sorted(puzzle_hash(template) for template in converted.values()))}

        for _trial in range(ARCAugmentRetriesFactor * aug_count):
            aug_name, map_grid = aug(name)
            augmented = {
                dest: ARCFullPuzzle(
                    id=aug_name,
                    pairs=[(map_grid(inp), map_grid(out)) for inp, out in template.pairs],
                    target_indices=list(template.target_indices),
                )
                for dest, template in converted.items()
            }

            # 変換の組み合わせ次第では元と同一になりうるので、ハッシュで重複排除する。
            h = "|".join(sorted(puzzle_hash(template) for template in augmented.values()))
            if h not in hashes:
                hashes.add(h)
                group.append(augmented)

            if len(group) >= aug_count + 1:
                break

    # split / subset ごとの結果配列へ、同じグループ単位で追加する。
    for dest in converted.keys():
        dest_split, dest_set = dest

        results.setdefault(dest_split, {})
        results[dest_split].setdefault(dest_set, [])
        results[dest_split][dest_set].append([converted_map[dest] for converted_map in group])

    return True


def load_puzzles_arcagi(config: DataProcessConfig):
    # full-context 学習の都合上、最低でも「文脈 2 + 予測対象 1」は欲しい。
    if config.min_context_pairs < 2:
        raise ValueError(f"min_context_pairs must be >= 2, got {config.min_context_pairs}")

    # train 側の例は常に train/all へ送る。
    train_examples_dest = ("train", "all")
    # test 側は指定サブセットだけ test/all に送り、それ以外は train/all に混ぜる。
    test_examples_map = {
        config.test_set_name: [(1.0, ("test", "all"))],
        "_default": [(1.0, ("train", "all"))],
    }

    test_puzzles = {}
    results = {}

    total_puzzles = 0
    skipped_puzzles = 0
    for subset_name in config.subsets:
        # まず challenges を読み込み、必要なら solutions を後から差し込む。
        with open(f"{config.input_file_prefix}_{subset_name}-challenges.json", "r") as f:
            puzzles = json.load(f)
            print(f"Loaded {len(puzzles)} puzzles from {subset_name} challenges")

        sols_filename = f"{config.input_file_prefix}_{subset_name}-solutions.json"
        if os.path.isfile(sols_filename):
            with open(sols_filename, "r") as f:
                sols = json.load(f)
                print(f"Loaded {len(sols)} solutions from {subset_name} solutions")

                # challenges 側の test に解答を埋め、train と同じ形式で扱えるようにする。
                for puzzle_id in puzzles.keys():
                    for idx, sol_grid in enumerate(sols[puzzle_id]):
                        puzzles[puzzle_id]["test"][idx]["output"] = sol_grid
        else:
            # 解答ファイルがないときも処理系を単純に保つため、ダミー出力を入れて形だけ揃える。
            print(f"{subset_name} solutions not found, filling with dummy")
            for puzzle_id, puzzle in puzzles.items():
                for example in puzzle["test"]:
                    example.setdefault("output", [[0]])

        # 問題順に偏りが出ないようシャッフルしてから split を割り振る。
        puzzles = list(puzzles.items())
        print(f"Shuffling {len(puzzles)} puzzles...")
        np.random.shuffle(puzzles)

        # インデックス比率を使って、問題ごとに出力先 split を決める。
        for idx, (name, puzzle) in enumerate(puzzles):
            fraction = idx / len(puzzles)
            test_examples_dest = None
            for f, dest in test_examples_map.get(subset_name, test_examples_map["_default"]):
                if fraction < f:
                    test_examples_dest = dest
                    break

            assert test_examples_dest is not None

            converted = convert_single_arc_puzzle(
                results,
                name,
                puzzle,
                config.num_aug,
                config.min_context_pairs,
                {"train": train_examples_dest, "test": test_examples_dest},
            )
            if not converted:
                skipped_puzzles += 1
                continue

            # test split に回した元問題だけは、後で可視化や評価に使えるよう保存しておく。
            if test_examples_dest[0] == "test":
                test_puzzles[name] = puzzle

            total_puzzles += 1

    print(f"Total convertible puzzles: {total_puzzles}")
    print(f"Skipped puzzles (not enough solved pairs): {skipped_puzzles}")
    print("results keys:", results.keys())
    return results, test_puzzles


def _sample_context_indices(
    num_pairs: int,
    target_idx: int,
    min_context_pairs: int,
):
    # ターゲット自身以外から文脈候補を集める。
    candidate_indices = [idx for idx in range(num_pairs) if idx != target_idx]
    if len(candidate_indices) < min_context_pairs:
        return None

    # 文脈数は固定せず、最小値以上でランダムに揺らす。
    num_context = np.random.randint(min_context_pairs, len(candidate_indices) + 1)
    context_indices = np.random.choice(candidate_indices, size=num_context, replace=False).tolist()
    return context_indices


def _pair_seq_upper_bound(
    inp: np.ndarray,
    out: np.ndarray,
    no_padding: bool,
):
    # 固定長なら 30x30 の input と output を 1 本ずつ持つので上限は一定。
    if not no_padding:
        return 2 * ARCMaxGridSize * ARCMaxGridSize

    # 可変長なら「入力/出力の大きい方 + EOS 分」を見積もれば十分。
    canvas_h = max(inp.shape[0], out.shape[0])
    canvas_w = max(inp.shape[1], out.shape[1])
    if canvas_h < ARCMaxGridSize:
        canvas_h += 1
    if canvas_w < ARCMaxGridSize:
        canvas_w += 1
    return 2 * canvas_h * canvas_w


def _full_sample_upper_bound(group: List[ARCFullPuzzle], no_padding: bool):
    # 同一グループ内で最も長くなりうるサンプル長を保存時の上限として使う。
    if not group:
        return 0
    return max(
        sum(_pair_seq_upper_bound(inp, out, no_padding) for inp, out in puzzle.pairs)
        for puzzle in group
    )


def _make_pair_sequences(
    inp: np.ndarray,
    out: np.ndarray,
    do_translation: bool,
    no_padding: bool,
):
    # 可変長モードでは最小キャンバス、固定長モードでは 30x30 + 平行移動拡張を使う。
    if no_padding:
        (inp_seq, out_seq), pair_shape = np_grids_to_unpadded_seq(inp, out)
    else:
        inp_seq, out_seq = np_grid_to_fixed_seq_translational_augment(
            inp,
            out,
            do_translation=do_translation,
        )
        pair_shape = (ARCMaxGridSize, ARCMaxGridSize)

    return (
        inp_seq.astype(np.uint8, copy=False),
        out_seq.astype(np.uint8, copy=False),
        pair_shape,
    )


def _make_pair_position_ids(pair_shape: Tuple[int, int], example_index: int):
    # position id は [ペア番号, 行, 列] の 3 軸で持つ。
    # これにより連結後も「どのペアのどの位置か」を復元しやすい。
    pair_h, pair_w = pair_shape
    rows = np.repeat(np.arange(pair_h, dtype=np.uint8), pair_w)
    cols = np.tile(np.arange(pair_w, dtype=np.uint8), pair_h)
    depth = np.full((pair_h * pair_w,), example_index, dtype=np.uint8)

    # 出力側は入力側と行番号が衝突しないよう、行方向をオフセットしておく。
    input_position_ids = np.stack([depth, rows, cols], axis=-1)
    output_position_ids = np.stack([depth, rows + ARCOutputRowOffset, cols], axis=-1)
    return input_position_ids, output_position_ids


def _build_full_context_example(
    puzzle: ARCFullPuzzle,
    target_idx: int,
    min_context_pairs: int,
    enable_translational_augment: bool,
    no_padding: bool,
):
    # まず、このターゲットを予測するために見せる文脈ペアをサンプリングする。
    context_indices = _sample_context_indices(len(puzzle.pairs), target_idx, min_context_pairs)
    if context_indices is None:
        return None

    # ターゲットを最後に固定せず混ぜることで、位置依存の学習を避ける。
    ordered_indices = [*context_indices, target_idx]
    np.random.shuffle(ordered_indices)
    # 少なくとも 1 ペアは平行移動させず、元の配置情報も常に残す。
    no_aug_pair_pos = np.random.randint(0, len(ordered_indices))

    input_parts = []
    label_parts = []
    position_parts = []

    for pair_pos, pair_idx in enumerate(ordered_indices):
        inp, out = puzzle.pairs[pair_idx]
        do_translation = enable_translational_augment and pair_pos != no_aug_pair_pos
        inp_seq, out_seq, pair_shape = _make_pair_sequences(inp, out, do_translation, no_padding)

        # ラベルが不要な箇所は 0 で埋め、loss 計算時に無視できる形へ揃える。
        zero_inp = np.zeros_like(inp_seq, dtype=np.uint8)
        zero_out = np.zeros_like(out_seq, dtype=np.uint8)
        inp_pos_ids, out_pos_ids = _make_pair_position_ids(pair_shape, pair_pos)

        if pair_idx == target_idx:
            # ターゲット問題では、モデル入力には output を見せず、
            # そのぶんラベル側にだけ正解 output を置く。
            input_parts.extend([inp_seq, zero_out])
            label_parts.extend([zero_inp, out_seq])
            position_parts.extend([inp_pos_ids, out_pos_ids])
        else:
            # 文脈問題では input/output の両方を入力へ入れ、ラベルは空にする。
            input_parts.extend([inp_seq, out_seq])
            label_parts.extend([zero_inp, zero_out])
            position_parts.extend([inp_pos_ids, out_pos_ids])

    # 連結して 1 サンプルぶんの 1 次元列へまとめる。
    sample_input = np.concatenate(input_parts).astype(np.uint8, copy=False)
    sample_label = np.concatenate(label_parts).astype(np.uint8, copy=False)
    sample_position_ids = np.concatenate(position_parts, axis=0).astype(np.uint8, copy=False)
    # seq_shape は後段で各サンプル長を復元するために使う。
    seq_shape = (1, int(sample_input.shape[0]))
    return sample_input, sample_label, seq_shape, sample_position_ids


def convert_dataset(config: DataProcessConfig):
    # 前処理全体で乱数を使うので、最初にシードを固定する。
    np.random.seed(config.seed)
    os.makedirs(config.output_dir, exist_ok=True)

    # まず ARC 問題を読み込み、train/test ごとのテンプレート群へ変換する。
    data, test_puzzles = load_puzzles_arcagi(config)
    
    # print(data)
    print("Splits found:", data.keys())
    print("train data:", len(data["train"]["all"]))
    print("test data:", len(data["test"]["all"]))
    print("Test puzzles:", len(test_puzzles))

    # 実際の puzzle id を埋め込むと問題そのものを暗記できてしまうので、
    # 全問題共通のダミー識別子だけを使う。
    num_identifiers = 2  # 0 is blank, 1 is shared dummy

    global_seq_len = 0
    for split in data.values():
        for subset in split.values():
            for group in subset:
                # 固定長保存時に備えて、全 split を通した最大長を先に調べる。
                global_seq_len = max(global_seq_len, _full_sample_upper_bound(group, config.no_padding))
    
    print(f"Global sequence length upper bound: {global_seq_len}")

    # split ごとに実サンプルを生成し、そのまま numpy 配列として保存する。
    for split_name, split in data.items():
        print("split: ", split_name)
        os.makedirs(os.path.join(config.output_dir, split_name), exist_ok=True)

        # train のみ平行移動拡張を有効にし、test では評価の一貫性を保つ。
        enable_translational_augment = split_name == "train"

        total_examples = 0
        total_puzzles = 0
        total_groups = 0
        split_max_position_id = np.zeros((3,), dtype=np.int32)

        for subset_name, subset in split.items():
            results = {
                "inputs": [],
                "labels": [],
                "puzzle_identifiers": [],
                "puzzle_indices": [0],
                "group_indices": [0],
            }
            if config.no_padding:
                results["seq_shapes"] = []
                results["position_ids"] = []

            example_id = 0
            puzzle_id = 0

            for group in subset:
                for puzzle in group:
                    # 1つの puzzle からは、target 候補ごとに別サンプルを作る。
                    for target_idx in puzzle.target_indices:
                        built = _build_full_context_example(
                            puzzle=puzzle,
                            target_idx=target_idx,
                            min_context_pairs=config.min_context_pairs,
                            enable_translational_augment=enable_translational_augment,
                            no_padding=config.no_padding,
                        )
                        if built is None:
                            continue

                        inp, out, seq_shape, position_ids = built
                        # 可変長保存では各サンプルをいったん Python list に積み、
                        # 後でフラット化して offsets 付きで保存する。
                        results["inputs"].append(inp)
                        results["labels"].append(out)
                        if config.no_padding:
                            results["seq_shapes"].append(seq_shape)
                            results["position_ids"].append(position_ids)
                            split_max_position_id = np.maximum(
                                split_max_position_id,
                                position_ids.max(axis=0).astype(np.int32) + 1,
                            )

                        example_id += 1
                        total_examples += 1

                    # `puzzle_indices` は「どこまでが同じ puzzle 由来か」を示す境界。
                    results["puzzle_indices"].append(example_id)
                    results["puzzle_identifiers"].append(DummyPuzzleIdentifier)

                    puzzle_id += 1
                    total_puzzles += 1

                # `group_indices` は元問題 + その拡張群のまとまり境界を示す。
                results["group_indices"].append(puzzle_id)
                total_groups += 1

            for key, value in results.items():
                if key in {"inputs", "labels"}:
                    if config.no_padding:
                        # 可変長モードでは全列を連結し、offsets で各サンプル範囲を復元する。
                        seq_lengths = np.array([seq.shape[0] for seq in value], dtype=np.int64)
                        seq_offsets = np.concatenate(
                            [np.array([0], dtype=np.int64), np.cumsum(seq_lengths, dtype=np.int64)]
                        )
                        flat_tokens = (
                            np.concatenate(value).astype(np.uint8, copy=False)
                            if value
                            else np.empty((0,), dtype=np.uint8)
                        )
                        np.save(
                            os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                            flat_tokens,
                        )
                        if key == "inputs":
                            np.save(
                                os.path.join(config.output_dir, split_name, f"{subset_name}__seq_offsets.npy"),
                                seq_offsets,
                            )
                    else:
                        # 固定長モードでは最大長まで PAD して 2 次元配列で保存する。
                        padded = [
                            np.pad(seq, (0, global_seq_len - seq.shape[0]), constant_values=0)
                            for seq in value
                        ]
                        array = (
                            np.stack(padded, 0).astype(np.uint8, copy=False)
                            if padded
                            else np.empty((0, global_seq_len), dtype=np.uint8)
                        )
                        np.save(
                            os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                            array,
                        )
                elif key == "position_ids":
                    # position_ids も可変長トークン列と同じ順序でフラット化する。
                    flat_positions = (
                        np.concatenate(value, axis=0).astype(np.uint8, copy=False)
                        if value
                        else np.empty((0, 3), dtype=np.uint8)
                    )
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                        flat_positions,
                    )
                elif key == "seq_shapes":
                    # 各サンプルの元の長さを保持して、復元可能にしておく。
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                        np.array(value, dtype=np.int32),
                    )
                else:
                    # インデックス類は int32 で十分なので、そのまま配列化して保存する。
                    np.save(
                        os.path.join(config.output_dir, split_name, f"{subset_name}__{key}.npy"),
                        np.array(value, dtype=np.int32),
                    )

        # split 単位のメタデータをまとめ、学習コード側が復元しやすいようにする。
        metadata = PuzzleDatasetMetadata(
            seq_len=global_seq_len,
            vocab_size=10 + 2,  # PAD + EOS + "0" ... "9"
            pad_id=0,
            ignore_label_id=0,
            blank_identifier_id=0,
            num_puzzle_identifiers=num_identifiers,
            total_groups=total_groups,
            mean_puzzle_examples=total_examples / max(total_puzzles, 1),
            sets=list(split.keys()),
            variable_seq_lengths=config.no_padding,
            position_id_shape=split_max_position_id.tolist() if config.no_padding and total_examples > 0 else None,
        )
        print(f"  Total puzzles: {total_puzzles}")
        print(f"  Total examples: {total_examples}")
        print(f"  Total groups: {total_groups}")
        print(f"  Mean examples per puzzle: {metadata.mean_puzzle_examples:.2f}")
        print(f"  Sequence length upper bound: {metadata.seq_len}")
        if metadata.position_id_shape is not None:
            print(f"  Position ID shape: {metadata.position_id_shape}")

        with open(os.path.join(config.output_dir, split_name, "dataset.json"), "w") as f:
            json.dump(metadata.model_dump(), f)
            print(f"  Saved metadata to {f.name}")

    # 識別子語彙と test 用の元問題も合わせて保存しておく。
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>", "<shared_dummy>"], f)

    with open(os.path.join(config.output_dir, "test_puzzles.json"), "w") as f:
        json.dump(test_puzzles, f)


@cli.command(singleton=True)
def main(config: DataProcessConfig):
    convert_dataset(config)


if __name__ == "__main__":
    cli()
