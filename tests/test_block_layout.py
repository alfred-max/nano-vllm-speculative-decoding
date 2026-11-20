import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

BS = 256
PROMPT_LENS = [1, 100, 255, 256, 257, 500, 513]


def decode_slot_check(prompt_len: int, steps: int = 520):
    bm = BlockManager(8192, BS)
    seq = Sequence([0] * prompt_len)
    seq.block_table = []
    bm.allocate(seq, 0)
    for _ in range(steps):
        bm.may_append(seq, 1)
        assert len(seq.block_table) == seq.num_blocks, (
            f"prompt_len={prompt_len} len={len(seq)}: "
            f"block_table={len(seq.block_table)} != num_blocks={seq.num_blocks}"
        )
        p = len(seq) - 1
        slot = seq.block_table[-1] * BS + seq.last_block_num_tokens - 1
        expected = seq.block_table[p // BS] * BS + p % BS
        assert slot == expected, (
            f"prompt_len={prompt_len} len={len(seq)}: slot={slot} != expected={expected}"
        )
        seq.append_token(0)


def reserve_check(prompt_len: int, num_tokens: int, steps: int = 80):
    bm = BlockManager(8192, BS)
    seq = Sequence([0] * prompt_len)
    seq.block_table = []
    bm.allocate(seq, 0)
    for _ in range(steps):
        need = (len(seq) + num_tokens - 2) // BS + 1
        bm.may_append(seq, num_tokens)
        assert len(seq.block_table) >= need, (
            f"prompt_len={prompt_len} len={len(seq)} n={num_tokens}: "
            f"需要 {need} 块，只有 {len(seq.block_table)}"
        )
        seq.append_token(0)


def main():
    for L in PROMPT_LENS:
        decode_slot_check(L)
    print(f"decode slot ok: prompt lens {PROMPT_LENS}")

    for L in [1, 100, 250, 256, 257, 500]:
        for n in (1, 2, 8):
            reserve_check(L, n)
    print("reserve ok: n in {1, 2, 8}")


if __name__ == "__main__":
    main()