import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm.spec.verify import verify_one


def main():
    torch.manual_seed(0)

    # 1) greedy：γ=3 → 需要 4 行
    rows = torch.tensor([[0.0, 5.0, 1.0, 0.0]] * 4)      # argmax 每行都是 1

    # 全接受 → γ 个 draft + 1 个 bonus
    assert verify_one(rows, [1, 1, 1], [0.0, 0.0, 0.0], 0.0) == [1, 1, 1, 1]
    # 第 2 个 draft 错 → 在此截断，用 argmax 顶替
    assert verify_one(rows, [1, 2, 1], [0.0, 0.0, 0.0], 0.0) == [1, 1]
    # γ=0 退化：等价于普通 decode
    assert verify_one(rows[:1], [], [], 0.0) == [1]

    # 2) 采样：γ=1，接受/拒绝合起来分布必须等于 p
    rows = torch.tensor([[0.0, 1.0, 2.0, -1.0]] * 2)     # 2 = 1 + γ
    p = torch.softmax(rows[0], dim=-1)
    d = 1
    counts = torch.zeros(4)
    n = 20000
    for _ in range(n):
        counts[verify_one(rows, [d], [0.0], 1.0)[0]] += 1
    freq = counts / n
    err = (freq - p).abs().max().item()
    print("p      =", [round(x, 4) for x in p.tolist()])
    print("freq   =", [round(x, 4) for x in freq.tolist()])
    print("max err=", round(err, 4))
    assert err < 0.02, "拒绝采样没有保持目标分布"
    print("PASS")


if __name__ == "__main__":
    main()