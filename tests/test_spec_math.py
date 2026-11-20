import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm.spec.verify import verify_one


def main():
    torch.manual_seed(0)

    # 1) greedy：draft 全对 → 提交 γ+1 个；draft 错一位 → 在此截断
    rows = torch.tensor([[0.0, 5.0, 1.0, 0.0]] * 3)
    assert verify_one(rows, [1, 1, 1], [0.0, 0.0, 0.0], 0.0) == [1, 1, 1, 1]
    assert verify_one(rows, [1, 2, 1], [0.0, 0.0, 0.0], 0.0) == [1, 1]

    # 2) 采样：γ=1，draft=d，接受/拒绝合起来分布必须等于 p
    rows = torch.tensor([[0.0, 1.0, 2.0, -1.0]] * 2)
    p = torch.softmax(rows[0], dim=-1)
    d = 1
    counts = torch.zeros(4)
    n = 20000
    for _ in range(n):
        out = verify_one(rows, [d], [0.0], 1.0)
        counts[out[0]] += 1
    freq = counts / n
    err = (freq - p).abs().max().item()
    print("p      =", [round(x, 4) for x in p.tolist()])
    print("freq   =", [round(x, 4) for x in freq.tolist()])
    print("max err=", round(err, 4))
    assert err < 0.02, "拒绝采样没有保持目标分布"
    print("PASS")


if __name__ == "__main__":
    main()