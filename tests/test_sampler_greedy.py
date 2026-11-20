import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 仓库根目录

import torch

from nanovllm.layers.sampler import Sampler


def main():
    torch.manual_seed(0)
    sampler = Sampler()
    logits = torch.randn(8, 128)

    out = sampler(logits, torch.zeros(8))
    argmax = logits.argmax(dim=-1)
    assert torch.equal(out, argmax), "greedy 路径不等于 argmax"
    assert out.dtype == torch.int64

    temps = torch.tensor([0.0, 0.7, 0.0, 1.0, 0.0, 0.3, 0.0, 0.9])
    mixed = sampler(logits, temps)
    assert mixed.shape == (8,)
    for i in (0, 2, 4, 6):
        assert mixed[i].item() == argmax[i].item(), f"第 {i} 行应该是 greedy"
    for i in (1, 3, 5, 7):
        assert mixed[i].item() != argmax[i].item() or True  # 只验证不崩

    repeated = sampler(logits, torch.zeros(8))
    assert torch.equal(out, repeated), "greedy 不可复现"
    print("sampler greedy tests passed")


if __name__ == "__main__":
    main()