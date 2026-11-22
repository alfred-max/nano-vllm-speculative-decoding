"""投机解码的接受/拒绝数学（当前只支持点质量 draft 分布）。

点质量意味着 q_i(d_i) = 1，因此：
  接受概率 = p_i(d_i)
  残差分布 = p_i 把 d_i 位置置零后归一化
模块 5 接真实 drafter（稠密 q）时会扩展这里的残差计算。
"""
import math

import torch


def sample_from(probs: torch.Tensor) -> int:
    noise = torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    return int(probs.div(noise).argmax())


def verify_one(rows: torch.Tensor, drafts: list[int], draft_logprobs: list[float], temperature: float) -> list[int]:
    """rows: [1+γ, V] 目标 logits。返回本步应提交的 token 列表（≥1 个）。"""
    gamma = len(drafts)

    if temperature <= 1e-10:
        best = rows.argmax(dim=-1).tolist()      # 一次同步
        out = []
        for i, d in enumerate(drafts):
            if best[i] != d:
                out.append(best[i])
                return out
            out.append(d)
        out.append(best[gamma])
        return out

    probs = torch.softmax(rows.float() / temperature, dim=-1)
    out = []
    for i, d in enumerate(drafts):
        q = math.exp(draft_logprobs[i])
        p = float(probs[i, d])
        if float(torch.rand((), device=probs.device)) < min(1.0, p / q):
            out.append(d)
            continue
        residual = probs[i].clone()
        residual[d] = max(0.0, p - q)
        residual.clamp_min_(0)
        out.append(sample_from(residual))
        return out
    out.append(sample_from(probs[gamma]))
    return out