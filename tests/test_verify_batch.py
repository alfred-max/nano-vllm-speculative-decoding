"""在 B>1、n=1+γ 的情况下，比较 verify 第 0 行与 decode 参考的 logits。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

PROMPTS = [
    "The capital of France is",
    "Write a Python function that returns the nth Fibonacci number.",
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May.",
    "1+1=2, 2+2=4, 3+3=",
]


def main():
    model_path = sys.argv[1]
    gamma = int(sys.argv[2]) if len(sys.argv) > 2 else 7
    n = 1 + gamma

    llm = LLM(model_path, enforce_eager=True, max_model_len=2048,
              spec_method="naive", num_speculative_tokens=gamma, gpu_memory_utilization=0.7)
    runner = llm.model_runner
    bm = llm.scheduler.block_manager
    tok = llm.tokenizer

    seqs = []
    for text in PROMPTS:
        ids = tok.encode(text)
        seq = Sequence(ids, SamplingParams(temperature=0.0))
        bm.allocate(seq, 0)
        bm.may_append(seq, n)
        seqs.append(seq)

    # 1) decode 参考（每序列 1 行）
    input_ids, positions = runner.prepare_decode(seqs)
    dec = runner.run_model(input_ids, positions, mode="decode").float()
    reset_context()

    # 2) verify（每序列 n 行，draft 随便填）
    for seq in seqs:
        seq.draft_token_ids = [seq.last_token] * gamma
    logits = runner.run_verify(seqs).float()
    assert logits.shape[0] == len(seqs) * n, (logits.shape, len(seqs), n)

    for i, seq in enumerate(seqs):
        a, b = dec[i], logits[i * n]
        diff = (a - b).abs().max().item()
        same = int(a.argmax()) == int(b.argmax())
        print(f"seq {i}: len={len(seq)} max|Δlogit|={diff:.4f} argmax一致={same}")

    # 3) 故意换一批完全不同的 draft，行 0 不应该变
    for seq in seqs:
        seq.draft_token_ids = [(seq.last_token + 111) % 1000] * gamma
    logits2 = runner.run_verify(seqs).float()

    print("\n换 draft 后，行 0 的变化：")
    leak = 0.0
    for i in range(len(seqs)):
        d = (logits[i * n] - logits2[i * n]).abs().max().item()
        leak = max(leak, d)
        print(f"  seq {i}: max|Δ| = {d:.6f}")
    print(f"  最大泄漏差 = {leak:.6f}")

    print(f"\nγ={gamma} n={n}")


if __name__ == "__main__":
    main()