"""模块 3 验收：verify 前向 == 逐 token decode 前向（逐行对比 logits）。

用法：
  python tests/test_verify_forward.py /path/to/Qwen3-8B 3
  python tests/test_verify_forward.py /path/to/Qwen3-8B 1
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Q: 1+1=2, 2+2=4, 3+3=",
]
ATOL = 5e-2


def decode_rows(runner, seq, draft):
    rows = []
    for i in range(len(draft) + 1):
        input_ids, positions = runner.prepare_decode([seq])
        rows.append(runner.run_model(input_ids, positions, mode="decode")[0].float().clone())
        if i < len(draft):
            seq.append_token(draft[i])
    return rows


def main():
    model = sys.argv[1]
    gamma = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    llm = LLM(model, enforce_eager=True, max_model_len=2048, num_speculative_tokens=gamma)
    runner = llm.model_runner
    bm = llm.scheduler.block_manager
    tok = llm.tokenizer

    ok = True
    for text in PROMPTS:
        ids = tok.encode(text)
        draft = ids[-gamma:] if len(ids) >= gamma else [ids[0]] * gamma

        seq_a = Sequence(ids, SamplingParams(temperature=0.0))
        bm.allocate(seq_a, 0)
        bm.may_append(seq_a, gamma + 1)
        rows = decode_rows(runner, seq_a, draft)

        seq_b = Sequence(ids, SamplingParams(temperature=0.0))
        bm.allocate(seq_b, 0)
        bm.may_append(seq_b, gamma + 1)
        seq_b.draft_token_ids = draft
        verify = runner.run_verify([seq_b]).float()

        assert verify.shape[0] == gamma + 1, f"logits 行数应为 {gamma + 1}，实际 {verify.shape}"

        diffs = [(verify[i] - rows[i]).abs().max().item() for i in range(gamma + 1)]
        top1 = [verify[i].argmax().item() == rows[i].argmax().item() for i in range(gamma + 1)]
        print(f"\n[{text!r}]  draft={draft}")
        print(f"  max|Δlogit| per row = {[round(d, 4) for d in diffs]}")
        print(f"  argmax 一致 per row = {top1}")

        if max(diffs) > ATOL or not all(top1):
            ok = False
            print("  ✗ 不一致")
        else:
            print("  ✓ 一致")

        bm.deallocate(seq_a)
        bm.deallocate(seq_b)

    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()