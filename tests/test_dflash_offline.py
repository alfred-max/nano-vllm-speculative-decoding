"""DFlash 离线验证：drafter 一次猜 7 个，与目标模型的手工贪心解码对比。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.target_hidden import TargetHiddenCollector
from nanovllm.models.block_drafter import BlockDrafter
from nanovllm.utils.loader import load_drafter

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Write a Python function that returns the nth Fibonacci number.",
    "The capital of France is",
]


def main():
    model_path, drafter_path = sys.argv[1], sys.argv[2]
    layer_offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    use_chat = len(sys.argv) > 4 and sys.argv[4] == "chat"

    llm = LLM(model_path, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.7)
    runner = llm.model_runner
    bm = llm.scheduler.block_manager
    tok = llm.tokenizer

    # 0) 引擎自检：必须在任何手工调用之前跑
    sanity = llm.generate(["The capital of France is"],
                          SamplingParams(temperature=0.0, max_tokens=8), use_tqdm=False)
    print("引擎自检 token:", sanity[0]["token_ids"])
    print("引擎自检文本:", repr(sanity[0]["text"]))
    print()

    dcfg = AutoConfig.from_pretrained(drafter_path)
    collector = TargetHiddenCollector(runner.model, dcfg.target_layer_ids, layer_offset)
    drafter = BlockDrafter(dcfg).to(device="cuda", dtype=torch.bfloat16)
    load_drafter(drafter, drafter_path)
    W = dcfg.block_size

    total_hit = total_pos = 0
    for text in PROMPTS:
        if use_chat:
            chat_text = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,      # ← 关键
            )
            prompt_ids = tok(chat_text, add_special_tokens=False)["input_ids"]   # 再自己编码
        else:
            prompt_ids = tok.encode(text)
        if use_chat and text == PROMPTS[0]:
            print("chat 模板前 30 个 token:", prompt_ids[:30])
            print("chat 模板解码:", repr(tok.decode(prompt_ids[-40:])))
            print("模板结尾:", repr(tok.decode(prompt_ids[-12:])))
        seq = Sequence(prompt_ids, SamplingParams(temperature=0.0))
        bm.allocate(seq, 0)

        # 1) prefill：拿最后位置的 logits + 5 层 hidden
        seq.num_scheduled_tokens = seq.num_tokens
        input_ids, positions = runner.prepare_prefill([seq])
        logits0 = runner.run_model(input_ids, positions, mode="prefill")
        aux = collector.take()
        print(f"[diag] logits0 argmax={int(logits0[0].argmax())} "
              f"max={float(logits0.max()):.2f} nan={bool(torch.isnan(logits0).any())}")
        print(f"[diag] aux shape={tuple(aux.shape)} "
              f"absmean={float(aux.abs().mean()):.4f} nan={bool(torch.isnan(aux).any())}")

        # 2) 参考值：手工逐步贪心解码 W 步（不走 generate）
        ref = []
        for _ in range(W):
            bm.may_append(seq, 1)
            ids, pos = runner.prepare_decode([seq])
            lg = runner.run_model(ids, pos, mode="decode")
            t = int(lg[0].argmax())
            ref.append(t)
            seq.append_token(t)

        # 3) drafter 一次猜 W 个
        ctx_raw = aux[:-1].contiguous()
        anchor = torch.tensor([prompt_ids[-1]], device="cuda")
        dlogits = drafter.draft_block(anchor, ctx_raw, [ctx_raw.shape[0]])
        drafts = dlogits[0].argmax(dim=-1).tolist()
        print(f"[diag] drafter logits max={float(dlogits.max()):.2f} "
              f"nan={bool(torch.isnan(dlogits).any())}")

        hits = 0
        for d, r in zip(drafts, ref):
            if d != r:
                break
            hits += 1
        total_hit += hits
        total_pos += len(ref)
        bm.deallocate(seq)

        print(f"prompt: {text[:40]!r}")
        print(f"  draft : {drafts}")
        print(f"  target: {ref}")
        print(f"  命中: {hits}/{W}\n")

    print(f"总计: {total_hit}/{total_pos}")


if __name__ == "__main__":
    main()