import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from transformers import AutoConfig, AutoTokenizer

from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.target_hidden import TargetHiddenCollector
from nanovllm.spec.dflash import DFlashDrafter

PROMPTS = [
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
    "Write a Python function that returns the nth Fibonacci number.",
    "The capital of France is",
]


def main():
    model_path, drafter_path = sys.argv[1], sys.argv[2]

    llm = LLM(model_path, enforce_eager=True, max_model_len=4096, gpu_memory_utilization=0.7)
    runner = llm.model_runner
    bm = llm.scheduler.block_manager
    tok = llm.tokenizer

    dcfg = AutoConfig.from_pretrained(drafter_path)
    collector = TargetHiddenCollector(runner.model, dcfg.target_layer_ids, 0)
    drafter = DFlashDrafter(dcfg, drafter_path, dcfg.block_size, num_blocks=256)
    W = dcfg.block_size

    total_hit = 0
    for text in PROMPTS:
        chat = tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True, tokenize=False,
                                       enable_thinking=False)
        prompt_ids = tok(chat, add_special_tokens=False)["input_ids"]
        seq = Sequence(prompt_ids, SamplingParams(temperature=0.0))
        bm.allocate(seq, 0)

        # 目标模型 prefill，拿 hidden
        seq.num_scheduled_tokens = seq.num_tokens
        input_ids, positions = runner.prepare_prefill([seq])
        runner.run_model(input_ids, positions, mode="prefill")
        aux = collector.take()

        # 写进 drafter cache
        drafter.ingest([seq], aux, positions, [0, len(positions)])

        # ★ 先猜，此时 len(seq) 还是 prompt 长度，anchor 是最后一个 prompt token
        assert len(seq) == len(prompt_ids), "propose 必须在序列被改长之前调用"
        proposal = drafter.propose([seq])

        # 再造参考值（会改长 seq，但已经不影响上面了）
        ref = []
        for _ in range(W):
            bm.may_append(seq, 1)
            ids, pos = runner.prepare_decode([seq])
            lg = runner.run_model(ids, pos, mode="decode")
            t = int(lg[0].argmax())
            ref.append(t)
            seq.append_token(t)

        hits = 0
        for d, r in zip(proposal.token_ids[0], ref):
            if d != r:
                break
            hits += 1
        total_hit += hits

        bm.deallocate(seq)
        print(f"\nprompt: {text[:40]!r}")
        print(f"  draft : {proposal.token_ids[0]}")
        print(f"  target: {ref}")
        print(f"  命中: {hits}/{W}")

    print(f"\n总计: {total_hit}/21")


if __name__ == "__main__":
    main()