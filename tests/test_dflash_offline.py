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

    llm = LLM(model_path, enforce_eager=True, max_model_len=4096)
    runner = llm.model_runner
    bm = llm.scheduler.block_manager
    tok = llm.tokenizer

    dcfg = AutoConfig.from_pretrained(drafter_path)
    collector = TargetHiddenCollector(runner.model, dcfg.target_layer_ids, layer_offset)
    drafter = BlockDrafter(dcfg).to(device="cuda", dtype=torch.bfloat16)
    load_drafter(drafter, drafter_path)
    print(f"drafter 载入完成: {dcfg.num_hidden_layers} 层, block={dcfg.block_size}, "
          f"target_layers={dcfg.target_layer_ids} (offset={layer_offset})")

    total_hit = total_pos = 0
    for text in PROMPTS:
        ids = tok.encode(text)
        seq = Sequence(ids, SamplingParams(temperature=0.0))
        bm.allocate(seq, 0)
        seq.num_scheduled_tokens = seq.num_tokens
        input_ids, positions = runner.prepare_prefill([seq])
        runner.run_model(input_ids, positions, mode="prefill")
        aux = collector.take()                       # [L, 5*H]

        ctx_raw = aux[:-1].contiguous()              # 0..L-2
        anchor = torch.tensor([seq.last_token], device="cuda")
        logits = drafter.draft_block(anchor, ctx_raw, [ctx_raw.shape[0]])
        drafts = logits[0].argmax(dim=-1).tolist()

        ref = llm.generate([text], SamplingParams(temperature=0.0, max_tokens=dcfg.block_size, ignore_eos=True),
                           use_tqdm=False)[0]["token_ids"]

        hits = 0
        for d, r in zip(drafts, ref):
            if d != r:
                break
            hits += 1
        total_hit += hits
        total_pos += len(ref)

        bm.deallocate(seq)
        print(f"\nprompt: {text[:50]!r}")
        print(f"  draft : {drafts}")
        print(f"  target: {ref}")
        print(f"  命中前缀: {hits}/{len(ref)}")

    print(f"\n总计: {total_hit}/{total_pos}")


if __name__ == "__main__":
    main()