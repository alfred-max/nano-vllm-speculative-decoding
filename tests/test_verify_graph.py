import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from nanovllm import LLM, SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context

def main():
    model_path = sys.argv[1]
    gamma = 7
    llm = LLM(model_path, enforce_eager=False, max_model_len=2048,
              spec_method="naive", num_speculative_tokens=gamma, gpu_memory_utilization=0.7)
    runner = llm.model_runner
    bm = llm.scheduler.block_manager
    tok = llm.tokenizer

    seqs = []
    for text in ["The capital of France is", "Write a Python function that returns"]:
        seq = Sequence(tok.encode(text), SamplingParams(temperature=0.0))
        bm.allocate(seq, 0)
        bm.may_append(seq, 1 + gamma)
        seq.draft_token_ids = [seq.last_token] * gamma
        seqs.append(seq)

    # eager 路径
    runner.verify_graphs = {}       # 临时禁用 graph
    ids, pos = runner.prepare_verify(seqs)
    eager = runner.run_model(ids, pos, mode="verify").float().clone()
    reset_context()

    # graph 路径
    llm2_graphs = runner.verify_graphs
    ...