import torch
import time

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.spec.drafter import build_drafter
from nanovllm.spec.verify import verify_one
from dataclasses import dataclass, field


@dataclass
class SpecStats:
    steps: int = 0
    seq_steps: int = 0
    committed: int = 0
    drafted: int = 0
    accepted: int = 0

    @property
    def accept_len(self) -> float:
        """τ：平均每步每序列提交的 token 数（1 + 平均接受数）"""
        return self.committed / self.seq_steps if self.seq_steps else 0.0

    @property
    def accept_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

class SpecRuntime:

    def __init__(self, config: Config, model_runner):
        self._dbg = 0
        self.config = config
        self.model_runner = model_runner
        self.gamma = config.num_speculative_tokens
        self.needs_hidden = config.spec_method in ("dflash", "dspark")
        self.stats = SpecStats()
        if self.needs_hidden:
            from transformers import AutoConfig
            from nanovllm.spec.dflash import DFlashDrafter
            dcfg = AutoConfig.from_pretrained(config.draft_model)
            self.drafter = DFlashDrafter(
                dcfg, config.draft_model, self.gamma,
                num_blocks=config.num_kvcache_blocks,
                block_size=config.kvcache_block_size,
            )
        else:
            self.drafter = build_drafter(config)

    def ingest(self, seqs, keep=None):
        if not self.needs_hidden:
            return
        aux, positions, cu = self.model_runner.take_aux()
        if aux is None:
            return
        self.drafter.ingest(seqs, aux, positions, cu, keep=keep)

    def step(self, seqs):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        proposal = self.drafter.propose(seqs)
        torch.cuda.synchronize(); t1 = time.perf_counter()

        keep_tokens, keep_logprobs = [], []
        for seq, toks, lps in zip(seqs, proposal.token_ids, proposal.logprobs):
            budget = max(0, min(
                self.gamma,
                self.config.max_model_len - len(seq),
            ))
            seq.draft_token_ids = toks[:budget]
            keep_tokens.append(seq.draft_token_ids)
            keep_logprobs.append(lps[:budget])

        logits = self.model_runner.call("run_verify", seqs)
        torch.cuda.synchronize(); t2 = time.perf_counter()

        committed, offset = [], 0
        for seq, drafts, lps in zip(seqs, keep_tokens, keep_logprobs):
            n = 1 + len(drafts)
            committed.append(verify_one(logits[offset:offset + n], drafts, lps, seq.temperature))
            offset += n
        torch.cuda.synchronize(); t3 = time.perf_counter()

        self.ingest(seqs, keep=[len(c) for c in committed])
        torch.cuda.synchronize(); t4 = time.perf_counter()

        self._dbg += 1
        if self._dbg <= 6:
            print(f"[timing] propose={1000*(t1-t0):6.1f}  verify={1000*(t2-t1):6.1f}  "
                  f"commit={1000*(t3-t2):5.1f}  ingest={1000*(t4-t3):5.1f} ms"
                  f"graph={self.model_runner._used_graph}")
                  
        self.stats.steps += 1
        self.stats.seq_steps += len(seqs)
        for seq, c, d in zip(seqs, committed, keep_tokens):
            keep = min(len(c), seq.max_tokens - seq.num_completion_tokens)
            self.stats.committed += keep
            self.stats.drafted += len(d)
            self.stats.accepted += max(0, keep - 1)
        return committed