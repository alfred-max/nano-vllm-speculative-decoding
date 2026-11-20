import torch

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
        self.config = config
        self.model_runner = model_runner
        self.gamma = config.num_speculative_tokens
        self.drafter = build_drafter(config)
        self.stats = SpecStats()

    def step(self, seqs: list[Sequence]) -> list[list[int]]:
        proposal = self.drafter.propose(seqs, self.gamma)

        keep_tokens, keep_logprobs = [], []
        for seq, toks, lps in zip(seqs, proposal.token_ids, proposal.logprobs):
            budget = min(
                self.gamma,
                seq.max_tokens - seq.num_completion_tokens,
                self.config.max_model_len - len(seq),
            )
            budget = max(0, budget)
            seq.draft_token_ids = toks[:budget]
            keep_tokens.append(seq.draft_token_ids)
            keep_logprobs.append(lps[:budget])

        logits = self.model_runner.call("run_verify", seqs)

        committed, offset = [], 0
        for seq, drafts, lps in zip(seqs, keep_tokens, keep_logprobs):
            n = 1 + len(drafts)
            rows = logits[offset:offset + n]
            offset += n
            committed.append(verify_one(rows, drafts, lps, seq.temperature))

        self.stats.steps += 1
        self.stats.seq_steps += len(seqs)
        for seq, c, d in zip(seqs, committed, keep_tokens):
            keep = min(len(c), seq.max_tokens - seq.num_completion_tokens)
            self.stats.committed += keep
            self.stats.drafted += len(d)
            self.stats.accepted += max(0, keep - 1)

        return committed