import torch
import time
import os

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
    rows: int = 0            # 累计验证行数（每序列）

    @property
    def mean_rows(self) -> float:
        return self.rows / self.steps if self.steps else 0.0

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
        self.adaptive = bool(getattr(config, "spec_adaptive", False))
        self.cost_a = model_runner.cost_a
        self.cost_b = model_runner.cost_b
        self.gamma = config.num_speculative_tokens
        self.needs_hidden = config.spec_method in ("dflash", "dspark")
        self.stats = SpecStats()
        if self.needs_hidden:
            from transformers import AutoConfig
            dcfg = AutoConfig.from_pretrained(config.draft_model)
            if config.spec_method == "dspark":
                from nanovllm.spec.dspark import DSparkDrafter
                cls = DSparkDrafter
            else:
                from nanovllm.spec.dflash import DFlashDrafter
                cls = DFlashDrafter
            self.drafter = cls(
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
        rows = self._choose_rows(seqs, proposal) if self.adaptive else (1 + self.gamma)
        max_drafts = rows - 1
        torch.cuda.synchronize(); t1 = time.perf_counter()

        keep_tokens, keep_logprobs = [], []
        for seq, toks, lps in zip(seqs, proposal.token_ids, proposal.logprobs):
            budget = max(0, min(max_drafts, self.config.max_model_len - len(seq)))
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
        self.stats.rows += rows
        for seq, c, d in zip(seqs, committed, keep_tokens):
            keep = min(len(c), seq.max_tokens - seq.num_completion_tokens)
            self.stats.committed += keep
            self.stats.drafted += len(d)
            self.stats.accepted += max(0, keep - 1)
        return committed

    def _choose_rows(self, seqs, proposal) -> int:
        max_rows = 1 + self.gamma
        if proposal.confidence is None:
            return max_rows

        conf = torch.tensor(proposal.confidence, dtype=torch.float32)
        surv = torch.cumprod(conf, dim=1).mean(dim=0).tolist()   # 批次平均存活率
        bs = len(seqs)
        a, b = self.cost_a, self.cost_b

        best_rows, best_score, cum = 1, 0.0, 0.0
        for L in range(1, max_rows):
            cum += surv[L - 1]
            rows = L + 1
            score = (1.0 + cum) / (a + b * (bs * rows))
            if score > best_score:
                best_score, best_rows = score, rows

        ns = getattr(self.model_runner, "verify_graph_ns", None)
        if ns and self.model_runner.verify_graphs:
            allowed = [x for x in ns if x <= best_rows]
            best_rows = allowed[-1] if allowed else ns[0]
        if os.environ.get("NANOVLLM_DEBUG_CHOOSE"):
            if getattr(self, "_dbg", 0) <= 3:
                print(f"[choose] bs={bs} surv={[round(x, 3) for x in surv]} "
                    f"a={a:.4f}s b={b:.2e} -> rows={best_rows}")
        return best_rows