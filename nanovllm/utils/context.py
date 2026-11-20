from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    mode: str = "decode"                    # "prefill" | "decode" | "verify"
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

    @property
    def is_prefill(self) -> bool:
        return self.mode == "prefill"

    @property
    def is_verify(self) -> bool:
        return self.mode == "verify"

    @property
    def is_varlen(self) -> bool:
        """prefill 和 verify 都走 flash_attn_varlen_func。"""
        return self.mode != "decode"


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(mode, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(mode, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                       slot_mapping, context_lens, block_tables)


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()