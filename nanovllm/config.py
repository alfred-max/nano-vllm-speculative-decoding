import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    num_speculative_tokens: int = 0       # γ：0 表示关闭投机解码
    spec_method: str = "none"              # none | naive | dflash | dspark
    draft_model: str | None = None
    num_speculative_tokens: int = 0
    # drafter 的 K/V 几何，由 LLMEngine 在加载后填入，用于给 KV 预算留空间
    drafter_layers: int = 0
    drafter_kv_heads: int = 0
    drafter_head_dim: int = 0
    spec_adaptive: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.num_speculative_tokens >= 0
        assert self.spec_method in ("none", "naive", "dflash", "dspark")
        if self.spec_method == "none":
            self.num_speculative_tokens = 0        # 关闭时必须归零
        else:
            assert self.num_speculative_tokens > 0, "开启投机解码需要 num_speculative_tokens > 0"
            if self.spec_method in ("dflash", "dspark"):
                assert self.draft_model and os.path.isdir(self.draft_model), "需要 draft_model 指向本地目录"