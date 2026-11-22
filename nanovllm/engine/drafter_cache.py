import torch


class DrafterKVCache:
    """独立的 K/V 缓存，和 target 共用 seq.block_table（位置一一对应）。"""

    def __init__(self, num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, device="cuda"):
        self.num_layers = num_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        shape = (num_layers, num_blocks * block_size, num_kv_heads, head_dim)
        self.k = torch.empty(shape, dtype=dtype, device=device)
        self.v = torch.empty(shape, dtype=dtype, device=device)

    def slots(self, seq, positions) -> list[int]:
        bs, bt = self.block_size, seq.block_table
        return [bt[p // bs] * bs + p % bs for p in positions]

    @torch.inference_mode()
    def write(self, layer: int, k: torch.Tensor, v: torch.Tensor, slots: list[int]):
        idx = torch.tensor(slots, dtype=torch.long, device=self.k.device)
        self.k[layer].index_copy_(0, idx, k)
        self.v[layer].index_copy_(0, idx, v)

    @torch.inference_mode()
    def gather(self, seq, length: int):
        """取该序列 0..length-1 的 K/V，返回 [num_layers, length, heads, dim]。"""
        bs = self.block_size
        nblocks = (length + bs - 1) // bs
        bt = torch.tensor(seq.block_table[:nblocks], dtype=torch.long, device=self.k.device)
        offs = torch.arange(bs, device=self.k.device)
        idx = (bt[:, None] * bs + offs[None, :]).reshape(-1)      # 展开成 token 槽位
        k = self.k[:, idx][:, :length]
        v = self.v[:, idx][:, :length]
        return k, v