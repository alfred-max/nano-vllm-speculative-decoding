import torch
import torch.nn.functional as F
from torch import nn
from flash_attn import flash_attn_varlen_func

from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import apply_rotary_emb, get_rope


def _apply_rope(rope, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    cos_sin = rope.cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    return apply_rotary_emb(x, cos, sin)


def _pack_kv(k_ctx, v_ctx, k_blk, v_blk, cu_q, cu_k):
    """把每个序列的 [ctx; block] 拼成 varlen 布局。"""
    total = int(cu_k[-1])
    k = k_ctx.new_empty((total,) + k_ctx.shape[1:])
    v = v_ctx.new_empty((total,) + v_ctx.shape[1:])
    ctx_off = 0
    for i in range(len(cu_q) - 1):
        q_len = int(cu_q[i + 1] - cu_q[i])
        kv_len = int(cu_k[i + 1] - cu_k[i])
        ctx_len = kv_len - q_len
        ks, qs = int(cu_k[i]), int(cu_q[i])
        k[ks:ks + ctx_len] = k_ctx[ctx_off:ctx_off + ctx_len]
        v[ks:ks + ctx_len] = v_ctx[ctx_off:ctx_off + ctx_len]
        k[ks + ctx_len:ks + kv_len] = k_blk[qs:qs + q_len]
        v[ks + ctx_len:ks + kv_len] = v_blk[qs:qs + q_len]
        ctx_off += ctx_len
    return k, v


class DrafterAttention(nn.Module):

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, eps):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = RMSNorm(head_dim, eps)
        self.k_norm = RMSNorm(head_dim, eps)

    def forward(self, x, ctx, rope, q_pos, ctx_pos, cu_q, cu_k, max_q, max_k):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        q = _apply_rope(rope, self.q_norm(q), q_pos)

        k_ctx = self.k_proj(ctx).view(-1, self.num_kv_heads, self.head_dim)
        v_ctx = self.v_proj(ctx).view(-1, self.num_kv_heads, self.head_dim)
        k_blk = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v_blk = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)

        k_ctx = _apply_rope(rope, self.k_norm(k_ctx), ctx_pos)
        k_blk = _apply_rope(rope, self.k_norm(k_blk), q_pos)

        k, v = _pack_kv(k_ctx, v_ctx, k_blk, v_blk, cu_q, cu_k)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=x.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=x.device)
        o = flash_attn_varlen_func(
            q, k, v, cu_q_t, cu_k_t, max_q, max_k,
            causal=False, softmax_scale=self.scaling,
        )
        return self.o_proj(o.reshape(-1, self.num_heads * self.head_dim))

    # DrafterAttention 里追加
    def forward_cached(self, x, k_ctx, v_ctx, rope, q_pos, cu_q, cu_k, max_q, max_k):
        q = self.q_proj(x).view(-1, self.num_heads, self.head_dim)
        q = _apply_rope(rope, self.q_norm(q), q_pos)

        k_blk = self.k_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        v_blk = self.v_proj(x).view(-1, self.num_kv_heads, self.head_dim)
        k_blk = _apply_rope(rope, self.k_norm(k_blk), q_pos)

        k, v = _pack_kv(k_ctx, v_ctx, k_blk, v_blk, cu_q, cu_k)
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32, device=x.device)
        cu_k_t = torch.tensor(cu_k, dtype=torch.int32, device=x.device)
        o = flash_attn_varlen_func(q, k, v, cu_q_t, cu_k_t, max_q, max_k,
                                   causal=False, softmax_scale=self.scaling)
        return self.o_proj(o.reshape(-1, self.num_heads * self.head_dim))

class MarkovHead(nn.Module):
    """一阶马尔可夫头：bias = W2 @ W1[prev_token]。"""

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def prev_embedding(self, token_ids):
        return self.markov_w1(token_ids.long())

    def step_bias(self, prev_token_ids):
        return self.markov_w2(self.prev_embedding(prev_token_ids))


class ConfidenceHead(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)

class DrafterMLP(nn.Module):

    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DrafterLayer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.self_attn = DrafterAttention(
            cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads,
            cfg.head_dim, cfg.rms_norm_eps,
        )
        self.mlp = DrafterMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, ctx, rope, q_pos, k_pos, cu_q, cu_k, max_q, max_k):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn(h, ctx, rope, q_pos, k_pos, cu_q, cu_k, max_q, max_k)
        x = residual + h
        residual = x
        h = self.post_attention_layernorm(x)
        return residual + self.mlp(h)

    # DrafterLayer 里追加
    def forward_cached(self, x, k_ctx, v_ctx, rope, q_pos, cu_q, cu_k, max_q, max_k):
        residual = x
        h = self.input_layernorm(x)
        h = self.self_attn.forward_cached(h, k_ctx, v_ctx, rope, q_pos, cu_q, cu_k, max_q, max_k)
        x = residual + h
        residual = x
        h = self.post_attention_layernorm(x)
        return residual + self.mlp(h)

class BlockDrafter(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.block_size = int(cfg.block_size)
        self.mask_token_id = int(cfg.mask_token_id)
        self.target_layer_ids = list(cfg.target_layer_ids)
        self.hidden_size = cfg.hidden_size

        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DrafterLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.fc = nn.Linear(len(self.target_layer_ids) * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

        rope_theta = getattr(cfg, "rope_theta", None)
        if rope_theta is None:
            rope_theta = getattr(cfg, "rope_parameters", {}).get("rope_theta", 1000000.0)
        self.rotary_emb = get_rope(cfg.head_dim, cfg.head_dim, cfg.max_position_embeddings, rope_theta)

        self.markov_rank = int(getattr(cfg, "markov_rank", 0) or 0)
        self.markov_head = (
            MarkovHead(cfg.vocab_size, self.markov_rank) if self.markov_rank > 0 else None
        )
        self.confidence_head = None
        self.confidence_with_markov = bool(getattr(cfg, "confidence_head_with_markov", False))
        if getattr(cfg, "enable_confidence_head", False):
            input_dim = cfg.hidden_size + (self.markov_rank if self.confidence_with_markov else 0)
            self.confidence_head = ConfidenceHead(input_dim)

    @torch.inference_mode()
    def draft_block(self, anchor_ids: torch.Tensor, ctx_raw: torch.Tensor, ctx_lens: list[int]) -> torch.Tensor:
        """anchor_ids: [B]；ctx_raw: [sum(ctx_lens), 5*H]；返回 logits [B, W, V]。"""
        B, W = anchor_ids.shape[0], self.block_size
        device = anchor_ids.device

        block_ids = torch.full((B, W), self.mask_token_id, dtype=torch.long, device=device)
        block_ids[:, 0] = anchor_ids
        ctx = self.hidden_norm(self.fc(ctx_raw))

        cu_q, cu_k, q_pos, ctx_pos = [], [], [], []
        q_off = k_off = 0
        for L in ctx_lens:
            cu_q.append(q_off)
            cu_k.append(k_off)
            ctx_pos.append(torch.arange(0, L, device=device))
            q_pos.append(torch.arange(L, L + W, device=device))
            q_off += W
            k_off += L + W
        cu_q.append(q_off)
        cu_k.append(k_off)
        max_k = max(L + W for L in ctx_lens)

        h = self.embed_tokens(block_ids).view(-1, self.hidden_size)
        for layer in self.layers:
            h = layer(h, ctx, self.rotary_emb, torch.cat(q_pos), torch.cat(ctx_pos),
                      cu_q, cu_k, W, max_k)
        h = self.norm(h)
        return self.lm_head(h).view(B, W, -1)
    
    # BlockDrafter 里追加
    @torch.inference_mode()
    def draft_block_cached(self, anchor_ids, ctx_k, ctx_v, ctx_lens, return_hidden=False):
        """ctx_k/ctx_v: [num_layers, sum(ctx_lens), kv_heads, dim]，已做完 k_norm + RoPE。"""
        B, W = anchor_ids.shape[0], self.block_size
        device = anchor_ids.device

        block_ids = torch.full((B, W), self.mask_token_id, dtype=torch.long, device=device)
        block_ids[:, 0] = anchor_ids

        cu_q, cu_k, q_pos = [], [], []
        q_off = k_off = 0
        for L in ctx_lens:
            cu_q.append(q_off)
            cu_k.append(k_off)
            q_pos.append(torch.arange(L, L + W, device=device))
            q_off += W
            k_off += L + W
        cu_q.append(q_off)
        cu_k.append(k_off)
        max_k = max(L + W for L in ctx_lens)

        h = self.embed_tokens(block_ids).view(-1, self.hidden_size)
        for i, layer in enumerate(self.layers):
            h = layer.forward_cached(h, ctx_k[i], ctx_v[i], self.rotary_emb,
                                     torch.cat(q_pos), cu_q, cu_k, W, max_k)
        h = self.norm(h)
        logits = self.lm_head(h).view(B, W, -1)
        if return_hidden:
            return logits, h.view(B, W, -1)
        return logits