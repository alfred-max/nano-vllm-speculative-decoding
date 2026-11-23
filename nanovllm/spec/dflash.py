import torch
import time

from nanovllm.engine.drafter_cache import DrafterKVCache
from nanovllm.models.block_drafter import BlockDrafter, _apply_rope
from nanovllm.spec.drafter import Drafter
from nanovllm.spec.types import DraftProposal
from nanovllm.utils.loader import load_drafter


class DFlashDrafter(Drafter):

    def __init__(self, dcfg, draft_path, gamma, num_blocks, block_size=256, device="cuda"):
        self.gamma = gamma
        self.block_size = block_size
        self.model = BlockDrafter(dcfg).to(device=device, dtype=torch.bfloat16)
        load_drafter(self.model, draft_path)
        self.num_kv_heads = dcfg.num_key_value_heads
        self.head_dim = dcfg.head_dim
        self.cache = DrafterKVCache(
            num_layers=dcfg.num_hidden_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            dtype=torch.bfloat16,
            device=device,
        )

    @torch.inference_mode()
    def ingest(self, seqs, aux, positions, cu_seqlens, keep=None):
        feats = self.model.hidden_norm(self.model.fc(aux))
        for i, seq in enumerate(seqs):
            s, e = cu_seqlens[i], cu_seqlens[i + 1]
            if keep is not None:
                e = s + keep[i]
            if e <= s:
                continue
            pos = positions[s:e]
            slots = self.cache.slots(seq, pos.tolist())
            f = feats[s:e]
            for l, layer in enumerate(self.model.layers):
                k = layer.self_attn.k_proj(f).view(-1, self.num_kv_heads, self.head_dim)
                k = _apply_rope(self.model.rotary_emb, layer.self_attn.k_norm(k), pos)
                v = layer.self_attn.v_proj(f).view(-1, self.num_kv_heads, self.head_dim)
                self.cache.write(l, k, v, slots)

    @torch.inference_mode()
    def propose(self, seqs) -> DraftProposal:
        if not seqs:
            return DraftProposal([], [], None)
        ctx_k, ctx_v, ctx_lens, anchors = self._gather(seqs)
        logits = self.model.draft_block_cached(anchors, ctx_k, ctx_v, ctx_lens)
        tokens = logits.argmax(dim=-1).tolist()
        logprobs = [[0.0] * len(t) for t in tokens]
        return DraftProposal(tokens, logprobs, None)

    def _gather(self, seqs):
        W = self.model.block_size
        device = self.cache.k.device
        ks, vs, ctx_lens = [], [], []
        for seq in seqs:
            L = len(seq) - 1
            k, v = self.cache.gather(seq, L)
            ks.append(k)
            vs.append(v)
            ctx_lens.append(L)
        ctx_k = torch.cat(ks, dim=1)
        ctx_v = torch.cat(vs, dim=1)
        anchors = torch.tensor([seq.last_token for seq in seqs], device=device)
        return ctx_k, ctx_v, ctx_lens, anchors