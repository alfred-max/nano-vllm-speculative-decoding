import torch

from nanovllm.spec.dflash import DFlashDrafter
from nanovllm.spec.types import DraftProposal


class DSparkDrafter(DFlashDrafter):
    """DSpark = 同一套 backbone + 马尔可夫顺序头 + 置信度头。"""

    @torch.inference_mode()
    def propose(self, seqs) -> DraftProposal:
        if not seqs:
            return DraftProposal([], [], [])

        ctx_k, ctx_v, ctx_lens, anchors = self._gather(seqs)
        logits, hidden = self.model.draft_block_cached(
            anchors, ctx_k, ctx_v, ctx_lens, return_hidden=True,
        )

        mh = self.model.markov_head
        ch = self.model.confidence_head
        with_markov_conf = self.model.confidence_with_markov

        B, W = logits.shape[:2]
        prev = anchors
        tokens, confs = [], []
        tok_cols, conf_cols = [], []

        for k in range(W):
            emb = mh.prev_embedding(prev)                 # [B, r]
            step_logits = logits[:, k] + mh.markov_w2(emb)  # [B, V]
            tok = step_logits.argmax(dim=-1)                # greedy
            tok_cols.append(tok)
            if ch is not None:
                feat = torch.cat([hidden[:, k], emb], dim=-1) if with_markov_conf else hidden[:, k]
                conf_cols.append(torch.sigmoid(ch(feat)))
            prev = tok

        tokens = torch.stack(tok_cols, dim=1).tolist()
        confidence = torch.stack(conf_cols, dim=1).tolist() if conf_cols else None
        logprobs = [[0.0] * len(t) for t in tokens]
        return DraftProposal(tokens, logprobs, confidence)