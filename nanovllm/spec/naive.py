from nanovllm.spec.drafter import Drafter
from nanovllm.spec.types import DraftProposal


class NaiveDrafter(Drafter):
    """假 drafter：每次都猜 last_token 重复 γ 次。

    点质量分布（log q = 0.0），所以 verify 里 q(d)=1。
    命中率很低，用途是验证接受/拒绝逻辑，不追求加速。
    """

    def __init__(self, config):
        self.gamma = config.num_speculative_tokens

    def propose(self, seqs, gamma: int) -> DraftProposal:
        tokens = [[seq.last_token] * gamma for seq in seqs]
        logprobs = [[0.0] * gamma for _ in seqs]
        return DraftProposal(tokens, logprobs)