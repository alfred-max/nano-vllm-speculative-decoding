from nanovllm.spec.types import DraftProposal


class Drafter:

    def propose(self, seqs) -> DraftProposal:
        raise NotImplementedError

    def reset(self):
        pass


def build_drafter(config) -> Drafter:
    if config.spec_method == "naive":
        from nanovllm.spec.naive import NaiveDrafter
        return NaiveDrafter(config)
    raise ValueError(f"暂不支持的 spec_method: {config.spec_method}")