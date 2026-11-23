from dataclasses import dataclass, field


@dataclass
class DraftProposal:
    token_ids: list[list[int]] = field(default_factory=list)
    logprobs: list[list[float]] = field(default_factory=list)   # 每个 draft 在其分布下的 log q
    confidence: list[list[float]] | None = None     # DSpark 才有