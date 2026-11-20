import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float()
        greedy = temperatures <= 1e-10
        scaled = logits / temperatures.clamp_min(1e-10).unsqueeze(dim=1)
        probs = torch.softmax(scaled, dim=-1)
        sampled = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return torch.where(greedy, logits.argmax(dim=-1), sampled)
