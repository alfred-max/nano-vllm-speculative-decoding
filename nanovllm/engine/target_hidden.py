import torch


class TargetHiddenCollector:
    """在目标模型指定层注册 forward hook，收集这些层的输出（residual + mlp_out）。"""

    def __init__(self, model, layer_ids, layer_offset: int = 0):
        self.layer_ids = [int(i) + layer_offset for i in layer_ids]
        self.outputs = {}
        self.handles = [
            model.model.layers[i].register_forward_hook(self._make_hook(i))
            for i in self.layer_ids
        ]

    def _make_hook(self, i):
        def hook(module, args, output):
            hidden, residual = output
            self.outputs[i] = hidden + residual if residual is not None else hidden
        return hook

    def take(self) -> torch.Tensor:
        """按 layer_ids 的顺序拼接成 [tokens, len(layer_ids) * hidden]。"""
        out = torch.cat([self.outputs[i] for i in self.layer_ids], dim=-1)
        self.outputs = {}
        return out

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []