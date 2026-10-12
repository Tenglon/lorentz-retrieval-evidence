"""Optional PyTorch output head matching the frozen H10 parameterization."""

from __future__ import annotations

import math

try:
    import torch
    from torch import nn
except ImportError as error:  # pragma: no cover - only used in full training
    raise ImportError("Install PyTorch to instantiate LorentzHead") from error


class LorentzHead(nn.Module):
    """LayerNorm -> row-normalized linear projection -> direct Lorentz lift."""

    def __init__(self, hidden_size: int = 2560, dimension: int = 128,
                 curvature: float = 20.0, output_scale: float = 1.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, dimension, bias=False)
        self.curvature = float(curvature)
        self.output_scale = float(output_scale)

    def pre_lift(self, hidden):
        weight = torch.nn.functional.normalize(
            self.projection.weight.float(), p=2, dim=1, eps=1e-12
        )
        weight = weight * (self.output_scale / math.sqrt(weight.shape[0]))
        return torch.nn.functional.linear(self.norm(hidden.float()), weight)

    def forward(self, hidden):
        spatial = self.pre_lift(hidden)
        time = torch.sqrt(
            spatial.square().sum(dim=-1, keepdim=True)
            + spatial.new_tensor(1.0 / self.curvature)
        )
        return torch.cat((time, spatial), dim=-1)
