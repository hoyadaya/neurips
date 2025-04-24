import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusion(nn.Module):
    """Lightweight Audio‑Visual Adaptive Fusion (AVadCLIP Eq. 1‑3)."""
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 2
        self.weight_proj = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.residual_proj = nn.Sequential(
            nn.Linear(dim * 2, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        """x_v, x_a: (B, T, D) → fused (B, T, D)"""
        x_cat = torch.cat([x_a, x_v], dim=-1)
        w = self.weight_proj(x_cat)
        x_res = self.residual_proj(x_cat)
        return x_v + w * x_res
