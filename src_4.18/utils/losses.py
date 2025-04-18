import torch
import torch.nn as nn

class UKDLoss(nn.Module):
    """Uncertainty‑Driven Knowledge Distillation (Eq. 11)."""
    def __init__(self):
        super().__init__()

    def forward(
        self,
        teacher_feat: torch.Tensor,
        student_feat: torch.Tensor,
        var_pred: torch.Tensor,
    ) -> torch.Tensor:
        """All tensors (B, T, D) – var_pred (B, T, 1)."""
        var_pred = var_pred.squeeze(-1).clamp(min=1e-6)  # safety
        mse = (teacher_feat - student_feat).pow(2).sum(dim=-1)  # (B,T)
        loss = (mse / var_pred + var_pred.log()).mean()
        return loss
