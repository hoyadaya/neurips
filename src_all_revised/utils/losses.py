# utils/losses.py
import torch
import torch.nn as nn

class UKDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, teacher_feat, student_feat, var_pred):
        # var_pred: σ²_i 예측치 (shape [B, T, 1])
        var_pred = var_pred.squeeze(-1).clamp(min=1e-6)
        
        # MSE를 차원별 합(sum) 대신 평균(mean)으로 계산 → 스케일 안정화
        mse = (teacher_feat - student_feat).pow(2).mean(dim=-1)  # [B, T]
        
        # Eq.(11): (mse/σ² + ln σ²)의 배치-시퀀스 평균
        loss = (mse / var_pred + var_pred.log()).mean()
        return loss