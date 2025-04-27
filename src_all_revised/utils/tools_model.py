# utils/tools_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveFusion(nn.Module):
    """AVadCLIP Eq.(1-3)에 따라 오디오-비주얼 특징을 가볍게 융합하는 모듈."""
    def __init__(self, dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or dim * 2  # 숨겨진 차원: 기본값 2D

        # (1) 융합 가중치 W 계산: 2D → D → Sigmoid
        self.weight_proj = nn.Sequential(
            nn.Linear(dim * 2, dim),  # 선형 변환
            nn.Sigmoid()              # 0~1 범위의 스칼라 가중치
        )

        # (2) 잔차 특징 X_res 계산: 2D → hidden → GELU → D
        self.residual_proj = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.GELU(),                # 비선형 활성화
            nn.Linear(hidden, dim)
        )

    def forward(self, x_v: torch.Tensor, x_a: torch.Tensor) -> torch.Tensor:
        """
        x_v, x_a: (B, T, D)  ➜  반환: 융합된 (B, T, D)
        """
        x_cat = torch.cat([x_a, x_v], dim=-1)  # 오디오·비디오 특징 결합
        w = self.weight_proj(x_cat)            # 식 (1) : 가중치 W
        x_res = self.residual_proj(x_cat)      # 식 (2) : 잔차 특징
        return x_v + w * x_res                 # 식 (3) : 적응적 융합

class VisualEnhancementModule(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv1d = nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: [B, T, D]
        identity = x
        x_t = x.transpose(1, 2)  # [B, D, T]
        x_t = self.conv1d(x_t)
        x_t = self.relu(x_t)
        x = x_t.transpose(1, 2) + identity  # 스킵 연결
        return x
    
class AVPrompt(nn.Module):
    def __init__(self, dim: int, num_class: int):
        super().__init__()
        # 두 단계 FFN: (dim → dim*4 → dim)으로 텍스트 임베딩 변환
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        # 정규화를 통해 Xp 안정화
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_av: torch.Tensor, conf: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        """
        x_av: 융합된 시퀀스 특징, shape=[B, T, D]
        conf: 원시 로짓 값 (이상 점수), shape=[B, T, 1]
        x_c: 클래스별 원본 텍스트 임베딩, shape=[B, C, D]
        반환: 강화된 클래스 임베딩, shape=[B, C, D]
        """
        # 로짓 값 처리 - 안정적인 가중치로 변환
        # 1. 절대값 적용 (부호 무시)
        # 2. softmax로 정규화하여 프레임 간 가중치 합이 1이 되도록 함
        attn_weights = F.softmax(conf.abs().squeeze(-1), dim=1).unsqueeze(-1)  # [B, T, 1]
        
        # Eq.4 수정: 정규화된 가중치로 시각·오디오 융합 특징 가중 합
        xp = (attn_weights * x_av).sum(dim=1)  # [B, D]
        
        # Norm 적용으로 분포 안정화
        xp = self.norm(xp)  # [B, D]

        # Eq.5: 클래스(x_c)와 전역 표현(xp) 간 유사도 행렬 계산
        # 벡터 정규화로 코사인 유사도 계산
        xp_norm = F.normalize(xp, p=2, dim=1).unsqueeze(-1)  # [B, D, 1]
        x_c_norm = F.normalize(x_c, p=2, dim=2)  # [B, C, D]
        
        # 코사인 유사도 계산 후 온도 스케일링 및 softmax
        sp = torch.softmax(
            torch.matmul(x_c_norm, xp_norm).squeeze(-1) / 0.1,  # 온도 파라미터 0.1
            dim=-1
        )  # [B, C]

        # Eq.6: 스코어(sp)로 시퀀스 특징 가중 합, 프롬프트 생성
        # 배치 내 각 샘플에 대해 처리
        x_mp_list = []
        for i in range(x_av.size(0)):
            # 현재 배치 항목의 가중치 및 특징
            curr_sp = sp[i]  # [C]
            curr_x_av = x_av[i]  # [T, D]
            
            # 클래스별 가중치로 가중 평균 계산
            # 각 클래스에 대한 가중치를 해당 클래스 인덱스의 특징에 적용
            weighted_feat = curr_sp.unsqueeze(1) * curr_x_av.mean(0, keepdim=True)  # [C, 1] * [1, D] = [C, D]
            x_mp_list.append(weighted_feat)
        
        x_mp = torch.stack(x_mp_list)  # [B, C, D]

        # Eq.7: FFN + 스킵 연결로 최종 클래스 임베딩 완성
        x_cp = self.ffn(x_mp + x_c) + x_c  # [B, C, D]
        return x_cp

class UncertaintyPredictor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, 1),
            nn.Softplus()            # 출력이 항상 >0 이 되도록
        )
    
    def forward(self, x):
        # x: [B, T, D] 또는 [B*T, D]
        # 반환: var_pred (σ²) shape [B, T, 1]
        return self.net(x) + 1e-6   # 수치 안정성 보강
