from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """Cross-modal attention fusion for sensor (1D) and vision (2D) signals.

    Given time-series features (B, L, D_t) and vision features (B, N, D_v),
    computes bidirectional cross-attention and fuses via gated summation.

    This is a key patent-level contribution: fusing heterogeneous industrial
    sensor streams with visual defect inspection within a single attention
    framework.
    """

    def __init__(
        self,
        ts_dim: int,
        vision_dim: int,
        fused_dim: int = 256,
        n_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Project both modalities to same dimension
        self.ts_proj = nn.Linear(ts_dim, fused_dim)
        self.vision_proj = nn.Linear(vision_dim, fused_dim)

        # Cross-attention: vision attends to time series
        self.cross_attn_v2t = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        # Cross-attention: time series attends to vision
        self.cross_attn_t2v = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=n_heads, batch_first=True, dropout=dropout
        )

        # Self-attention fusion
        self.self_attn = nn.MultiheadAttention(
            embed_dim=fused_dim, num_heads=n_heads, batch_first=True, dropout=dropout
        )

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.Sigmoid(),
        )

        self.layer_norm1 = nn.LayerNorm(fused_dim)
        self.layer_norm2 = nn.LayerNorm(fused_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        ts_features: torch.Tensor,
        vision_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse time-series and vision features.

        Args:
            ts_features: (B, L, D_t) temporal features.
            vision_features: (B, N, D_v) vision patch/token features.

        Returns:
            (B, fused_dim) fused representation.
        """
        # Project to common space
        t = self.ts_proj(ts_features)      # (B, L, F)
        v = self.vision_proj(vision_features)  # (B, N, F)

        # Bidirectional cross-attention
        t_enhanced, _ = self.cross_attn_t2v(t, v, v)    # vision -> time series
        v_enhanced, _ = self.cross_attn_v2t(v, t, t)    # time series -> vision

        # Pool
        t_pool = t_enhanced.mean(dim=1)  # (B, F)
        v_pool = v_enhanced.mean(dim=1)  # (B, F)

        # Gated fusion
        gate = self.gate(torch.cat([t_pool, v_pool], dim=-1))  # (B, F)
        fused = gate * t_pool + (1 - gate) * v_pool

        # Self-attention refinement
        fused_expanded = fused.unsqueeze(1)  # (B, 1, F)
        refined, _ = self.self_attn(fused_expanded, fused_expanded, fused_expanded)
        refined = self.layer_norm1(fused_expanded + self.dropout(refined))

        return refined.squeeze(1)  # (B, F)


class MultimodalDiagnosisHead(nn.Module):
    """Diagnosis head for multi-modal industrial health assessment.

    Produces: anomaly probability, risk level, and remaining useful life (RUL).
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )
        self.risk_head = nn.Sequential(
            nn.Linear(hidden_dim, 5),  # 5 risk levels
        )
        self.rul_head = nn.Linear(hidden_dim, 1)  # remaining useful life

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.shared(x)
        return {
            "anomaly_prob": self.anomaly_head(h),
            "risk_logits": self.risk_head(h),
            "rul": self.rul_head(h),
        }
