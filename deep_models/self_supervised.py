from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesContrastive(nn.Module):
    """Self-supervised contrastive learning for time series representations.

    Uses instance-level contrastive loss (SimCLR-style) with temporal
    augmentations: jitter, scaling, time warping, and window cropping.

    The learned representations serve as universal features for downstream
    tasks (forecasting, anomaly detection, classification) on unlabeled
    industrial sensor streams.

    Reference: Chen et al. (2020) SimCLR; Yue et al. (2022) TS2Vec.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        proj_dim: int = 64,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.temperature = temperature

        # Encoder backbone (1D CNN for inductive bias on temporal locality)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, V) -> representation (B, proj_dim)"""
        x = x.permute(0, 2, 1)  # (B, V, L)
        h = self.encoder(x).squeeze(-1)  # (B, hidden_dim)
        z = self.projection(h)
        return F.normalize(z, dim=-1)

    def contrastive_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """NT-Xent loss between two augmented views."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
        sim = sim.masked_fill(mask, float("-inf"))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(B)], dim=0).to(z.device)

        return F.cross_entropy(sim, labels)


class TemporalAugmentation:
    """Deterministic temporal augmentations for contrastive learning."""

    @staticmethod
    def jitter(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
        return x + torch.randn_like(x) * sigma

    @staticmethod
    def scaling(x: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
        factor = torch.randn(x.size(0), 1, 1, device=x.device) * sigma + 1.0
        return x * factor

    @staticmethod
    def window_crop(x: torch.Tensor, crop_ratio: float = 0.8) -> torch.Tensor:
        B, L, V = x.shape
        crop_len = int(L * crop_ratio)
        start = torch.randint(0, L - crop_len + 1, (1,)).item()
        cropped = x[:, start : start + crop_len, :]
        return F.interpolate(
            cropped.permute(0, 2, 1),
            size=L,
            mode="linear",
            align_corners=False,
        ).permute(0, 2, 1)

    @classmethod
    def strong(cls, x: torch.Tensor) -> torch.Tensor:
        x = cls.jitter(x)
        x = cls.scaling(x)
        x = cls.window_crop(x)
        return x

    @classmethod
    def weak(cls, x: torch.Tensor) -> torch.Tensor:
        return cls.jitter(x, sigma=0.01)
