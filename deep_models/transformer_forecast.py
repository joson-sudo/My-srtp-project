from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer time-series models."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1), :])


class TimeSeriesTransformer(nn.Module):
    """PatchTST-inspired transformer for multivariate time series forecasting.

    Splits input sequence into overlapping/non-overlapping patches, applies
    Transformer encoder, and projects to forecast horizon.
    """

    def __init__(
        self,
        n_vars: int,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 3,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        forecast_horizon: int = 96,
    ):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.stride = stride
        self.forecast_horizon = forecast_horizon

        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=2000, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.head = nn.Linear(d_model, forecast_horizon)

    def _create_patches(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """x: (B, L, V) -> patches: (B, V, N, patch_len)"""
        B, L, V = x.shape
        pad_len = (self.patch_len - (L % self.patch_len)) % self.patch_len
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        L_padded = L + pad_len
        num_patches = max(1, (L_padded - self.patch_len) // self.stride + 1)
        patches = []
        for i in range(num_patches):
            start = i * self.stride
            patches.append(x[:, start : start + self.patch_len, :])
        patches = torch.stack(patches, dim=1)  # (B, N, patch_len, V)
        patches = patches.permute(0, 3, 1, 2)  # (B, V, N, patch_len)
        return patches, num_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, V = x.shape
        patches, num_patches = self._create_patches(x)  # (B, V, N, patch_len)
        patches = self.patch_embedding(patches)  # (B, V, N, d_model)
        patches = patches.mean(dim=1)  # (B, N, d_model) — pool over variables
        patches = self.positional_encoding(patches)
        encoded = self.encoder(patches)
        pooled = encoded.mean(dim=1)  # (B, d_model)
        return self.head(pooled)  # (B, forecast_horizon)


class BiLSTMAttention(nn.Module):
    """BiLSTM encoder with multi-head self-attention for sequence modelling."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        forecast_horizon: int = 96,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2, num_heads=n_heads, batch_first=True, dropout=dropout
        )
        self.layer_norm = nn.LayerNorm(hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)  # (B, L, 2H)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.layer_norm(lstm_out + self.dropout(attn_out))
        pooled = attn_out.mean(dim=1)  # (B, 2H)
        return self.head(pooled)


class TCNBlock(nn.Module):
    """Dilated causal convolution block for TCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        out = out[..., : residual.shape[-1]]  # trim causal padding
        return self.relu(out + residual)


class TCN(nn.Module):
    """Temporal Convolutional Network for time series forecasting."""

    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
        forecast_horizon: int = 96,
    ):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = input_size if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(num_channels[-1], forecast_horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (B, V, L)
        out = self.network(x)
        out = self.pool(out).squeeze(-1)  # (B, C_last)
        return self.head(out)
