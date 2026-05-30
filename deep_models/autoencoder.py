from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalAutoencoder(nn.Module):
    """VAE for unsupervised anomaly detection on multivariate time series.

    Reconstruction probability is used as the anomaly score: low probability
    under the learned latent distribution indicates an anomaly.

    Reference: An & Cho (2015), Variational Autoencoder based Anomaly Detection
    using Reconstruction Probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        # -------- Encoder --------
        enc_layers: list[nn.Module] = []
        prev_dim = input_dim
        for hd in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        self.encoder = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # -------- Decoder --------
        dec_layers: list[nn.Module] = []
        prev_dim = latent_dim
        for hd in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd
        dec_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def reconstruction_probability(self, x: torch.Tensor, n_samples: int = 10) -> torch.Tensor:
        """Compute anomaly score via reconstruction probability.

        Higher score -> more likely anomalous.
        """
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(x)
            scores = torch.zeros(x.size(0), device=x.device)
            for _ in range(n_samples):
                z = self.reparameterize(mu, logvar)
                recon = self.decode(z)
                # log p(x|z) under Gaussian assumption
                log_pxz = -0.5 * F.mse_loss(recon, x, reduction="none").sum(dim=1)
                scores += log_pxz
            scores /= n_samples
        return -scores  # negative log-likelihood


class AutoencoderAnomaly(nn.Module):
    """Deep autoencoder for reconstruction-error-based anomaly detection."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] | None = None,
        latent_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        enc_layers: list[nn.Module] = []
        prev = input_dim
        for hd in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hd
        enc_layers.append(nn.Linear(hidden_dims[-1], latent_dim))
        enc_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: list[nn.Module] = []
        prev = latent_dim
        for hd in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev, hd),
                nn.BatchNorm1d(hd),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = hd
        dec_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample reconstruction MSE."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x)
        return F.mse_loss(recon, x, reduction="none").mean(dim=1)
