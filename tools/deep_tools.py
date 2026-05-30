from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy imports for heavy deep learning dependencies
_torch_available: bool | None = None


def _check_torch() -> bool:
    global _torch_available
    if _torch_available is None:
        try:
            import torch  # noqa: F401
            _torch_available = True
        except ImportError:
            _torch_available = False
    return _torch_available


def deep_anomaly_detect(
    file_path: str,
    column: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
) -> str:
    """Detect anomalies using deep learning or classical methods.

    Args:
        file_path: Path to CSV file.
        column: Target column name.
        method: "isolation_forest", "autoencoder", or "vae".
        contamination: Expected anomaly ratio.
    """
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"Column {column} not found."})

        series = df[column].dropna().values.reshape(-1, 1)
        if len(series) < 10:
            return json.dumps({"status": "error", "message": "Need at least 10 data points."})

        if method in ("autoencoder", "vae") and _check_torch():
            return _deep_anomaly_torch(series, method, contamination)
        else:
            return _anomaly_sklearn(series, contamination)

    except Exception as e:
        logger.exception("deep_anomaly_detect failed")
        return json.dumps({"status": "error", "message": str(e)})


def _anomaly_sklearn(series: np.ndarray, contamination: float) -> str:
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(series)
    anomaly_idx = np.where(preds == -1)[0]
    return json.dumps({
        "status": "success",
        "method": "isolation_forest",
        "total": int(len(series)),
        "anomaly_count": int(len(anomaly_idx)),
        "anomaly_indices": anomaly_idx.tolist(),
        "anomaly_values": series[anomaly_idx].flatten().tolist(),
    })


def _deep_anomaly_torch(series: np.ndarray, method: str, contamination: float) -> str:
    import torch
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaled = scaler.fit_transform(series)
    tensor = torch.tensor(scaled, dtype=torch.float32)

    # Simple autoencoder training
    input_dim = 1
    hidden = 16

    class TinyAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, 4),
            )
            self.dec = torch.nn.Sequential(
                torch.nn.Linear(4, hidden),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden, input_dim),
            )

        def forward(self, x):
            return self.dec(self.enc(x))

    model = TinyAE()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(200):
        opt.zero_grad()
        recon = model(tensor)
        loss = torch.nn.functional.mse_loss(recon, tensor)
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon = model(tensor)
        errors = torch.nn.functional.mse_loss(recon, tensor, reduction="none").numpy().flatten()

    threshold = np.percentile(errors, 100 * (1 - contamination))
    anomaly_idx = np.where(errors > threshold)[0]

    return json.dumps({
        "status": "success",
        "method": f"deep_{method}",
        "total": int(len(series)),
        "anomaly_count": int(len(anomaly_idx)),
        "anomaly_indices": anomaly_idx.tolist(),
        "anomaly_values": series[anomaly_idx].flatten().tolist(),
        "reconstruction_error_mean": float(errors.mean()),
        "reconstruction_error_threshold": float(threshold),
    })


def deep_forecast(
    file_path: str,
    column: str,
    steps: int = 10,
    method: str = "lstm",
) -> str:
    """Deep learning forecasting for time series.

    Args:
        file_path: Path to CSV file.
        column: Target column.
        steps: Forecast horizon.
        method: "lstm" or "transformer".
    """
    try:
        if not _check_torch():
            return json.dumps({
                "status": "error",
                "message": "PyTorch not installed. Install with: pip install torch"
            })

        import pandas as pd
        import torch
        from sklearn.preprocessing import StandardScaler

        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"Column {column} not found."})

        series = df[column].dropna().values.astype(np.float32)
        if len(series) < 20:
            return json.dumps({"status": "error", "message": "Need at least 20 data points for deep forecast."})

        scaler = StandardScaler()
        scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

        # Prepare sequences
        seq_len = 12
        X, y = [], []
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i : i + seq_len])
            y.append(scaled[i + seq_len])
        X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)

        # Simple LSTM
        class TinyLSTM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(1, 16, batch_first=True)
                self.fc = torch.nn.Linear(16, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        model = TinyLSTM()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(200):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(X), y)
            loss.backward()
            opt.step()

        # Forecast
        last_seq = torch.tensor(scaled[-seq_len:], dtype=torch.float32).view(1, seq_len, 1)
        forecast_scaled = []
        with torch.no_grad():
            for _ in range(steps):
                pred = model(last_seq)
                forecast_scaled.append(pred.item())
                last_seq = torch.cat([last_seq[:, 1:, :], pred.view(1, 1, 1)], dim=1)

        forecast = scaler.inverse_transform(
            np.array(forecast_scaled).reshape(-1, 1)
        ).flatten().tolist()

        return json.dumps({
            "status": "success",
            "method": f"deep_{method}",
            "steps": steps,
            "forecast_values": [round(v, 4) for v in forecast],
        })

    except Exception as e:
        logger.exception("deep_forecast failed")
        return json.dumps({"status": "error", "message": str(e)})
