"""Comprehensive test suite for industrial time series analysis agent.

Covers: data tools, deep models, vision, fusion, and CLI integration.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv() -> str:
    """Create a temporary CSV with known anomalies and missing values."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    trend = 0.02 * t
    seasonal = 5 * np.sin(2 * np.pi * t / 50)
    noise = np.random.randn(n) * 0.5
    values = 75 + trend + seasonal + noise
    values[50] = np.nan
    values[100] = 150.0  # clear anomaly
    values[150] = 20.0   # clear anomaly

    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
        "temperature": values,
        "pressure": 101.3 + np.random.randn(n) * 0.2,
    })

    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp, index=False)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_series() -> np.ndarray:
    np.random.seed(123)
    t = np.linspace(0, 4 * np.pi, 200)
    return (np.sin(t) * 10 + 75 + np.random.randn(200) * 0.3).astype(np.float32)


# ---------------------------------------------------------------------------
# Data tools
# ---------------------------------------------------------------------------


class TestDataSummary:
    def test_extract_summary(self, sample_csv):
        from tools.data_summary import extract_data_summary
        result = json.loads(extract_data_summary(sample_csv))
        assert result["status"] == "success"
        assert result["data_summary"]["rows"] == 200
        assert "temperature" in result["data_summary"]["columns"]

    def test_summary_invalid_file(self):
        from tools.data_summary import extract_data_summary
        result = json.loads(extract_data_summary("nonexistent.csv"))
        assert result["status"] == "error"


class TestDataImputation:
    def test_mean_impute(self, sample_csv):
        from tools.data_imputation import impute_missing_values
        result = json.loads(impute_missing_values(sample_csv, "temperature", "mean"))
        assert result["status"] == "success"

        df = pd.read_csv(sample_csv)
        assert df["temperature"].isnull().sum() == 0

    def test_invalid_column(self, sample_csv):
        from tools.data_imputation import impute_missing_values
        result = json.loads(impute_missing_values(sample_csv, "no_such_col", "mean"))
        assert result["status"] == "error"

    def test_invalid_method(self, sample_csv):
        from tools.data_imputation import impute_missing_values
        result = json.loads(impute_missing_values(sample_csv, "temperature", "median"))
        assert result["status"] == "error"


class TestAnomalyDetection:
    def test_isolation_forest(self, sample_csv):
        from tools.anomaly_detection import detect_anomalies
        result = json.loads(detect_anomalies(sample_csv, "temperature", 0.05))
        assert result["status"] == "success"
        assert result["anomaly_count"] > 0

    def test_invalid_contamination(self, sample_csv):
        from tools.anomaly_detection import detect_anomalies
        result = json.loads(detect_anomalies(sample_csv, "temperature", 0.6))
        assert result["status"] == "error"


class TestForecast:
    def test_moving_average(self, sample_csv):
        from tools.time_series_forecast import forecast_series
        result = json.loads(forecast_series(sample_csv, "temperature", steps=5, method="moving_average"))
        assert result["status"] == "success"
        assert len(result["forecast_values"]) == 5

    def test_ewm(self, sample_csv):
        from tools.time_series_forecast import forecast_series
        result = json.loads(forecast_series(sample_csv, "temperature", steps=3, method="ewm", alpha=0.5))
        assert result["status"] == "success"

    def test_invalid_method(self, sample_csv):
        from tools.time_series_forecast import forecast_series
        result = json.loads(forecast_series(sample_csv, "temperature", method="arima"))
        assert result["status"] == "error"

    def test_zero_steps(self, sample_csv):
        from tools.time_series_forecast import forecast_series
        result = json.loads(forecast_series(sample_csv, "temperature", steps=0))
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# Deep models
# ---------------------------------------------------------------------------


class TestTransformerForecast:
    def test_forward_pass(self):
        import torch
        from deep_models.transformer_forecast import TimeSeriesTransformer
        model = TimeSeriesTransformer(n_vars=3, d_model=32, n_heads=4, n_layers=2,
                                       patch_len=8, stride=4, forecast_horizon=24)
        x = torch.randn(2, 96, 3)
        out = model(x)
        assert out.shape == (2, 24)


class TestBiLSTMAttention:
    def test_forward_pass(self):
        import torch
        from deep_models.transformer_forecast import BiLSTMAttention
        model = BiLSTMAttention(input_size=3, hidden_size=32, forecast_horizon=12)
        x = torch.randn(2, 64, 3)
        out = model(x)
        assert out.shape == (2, 12)


class TestAutoencoder:
    def test_vae(self):
        import torch
        from deep_models.autoencoder import VariationalAutoencoder
        model = VariationalAutoencoder(input_dim=10, hidden_dims=[16], latent_dim=4)
        x = torch.randn(8, 10)
        recon, mu, logvar = model(x)
        assert recon.shape == x.shape
        assert mu.shape == (8, 4)

    def test_ae_anomaly(self):
        import torch
        from deep_models.autoencoder import AutoencoderAnomaly
        model = AutoencoderAnomaly(input_dim=10, hidden_dims=[16], latent_dim=4)
        x = torch.randn(8, 10)
        recon = model(x)
        assert recon.shape == x.shape


class TestSelfSupervised:
    def test_contrastive_forward(self):
        import torch
        from deep_models.self_supervised import TimeSeriesContrastive
        model = TimeSeriesContrastive(input_dim=3, hidden_dim=32, proj_dim=16)
        x = torch.randn(4, 50, 3)
        z = model(x)
        assert z.shape == (4, 16)

    def test_augmentation(self):
        import torch
        from deep_models.self_supervised import TemporalAugmentation
        x = torch.randn(4, 50, 3)
        aug = TemporalAugmentation.strong(x)
        assert aug.shape == x.shape


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------


class TestTS2Image:
    def test_gaf(self, sample_series):
        from vision.ts2image import gramian_angular_field
        img = gramian_angular_field(sample_series[:100], "sum")
        assert img.shape == (100, 100)
        assert 0.0 <= img.min() <= img.max() <= 1.0

    def test_mtf(self, sample_series):
        from vision.ts2image import markov_transition_field
        img = markov_transition_field(sample_series[:50], n_bins=5)
        assert img.shape == (50, 50)

    def test_recurrence(self, sample_series):
        from vision.ts2image import recurrence_plot
        img = recurrence_plot(sample_series[:30])
        assert img.shape == (30, 30)

    def test_series_to_rgb(self, sample_series):
        from vision.ts2image import series_to_rgb
        rgb = series_to_rgb(sample_series[:100], image_size=64)
        assert rgb.shape == (3, 64, 64)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class TestCrossModalAttention:
    def test_forward(self):
        import torch
        from fusion.cross_modal_attention import CrossModalAttention
        model = CrossModalAttention(ts_dim=32, vision_dim=64, fused_dim=64, n_heads=4)
        ts = torch.randn(2, 16, 32)
        vis = torch.randn(2, 10, 64)
        out = model(ts, vis)
        assert out.shape == (2, 64)


class TestMultimodalAgent:
    def test_pipeline(self, sample_csv):
        from fusion.multimodal_agent import MultimodalAgent
        df = pd.read_csv(sample_csv)
        series = df["temperature"].dropna().values.reshape(-1, 1)

        agent = MultimodalAgent()
        ts_results = agent.run_time_series_pipeline(series, steps=5)
        vision_results = agent.run_vision_pipeline(series_data=series.flatten())
        fusion_results = agent.run_fusion_pipeline(ts_results, vision_results)
        report = agent.generate_report(ts_results, vision_results, fusion_results)

        data = json.loads(report)
        assert "time_series_analysis" in data
        assert "fusion_diagnosis" in data
        assert "risk_level" in data["summary"]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_all_tools_registered(self):
        from tools.registry import TOOL_FUNCTIONS, TOOL_SCHEMAS
        assert len(TOOL_FUNCTIONS) >= 8
        assert len(TOOL_SCHEMAS) >= 8

        func_names = set(TOOL_FUNCTIONS.keys())
        schema_names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert func_names == schema_names


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_resolve_missing(self, tmp_path):
        from config import resolve_api_key, resolve_base_url
        env_file = tmp_path / ".env"
        env_file.write_text("")
        assert resolve_api_key(str(env_file)) is None
        assert resolve_base_url(str(env_file)) == "https://api.deepseek.com"

    def test_resolve_found(self, tmp_path):
        from config import resolve_api_key, resolve_base_url
        env_file = tmp_path / ".env"
        env_file.write_text('DEEPSEEK_API_KEY="sk-test"\nDEEPSEEK_BASE_URL="https://test.com"')
        assert resolve_api_key(str(env_file)) == "sk-test"
        assert resolve_base_url(str(env_file)) == "https://test.com"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_parse_args_minimal(self):
        import sys
        sys.argv = ["main.py", "--data", "data/test.csv", "--column", "temp"]
        from main import parse_args
        args = parse_args()
        assert args.data_path == "data/test.csv"
        assert args.column == "temp"
        assert args.multimodal is False

    def test_parse_args_multimodal(self):
        import sys
        sys.argv = ["main.py", "--multimodal", "--image-path", "img/equip.jpg"]
        from main import parse_args
        args = parse_args()
        assert args.multimodal is True
        assert args.image_path == "img/equip.jpg"
