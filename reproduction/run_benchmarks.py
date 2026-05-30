"""
Comprehensive reproduction benchmark across all project modules.

Runs the full 8-stage pipeline on ETTh1 benchmark data:
  1. Data Summary & Profiling
  2. Missing Value Imputation
  3. Classical Anomaly Detection (Isolation Forest)
  4. Baseline Forecasting
  5. Deep Anomaly Detection (Autoencoder)
  6. Deep Forecasting (LSTM)
  7. Vision Analysis (TS-to-Image + CNN Scoring)
  8. Multi-Modal Fusion Diagnosis

Outputs JSON traces, figures, and benchmark metrics.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "reproduction" / "data"
RESULTS_DIR = PROJECT_ROOT / "reproduction" / "results" / "etth1_benchmark"
FIGURES_DIR = RESULTS_DIR / "figures"
TRACES_DIR = RESULTS_DIR / "traces"
REPORTS_DIR = RESULTS_DIR / "reports"

for d in [FIGURES_DIR, TRACES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


class BenchmarkRunner:
    """Run and record each pipeline stage."""

    def __init__(self):
        self.results: dict = {"stages": {}, "metrics": {}, "timestamp": str(datetime.now())}
        self.etth1_path = DATA_DIR / "etth1" / "ETTh1.csv"

    def run_all(self) -> None:
        if not self.etth1_path.exists():
            print(f"[ERROR] ETTh1 not found at {self.etth1_path}. Run download_data.py first.")
            return

        stages = [
            ("1_data_summary",       self.stage_data_summary),
            ("2_imputation",          self.stage_imputation),
            ("3_classical_anomaly",   self.stage_classical_anomaly),
            ("4_baseline_forecast",   self.stage_baseline_forecast),
            ("5_deep_anomaly",        self.stage_deep_anomaly),
            ("6_deep_forecast",       self.stage_deep_forecast),
            ("7_vision_analysis",     self.stage_vision_analysis),
            ("8_multimodal_fusion",   self.stage_multimodal_fusion),
        ]

        for name, func in stages:
            print(f"\n{'='*60}\n  Stage: {name}\n{'='*60}")
            t0 = time.time()
            try:
                result = func()
                elapsed = time.time() - t0
                self.results["stages"][name] = {"status": "ok", "elapsed_s": round(elapsed, 2)}
                self.results["stages"][name].update(result)
                print(f"  [PASS] {name} ({elapsed:.1f}s)")
            except Exception as e:
                import traceback
                elapsed = time.time() - t0
                self.results["stages"][name] = {"status": "error", "error": str(e), "elapsed_s": round(elapsed, 2)}
                print(f"  [FAIL] {name}: {e}")
                traceback.print_exc()

        self._save_trace()
        self._compute_summary()

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    def stage_data_summary(self) -> dict:
        from tools.data_summary import extract_data_summary
        raw = json.loads(extract_data_summary(str(self.etth1_path), head_rows=5))
        return {"summary": raw.get("data_summary", {})}

    def stage_imputation(self) -> dict:
        from tools.data_imputation import impute_missing_values
        # Check if any missing values exist
        df = pd.read_csv(self.etth1_path)
        missing_cols = [c for c in df.columns if df[c].isnull().any() and c != "date"]
        results = {}
        for col in missing_cols[:3]:  # limit to 3 cols
            raw = json.loads(impute_missing_values(str(self.etth1_path), col, "mean"))
            results[col] = raw
        return {"imputed_columns": list(results.keys()), "details": results}

    def stage_classical_anomaly(self) -> dict:
        from tools.anomaly_detection import detect_anomalies
        # Use OT (oil temperature) as the target
        raw = json.loads(detect_anomalies(str(self.etth1_path), "OT", 0.05))
        self._plot_anomaly(raw, "classical_anomaly.png")
        return {"detection": raw}

    def stage_baseline_forecast(self) -> dict:
        from tools.time_series_forecast import forecast_series
        methods = ["moving_average", "ewm"]
        results = {}
        for method in methods:
            raw = json.loads(forecast_series(str(self.etth1_path), "OT", steps=24, method=method))
            results[method] = raw
        return {"forecasts": results}

    def stage_deep_anomaly(self) -> dict:
        from tools.deep_tools import deep_anomaly_detect
        raw = json.loads(deep_anomaly_detect(str(self.etth1_path), "OT", method="autoencoder", contamination=0.05))
        return {"detection": raw}

    def stage_deep_forecast(self) -> dict:
        from tools.deep_tools import deep_forecast
        raw = json.loads(deep_forecast(str(self.etth1_path), "OT", steps=24, method="lstm"))
        self._plot_forecast(raw, "lstm_forecast.png")
        return {"forecast": raw}

    def stage_vision_analysis(self) -> dict:
        from vision.ts2image import series_to_rgb, gramian_angular_field, markov_transition_field, recurrence_plot
        # Sample OT column for transformation
        df = pd.read_csv(self.etth1_path)
        sample = df["OT"].values[-500:].astype(np.float32)

        gasf = gramian_angular_field(sample, "sum")
        mtf = markov_transition_field(sample, n_bins=10)
        rp = recurrence_plot(sample)
        rgb = series_to_rgb(sample, image_size=224)

        self._plot_ts_images(gasf, mtf, rp, rgb)

        return {
            "ts_length": len(sample),
            "gasf_shape": list(gasf.shape),
            "mtf_shape": list(mtf.shape),
            "rp_shape": list(rp.shape),
            "rgb_shape": list(rgb.shape),
            "gasf_stats": {"min": float(gasf.min()), "max": float(gasf.max()), "mean": float(gasf.mean())},
        }

    def stage_multimodal_fusion(self) -> dict:
        from tools.vision_tools import multimodal_diagnosis
        raw = json.loads(multimodal_diagnosis(str(self.etth1_path), "OT", forecast_steps=24))
        return {"diagnosis": raw}

    # ------------------------------------------------------------------
    # Visualization helpers
    # ------------------------------------------------------------------

    def _plot_anomaly(self, result: dict, filename: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df = pd.read_csv(self.etth1_path)
            ot = df["OT"].values
            anomaly_idx = result.get("anomaly_indices", [])

            fig, ax = plt.subplots(figsize=(14, 4))
            ax.plot(ot[-500:], color="steelblue", linewidth=0.6, label="OT")
            rel_idx = [i for i in anomaly_idx if i >= len(ot) - 500]
            if rel_idx:
                ax.scatter([i - (len(ot) - 500) for i in rel_idx],
                          ot[rel_idx], color="red", s=20, zorder=5, label="Anomaly")
            ax.set_title("Isolation Forest Anomaly Detection on ETTh1 (OT, last 500 points)")
            ax.set_xlabel("Time step (relative)")
            ax.set_ylabel("Oil Temperature")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / filename, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Plot anomaly failed: {e}")

    def _plot_forecast(self, result: dict, filename: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            df = pd.read_csv(self.etth1_path)
            ot = df["OT"].values
            forecast = result.get("forecast_values", [])

            fig, ax = plt.subplots(figsize=(14, 4))
            # Show last 200 actual points
            actual_range = range(-200, 0)
            ax.plot(actual_range, ot[-200:], color="steelblue", linewidth=0.8, label="Actual")
            if forecast:
                fc_range = range(len(forecast))
                ax.plot(fc_range, forecast, color="orange", linewidth=2, marker="o",
                       markersize=4, label=f"LSTM Forecast ({len(forecast)} steps)")
            ax.set_title("LSTM Forecast on ETTh1 (OT)")
            ax.set_xlabel("Time step (0 = forecast start)")
            ax.set_ylabel("Oil Temperature")
            ax.legend()
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / filename, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Plot forecast failed: {e}")

    def _plot_ts_images(self, gasf: np.ndarray, mtf: np.ndarray,
                        rp: np.ndarray, rgb: np.ndarray) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

            axes[0].imshow(gasf, cmap="jet", aspect="auto")
            axes[0].set_title("GASF (Gramian Angular Sum Field)")
            axes[0].axis("off")

            axes[1].imshow(mtf, cmap="viridis", aspect="auto")
            axes[1].set_title("MTF (Markov Transition Field)")
            axes[1].axis("off")

            axes[2].imshow(rp, cmap="binary", aspect="auto")
            axes[2].set_title("RP (Recurrence Plot)")
            axes[2].axis("off")

            axes[3].imshow(rgb.transpose(1, 2, 0), aspect="auto")
            axes[3].set_title("RGB Composite (GASF+GADF+MTF)")
            axes[3].axis("off")

            fig.suptitle("ETTh1 Time Series to Image Transformations", fontsize=13)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR / "ts2image_transforms.png", dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"  [WARN] Plot ts2image failed: {e}")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _save_trace(self) -> None:
        trace_path = TRACES_DIR / "benchmark_trace.json"
        trace_path.write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[TRACE] Saved -> {trace_path}")

    def _compute_summary(self) -> None:
        metrics = {}
        stages = self.results["stages"]
        passed = sum(1 for v in stages.values() if v.get("status") == "ok")
        total = len(stages)
        metrics["stages_passed"] = f"{passed}/{total}"
        metrics["total_elapsed_s"] = round(sum(v.get("elapsed_s", 0) for v in stages.values()), 2)

        # Extract key numbers from results
        s3 = stages.get("3_classical_anomaly", {})
        if "detection" in s3:
            metrics["classical_anomaly_count"] = s3["detection"].get("anomaly_count", "N/A")

        s5 = stages.get("5_deep_anomaly", {})
        if "detection" in s5:
            d = s5["detection"]
            metrics["deep_anomaly_count"] = d.get("anomaly_count", "N/A")
            metrics["recon_error_mean"] = d.get("reconstruction_error_mean", "N/A")

        s6 = stages.get("6_deep_forecast", {})
        if "forecast" in s6:
            f = s6["forecast"]
            metrics["forecast_values"] = f.get("forecast_values", [])

        s8 = stages.get("8_multimodal_fusion", {})
        if "diagnosis" in s8:
            diag = s8["diagnosis"]
            if isinstance(diag, dict):
                summary = diag.get("summary", {})
                metrics["risk_level"] = summary.get("risk_level", "N/A")
                metrics["composite_score"] = summary.get("composite_score", "N/A")

        self.results["metrics"] = metrics
        self._save_trace()

        # Print summary
        print(f"\n{'='*60}")
        print("  BENCHMARK SUMMARY")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"  {k:30s}: {v}")
        print(f"{'='*60}")


def main() -> None:
    runner = BenchmarkRunner()
    runner.run_all()

    # Also generate the HTML report
    from reproduction.generate_report import generate_html_report
    html_path = generate_html_report(RESULTS_DIR, FIGURES_DIR)
    print(f"\n[REPORT] {html_path}")


if __name__ == "__main__":
    main()
