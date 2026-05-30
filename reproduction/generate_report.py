"""
Generate a comprehensive HTML reproduction report with embedded figures.

The report is self-contained: figures are base64-encoded inline,
so a single HTML file can be shared as a standalone artifact.
"""

import base64
import json
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _img_to_b64(path: Path) -> str:
    """Convert an image to a base64 data URI."""
    if not path.exists():
        return ""
    ext = path.suffix.lower().lstrip(".")
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "svg": "image/svg+xml"}.get(ext, "image/png")
    data = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def _load_trace(trace_path: Path) -> dict:
    if trace_path.exists():
        return json.loads(trace_path.read_text(encoding="utf-8"))
    return {}


def generate_html_report(results_dir: Path, figures_dir: Path) -> str:
    trace = _load_trace(results_dir / "traces" / "benchmark_trace.json")
    stages = trace.get("stages", {})
    metrics = trace.get("metrics", {})

    # Gather figure data URIs
    figs = {}
    for name in ["classical_anomaly.png", "lstm_forecast.png", "ts2image_transforms.png"]:
        fpath = figures_dir / name
        figs[name] = _img_to_b64(fpath) if fpath.exists() else ""

    # Build summary table rows
    stage_rows = ""
    for i, (name, info) in enumerate(stages.items(), 1):
        status = info.get("status", "?")
        elapsed = info.get("elapsed_s", 0)
        badge = '<span class="pass">PASS</span>' if status == "ok" else '<span class="fail">FAIL</span>'
        stage_rows += f"""
        <tr>
          <td>{i}</td>
          <td>{name}</td>
          <td>{badge}</td>
          <td>{elapsed:.1f}s</td>
        </tr>"""

    metric_rows = ""
    for k, v in metrics.items():
        metric_rows += f"<tr><td>{k}</td><td>{v}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ETTh1 Reproduction Benchmark - Industrial Time Series Analysis Agent</title>
<style>
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; margin: 40px auto; max-width: 1100px; line-height: 1.6; color: #222; background: #f9fafb; }}
  h1 {{ border-bottom: 3px solid #2563eb; padding-bottom: 10px; color: #1e3a5f; }}
  h2 {{ color: #1e40af; margin-top: 40px; border-bottom: 1px solid #d1d5db; padding-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
  th {{ background: #2563eb; color: white; font-weight: 600; }}
  tr:hover {{ background: #f1f5f9; }}
  .pass {{ color: #059669; font-weight: bold; }}
  .fail {{ color: #dc2626; font-weight: bold; }}
  .figure-container {{ margin: 25px 0; text-align: center; background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  .figure-container img {{ max-width: 100%; height: auto; border-radius: 4px; }}
  .figure-caption {{ color: #6b7280; font-size: 0.9em; margin-top: 8px; }}
  .meta {{ color: #6b7280; font-size: 0.9em; margin-bottom: 30px; }}
  .code-block {{ background: #1e293b; color: #e2e8f0; padding: 15px 20px; border-radius: 6px; font-family: 'Cascadia Code', 'Fira Code', monospace; font-size: 0.85em; overflow-x: auto; }}
  .dataset-card {{ background: white; border-left: 4px solid #2563eb; padding: 15px 20px; margin: 15px 0; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
</style>
</head>
<body>

<h1>ETTh1 Reproduction Benchmark</h1>
<p class="meta">
  <strong>Project:</strong> Autonomous Industrial Time Series &amp; Vision Analysis Agent<br>
  <strong>Dataset:</strong> ETTh1 (Electricity Transformer Temperature) &mdash; Zhou et al., AAAI 2021 Best Paper<br>
  <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
  <strong>Pipeline:</strong> 8-stage LLM-orchestrated deep learning &amp; vision pipeline
</p>

<div class="dataset-card">
  <strong>ETTh1 Dataset</strong><br>
  Source: <a href="https://github.com/zhouhaoyi/ETDataset">github.com/zhouhaoyi/ETDataset</a><br>
  17,420 hourly records | 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT<br>
  Time range: 2016-07-01 00:00 to 2018-06-26 19:00<br>
  Task: Long-sequence time series forecasting (Informer, AAAI 2021 Best Paper)
</div>

<h2>Pipeline Results</h2>
<table>
  <tr><th>#</th><th>Stage</th><th>Status</th><th>Time</th></tr>
  {stage_rows}
</table>

<h2>Key Metrics</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  {metric_rows}
</table>

<h2>Figure 1: Classical Anomaly Detection</h2>
<div class="figure-container">
  <img src="{figs.get('classical_anomaly.png', '')}" alt="Anomaly Detection">
  <p class="figure-caption">Isolation Forest anomaly detection on ETTh1 Oil Temperature (OT). Red dots indicate detected anomalies (contamination=5%).</p>
</div>

<h2>Figure 2: Deep LSTM Forecast</h2>
<div class="figure-container">
  <img src="{figs.get('lstm_forecast.png', '')}" alt="LSTM Forecast">
  <p class="figure-caption">LSTM sequence-to-sequence forecast of ETTh1 Oil Temperature. Orange markers show 24-step ahead predictions from the trained model.</p>
</div>

<h2>Figure 3: Time Series to Image Transformations</h2>
<div class="figure-container">
  <img src="{figs.get('ts2image_transforms.png', '')}" alt="TS2Image Transformations">
  <p class="figure-caption">
    Four representations of a 500-point OT segment. From left: GASF (Gramian Angular Summation Field),
    MTF (Markov Transition Field), RP (Recurrence Plot), and RGB Composite for pretrained CNN input.
    Reference: Wang &amp; Oates, AAAI 2015.
  </p>
</div>

<h2>Reproduction Instructions</h2>
<div class="code-block">
<pre>
# 1. Prepare data
python reproduction/download_data.py

# 2. Run full benchmark (all 8 stages)
python reproduction/run_benchmarks.py

# 3. View report
open reproduction/results/etth1_benchmark/reports/benchmark_report.html

# 4. Run individual stages via CLI
python main.py --data reproduction/data/etth1/ETTh1.csv --column OT \
  --deep-anomaly-method autoencoder --deep-forecast-method lstm \
  --forecast-steps 24 --multimodal
</pre>
</div>

<h2>References</h2>
<ol>
  <li>Zhou, H. et al. <em>Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.</em> AAAI 2021 (Best Paper).</li>
  <li>Wang, Z. &amp; Oates, T. <em>Encoding Time Series as Images for Visual Inspection and Classification Using Tiled CNNs.</em> AAAI 2015.</li>
  <li>An, J. &amp; Cho, S. <em>Variational Autoencoder based Anomaly Detection using Reconstruction Probability.</em> SNU 2015.</li>
  <li>Song, K. &amp; Yan, Y. <em>A Noise Robust Method Based on Completed Local Binary Patterns for Hot-Rolled Steel Strip Surface Defects.</em> Applied Surface Science, 2014.</li>
</ol>

</body>
</html>"""

    report_path = results_dir / "reports" / "benchmark_report.html"
    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


if __name__ == "__main__":
    from pathlib import Path
    results = Path(__file__).resolve().parent / "results" / "etth1_benchmark"
    figures = results / "figures"
    path = generate_html_report(results, figures)
    print(f"Report generated: {path}")
