# Reproduction Experiments

Comprehensive benchmark reproduction using public datasets and paper-derived
methods. Demonstrates every module of the Autonomous Industrial Time Series &
Vision Analysis Agent on real-world data.

## Datasets

| Dataset | Source | Paper | Size |
|---------|--------|-------|------|
| ETTh1 | [ETDataset](https://github.com/zhouhaoyi/ETDataset) | Informer (AAAI 2021 Best Paper) | 17,420 x 7 |
| Synthetic Steel Defects | Generated locally | NEU Surface Defect Dataset (Song & Yan, 2014) | 90 images, 6 classes |

## Quick Start

```bash
# 1. Download / generate data
python reproduction/download_data.py

# 2. Run all 8 pipeline stages
python reproduction/run_benchmarks.py

# 3. Open the HTML report
start reproduction/results/etth1_benchmark/reports/benchmark_report.html
```

## Pipeline Stages

1. **Data Summary** - Descriptive statistics, missing value profiling
2. **Imputation** - Mean/forward fill for missing values
3. **Classical Anomaly** - Isolation Forest (contamination=5%)
4. **Baseline Forecast** - Moving average + EWM (24-step horizon)
5. **Deep Anomaly** - Autoencoder reconstruction probability
6. **Deep Forecast** - LSTM sequence-to-sequence (24-step horizon)
7. **Vision Analysis** - GASF + MTF + RP transformation, CNN scoring
8. **Multi-Modal Fusion** - Cross-modal risk assessment + RUL estimation

## Output Structure

```
reproduction/
  data/
    etth1/ETTh1.csv                        # Benchmark time series
    vision/
      normal/   (30 images)                # Normal steel surfaces
      defect/   (60 images, 6 classes)     # Synthetic defects
  results/
    etth1_benchmark/
      figures/
        classical_anomaly.png              # Anomaly detection plot
        lstm_forecast.png                  # Forecast comparison
        ts2image_transforms.png            # TS-to-image visualizations
      traces/
        benchmark_trace.json               # Full execution trace
      reports/
        benchmark_report.html              # Self-contained HTML report
```

## References

- Zhou, H. et al. *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting.* AAAI 2021.
- Song, K. & Yan, Y. *A Noise Robust Method Based on Completed Local Binary Patterns for Hot-Rolled Steel Strip Surface Defects.* Applied Surface Science, 2014.
- Wang, Z. & Oates, T. *Encoding Time Series as Images for Visual Inspection and Classification Using Tiled CNNs.* AAAI 2015.
