# Industrial Time-Series Analysis Agent (Adaptive Core Version)

This branch contains a small, explainable SRTP prototype for **LLM-guided industrial time-series analysis**.

The large multimodal / deep-learning prototype remains on the `main` branch. The current `core-clean` branch focuses on one research question: can an LLM use tool feedback to make useful analysis decisions instead of merely executing a hard-coded pipeline?

## Core idea

The LLM does **not** directly calculate numerical results from the CSV. Python tools perform data processing, anomaly detection, validation, and forecasting. The LLM observes tool outputs and decides what action should come next.

The workflow is intentionally constrained but adaptive:

```text
raw CSV
   |
   v
inspect data and identify suitable numeric targets
   |
   v
LLM decides missing-value strategy (if needed)
   |
   v
LLM chooses anomaly detection method/parameters
   |
   v
LLM decides whether detected anomalies should be kept or handled
   |
   v
Python evaluates several lightweight forecasting candidates
   |
   v
LLM selects a final forecast method from validation evidence
   |
   v
final report with decisions, evidence, and limitations
```

This is not unrestricted autonomy. The agent operates inside a registered tool set, while important strategy choices are left to the LLM and numerical claims are grounded in tool results.

## Multi-column behavior

If `--column` is omitted, the agent inspects the dataset and analyzes all suitable numeric measurement columns while ignoring metadata-like fields such as timestamps, dates, IDs, and indices.

```powershell
python main.py --data data/sample_data.csv
```

To analyze only one target, specify it explicitly:

```powershell
python main.py --data data/sample_data.csv --column temperature
```

When preprocessing multiple columns, the agent maintains one current working CSV. Each returned `output_file` becomes the input for later actions so changes to earlier columns are not lost.

## Available tools

- `extract_data_summary`: rows, columns, numeric target candidates, missing locations, statistics, sample rows
- `impute_missing_values`: mean / median / forward fill / interpolation
- `detect_anomalies`: Isolation Forest / IQR rule
- `handle_anomalies`: keep / interpolate / median replacement / IQR clipping
- `evaluate_forecast_methods`: rolling validation with MAE and RMSE for lightweight candidates
- `forecast_series`: last value / moving average / exponential smoothing / linear trend

Preprocessing tools preserve the input file and return a new `output_file`; later tools should continue from that file.

## Core architecture

```text
main.py
  |
  +--> agent/prompts.py        defines goals and decision principles
  |
  +--> agent/core_brain.py     LLM <-> tool-calling loop
              |
              +--> tools/registry.py
                        |
                        +--> data_summary.py
                        +--> data_imputation.py
                        +--> anomaly_detection.py
                        +--> anomaly_handling.py
                        +--> forecast_evaluation.py
                        +--> time_series_forecast.py
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
DEEPSEEK_API_KEY=your_key_here
```

## Larger ETTh1 test data

A reproducible preparation script is included instead of committing a large public benchmark CSV directly.

From the project root run:

```powershell
python scripts/prepare_etth1.py
```

This downloads the public ETTh1 benchmark and creates:

```text
data/ETTh1_1000.csv
data/ETTh1_agent_test.csv
data/ETTh1_agent_test_ground_truth.json
```

`ETTh1_1000.csv` is the untouched first 1000 hourly rows. `ETTh1_agent_test.csv` additionally contains controlled missing values and artificial spikes so the agent's decisions can be checked against known ground truth. See `data/ETTh1_TESTING.md` for details.

Example all-column test:

```powershell
python main.py --data data/ETTh1_agent_test.csv
```

Example single-column test on transformer oil temperature:

```powershell
python main.py --data data/ETTh1_agent_test.csv --column OT
```

## Run options

Analysis strategies are deliberately not exposed as command-line arguments. The user specifies task-level requirements and the agent chooses analysis strategies from tool evidence.

```text
--data <csv path>
--column <optional target column>
--forecast-steps 5
--model deepseek-chat
--max-steps 20
```

The final LLM report is printed in the terminal. The complete interaction trace is saved under:

```text
outputs/run_YYYYMMDD_HHMMSS.json
```

## Current scope and limitations

This is a research prototype, not a production industrial system and not a claim of a novel forecasting model. The current focus is the **agent decision loop**: observe tool evidence, choose an action, execute it, inspect the result, and adapt.

The forecasting library is intentionally small. More advanced statistical or deep-learning models can be added later, but only after the current decision process and evaluation method are understood and tested on larger public datasets.
