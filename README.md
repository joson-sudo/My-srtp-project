# Industrial Time-Series Analysis Agent (Core Version)

This branch contains a minimal, explainable SRTP prototype focused on **LLM tool calling for industrial time-series analysis**.

The large multimodal / deep-learning prototype is preserved on the `main` branch. This `core-clean` branch intentionally keeps only the core workflow that is currently being studied and maintained.

## What the system does

Given a CSV file and a target numeric column, the agent can:

1. inspect the dataset summary;
2. fill missing values when needed;
3. detect anomalies with Isolation Forest;
4. produce a simple baseline forecast;
5. let the LLM summarize the tool results.

The LLM does **not** directly calculate on the CSV. It decides which registered Python tool to call, receives that tool's result, and then decides the next step.

## Core architecture

```text
main.py
  |
  +--> agent/prompts.py        build the task prompt
  |
  +--> agent/core_brain.py     LLM <-> tool calling loop
              |
              +--> tools/registry.py
                        |
                        +--> data_summary.py
                        +--> data_imputation.py
                        +--> anomaly_detection.py
                        +--> time_series_forecast.py
```

## Project layout

```text
.
├── main.py
├── config.py
├── requirements.txt
├── agent/
│   ├── __init__.py
│   ├── core_brain.py
│   └── prompts.py
├── tools/
│   ├── __init__.py
│   ├── registry.py
│   ├── data_summary.py
│   ├── data_imputation.py
│   ├── anomaly_detection.py
│   └── time_series_forecast.py
└── data/
    └── sample_data.csv
```

## Setup

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```text
DEEPSEEK_API_KEY=your_key_here
```

The default API endpoint is `https://api.deepseek.com` and the default model is `deepseek-chat`.

## Run

```powershell
python main.py --data data/sample_data.csv --column temperature
```

Useful options:

```text
--impute-method mean|forward
--contamination 0.1
--forecast-steps 5
--forecast-method moving_average|ewm|last
--forecast-window 5
--forecast-alpha 0.4
```

The final LLM summary is printed in the terminal. The complete trace is also saved under:

```text
outputs/run_YYYYMMDD_HHMMSS.json
```

## Current scope

This branch is deliberately a small prototype. It does not claim a novel forecasting model or a production-ready industrial system. The present goal is to make the agent workflow understandable, reproducible, and easy to evaluate before adding more advanced models.

Future work can reintroduce selected deep-learning modules from the `main` branch after they are understood and experimentally validated.
