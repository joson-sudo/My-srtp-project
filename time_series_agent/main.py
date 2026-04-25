#!/usr/bin/env python3
"""
Time Series Prediction Agent entry point.
Provides reproducible CLI execution for multi-agent forecasting workflow.
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

from graph.agent_graph import TimeSeriesAgentGraph
from config.default_config import create_config_from_args


logger = logging.getLogger("IndustrialTimeSeriesAgent")


def _default_data_path() -> str:
    return str((Path(__file__).resolve().parents[1] / "dataset" / "ETTh1.csv").resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the industrial time-series multi-agent workflow")
    parser.add_argument("--data", dest="data_path", default=None, help="Input csv data path")
    parser.add_argument("--output-dir", default=None, help="Output directory for reports and artifacts")
    parser.add_argument("--num-slices", type=int, default=None, help="Number of rolling slices")
    parser.add_argument("--input-length", type=int, default=None, help="Validation window length")
    parser.add_argument("--horizon", type=int, default=None, help="Forecast horizon")
    parser.add_argument("--k-models", type=int, default=None, help="Top-k models to keep after validation")
    parser.add_argument("--llm-provider", choices=["anthropic", "openai"], default=None, help="LLM provider")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--slice-delay-seconds", type=int, default=None, help="Delay between slices")
    parser.add_argument("--debug", action="store_true", help="Enable graph stream debug mode")
    parser.add_argument("--date-column", default="date", help="Datetime column name")
    parser.add_argument("--value-column", default="OT", help="Target value column name")
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> dict:
    overrides = {
        "data_path": args.data_path or _default_data_path(),
        "date_column": args.date_column,
        "value_column": args.value_column,
    }

    optional_overrides = {
        "output_dir": args.output_dir,
        "num_slices": args.num_slices,
        "input_length": args.input_length,
        "horizon": args.horizon,
        "k_models": args.k_models,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model,
        "slice_delay_seconds": args.slice_delay_seconds,
    }

    for key, value in optional_overrides.items():
        if value is not None:
            overrides[key] = value

    config = create_config_from_args(**overrides)
    config["debug"] = bool(args.debug)

    # Keep nested model config in sync for agent-level readers.
    if config.get("models") and config.get("k_models") is not None:
        config["models"]["k_models"] = config["k_models"]

    return config


def validate_environment(config: dict) -> None:
    provider = str(config.get("llm_provider", "anthropic")).lower()
    key_map = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    required_key = key_map.get(provider)
    if not required_key:
        raise ValueError(f"Unsupported llm_provider: {provider}")
    if not os.environ.get(required_key):
        raise EnvironmentError(f"Missing required environment variable: {required_key}")


def save_results(results: dict, output_dir: str) -> None:
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    complete_report_path = reports_dir / f"complete_time_series_report_{timestamp}.json"
    with open(complete_report_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False, default=str)
    logger.info("Complete workflow result saved: %s", complete_report_path)

    aggregated = results.get("aggregated_results")
    if aggregated:
        aggregated_report_path = reports_dir / f"aggregated_forecast_results_{timestamp}.json"
        with open(aggregated_report_path, "w", encoding="utf-8") as handle:
            json.dump(aggregated, handle, indent=2, ensure_ascii=False, default=str)
        logger.info("Aggregated forecast result saved: %s", aggregated_report_path)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    config = build_runtime_config(args)

    try:
        validate_environment(config)
    except Exception as exc:
        logger.error("Environment validation failed: %s", exc)
        return 2

    logger.info(
        "Launching workflow provider=%s model=%s slices=%s horizon=%s",
        config.get("llm_provider"),
        config.get("llm_model"),
        config.get("num_slices"),
        config.get("horizon"),
    )

    graph = TimeSeriesAgentGraph(config=config, model=config["llm_model"], debug=config["debug"])
    results = graph.run()

    save_results(results, config.get("output_dir", "results"))

    aggregated = results.get("aggregated_results", {})
    if aggregated.get("test_metrics", {}).get("ensemble"):
        metrics = aggregated["test_metrics"]["ensemble"]
        logger.info(
            "Final ensemble metrics: MSE=%.6f MAE=%.6f MAPE=%.4f%%",
            metrics.get("mse", float("nan")),
            metrics.get("mae", float("nan")),
            metrics.get("mape", float("nan")),
        )

    logger.info("Workflow finished successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
