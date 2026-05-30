import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from agent.core_brain import run_tool_loop
from agent.prompts import build_user_prompt
from config import resolve_api_key, resolve_base_url
from tools.registry import TOOL_FUNCTIONS, TOOL_SCHEMAS

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industrial time-series agent")
    parser.add_argument("--data", dest="data_path", default="data/sample_data.csv")
    parser.add_argument("--column", default="temperature")
    parser.add_argument("--impute-method", choices=["mean", "forward"], default="mean")
    parser.add_argument("--contamination", type=float, default=0.1)
    parser.add_argument("--forecast-steps", type=int, default=5)
    parser.add_argument("--forecast-method", choices=["moving_average", "ewm", "last"], default="moving_average")
    parser.add_argument("--forecast-window", type=int, default=5)
    parser.add_argument("--forecast-alpha", type=float, default=0.4)
    parser.add_argument("--deep-anomaly-method", choices=["isolation_forest", "autoencoder", "vae"], default="isolation_forest")
    parser.add_argument("--deep-forecast-method", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--multimodal", action="store_true", help="Enable cross-modal diagnosis")
    parser.add_argument("--image-path", default=None, help="Equipment image for vision analysis")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(message)s"
    )


def save_run(messages: list[dict], output_dir: str) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = output_path / f"run_{timestamp}.json"
    payload = {"created_at": timestamp, "messages": messages}
    run_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    return run_file


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    api_key = resolve_api_key(args.env_file)
    if not api_key:
        raise SystemExit("API key not found. Set DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    base_url = args.base_url or resolve_base_url(args.env_file)
    client = OpenAI(api_key=api_key, base_url=base_url)

    user_prompt = build_user_prompt(
        data_path=args.data_path,
        target_column=args.column,
        impute_method=args.impute_method,
        contamination=args.contamination,
        forecast_steps=args.forecast_steps,
        forecast_method=args.forecast_method,
        forecast_window=args.forecast_window,
        forecast_alpha=args.forecast_alpha,
        deep_anomaly_method=args.deep_anomaly_method,
        deep_forecast_method=args.deep_forecast_method,
        use_multimodal=args.multimodal,
        image_path=args.image_path,
    )

    messages = run_tool_loop(
        client=client,
        tool_schemas=TOOL_SCHEMAS,
        tool_functions=TOOL_FUNCTIONS,
        user_prompt=user_prompt,
        model=args.model,
        max_steps=args.max_steps
    )

    run_file = save_run(messages, args.output_dir)
    logger.info("Run saved to %s", run_file)


if __name__ == "__main__":
    main()


