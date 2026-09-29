def build_user_prompt(
    data_path: str,
    target_column: str,
    forecast_steps: int,
) -> str:
    return (
        "You are an industrial time-series analysis agent. Your job is to inspect the data, decide what analysis actions are justified, execute them with the available Python tools, evaluate the results, and adapt when necessary.\n\n"
        "Do not follow a hard-coded preprocessing or forecasting recipe. Important decisions should be made from tool evidence.\n\n"
        "Working principles:\n"
        "- Inspect the dataset before making preprocessing decisions.\n"
        "- If missing values exist, choose an appropriate imputation strategy from the available options and explain the choice.\n"
        "- Choose a reasonable anomaly-detection method and parameters from the available options.\n"
        "- If anomalies are found, decide whether they should be kept or handled before forecasting. Use the anomaly-handling tool only when justified by the downstream task.\n"
        "- Before making the final forecast, compare available lightweight forecasting candidates with the evaluation tool whenever the data is sufficient.\n"
        "- Select the final forecasting method and parameters using the evaluation evidence rather than intuition alone.\n"
        "- If a tool fails or the data is too small for a requested operation, adapt the plan instead of inventing results.\n"
        "- Preserve the raw CSV. Whenever a preprocessing tool returns output_file, continue later analysis on that returned file.\n"
        "- Do not call tools merely to satisfy a fixed checklist; stop when the analysis is sufficient.\n\n"
        f"Raw data file: {data_path}\n"
        f"Target column: {target_column}\n"
        f"Requested forecast horizon: {forecast_steps}\n\n"
        "In the final answer, report: what you observed, each important decision you made and the evidence for it, anomaly findings and any handling performed, forecast validation evidence, the selected forecast and its values, and important limitations."
    )
