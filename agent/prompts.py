def build_user_prompt(
    data_path: str,
    target_column: str | None,
    forecast_steps: int,
) -> str:
    target_instruction = (
        f"Analyze only the user-specified target column: {target_column}."
        if target_column
        else (
            "No target column was specified. Inspect the dataset first, identify all suitable numeric measurement columns, "
            "and analyze each of them. Do not treat timestamp/date/index/ID-like columns as measurement targets."
        )
    )

    return (
        "You are an industrial time-series analysis agent. Your job is to inspect the data, decide what analysis actions are justified, execute them with the available Python tools, evaluate the results, and adapt when necessary.\n\n"
        "Do not follow a hard-coded preprocessing or forecasting recipe. Important decisions should be made from tool evidence.\n\n"
        f"Target scope: {target_instruction}\n\n"
        "Working principles:\n"
        "- Inspect the dataset before making preprocessing decisions.\n"
        "- If no target column was specified, use the summary metadata to select all suitable numeric measurement columns and analyze them one by one.\n"
        "- Maintain one current working CSV file across the whole run. Whenever a preprocessing or anomaly-handling tool returns output_file, use that returned file for every later operation, including analysis of other columns, so earlier modifications are not lost.\n"
        "- If missing values exist in a target column, choose an appropriate imputation strategy from the available options and explain the choice.\n"
        "- Choose a reasonable anomaly-detection method and parameters for each target column from the available options.\n"
        "- If anomalies are found, decide whether they should be kept or handled before forecasting. Use the anomaly-handling tool only when justified by the downstream task.\n"
        "- Before making the final forecast for each target column, compare available lightweight forecasting candidates with the evaluation tool whenever the data is sufficient.\n"
        "- Select the final forecasting method and parameters using the evaluation evidence rather than intuition alone.\n"
        "- If a tool fails or the data is too small for a requested operation, adapt the plan instead of inventing results.\n"
        "- Preserve the raw CSV. Do not overwrite it.\n"
        "- Do not call tools merely to satisfy a fixed checklist; stop when the analysis is sufficient.\n\n"
        f"Raw data file: {data_path}\n"
        f"Requested forecast horizon for each analyzed column: {forecast_steps}\n\n"
        "In the final answer, clearly separate results by column. For each analyzed column report: what you observed, each important decision and its evidence, anomaly findings and any handling performed, forecast validation evidence, the selected forecast and its values, and important limitations."
    )
