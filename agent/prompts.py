def build_user_prompt(
    data_path: str,
    target_column: str,
    contamination: float,
    forecast_steps: int,
    forecast_method: str,
    forecast_window: int,
    forecast_alpha: float,
) -> str:
    return (
        "You are an industrial time-series analysis agent. Use the available tools to analyze the dataset and make reasonable decisions based on tool results.\n\n"
        "Start by inspecting the dataset summary before deciding how to handle missing values.\n"
        "If the target column contains missing values, choose ONE imputation method from the available tool options based on the observed data characteristics.\n"
        "Do not assume a fixed imputation method in advance. Briefly explain why you chose that method.\n"
        "After imputation, continue subsequent analysis using the processed output file returned by the imputation tool, not the original raw file.\n"
        "Then perform anomaly detection and forecasting.\n\n"
        f"Data file: {data_path}\n"
        f"Target column: {target_column}\n"
        f"Anomaly contamination: {contamination}\n"
        f"Forecast steps: {forecast_steps}\n"
        f"Forecast method: {forecast_method}\n"
        f"Forecast window: {forecast_window}\n"
        f"Forecast alpha: {forecast_alpha}\n\n"
        "After using the tools, summarize: (1) what you observed, (2) which imputation method you selected and why, "
        "(3) anomaly results, (4) forecast results, and (5) any limitations or tool errors."
    )
