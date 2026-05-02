def build_user_prompt(
    data_path: str,
    target_column: str,
    impute_method: str,
    contamination: float,
    forecast_steps: int,
    forecast_method: str,
    forecast_window: int,
    forecast_alpha: float
) -> str:
    return (
        "You are an industrial time-series analyst. "
        "Use the available tools to complete the workflow in order.\n"
        "1) Read dataset summary.\n"
        "2) If missing values exist in the target column, impute them.\n"
        "3) Detect anomalies in the target column.\n"
        "4) Forecast the next values for the target column.\n\n"
        f"Data file: {data_path}\n"
        f"Target column: {target_column}\n"
        f"Imputation method: {impute_method}\n"
        f"Anomaly contamination: {contamination}\n"
        f"Forecast steps: {forecast_steps}\n"
        f"Forecast method: {forecast_method}\n"
        f"Forecast window: {forecast_window}\n"
        f"Forecast alpha: {forecast_alpha}\n"
        "Report the outcome after each step."
    )
