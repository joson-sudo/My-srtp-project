def build_user_prompt(
    data_path: str,
    target_column: str,
    impute_method: str,
    contamination: float,
    forecast_steps: int,
    forecast_method: str,
    forecast_window: int,
    forecast_alpha: float,
    deep_anomaly_method: str = "isolation_forest",
    deep_forecast_method: str = "lstm",
    use_multimodal: bool = False,
    image_path: str | None = None,
) -> str:
    base = (
        "You are an industrial time-series analyst with access to deep learning and vision tools. "
        "Use the available tools to complete the workflow in order.\n"
        "1) Read dataset summary.\n"
        "2) If missing values exist in the target column, impute them.\n"
        "3) Detect anomalies in the target column.\n"
        "4) Forecast the next values for the target column.\n"
        "5) Run deep anomaly detection (autoencoder-based).\n"
        "6) Run deep learning forecast (LSTM/Transformer).\n"
    )
    if use_multimodal:
        base += "7) Run multimodal diagnosis fusing sensor and vision data.\n"
    base += (
        f"\nData file: {data_path}\n"
        f"Target column: {target_column}\n"
        f"Imputation method: {impute_method}\n"
        f"Anomaly contamination: {contamination}\n"
        f"Forecast steps: {forecast_steps}\n"
        f"Forecast method: {forecast_method}\n"
        f"Forecast window: {forecast_window}\n"
        f"Forecast alpha: {forecast_alpha}\n"
        f"Deep anomaly method: {deep_anomaly_method}\n"
        f"Deep forecast method: {deep_forecast_method}\n"
    )
    if image_path:
        base += f"Equipment image path: {image_path}\n"
    base += "Report the outcome after each step."
    return base
