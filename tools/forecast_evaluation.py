import json
import math

import numpy as np
import pandas as pd


def _predict_one(history: pd.Series, method: str, window: int = 5, alpha: float = 0.4) -> float:
    values = history.astype(float).reset_index(drop=True)
    if values.empty:
        raise ValueError("history is empty")

    if method == "last":
        return float(values.iloc[-1])

    if method == "moving_average":
        window = max(1, min(int(window), len(values)))
        return float(values.tail(window).mean())

    if method == "ewm":
        if not 0 < float(alpha) < 1:
            raise ValueError("alpha must be between 0 and 1")
        return float(values.ewm(alpha=float(alpha), adjust=False).mean().iloc[-1])

    if method == "linear_trend":
        if len(values) < 2:
            return float(values.iloc[-1])
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values.to_numpy(dtype=float), 1)
        return float(intercept + slope * len(values))

    raise ValueError(f"unsupported method: {method}")


def evaluate_forecast_methods(file_path: str, column: str, validation_points: int = 3) -> str:
    """Compare simple forecasting candidates with rolling one-step validation."""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)

        series = pd.to_numeric(df[column], errors="coerce")
        if series.isnull().any():
            return json.dumps({
                "status": "error",
                "message": f"列 {column} 仍有缺失值，请先处理后再做预测验证。"
            }, ensure_ascii=False)

        n = len(series)
        if n < 5:
            return json.dumps({
                "status": "error",
                "message": f"只有 {n} 个数据点，至少需要 5 个点才能进行简单滚动验证。"
            }, ensure_ascii=False)

        validation_points = max(2, int(validation_points))
        validation_points = min(validation_points, max(2, n // 3), n - 2)
        split = n - validation_points

        candidates = [
            {"method": "last"},
            {"method": "moving_average", "window": 3},
            {"method": "moving_average", "window": 5},
            {"method": "ewm", "alpha": 0.2},
            {"method": "ewm", "alpha": 0.4},
            {"method": "ewm", "alpha": 0.7},
            {"method": "linear_trend"},
        ]

        results = []
        for candidate in candidates:
            predictions = []
            actuals = []
            for i in range(split, n):
                history = series.iloc[:i]
                pred = _predict_one(
                    history,
                    method=candidate["method"],
                    window=candidate.get("window", 5),
                    alpha=candidate.get("alpha", 0.4),
                )
                predictions.append(pred)
                actuals.append(float(series.iloc[i]))

            errors = np.array(predictions) - np.array(actuals)
            mae = float(np.mean(np.abs(errors)))
            rmse = float(math.sqrt(np.mean(errors ** 2)))
            results.append({
                **candidate,
                "mae": round(mae, 6),
                "rmse": round(rmse, 6),
                "predictions": [round(float(x), 6) for x in predictions],
                "actuals": [round(float(x), 6) for x in actuals],
            })

        results.sort(key=lambda item: (item["mae"], item["rmse"]))
        return json.dumps({
            "status": "success",
            "validation_points": validation_points,
            "evaluation": results,
            "best_by_mae": results[0],
            "note": "Metrics are from rolling one-step validation on a very small recent holdout; use them as evidence, not as proof of general superiority."
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
