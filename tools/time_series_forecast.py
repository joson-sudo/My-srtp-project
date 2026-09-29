import json

import numpy as np
import pandas as pd


def forecast_series(
    file_path: str,
    column: str,
    steps: int = 5,
    method: str = "moving_average",
    window: int = 5,
    alpha: float = 0.4
) -> str:
    """Forecast future values with a selected lightweight baseline method."""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)

        try:
            steps = int(steps)
        except (TypeError, ValueError):
            return json.dumps({"status": "error", "message": "steps 必须是整数。"}, ensure_ascii=False)

        if steps <= 0:
            return json.dumps({"status": "error", "message": "steps 必须大于 0。"}, ensure_ascii=False)

        series = pd.to_numeric(df[column], errors="coerce").dropna().reset_index(drop=True)
        if series.empty:
            return json.dumps({"status": "error", "message": "没有足够的数据来进行预测。"}, ensure_ascii=False)

        method = method.lower().strip()
        supported = {"moving_average", "ewm", "last", "linear_trend"}
        if method not in supported:
            return json.dumps({"status": "error", "message": f"不支持的预测方法: {method}。"}, ensure_ascii=False)

        if method == "last":
            forecast = [float(series.iloc[-1])] * steps

        elif method == "ewm":
            alpha = float(alpha)
            if not 0 < alpha < 1:
                return json.dumps({"status": "error", "message": "alpha 必须在 0 和 1 之间。"}, ensure_ascii=False)
            value = float(series.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
            forecast = [value] * steps

        elif method == "moving_average":
            window = max(1, min(int(window), len(series)))
            value = float(series.tail(window).mean())
            forecast = [value] * steps

        else:
            if len(series) < 2:
                return json.dumps({"status": "error", "message": "linear_trend 至少需要 2 个数据点。"}, ensure_ascii=False)
            x = np.arange(len(series), dtype=float)
            slope, intercept = np.polyfit(x, series.to_numpy(dtype=float), 1)
            forecast = [float(intercept + slope * (len(series) + i)) for i in range(steps)]

        return json.dumps({
            "status": "success",
            "method": method,
            "parameters": {
                "window": int(window) if method == "moving_average" else None,
                "alpha": float(alpha) if method == "ewm" else None,
            },
            "message": f"已使用 {method} 方法预测接下来 {steps} 个时间点。",
            "forecast_values": [round(v, 4) for v in forecast]
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
