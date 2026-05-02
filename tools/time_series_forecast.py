import json
import pandas as pd

def forecast_series(
    file_path: str,
    column: str,
    steps: int = 5,
    method: str = "moving_average",
    window: int = 5,
    alpha: float = 0.4
) -> str:
    """使用基线方法预测未来趋势"""
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

        series = df[column].dropna()
        if series.empty:
            return json.dumps({"status": "error", "message": "没有足够的数据来进行预测。"}, ensure_ascii=False)

        method = method.lower().strip()
        if method not in {"moving_average", "ewm", "last"}:
            return json.dumps({"status": "error", "message": f"不支持的预测方法: {method}。"}, ensure_ascii=False)

        if method == "last":
            value = float(series.iloc[-1])
        elif method == "ewm":
            if not 0 < alpha < 1:
                return json.dumps({"status": "error", "message": "alpha 必须在 0 和 1 之间。"}, ensure_ascii=False)
            value = float(series.ewm(alpha=alpha, adjust=False).mean().iloc[-1])
        else:
            window = max(1, int(window))
            value = float(series.tail(window).mean())

        forecast = [round(value, 2)] * int(steps)

        return json.dumps({
            "status": "success",
            "message": f"已使用 {method} 方法预测接下来 {steps} 个时间点的数值。",
            "forecast_values": forecast
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
