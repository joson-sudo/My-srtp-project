import pandas as pd
import json

def forecast_series(file_path: str, column: str, steps: int = 5) -> str:
    """使用简单移动平均方法预测未来趋势 (简化版占位符)"""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)
            
        # 这里为了快速部署，暂用 Pandas 的指数平滑或移动平均模拟 ARIMA
        # 后续可以升级为来自 statsmodels 的 ARIMA
        recent_data = df[column].dropna().tail(3)
        if len(recent_data) == 0:
            return json.dumps({"status": "error", "message": "没有足够的数据来进行预测。"}, ensure_ascii=False)
            
        trend_value = recent_data.mean()
        forecast = [round(trend_value, 2)] * steps
        
        return json.dumps({
            "status": "success",
            "message": f"已成功预测接下来 {steps} 个时间点的数值。",
            "forecast_values": forecast
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
