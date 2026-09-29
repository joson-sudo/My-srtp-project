import json

import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    file_path: str,
    column: str,
    method: str = "isolation_forest",
    contamination: float = 0.1,
    iqr_multiplier: float = 1.5,
) -> str:
    """Detect anomalies with an agent-selected lightweight method."""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)

        col_data = pd.to_numeric(df[column], errors="coerce")
        if col_data.isnull().any():
            return json.dumps({
                "status": "error",
                "message": f"列 {column} 存在缺失值，请先完成缺失值处理。"
            }, ensure_ascii=False)

        method = method.lower().strip()
        if method not in {"isolation_forest", "iqr"}:
            return json.dumps({"status": "error", "message": f"不支持的异常检测方法: {method}"}, ensure_ascii=False)

        details = {}
        if method == "isolation_forest":
            contamination = float(contamination)
            if not 0 < contamination < 0.5:
                return json.dumps({"status": "error", "message": "contamination 必须在 0 和 0.5 之间。"}, ensure_ascii=False)
            model = IsolationForest(contamination=contamination, random_state=42)
            preds = model.fit_predict(col_data.to_numpy().reshape(-1, 1))
            anomaly_indices = df.index[preds == -1].tolist()
            details = {"contamination": contamination}
        else:
            iqr_multiplier = float(iqr_multiplier)
            if iqr_multiplier <= 0:
                return json.dumps({"status": "error", "message": "iqr_multiplier 必须大于 0。"}, ensure_ascii=False)
            q1 = float(col_data.quantile(0.25))
            q3 = float(col_data.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr
            mask = (col_data < lower) | (col_data > upper)
            anomaly_indices = df.index[mask].tolist()
            details = {
                "iqr_multiplier": iqr_multiplier,
                "lower_bound": round(lower, 6),
                "upper_bound": round(upper, 6),
            }

        anomaly_values = [float(df.loc[i, column]) for i in anomaly_indices]
        return json.dumps({
            "status": "success",
            "method": method,
            "total_checked": len(df),
            "anomaly_count": len(anomaly_indices),
            "anomalies_indices": anomaly_indices,
            "anomalies_values": anomaly_values,
            "details": details,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
