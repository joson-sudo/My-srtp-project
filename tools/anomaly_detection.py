import pandas as pd
import json
from sklearn.ensemble import IsolationForest

def detect_anomalies(file_path: str, column: str, contamination: float = 0.1) -> str:
    """使用孤立森林检测某列的异常值"""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)
        
        # 提取需要检测的列
        col_data = df[column].copy()
        
        # 孤立森林不能处理NaN，必须在处理前填充（为了检测临时填充）
        if col_data.isnull().any():
            return json.dumps({
                "status": "error", 
                "message": f"列 {column} 存在缺失值(NaN)，孤立森林无法直接处理！请先调用数据填补工具进行处理。"
            }, ensure_ascii=False)
        
        # 将数据转为 sklearn 需要的二维数组结构
        X = col_data.values.reshape(-1, 1)
        
        # 实例化并训练模型
        model = IsolationForest(contamination=contamination, random_state=42)
        preds = model.fit_predict(X)
        
        # preds 返回 1 为正常点， -1 为异常点
        anomalies_indices = df.index[preds == -1].tolist()
        anomalies_values = df.loc[anomalies_indices, column].tolist()
        
        return json.dumps({
            "status": "success",
            "message": f"孤立森林算法已经完成运算。",
            "total_checked": len(df),
            "anomaly_count": len(anomalies_indices),
            "anomalies_indices": anomalies_indices,
            "anomalies_values": anomalies_values
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
