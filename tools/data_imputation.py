import pandas as pd
import json

def impute_missing_values(file_path: str, column: str, method: str = 'mean') -> str:
    """使用指定方法填补指定列的缺失值"""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)
        
        missing_count = df[column].isnull().sum()
        if missing_count == 0:
            return json.dumps({"status": "success", "message": f"列 {column} 暂无缺失值。"}, ensure_ascii=False)
        
        if method == 'mean':
            fill_value = df[column].mean()
            df = df.copy() # Avoid SettingWithCopyWarning
            df[column] = df[column].fillna(fill_value)
            action = f"均值 ({fill_value:.2f})"
        else:
            df = df.copy()
            df[column] = df[column].ffill()
            action = "前向填充 (forward fill)"
        
        # 覆盖原文件以保存修改
        df.to_csv(file_path, index=False)
        
        return json.dumps({
            "status": "success", 
            "message": f"成功使用 {action} 填补了 {column} 列的 {missing_count} 个缺失值，并已保存到 {file_path}。"
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
