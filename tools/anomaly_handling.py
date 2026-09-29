import json
from pathlib import Path

import pandas as pd


def handle_anomalies(file_path: str, column: str, indices: list, method: str) -> str:
    """Handle already-detected anomalies and save a new processed CSV file."""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)

        try:
            anomaly_indices = sorted({int(i) for i in indices})
        except (TypeError, ValueError):
            return json.dumps({"status": "error", "message": "indices 必须是整数索引列表。"}, ensure_ascii=False)

        valid_indices = [i for i in anomaly_indices if 0 <= i < len(df)]
        if not valid_indices:
            return json.dumps({
                "status": "success",
                "message": "没有有效异常索引需要处理。",
                "method": "keep",
                "output_file": file_path,
                "changed_count": 0,
            }, ensure_ascii=False)

        method = method.lower().strip()
        supported = {"keep", "interpolate", "median", "clip"}
        if method not in supported:
            return json.dumps({
                "status": "error",
                "message": f"不支持的异常处理方法: {method}。可选: {sorted(supported)}"
            }, ensure_ascii=False)

        if method == "keep":
            return json.dumps({
                "status": "success",
                "message": "根据当前决策保留异常点，不修改数据。",
                "method": "keep",
                "output_file": file_path,
                "changed_count": 0,
                "handled_indices": valid_indices,
            }, ensure_ascii=False)

        if not pd.api.types.is_numeric_dtype(df[column]):
            return json.dumps({"status": "error", "message": "异常处理仅支持数值列。"}, ensure_ascii=False)

        processed = df.copy()
        original_values = [float(processed.loc[i, column]) for i in valid_indices]
        normal_mask = ~processed.index.isin(valid_indices)
        normal_values = processed.loc[normal_mask, column].dropna()

        if normal_values.empty:
            return json.dumps({"status": "error", "message": "没有足够的正常数据用于异常处理。"}, ensure_ascii=False)

        if method == "interpolate":
            processed.loc[valid_indices, column] = pd.NA
            processed[column] = pd.to_numeric(processed[column], errors="coerce")
            processed[column] = processed[column].interpolate(method="linear", limit_direction="both")
        elif method == "median":
            replacement = float(normal_values.median())
            processed.loc[valid_indices, column] = replacement
        else:
            q1 = float(normal_values.quantile(0.25))
            q3 = float(normal_values.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            processed.loc[valid_indices, column] = processed.loc[valid_indices, column].clip(lower=lower, upper=upper)

        if processed[column].isnull().any():
            return json.dumps({"status": "error", "message": "异常处理后仍存在缺失值。"}, ensure_ascii=False)

        source = Path(file_path)
        output_dir = Path("outputs") / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{source.stem}_{column}_anomaly_{method}{source.suffix or '.csv'}"
        processed.to_csv(output_file, index=False)

        new_values = [float(processed.loc[i, column]) for i in valid_indices]
        return json.dumps({
            "status": "success",
            "message": f"已使用 {method} 处理 {len(valid_indices)} 个异常点。",
            "method": method,
            "handled_indices": valid_indices,
            "original_values": original_values,
            "new_values": new_values,
            "changed_count": len(valid_indices),
            "output_file": output_file.as_posix(),
            "source_file_unchanged": True,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
