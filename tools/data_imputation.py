import json
from pathlib import Path

import pandas as pd


def impute_missing_values(file_path: str, column: str, method: str) -> str:
    """Fill missing values in one column and save to a new processed CSV file."""
    try:
        df = pd.read_csv(file_path)
        if column not in df.columns:
            return json.dumps({"status": "error", "message": f"列名 {column} 不存在。"}, ensure_ascii=False)

        missing_count = int(df[column].isnull().sum())
        if missing_count == 0:
            return json.dumps({
                "status": "success",
                "message": f"列 {column} 暂无缺失值，无需填补。",
                "output_file": file_path,
                "method": None,
                "filled_count": 0,
            }, ensure_ascii=False)

        method = method.lower().strip()
        supported = {"mean", "median", "forward", "interpolate"}
        if method not in supported:
            return json.dumps({
                "status": "error",
                "message": f"不支持的填补方法: {method}。可选: {sorted(supported)}"
            }, ensure_ascii=False)

        df = df.copy()

        if method == "mean":
            fill_value = df[column].mean()
            df[column] = df[column].fillna(fill_value)
            action = f"mean ({fill_value:.4f})"
        elif method == "median":
            fill_value = df[column].median()
            df[column] = df[column].fillna(fill_value)
            action = f"median ({fill_value:.4f})"
        elif method == "forward":
            df[column] = df[column].ffill().bfill()
            action = "forward fill (with backward fill for leading gaps)"
        else:
            if not pd.api.types.is_numeric_dtype(df[column]):
                return json.dumps({
                    "status": "error",
                    "message": "interpolate 仅支持数值列。"
                }, ensure_ascii=False)
            df[column] = df[column].interpolate(method="linear", limit_direction="both")
            action = "linear interpolation"

        remaining_missing = int(df[column].isnull().sum())
        if remaining_missing > 0:
            return json.dumps({
                "status": "error",
                "message": f"填补后仍有 {remaining_missing} 个缺失值。"
            }, ensure_ascii=False)

        source = Path(file_path)
        output_dir = Path("outputs") / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{source.stem}_{column}_{method}{source.suffix or '.csv'}"
        df.to_csv(output_file, index=False)

        return json.dumps({
            "status": "success",
            "message": f"使用 {action} 填补了 {column} 列的 {missing_count} 个缺失值。",
            "method": method,
            "filled_count": missing_count,
            "output_file": output_file.as_posix(),
            "source_file_unchanged": True,
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
