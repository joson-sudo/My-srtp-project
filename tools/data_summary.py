import json
import pandas as pd


def extract_data_summary(file_path: str = "data/sample_data.csv", head_rows: int = 5) -> str:
    """Return summary metadata for a CSV file, including candidate numeric targets."""
    try:
        df = pd.read_csv(file_path)
        head_rows = max(0, int(head_rows))

        missing = {col: int(val) for col, val in df.isnull().sum().items()}
        missing_indices = {
            col: [int(i) for i in df.index[df[col].isnull()].tolist()]
            for col in df.columns
            if df[col].isnull().any()
        }

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        excluded_name_tokens = ("id", "index", "time", "date", "timestamp")
        candidate_numeric_targets = [
            col for col in numeric_columns
            if not any(token in col.lower() for token in excluded_name_tokens)
        ]

        summary = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "numeric_columns": numeric_columns,
            "candidate_numeric_targets": candidate_numeric_targets,
            "missing": missing,
            "missing_indices": missing_indices,
            "describe": json.loads(df.describe().to_json())
        }

        if head_rows:
            summary["head"] = json.loads(df.head(head_rows).to_json(orient="records"))

        return json.dumps({"status": "success", "data_summary": summary}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
