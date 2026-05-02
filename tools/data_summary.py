import json
import pandas as pd


def extract_data_summary(file_path: str = "data/sample_data.csv", head_rows: int = 5) -> str:
    """Return summary metadata for a CSV file."""
    try:
        df = pd.read_csv(file_path)
        head_rows = max(0, int(head_rows))

        missing = {col: int(val) for col, val in df.isnull().sum().items()}
        summary = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "missing": missing,
            "describe": json.loads(df.describe().to_json())
        }

        if head_rows:
            summary["head"] = json.loads(df.head(head_rows).to_json(orient="records"))

        return json.dumps({"status": "success", "data_summary": summary}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
