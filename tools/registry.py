from tools.anomaly_detection import detect_anomalies
from tools.anomaly_handling import handle_anomalies
from tools.data_imputation import impute_missing_values
from tools.data_summary import extract_data_summary
from tools.forecast_evaluation import evaluate_forecast_methods
from tools.time_series_forecast import forecast_series

TOOL_FUNCTIONS = {
    "extract_data_summary": extract_data_summary,
    "impute_missing_values": impute_missing_values,
    "detect_anomalies": detect_anomalies,
    "handle_anomalies": handle_anomalies,
    "evaluate_forecast_methods": evaluate_forecast_methods,
    "forecast_series": forecast_series,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_data_summary",
            "description": "Inspect a CSV file and return rows, columns, missing-value counts and locations, summary statistics, and sample rows. Use this before choosing preprocessing or modeling actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "head_rows": {"type": "integer", "description": "Number of head rows to include"}
                },
                "required": ["file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "impute_missing_values",
            "description": "Fill missing values in a target column. Choose the strategy from evidence in the data. The raw file is preserved and output_file should be used by later tools.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Current CSV file"},
                    "column": {"type": "string", "description": "Target numeric column"},
                    "method": {
                        "type": "string",
                        "enum": ["mean", "median", "forward", "interpolate"],
                        "description": "mean: global average; median: robust global center; forward: preserve recent temporal level; interpolate: estimate gaps from neighboring observations"
                    }
                },
                "required": ["file_path", "column", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect anomalies with a selected lightweight method. Choose the method and parameters based on the data size and distribution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Current processed CSV file"},
                    "column": {"type": "string", "description": "Target numeric column"},
                    "method": {
                        "type": "string",
                        "enum": ["isolation_forest", "iqr"],
                        "description": "Isolation Forest is model-based; IQR is a simple robust statistical rule"
                    },
                    "contamination": {"type": "number", "description": "Expected anomaly fraction for Isolation Forest; must be between 0 and 0.5"},
                    "iqr_multiplier": {"type": "number", "description": "IQR fence multiplier, commonly around 1.5"}
                },
                "required": ["file_path", "column", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_anomalies",
            "description": "Optionally handle anomaly indices already returned by anomaly detection. Use only if changing those points is justified for the downstream task. Raw/current input is preserved and output_file is returned when data is changed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Current CSV file"},
                    "column": {"type": "string", "description": "Target numeric column"},
                    "indices": {"type": "array", "items": {"type": "integer"}, "description": "Anomaly row indices returned by detect_anomalies"},
                    "method": {
                        "type": "string",
                        "enum": ["keep", "interpolate", "median", "clip"],
                        "description": "keep: do not modify; interpolate: replace using neighboring trend; median: replace with robust normal median; clip: cap using IQR bounds from non-anomalous points"
                    }
                },
                "required": ["file_path", "column", "indices", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_forecast_methods",
            "description": "Compare several lightweight forecasting candidates with rolling one-step validation and return MAE/RMSE evidence. Candidates include last value, moving averages, exponential smoothing, and a linear trend baseline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Current processed CSV file after any justified preprocessing"},
                    "column": {"type": "string", "description": "Target numeric column"},
                    "validation_points": {"type": "integer", "description": "Number of recent points used for rolling validation; tool adapts this to small datasets"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_series",
            "description": "Generate the final forecast using a method selected from evaluation evidence or other justified tool results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Current processed CSV file"},
                    "column": {"type": "string", "description": "Target numeric column"},
                    "steps": {"type": "integer", "description": "Forecast horizon requested by the user"},
                    "method": {"type": "string", "enum": ["moving_average", "ewm", "last", "linear_trend"], "description": "Forecast method chosen after evaluation"},
                    "window": {"type": "integer", "description": "Moving-average window when method=moving_average"},
                    "alpha": {"type": "number", "description": "Smoothing factor when method=ewm"}
                },
                "required": ["file_path", "column", "steps", "method"]
            }
        }
    }
]
