from tools.anomaly_detection import detect_anomalies
from tools.data_imputation import impute_missing_values
from tools.data_summary import extract_data_summary
from tools.time_series_forecast import forecast_series
from tools.deep_tools import deep_anomaly_detect, deep_forecast
from tools.vision_tools import vision_inspect, multimodal_diagnosis

TOOL_FUNCTIONS = {
    "extract_data_summary": extract_data_summary,
    "impute_missing_values": impute_missing_values,
    "detect_anomalies": detect_anomalies,
    "forecast_series": forecast_series,
    "deep_anomaly_detect": deep_anomaly_detect,
    "deep_forecast": deep_forecast,
    "vision_inspect": vision_inspect,
    "multimodal_diagnosis": multimodal_diagnosis,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "extract_data_summary",
            "description": "Read a CSV file and return summary statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "head_rows": {"type": "integer", "description": "Number of head rows to include"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "impute_missing_values",
            "description": "Fill missing values in a target column.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "column": {"type": "string", "description": "Target column"},
                    "method": {"type": "string", "enum": ["mean", "forward"], "description": "Imputation method"}
                },
                "required": ["file_path", "column", "method"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_anomalies",
            "description": "Detect anomalies in a target column using Isolation Forest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "column": {"type": "string", "description": "Target column"},
                    "contamination": {"type": "number", "description": "Expected anomaly ratio"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "forecast_series",
            "description": "Forecast a target column with baseline methods.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "column": {"type": "string", "description": "Target column"},
                    "steps": {"type": "integer", "description": "Forecast steps"},
                    "method": {"type": "string", "enum": ["moving_average", "ewm", "last"], "description": "Forecast method"},
                    "window": {"type": "integer", "description": "Moving average window"},
                    "alpha": {"type": "number", "description": "Exponential smoothing alpha"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_anomaly_detect",
            "description": "Detect anomalies with deep autoencoder or isolation forest.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "column": {"type": "string", "description": "Target column"},
                    "method": {"type": "string", "enum": ["isolation_forest", "autoencoder", "vae"], "description": "Detection method"},
                    "contamination": {"type": "number", "description": "Expected anomaly ratio"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "deep_forecast",
            "description": "Deep learning forecasting with LSTM or Transformer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to CSV file"},
                    "column": {"type": "string", "description": "Target column"},
                    "steps": {"type": "integer", "description": "Forecast horizon"},
                    "method": {"type": "string", "enum": ["lstm", "transformer"], "description": "Model architecture"}
                },
                "required": ["file_path", "column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "vision_inspect",
            "description": "Inspect industrial equipment image for defects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string", "description": "Path to image file"},
                    "task": {"type": "string", "enum": ["anomaly_score", "classify"], "description": "Vision task"},
                    "reference_dir": {"type": "string", "description": "Directory of normal reference images"}
                },
                "required": ["image_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multimodal_diagnosis",
            "description": "Cross-modal fusion of sensor and vision analysis for comprehensive equipment diagnosis with risk level and RUL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data_path": {"type": "string", "description": "Path to sensor CSV file"},
                    "column": {"type": "string", "description": "Target sensor column"},
                    "image_path": {"type": "string", "description": "Optional equipment image path"},
                    "forecast_steps": {"type": "integer", "description": "Forecast horizon"}
                },
                "required": ["data_path", "column"]
            }
        }
    }
]
