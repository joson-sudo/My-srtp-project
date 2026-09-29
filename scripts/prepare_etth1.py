"""Prepare a reproducible ETTh1 subset for the SRTP agent.

This script downloads the public ETTh1 benchmark, keeps the first 1000 hourly
rows as a reference subset, and creates a second controlled test file with
deterministic missing values and artificial spikes.

Run from the project root:
    python scripts/prepare_etth1.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/"
    "main/ETT-small/ETTh1.csv"
)
ROWS = 1000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
BASE_OUTPUT = DATA_DIR / "ETTh1_1000.csv"
TEST_OUTPUT = DATA_DIR / "ETTh1_agent_test.csv"
GROUND_TRUTH_OUTPUT = DATA_DIR / "ETTh1_agent_test_ground_truth.json"

REQUIRED_COLUMNS = ["date", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def python_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading first {ROWS} rows of ETTh1...")
    base = pd.read_csv(SOURCE_URL, nrows=ROWS)

    missing_columns = [c for c in REQUIRED_COLUMNS if c not in base.columns]
    if missing_columns:
        raise RuntimeError(f"Unexpected ETTh1 schema; missing columns: {missing_columns}")
    if len(base) != ROWS:
        raise RuntimeError(f"Expected {ROWS} rows, received {len(base)} rows")

    base.to_csv(BASE_OUTPUT, index=False)

    test = base.copy()
    changes: list[dict] = []

    def record_change(row: int, column: str, new_value, kind: str, note: str) -> None:
        original = python_value(test.at[row, column])
        test.at[row, column] = new_value
        changes.append(
            {
                "row_index": row,
                "date": str(base.at[row, "date"]),
                "column": column,
                "kind": kind,
                "original_value": original,
                "new_value": python_value(new_value),
                "note": note,
            }
        )

    # Isolated missing values in different variables.
    record_change(120, "OT", np.nan, "missing", "isolated missing value")
    record_change(275, "HUFL", np.nan, "missing", "isolated missing value")
    record_change(620, "MUFL", np.nan, "missing", "isolated missing value")

    # A short contiguous missing block, useful for testing whether the agent treats
    # a gap differently from a single missing observation.
    for row in range(430, 435):
        record_change(row, "LULL", np.nan, "missing", "5-point contiguous missing block")

    def inject_high_spike(row: int, column: str, iqr_factor: float = 8.0) -> None:
        series = pd.to_numeric(base[column], errors="coerce").dropna()
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr > 0:
            new_value = q3 + iqr_factor * iqr
        else:
            std = float(series.std())
            new_value = float(series.max()) + max(1.0, 6.0 * std)
        record_change(
            row,
            column,
            float(new_value),
            "artificial_spike",
            f"set to Q3 + {iqr_factor:g}*IQR to create a controlled high outlier",
        )

    # Several controlled spikes; some columns remain untouched as controls.
    inject_high_spike(180, "OT")
    inject_high_spike(510, "HUFL")
    inject_high_spike(760, "MULL")
    inject_high_spike(845, "LULL")

    test.to_csv(TEST_OUTPUT, index=False)

    ground_truth = {
        "source_url": SOURCE_URL,
        "subset": "first 1000 rows of ETTh1",
        "purpose": (
            "Controlled agent debugging only. Artificial corruption is injected so "
            "we know which missing values and spikes the agent should encounter."
        ),
        "base_file": BASE_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "test_file": TEST_OUTPUT.relative_to(PROJECT_ROOT).as_posix(),
        "changes": changes,
    }
    GROUND_TRUTH_OUTPUT.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Created: {BASE_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Created: {TEST_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Created: {GROUND_TRUTH_OUTPUT.relative_to(PROJECT_ROOT)}")
    print(f"Injected {len(changes)} controlled changes.")


if __name__ == "__main__":
    main()
