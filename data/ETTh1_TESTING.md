# ETTh1 test data for the adaptive agent

The project keeps the data-generation process reproducible instead of committing a large benchmark CSV directly.

## Source

`scripts/prepare_etth1.py` downloads the public ETTh1 benchmark from the original ETT dataset repository:

`https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv`

The ETTh1 schema is:

`date, HUFL, HULL, MUFL, MULL, LUFL, LULL, OT`

The script keeps the first 1000 hourly observations.

## Generated files

Run from the project root:

```powershell
python scripts/prepare_etth1.py
```

It creates:

- `data/ETTh1_1000.csv`: an untouched 1000-row subset for normal analysis.
- `data/ETTh1_agent_test.csv`: the same subset with deterministic missing values and artificial spikes for testing agent decisions.
- `data/ETTh1_agent_test_ground_truth.json`: exact injected locations, original values, and modified values.

These generated files are ignored by Git because they can be reproduced at any time.

## Controlled corruption in the agent test file

The test generator injects:

- isolated missing values in `OT`, `HUFL`, and `MUFL`;
- a five-point contiguous missing block in `LULL`;
- artificial high spikes in `OT`, `HUFL`, `MULL`, and `LULL`.

The spikes are generated using `Q3 + 8 * IQR` from the untouched 1000-row subset, so they are deliberately strong outliers rather than manually chosen arbitrary numbers.

This file is for controlled debugging of the agent's preprocessing and anomaly decisions. It should not be presented as an untouched benchmark or used to claim forecasting performance on the original ETTh1 dataset.

## Run examples

Analyze all suitable numeric columns:

```powershell
python main.py --data data/ETTh1_agent_test.csv
```

Analyze only transformer oil temperature (`OT`):

```powershell
python main.py --data data/ETTh1_agent_test.csv --column OT
```

Analyze the unmodified 1000-row subset:

```powershell
python main.py --data data/ETTh1_1000.csv --column OT
```
