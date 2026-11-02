# CSE 6242 Team 33 — Food Processing Analysis

This repository analyzes US Open Food Facts products modified from July 2024
through June 2026. The workflow evaluates NOVA-group classification, completes
Nutri-Score labels conservatively, and exports product-level results for Tableau.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the full processed CSVs are already present in `data/processed/`, start the
model workflow:

```powershell
python -m src.modeling
```

To download/refresh the Open Food Facts source and regenerate both processed
datasets first (about 1.3 GB compressed raw input in the current snapshot):

```powershell
python -m src.data_loader
python -m src.modeling
```

The Tableau-ready file is `results/tables/tableau_products.csv`. Open
[`src/Project Dashboard.twb`](src/Project%20Dashboard.twb) in Tableau after the
CSV has been generated. The workbook currently uses an absolute local path to
that CSV; see [Project Overview](docs/PROJECT_OVERVIEW.md) if the repository is
moved to another machine or folder.

The modeling run also refreshes `data/processed/sample_5000_us_food.csv` and
`data/processed/sample_tableau_mock_data.csv` from the current inputs and
Tableau export. Product codes are preserved as text in both samples.

Run the test suite with:

```powershell
python -m pytest -q
```

## Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) — complete file inventory, data
  dictionary, data structures, formulas, modeling approach, run sequence,
  outputs, and Tableau connection details.

The two checked-in sample CSVs are for inspection and Tableau prototyping. The
modeling command requires both full processed CSVs and will not silently
substitute either sample.
