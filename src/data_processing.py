"""Data Processing Module

Handles loading, cleaning, and transforming data.
"""

from pathlib import Path
import pandas as pd

def load_raw_data(filepath):
    """Load raw data from a CSV or Excel file."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

def clean_data(df):
    """Remove duplicates and basic invalid entries."""
    df = df.drop_duplicates()
    return df

def save_processed_data(df, output_path):
    """Save processed data to a CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
