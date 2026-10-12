"""Utility Functions

General helper utilities for the project.
"""

import json
from pathlib import Path
import yaml

def load_config(config_path="config/config.yaml"):
    """Load configuration from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_config(config, output_path="config/config.yaml"):
    """Save config to YAML."""
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False)

def create_directory_structure(base_path="."):
    """Ensure project folders exist."""
    directories = [
        "data/raw",
        "data/processed",
        "results/figures",
        "results/tables",
        "results/models",
        "notebooks",
    ]
    for directory in directories:
        path = Path(base_path) / directory
        path.mkdir(parents=True, exist_ok=True)

def print_summary(data):
    """Print a summary of data or dictionary content."""
    if isinstance(data, dict):
        print(json.dumps(data, indent=2))
    else:
        print(data.to_string())
