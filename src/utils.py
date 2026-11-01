"""Shared nutrient names and CSV output."""
from pathlib import Path

CORE_NUTRIENTS = [
    "energy-kcal_100g", "fat_100g", "saturated-fat_100g",
    "sugars_100g", "fiber_100g", "proteins_100g", "sodium_100g",
]

def save_table(table, output_dir, name):
    destination = Path(output_dir) / "tables" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    return table
