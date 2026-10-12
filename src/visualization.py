"""Visualization Module

Contains functions for chart generation.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Set plot style."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (12, 6)

def plot_distributions(df, output_dir="results/figures/"):
    """Create distribution plots for numeric columns."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        df[col].hist(bins=50, ax=ax)
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"{col}_distribution.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_correlation_heatmap(correlation_matrix, output_path="results/figures/correlation_heatmap.png"):
    """Create a correlation heatmap."""
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
