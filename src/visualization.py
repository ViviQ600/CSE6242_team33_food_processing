"""Figure persistence and Tableau exports based on real model outputs.

This module creates no mock probabilities, approximate grades, or proxy SHAP.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from .utils import WorkflowState


def show_figure(name, figure_dir):
    """Save the current figure, close it, and display the saved image."""
    plt.tight_layout()
    destination = Path(figure_dir) / name
    plt.savefig(destination, dpi=150, bbox_inches="tight")
    plt.close()
    display(Image(filename=str(destination)))


def export_tableau(state: WorkflowState) -> WorkflowState:
    """Export actual validation predictions, nutrition baselines, and provenance for Tableau.

    Required artifacts: NUTRIENTS, baseline_grades, best_name, save_table, scenario_results, val, validation_export.
    Published artifacts: tableau.
    """
    NUTRIENTS = state.require("NUTRIENTS")
    baseline_grades = state.require("baseline_grades")
    best_name = state.require("best_name")
    save_table = state.require("save_table")
    scenario_results = state.require("scenario_results")
    val = state.require("val")
    validation_export = state.require("validation_export")
    tableau = validation_export.merge(
        baseline_grades[["code", "grade_display", "grade_provenance"]],
        on="code",
        how="left",
        validate="one_to_one",
    )
    tableau = tableau.merge(
        val[["code"] + NUTRIENTS + ["category_source"]],
        on="code",
        how="left",
        validate="one_to_one",
    )
    tableau["prediction_model"] = best_name
    tableau["evaluation_split"] = "validation"
    save_table(tableau, "24_tableau_products.csv")
    print("Dashboard products:", len(tableau), "Scenario rows:", len(scenario_results))
    print("No proxy probabilities, proxy SHAP, or approximate Nutri-Score formula used.")
    state.update({"tableau": tableau})
    return state


def plot_target_distributions(nova_dist, grade_dist, show_figure):
    """Plot the observed NOVA and Nutri-Score target distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].bar(nova_dist.NOVA.astype(str), nova_dist.rows, color="#315b3a")
    axes[0].set(title="Valid NOVA labels", xlabel="NOVA", ylabel="Products")
    axes[1].bar(grade_dist.grade.str.upper(), grade_dist.rows, color="#667f40")
    axes[1].set(title="Valid Nutri-Score labels", xlabel="Grade", ylabel="Products")
    show_figure("01_target_distributions.png")


def plot_nutrient_coverage(coverage, show_figure):
    """Plot valid coverage for the seven baseline nutrient fields."""
    plt.figure(figsize=(8, 3.5))
    plt.barh(coverage.field, coverage.usable_pct, color="#315b3a")
    plt.xlabel("Usable coverage (%)")
    plt.xlim(0, 100)
    show_figure("02_nutrient_coverage.png")


def plot_nutrient_correlations(pearson_corr, spearman_corr, FIG_DIR):
    """Plot training-only Pearson and Spearman matrices with common color limits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    short_labels = ["Energy (kcal)", "Fat", "Saturated fat", "Sugars", "Fiber", "Protein", "Sodium"]
    for ax, matrix, title in zip(
        axes, [pearson_corr, spearman_corr], ["Pearson correlation", "Spearman correlation"]
    ):
        image_corr = ax.imshow(matrix.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(short_labels)), short_labels, rotation=65, ha="right")
        ax.set_yticks(range(len(short_labels)), short_labels)
        ax.set_title(title)
        for r in range(len(short_labels)):
            for c in range(len(short_labels)):
                value = matrix.iloc[r, c]
                if pd.notna(value):
                    ax.text(
                        c,
                        r,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(value) > 0.65 else "black",
                    )
    fig.colorbar(image_corr, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    fig.subplots_adjust(bottom=0.25, left=0.11, right=0.88, wspace=0.45)
    fig.savefig(FIG_DIR / "07_nutrient_correlations.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    display(Image(filename=str(FIG_DIR / "07_nutrient_correlations.png")))


def plot_pca(pca_variance, pca_val_export, SEED, show_figure):
    """Plot explained variance and a validation-only PCA projection."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(
        pca_variance.component,
        pca_variance.explained_variance_ratio,
        color="#315b3a",
        label="Individual",
    )
    axes[0].plot(
        pca_variance.component,
        pca_variance.cumulative_variance_ratio,
        "o-",
        color="#b36c24",
        label="Cumulative",
    )
    axes[0].axhline(0.8, linestyle="--", color="gray", linewidth=0.8)
    axes[0].set(
        ylabel="Explained variance ratio", ylim=(0, 1.05), title="PCA fitted on training nutrients"
    )
    axes[0].legend()
    plot_pca = pca_val_export.sample(n=min(5000, len(pca_val_export)), random_state=SEED)
    colors = ["#4477aa", "#eeaa33", "#228833", "#cc6677"]
    for group, color in zip([1, 2, 3, 4], colors):
        part = plot_pca.loc[plot_pca.nova_target.eq(group)]
        axes[1].scatter(
            part.PC1, part.PC2, s=7, alpha=0.35, color=color, label=f"NOVA {group}", rasterized=True
        )
    axes[1].set(xlabel="PC1", ylabel="PC2", title="Validation projection (up to 5,000 points)")
    axes[1].legend(markerscale=2)
    show_figure("08_nutrient_pca.png")


def plot_feature_ablation(ablation, show_figure):
    """Plot same-split validation macro-F1 for the structured input groups."""
    plt.figure(figsize=(7, 3.5))
    plt.bar(ablation.feature_set, ablation.macro_f1, color=["#627e52", "#9da96d", "#315b3a"])
    plt.ylim(0, 1)
    plt.ylabel("Validation macro-F1")
    show_figure("03_feature_ablation.png")


def plot_calibration(mean_prob, observed, show_figure):
    """Plot reliability bins without claiming the probabilities are calibrated."""
    plt.figure(figsize=(4.5, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.plot(mean_prob, observed, "o-", color="#315b3a")
    plt.xlabel("Mean predicted P(NOVA4)")
    plt.ylabel("Observed NOVA4 fraction")
    show_figure("05_calibration.png")


def plot_confusion_matrix(cm, show_figure):
    """Plot observed-versus-predicted NOVA validation counts."""
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Greens")
    plt.xticks(range(4), [1, 2, 3, 4])
    plt.yticks(range(4), [1, 2, 3, 4])
    plt.xlabel("Predicted")
    plt.ylabel("True NOVA")
    for r in range(4):
        for c in range(4):
            plt.text(
                c,
                r,
                f"{cm.iloc[r, c]:,}",
                ha="center",
                va="center",
                color="white" if cm.iloc[r, c] > cm.to_numpy().max() / 2 else "black",
            )
    show_figure("04_confusion_matrix.png")


def plot_permutation_importance(importance, show_figure):
    """Plot actual validation permutation importance for the leading features."""
    plot_imp = importance.head(12).sort_values("macro_f1_drop_mean")
    plt.figure(figsize=(7, 4))
    plt.barh(plot_imp.feature, plot_imp.macro_f1_drop_mean, color="#315b3a")
    plt.xlabel("Validation macro-F1 decrease after permutation")
    show_figure("06_permutation_importance.png")
