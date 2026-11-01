"""Basic figures and Tableau CSV exports."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .utils import CORE_NUTRIENTS, save_table


def plot_eda(correlations, scores, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    image = axes[0].imshow(correlations, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(correlations)), correlations.columns, rotation=90)
    axes[0].set_yticks(range(len(correlations)), correlations.index)
    axes[0].set_title("Training nutrient correlations")
    fig.colorbar(image, ax=axes[0])
    # Every validation point is plotted; training and metrics also use full inputs.
    for group, part in scores.groupby("nova_target"):
        axes[1].scatter(part.PC1, part.PC2, s=3, alpha=0.15, label=f"NOVA {int(group)}")
    axes[1].set(xlabel="PC1", ylabel="PC2", title="Validation nutrients projected onto training PCA")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures/nutrient_eda.png", dpi=150)
    plt.close(fig)


def plot_model_results(model, validation, comparison, output_dir):
    matrix = confusion_matrix(validation.nova_target.astype(int), model.predict(validation), labels=[1, 2, 3, 4])
    save_table(pd.DataFrame(matrix, index=[1, 2, 3, 4], columns=[1, 2, 3, 4]).rename_axis("observed_NOVA").reset_index(), output_dir, "nova_validation_confusion.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].barh(comparison.model, comparison.macro_f1)
    axes[0].set(xlim=(0, 1), xlabel="Validation macro-F1")
    axes[1].imshow(matrix, cmap="Blues")
    axes[1].set_xticks(range(4), [1, 2, 3, 4])
    axes[1].set_yticks(range(4), [1, 2, 3, 4])
    axes[1].set(xlabel="Predicted NOVA", ylabel="Observed NOVA")
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, str(matrix[i, j]), ha="center", color="white" if matrix[i, j] > matrix.max()/2 else "black")
    fig.tight_layout()
    fig.savefig(Path(output_dir) / "figures/nova_models.png", dpi=150)
    plt.close(fig)


def export_tableau(df, model, model_name, grades, scenarios, importance, output_dir, run_final_test=False):
    context = [
        "product_id", "code", "product_name", "brands", "category", "category_source",
        "ingredients_text", "additives_tags", "additives_n", "family_group",
        "nova_split", "nutri_split", "observed_core_nutrients", "needs_review",
        "flag_satfat_gt_fat", "flag_sugars_gt_carbs",
    ]
    # Optional context such as brands remains empty if absent from the input.
    products = df.reindex(columns=context + CORE_NUTRIENTS).copy()
    products["nova_observed"] = df.nova_target
    products["nova_prediction"] = np.nan
    products["predicted_NOVA4_probability"] = np.nan
    support = df.ingredient_present | df.observed_core_nutrients.ge(4) | df.additives_n_feature.notna()
    predict = support & ~df.needs_review
    if not run_final_test:
        predict &= ~df.nova_split.eq("test")
        products.loc[df.nova_split.eq("test"), "nova_observed"] = np.nan
    if predict.any():
        products.loc[predict, "nova_prediction"] = model.predict(df.loc[predict])
        products.loc[predict, "predicted_NOVA4_probability"] = model.predict_proba(df.loc[predict])[:, list(model.classes_).index(4)]
    products["nova_display"] = products.nova_observed.fillna(products.nova_prediction)
    products["nova_source"] = np.select(
        [products.nova_observed.notna(), products.nova_prediction.notna()],
        ["OFF_observed", "model_estimated"], default="unknown")
    products.loc[df.needs_review & products.nova_observed.isna(), "nova_source"] = "needs_review"
    products["prediction_basis"] = np.select(
        [df.needs_review,
         df.nova_split.eq("test") & (not run_final_test),
         ~predict,
         df.nova_split.eq("train"),
         df.nova_split.eq("validation"),
         df.nova_split.eq("test")],
        ["excluded_quality", "test_hidden", "insufficient_input", "training_in_sample",
         "validation_held_out", "test_held_out"],
        default="missing_label_application",
    )
    if not run_final_test:
        products.loc[df.nova_split.eq("test"), "nova_source"] = "held_out_test_hidden"
    grades = grades.copy()
    if not run_final_test:
        held = grades.nutri_split.eq("test")
        grades.loc[held, ["grade_display", "formula_grade_2023", "formula_score_2023"]] = pd.NA
        grades.loc[held, "grade_source"] = "held_out_test_hidden"
    products = products.merge(grades.drop(columns="nutri_split"), on="product_id", how="left", validate="one_to_one")
    products["model"] = model_name
    products["probabilities_calibrated"] = False
    products["nutrition_basis"] = "original_OFF_snapshot"
    save_table(products, output_dir, "tableau_products.csv")
    save_table(scenarios, output_dir, "tableau_scenarios.csv")
    save_table(importance.assign(importance_type="permutation_validation_macro_f1"), output_dir, "tableau_feature_importance.csv")
    long = scenarios[["product_id", "scenario_id"]].merge(
        df[["product_id"] + CORE_NUTRIENTS].melt(id_vars="product_id", var_name="nutrient", value_name="baseline_value"),
        on="product_id", how="left", validate="many_to_many")
    long["scenario_value"] = long.baseline_value
    long["value_basis"] = "assumed_unchanged_not_measured"
    save_table(long, output_dir, "tableau_scenario_nutrients.csv")
