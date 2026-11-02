"""EDA plots, model diagnostics, and the Tableau-compatible product export."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

TABLEAU_EXPORT_COLUMNS = [
    "code",
    "product_name",
    "unified_category",
    "url",
    "nutriscore_provenance",
    "nutriscore_grade_final",
    "nutriscore_score_final",
    "pred_nova_group",
    "pred_nova4_prob",
    "sugars_100g",
    "sodium_100g",
    "saturated-fat_100g",
    "fiber_100g",
    "proteins_100g",
    "additives_n",
    "has_emulsifier",
    "has_sweetener",
    "shap_sugars_nova",
    "shap_sodium_nova",
    "shap_additives_nova",
    "nutri_sugar_penalty_pts",
    "nutri_sodium_penalty_pts",
    "is_healthy_benchmark",
    "has_preservative",
    "has_artificial_color",
    "has_flavor_enhancer",
    "has_other_additive",
    "sub_category",
]


def plot_eda(correlations, scores, output_dir):
    """Save nutrient correlation and validation PCA plots."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    output_dir = Path(output_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(9, 7))
    sns.heatmap(correlations, annot=True, fmt=".2f", cmap="vlag", center=0, ax=axis)
    axis.set_title("Nutrient correlations (training data)")
    fig.tight_layout()
    fig.savefig(output_dir / "nutrient_correlations.png", dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(9, 7))
    scatter = axis.scatter(
        scores["PC1"],
        scores["PC2"],
        c=scores["nova_target"],
        cmap="viridis",
        alpha=0.55,
        s=12,
    )
    axis.set(title="Nutrient PCA (validation data)", xlabel="PC1", ylabel="PC2")
    fig.colorbar(scatter, ax=axis, label="Observed NOVA group")
    fig.tight_layout()
    fig.savefig(output_dir / "nutrient_pca_validation.png", dpi=200)
    plt.close(fig)


def plot_model_results(model, validation, comparison, output_dir):
    """Save NOVA validation confusion matrix and model score comparison."""
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir) / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    truth = validation.nova_target.astype(int)
    prediction = model.predict(validation).astype(int)
    matrix = confusion_matrix(truth, prediction, labels=[1, 2, 3, 4])
    fig, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrix, display_labels=[1, 2, 3, 4]).plot(
        ax=axis, cmap="Blues", colorbar=False
    )
    axis.set_title("NOVA validation confusion matrix")
    fig.tight_layout()
    fig.savefig(output_dir / "nova_validation_confusion_matrix.png", dpi=200)
    plt.close(fig)

    ordered = comparison.sort_values("macro_f1", ascending=True)
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.barh(ordered["model"], ordered["macro_f1"], color="#4472C4")
    axis.set(xlim=(0, 1), xlabel="Macro F1", title="NOVA validation model comparison")
    fig.tight_layout()
    fig.savefig(output_dir / "nova_validation_model_comparison.png", dpi=200)
    plt.close(fig)


def _probability_for_class(model, frame, target_class, batch_size=50000):
    classes = list(model.classes_)
    if target_class not in classes:
        raise ValueError(f"Trained model has no class {target_class}.")
    class_index = classes.index(target_class)
    return np.concatenate(
        [
            model.predict_proba(frame.iloc[start : start + batch_size])[:, class_index]
            for start in range(0, len(frame), batch_size)
        ]
    )


def _legacy_sensitivity_fields(model, df, base_probability):
    """Calculate local probability deltas while retaining Tableau's field names."""
    from .analysis import INGREDIENT_FEATURES

    results = pd.DataFrame(index=df.index)
    for nutrient, output_name in [
        ("sugars_100g", "shap_sugars_nova"),
        ("sodium_100g", "shap_sodium_nova"),
    ]:
        usable = df[nutrient].notna()
        results[output_name] = np.nan
        if usable.any():
            changed = df.loc[usable].copy()
            changed[nutrient] = 0
            results.loc[usable, output_name] = (
                base_probability[usable]
                - _probability_for_class(model, changed, 4)
            )

    results["shap_additives_nova"] = np.nan
    marker_columns = [
        feature for feature in INGREDIENT_FEATURES if feature.startswith("marker_")
    ]
    additive_evidence = df["additives_n_feature"].notna()
    if marker_columns:
        additive_evidence |= df[marker_columns].notna().any(axis=1)
    masks = {
        "shap_sugars_nova": df["sugars_100g"].notna(),
        "shap_sodium_nova": df["sodium_100g"].notna(),
        "shap_additives_nova": additive_evidence,
    }

    def zero_nutrient(frame, column):
        frame[column] = 0

    def zero_additives(frame):
        frame["additives_n_feature"] = 0
        frame[marker_columns] = 0

    for output_name, column, mask, mutate in [
        ("shap_sugars_nova", "sugars_100g", masks["shap_sugars_nova"], zero_nutrient),
        ("shap_sodium_nova", "sodium_100g", masks["shap_sodium_nova"], zero_nutrient),
        (
            "shap_additives_nova",
            "additives_n_feature",
            masks["shap_additives_nova"],
            zero_additives,
        ),
    ]:
        positions = np.flatnonzero(mask.to_numpy())
        deltas = np.full(len(positions), np.nan)
        for start in range(0, len(positions), 50000):
            batch_positions = positions[start : start + 50000]
            changed = df.iloc[batch_positions].copy()
            if column == "additives_n_feature":
                mutate(changed)
            else:
                mutate(changed, column)
            deltas[start : start + len(batch_positions)] = (
                base_probability[batch_positions]
                - _probability_for_class(model, changed, 4)
            )
        results.iloc[
            positions, results.columns.get_loc(output_name)
        ] = deltas
    return results


def export_tableau(
    df,
    model,
    model_name,
    grades,
    scenarios,
    importance_table,
    output_dir,
    run_final_test,
):
    """Export model results for all deduplicated labeled and unlabeled products.

    Tableau's established column names are retained. The historical ``shap_*``
    columns contain local model probability deltas, not SHAP attributions.
    """
    del model_name, scenarios, importance_table
    from .analysis import tableau_additive_features, tableau_categories

    output_dir = Path(output_dir)
    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    if df.product_id.duplicated().any():
        raise ValueError("Tableau export requires unique product_id values.")
    if grades.product_id.duplicated().any():
        raise ValueError("Nutri-Score results must have unique product_id values.")

    result = df.merge(
        grades[
            [
                "product_id",
                "grade_display",
                "grade_source",
                "formula_score_2023",
            ]
        ],
        on="product_id",
        how="left",
        validate="one_to_one",
    )
    if result.grade_source.isna().any():
        raise ValueError("Nutri-Score results are missing products from the Tableau export.")

    base_probability = _probability_for_class(model, df, 4)
    output = pd.DataFrame(index=df.index)
    output["code"] = df["code"].astype("string")
    output["product_name"] = (
        df["product_name"].fillna("").astype("string").str.strip()
    )
    missing_name = output.product_name.eq("")
    output.loc[missing_name, "product_name"] = (
        "Product " + df.loc[missing_name, "product_id"].astype("string")
    )
    categories = tableau_categories(df)
    output["unified_category"] = categories.unified_category
    output["sub_category"] = categories.sub_category
    output["url"] = df["url"] if "url" in df else pd.NA

    final_test = df.nutri_split.eq("test") & (not run_final_test)
    grade = result.grade_display.astype("string").str.upper().copy()
    grade.loc[final_test] = pd.NA
    source = result.grade_source.astype("string").copy()
    source.loc[final_test] = "hidden_final_test"
    output["nutriscore_provenance"] = source
    output["nutriscore_grade_final"] = grade

    observed_score = pd.to_numeric(
        df["nutriscore_score"], errors="coerce"
    ) if "nutriscore_score" in df else pd.Series(np.nan, index=df.index)
    formula_score = pd.to_numeric(result.formula_score_2023, errors="coerce")
    score = observed_score.where(result.grade_source.eq("OFF_observed"), formula_score)
    output["nutriscore_score_final"] = score.mask(final_test)

    output["pred_nova_group"] = np.concatenate(
        [
            model.predict(df.iloc[start : start + 50000]).astype(int)
            for start in range(0, len(df), 50000)
        ]
    )
    output["pred_nova4_prob"] = base_probability

    for nutrient in [
        "sugars_100g",
        "sodium_100g",
        "saturated-fat_100g",
        "fiber_100g",
        "proteins_100g",
    ]:
        output[nutrient] = (
            pd.to_numeric(df[nutrient], errors="coerce")
            if nutrient in df
            else np.nan
        )
    output["additives_n"] = pd.to_numeric(
        df["additives_n_feature"], errors="coerce"
    )

    additive_features = tableau_additive_features(df)
    output[additive_features.columns] = additive_features
    output = output.join(_legacy_sensitivity_fields(model, df, base_probability))

    sugar = output["sugars_100g"]
    sodium = output["sodium_100g"]
    output["nutri_sugar_penalty_pts"] = (sugar / 4.5).clip(0, 10)
    output["nutri_sodium_penalty_pts"] = (sodium / 0.09).clip(0, 10)
    output["is_healthy_benchmark"] = np.where(
        output.pred_nova_group.le(2) & output.nutriscore_grade_final.isin(["A", "B"]),
        "Yes",
        "No",
    )

    tableau = output[TABLEAU_EXPORT_COLUMNS]
    if len(tableau) != len(df) or tableau.columns.tolist() != TABLEAU_EXPORT_COLUMNS:
        raise AssertionError("Tableau export does not match the established data contract.")
    destination = table_dir / "tableau_products.csv"
    tableau.to_csv(destination, index=False)
    print(f"Tableau product data saved to: {destination} ({len(tableau):,} rows)")
    return tableau
