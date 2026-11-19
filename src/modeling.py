"""Run the complete dataset workflow: python -m src.modeling."""
import argparse
from pathlib import Path
import sys
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "src"

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits
from .data_processing import clean_data
from .analysis import (field, normalize, load_processed_data, engineer_features,
                       assign_splits, select_nutrients, exploratory_analysis,
                       INGREDIENT_FEATURES, MARKERS, parse_tags, ingredient_features)
from .utils import CORE_NUTRIENTS, save_table
from .visualization import (
    calibrate_additive_weights,
    export_tableau,
    plot_model_results,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results"
SEED = 33


def make_pipeline(
    name, nutrients, include_ingredients=True, ingredient_columns=None
):
    if ingredient_columns is None:
        ingredient_columns = INGREDIENT_FEATURES if include_ingredients else []
    numeric = list(ingredient_columns) + list(nutrients)
    preparation = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("category", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, max_categories=40, sparse_output=False), ["category"]),
    ])
    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(n_estimators=120, max_depth=18, min_samples_leaf=5, class_weight="balanced_subsample", random_state=SEED, n_jobs=2),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15, learning_rate=0.08, class_weight="balanced", early_stopping=False, random_state=SEED),
    }
    return Pipeline([("preprocess", preparation), ("model", models[name])])


def nova_metrics(model, frame):
    truth = frame.nova_target.astype(int)
    prediction = model.predict(frame).astype(int)
    p4 = model.predict_proba(frame)[:, list(model.classes_).index(4)]
    return {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, labels=[1, 2, 3, 4], average="macro", zero_division=0),
        "nova4_precision": precision_score(truth.eq(4), prediction == 4, zero_division=0),
        "nova4_recall": recall_score(truth.eq(4), prediction == 4, zero_division=0),
        "nova4_f1": f1_score(truth.eq(4), prediction == 4, zero_division=0),
        "nova4_average_precision": average_precision_score(truth.eq(4), p4),
        "nova4_brier": brier_score_loss(truth.eq(4), p4),
    }


def optimal_nova4_threshold(model, validation):
    """Choose the NOVA-4 probability cutoff maximizing validation-set F1."""
    truth = validation.nova_target.astype(int).eq(4)
    class_index = list(model.classes_).index(4)
    probability = model.predict_proba(validation)[:, class_index]
    precision, recall, thresholds = precision_recall_curve(truth, probability)
    if not len(thresholds):
        raise ValueError("Validation data does not support a NOVA 4 threshold.")
    f1 = (
        2 * precision[:-1] * recall[:-1]
        / np.maximum(precision[:-1] + recall[:-1], np.finfo(float).eps)
    )
    best = int(np.argmax(f1))
    return float(thresholds[best])


def thresholded_nova_predictions(model, frame, threshold):
    """Apply a calibrated NOVA-4 cutoff while retaining the other class argmax."""
    classes = np.asarray(model.classes_)
    probabilities = model.predict_proba(frame)
    nova4_column = int(np.flatnonzero(classes == 4)[0])
    predictions = classes[probabilities.argmax(axis=1)].copy()
    is_nova4 = probabilities[:, nova4_column] >= threshold
    predictions[is_nova4] = 4
    below_threshold = ~is_nova4 & (predictions == 4)
    if below_threshold.any():
        other_columns = np.flatnonzero(classes != 4)
        predictions[below_threshold] = classes[
            other_columns[
                probabilities[below_threshold][:, other_columns].argmax(axis=1)
            ]
        ]
    return predictions


def final_test_metrics(model, frame, threshold):
    """Export per-class metrics and calibrated NOVA-4 probability metrics."""
    truth = frame.nova_target.astype(int)
    prediction = thresholded_nova_predictions(model, frame, threshold)
    class_rows = []
    for label in [1, 2, 3, 4]:
        actual = truth.eq(label)
        predicted = prediction == label
        class_rows.extend(
            [
                {
                    "metric": "precision",
                    "class": label,
                    "value": precision_score(actual, predicted, zero_division=0),
                },
                {
                    "metric": "recall",
                    "class": label,
                    "value": recall_score(actual, predicted, zero_division=0),
                },
                {
                    "metric": "f1",
                    "class": label,
                    "value": f1_score(actual, predicted, zero_division=0),
                },
            ]
        )
    class_rows.extend(
        [
            {
                "metric": "macro_f1",
                "class": "all",
                "value": f1_score(
                    truth, prediction, labels=[1, 2, 3, 4],
                    average="macro", zero_division=0,
                ),
            },
            {
                "metric": "nova4_average_precision",
                "class": 4,
                "value": average_precision_score(
                    truth.eq(4),
                    model.predict_proba(frame)[:, list(model.classes_).index(4)],
                ),
            },
            {
                "metric": "nova4_brier",
                "class": 4,
                "value": brier_score_loss(
                    truth.eq(4),
                    model.predict_proba(frame)[:, list(model.classes_).index(4)],
                ),
            },
        ]
    )
    metrics = pd.DataFrame(class_rows)
    metrics["nova4_threshold"] = threshold
    return metrics


def train_nova(train, validation, nutrients, output_dir):
    models, rows = {}, []
    for name in ["majority", "logistic", "random_forest", "gradient_boosting"]:
        model = make_pipeline(name, nutrients)
        model.fit(train, train.nova_target.astype(int))
        models[name] = model
        rows.append({"model": name, **nova_metrics(model, validation)})
        print(name, "validation macro-F1:", round(rows[-1]["macro_f1"], 4))
    comparison = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    best_name = comparison.loc[comparison.model.ne("majority"), "model"].iloc[0]
    save_table(comparison, output_dir, "nova_validation_metrics.csv")
    return models[best_name], best_name, comparison


def ns_points(value, thresholds, inclusive=False):
    """Count crossed scoring thresholds with the specified boundary convention."""
    return (
        int(np.sum(float(value) >= np.array(thresholds)))
        if inclusive
        else int(np.sum(float(value) > np.array(thresholds)))
    )

def nutriscore_2023_one(values, group):
    """Score only supported general foods, cheese, and plain water.

    Energy is kJ per 100 g; salt/macronutrients are grams per 100 g; FVL is
    the fruit/vegetable/legume percentage. Unsupported branches return no grade.
    """
    if group == "water":
        return None, "A"
    if group not in {"food", "cheese"}:
        return None, None
    required = ["energy_kj", "sugars", "satfat", "salt", "fiber", "protein", "fvl"]
    if any(key not in values or pd.isna(values[key]) for key in required):
        return None, None
    if any(not np.isfinite(float(values[key])) or values[key] < 0 for key in required):
        return None, None
    if values["fvl"] > 100:
        return None, None

    energy = ns_points(values["energy_kj"], [335 * i for i in range(1, 11)])
    sugars = ns_points(values["sugars"], [3.4, 6.8, 10, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51])
    saturated_fat = ns_points(values["satfat"], list(range(1, 11)))
    salt = ns_points(values["salt"], [round(0.2 * i, 1) for i in range(1, 21)])
    negative = energy + sugars + saturated_fat + salt

    fiber = ns_points(values["fiber"], [3, 4.1, 5.2, 6.3, 7.4])
    fruit = ns_points(values["fvl"], [40, 60, 80, 80, 80])
    protein = ns_points(values["protein"], [2.4, 4.8, 7.2, 9.6, 12, 14, 17])
    count_protein = group == "cheese" or negative < 11
    score = negative - fiber - fruit - (protein if count_protein else 0)
    grade = "ABCDE"[int(np.sum(score > np.array([0, 2, 10, 18])))]
    return int(score), grade

def formula_grades(df):
    """Resolve supported rule branches conservatively using observed context."""
    pnns = normalize(field(df, "pnns_groups_2"))
    category = normalize(field(df, "categories_en"))
    safe = {
        "bread",
        "breakfast cereals",
        "cereals",
        "legumes",
        "pasta",
        "rice",
        "potatoes",
        "vegetables",
        "fruits",
        "soups",
        "biscuits and cakes",
        "pastries",
        "chocolate products",
        "sweets",
        "ice cream",
        "sandwiches",
        "pizza pies and quiche",
        "one-dish meals",
        "salty and fatty products",
        "fish and seafood",
        "eggs",
        "dairy desserts",
    }
    branch = pd.Series("unknown", index=df.index, dtype="string")
    branch.loc[pnns.isin(safe)] = "food"
    branch.loc[pnns.eq("cheese")] = "cheese"
    water = category.str.contains(
        r"(?:^|,)\s*(?:natural mineral waters|spring waters)\s*(?:,|$)", regex=True
    )
    water &= ~category.str.contains(r"flavou?red|sweetened", regex=True)
    branch.loc[water] = "water"
    excluded = df.nutri_not_applicable | category.str.contains(
        r"(?:^|,)\s*(?:baby foods|baby milks|infant formulas|alcoholic beverages)\s*(?:,|$)",
        regex=True,
    )
    branch.loc[excluded] = "not_applicable"
    numerics = pd.DataFrame(index=df.index)
    numerics["energy_kj"] = (
        df["energy-kj_100g"]
        .fillna(df["energy_100g"])
        .fillna(df["energy-kcal_100g"] * 4.184)
    )
    numerics["sugars"] = df["sugars_100g"]
    numerics["satfat"] = df["saturated-fat_100g"]
    numerics["salt"] = (
        df["salt_100g"].fillna(df["sodium_100g"] * 2.5).where(lambda x: x.le(100))
    )
    numerics["fiber"] = df["fiber_100g"]
    numerics["protein"] = df["proteins_100g"]
    numerics["fvl"] = df["fruits-vegetables-legumes_100g"]
    numerics["fat"] = df["fat_100g"]
    numerics["sweeteners"] = np.nan
    required = ["energy_kj", "sugars", "satfat", "salt", "fiber", "protein", "fvl"]
    complete = branch.isin(["food", "cheese"]) & numerics[required].notna().all(axis=1)
    inconsistent = df.flag_satfat_gt_fat | df.flag_sugars_gt_carbs
    complete &= ~inconsistent
    complete |= branch.eq("water")
    complete &= ~df.needs_review
    result = pd.DataFrame(
        {
            "product_id": df.product_id,
            "rule_branch": branch,
            "formula_score_2023": np.nan,
            "formula_grade_2023": pd.Series(pd.NA, index=df.index, dtype="string"),
            "formula_status": "missing_required_inputs",
        }
    )
    result.loc[branch.eq("unknown"), "formula_status"] = "unresolved_scoring_branch"
    result.loc[branch.eq("not_applicable"), "formula_status"] = "not_applicable"
    result.loc[
        branch.isin(["food", "cheese"]) & numerics.fvl.isna(), "formula_status"
    ] = "missing_fruit_vegetable_legume_proportion"
    result.loc[inconsistent & ~branch.eq("water"), "formula_status"] = (
        "inconsistent_nutrient_values"
    )
    result.loc[df.needs_review, "formula_status"] = "needs_review"
    rows = df.index[complete]
    for i, values, group in zip(
        rows, numerics.loc[rows].to_dict("records"), branch.loc[rows]
    ):
        score, grade = nutriscore_2023_one(values, group)
        if grade is not None:
            result.loc[i, "formula_score_2023"] = np.nan if score is None else score
            result.loc[i, "formula_grade_2023"] = grade
            result.loc[i, "formula_status"] = "calculated_2023"
    return result


def grade_metrics(truth, prediction):
    order = {grade: number for number, grade in enumerate("abcde")}
    return {
        "accuracy": accuracy_score(truth, prediction),
        "macro_f1": f1_score(truth, prediction, labels=list("abcde"), average="macro", zero_division=0),
        "ordinal_mae": mean_absolute_error(truth.map(order), pd.Series(prediction).map(order)),
    }


def complete_nutriscore(df, output_dir, run_final_test=False):
    """Observed grade -> supported 2023 formula -> classifier -> unknown."""
    formulas = formula_grades(df)
    # A text label alone is not sufficient for this nutrition-completion model.
    supported = df.observed_core_nutrients.ge(4) & ~df.needs_review
    train = df.loc[df.nutri_target.notna() & supported & df.nutri_split.eq("train")]
    validation = df.loc[df.nutri_target.notna() & supported & df.nutri_split.eq("validation")]
    nutrients, profile = select_nutrients(train)
    save_table(profile, output_dir, "nutriscore_feature_selection.csv")
    model = make_pipeline("logistic", nutrients)
    model.fit(train, train.nutri_target)
    rows = [{"split": "validation", "coverage": "all_supported", "rows": len(validation),
             **grade_metrics(validation.nutri_target, model.predict(validation))}]
    # Report whether performance varies with nutrient availability.
    for label, lower, upper in [("4-5 nutrients", 4, 5), ("6-7 nutrients", 6, 7)]:
        part = validation.loc[validation.observed_core_nutrients.between(lower, upper)]
        if len(part):
            rows.append({"split": "validation", "coverage": label, "rows": len(part),
                         **grade_metrics(part.nutri_target, model.predict(part))})
    if run_final_test:
        test = df.loc[df.nutri_target.notna() & supported & df.nutri_split.eq("test")]
        rows.append({"split": "test", "coverage": "all_supported", "rows": len(test),
                     **grade_metrics(test.nutri_target, model.predict(test))})
    save_table(pd.DataFrame(rows), output_dir, "nutriscore_metrics.csv")
    joblib.dump(model, Path(output_dir) / "models/nutriscore_model.joblib")

    result = df[["product_id", "nutri_split"]].copy()
    result["grade_display"] = df.nutri_target.str.upper()
    result["grade_source"] = "unknown"
    result.loc[formulas.rule_branch.eq("not_applicable"), "grade_source"] = "not_applicable"
    result.loc[df.needs_review & result.grade_source.eq("unknown"), "grade_source"] = "needs_review"
    observed = df.nutri_target.notna()
    result.loc[observed, "grade_source"] = "OFF_observed"
    result["grade_version"] = "unknown"
    result.loc[observed, "grade_version"] = "OFF_version_unverified"
    calculate = ~observed & formulas.formula_grade_2023.notna() & ~formulas.rule_branch.eq("not_applicable")
    result.loc[calculate, "grade_display"] = formulas.loc[calculate, "formula_grade_2023"]
    result.loc[calculate, "grade_source"] = "formula_calculated"
    result.loc[calculate, "grade_version"] = "2023"
    estimate = result.grade_source.eq("unknown") & supported
    result["estimate_probability_uncalibrated"] = np.nan
    if estimate.any():
        result.loc[estimate, "grade_display"] = pd.Series(model.predict(df.loc[estimate]), index=df.index[estimate]).str.upper()
        result.loc[estimate, "grade_source"] = "model_estimated"
        result.loc[estimate, "grade_version"] = "OFF_training_version_unverified"
        result.loc[estimate, "estimate_probability_uncalibrated"] = model.predict_proba(df.loc[estimate]).max(axis=1)
    assert result.loc[observed, "grade_display"].str.lower().equals(df.loc[observed, "nutri_target"])
    result = result.merge(formulas, on="product_id", validate="one_to_one")
    save_table(result, output_dir, "nutriscore_completion.csv")
    save_table(result.grade_source.value_counts().rename_axis("source").reset_index(name="rows"), output_dir, "nutriscore_completion_counts.csv")
    return result


def ingredient_scenarios(model, validation, output_dir):
    """Evaluate supported single and paired additive removals on NOVA 4 products.

    These are model-input sensitivity tests, not verified recipe changes.
    Nutrients remain fixed; no post-change Nutri-Score is claimed.
    """
    import re

    candidates = validation.loc[validation.nova_target.eq(4) & ~validation.needs_review]
    if candidates.empty:
        raise ValueError("No NOVA 4 validation products for scenarios.")
    p4_index = list(model.classes_).index(4)
    baseline = pd.Series(model.predict_proba(candidates)[:, p4_index], index=candidates.index)
    scenario_markers = {
        "sucralose": ["sucralose"],
        "carrageenan": ["carrageenan"],
        "lecithin": ["lecithin"],
        "sucralose_lecithin": ["sucralose", "lecithin"],
        "carrageenan_lecithin": ["carrageenan", "lecithin"],
        "sucralose_carrageenan": ["sucralose", "carrageenan"],
    }
    altered_rows, records, audits = [], [], []
    for index, row in candidates.iterrows():
        tags = parse_tags(row.additives_tags)
        count = pd.to_numeric(row.additives_n, errors="coerce")
        if pd.isna(count) or count != len(tags):
            audits.append({"product_id": row.product_id, "status": "inconsistent_additive_evidence"})
            continue
        supported = {}
        for marker, (pattern, codes) in MARKERS.items():
            if marker not in {"sucralose", "carrageenan", "lecithin"}:
                continue
            matched_codes = tags & codes
            if re.search(pattern, row.ingredients_text_norm) and matched_codes:
                supported[marker] = (pattern, matched_codes)

        eligible_scenarios = 0
        for scenario, markers in scenario_markers.items():
            if not all(marker in supported for marker in markers):
                continue
            altered = row.copy()
            ingredient_text = row.ingredients_text_norm
            removed_codes = set()
            for marker in markers:
                pattern, codes = supported[marker]
                ingredient_text = re.sub(pattern, " ", ingredient_text)
                removed_codes.update(codes)
            altered["ingredients_text_norm"] = re.sub(
                r"\s+", " ", ingredient_text
            ).strip()
            remaining = tags - removed_codes
            altered["additives_tags"] = ",".join("en:" + tag for tag in sorted(remaining))
            altered["additives_n"] = len(remaining)
            altered_rows.append(altered)
            records.append(
                {
                    "product_id": row.product_id,
                    "scenario_id": "remove_" + "_".join(markers),
                    "baseline_NOVA4_probability": baseline.loc[index],
                    "nutrition_assumption": "held_constant_not_measured",
                    "interpretation": "model_sensitivity_only",
                }
            )
            eligible_scenarios += 1
        audits.append(
            {
                "product_id": row.product_id,
                "status": "eligible" if eligible_scenarios else "no_supported_marker",
                "eligible_scenarios": eligible_scenarios,
            }
        )
    columns = ["product_id", "scenario_id", "baseline_NOVA4_probability", "nutrition_assumption", "interpretation"]
    result = pd.DataFrame(records, columns=columns)
    result["scenario_NOVA4_probability"] = np.nan
    if altered_rows:
        altered = pd.DataFrame(altered_rows).reset_index(drop=True)
        altered[INGREDIENT_FEATURES] = ingredient_features(altered)[INGREDIENT_FEATURES]
        original = validation.set_index("product_id").loc[result.product_id].reset_index(drop=True)
        pd.testing.assert_frame_equal(original[CORE_NUTRIENTS], altered[CORE_NUTRIENTS], check_dtype=False)
        result["scenario_NOVA4_probability"] = model.predict_proba(altered)[:, p4_index]
    result["probability_change"] = result.scenario_NOVA4_probability - result.baseline_NOVA4_probability
    result["lower_prediction"] = result.probability_change.lt(-1e-8)
    save_table(pd.DataFrame(audits), output_dir, "scenario_eligibility.csv")
    save_table(result, output_dir, "ingredient_scenarios.csv")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the food processing model pipeline.")
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        default=False,
        help="Evaluate the frozen test split after validation settings are fixed.",
    )
    run_final_test = parser.parse_args(argv).run_final_test

    for folder in ["tables", "figures", "models"]:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    # 1. Read both complete files. No sample file is used.
    raw, audit = load_processed_data(DATA_DIR, clean_function=clean_data)
    save_table(audit, OUTPUT_DIR, "input_counts.csv")
    print("Full input rows:", len(raw))

    # 2. Check measurements and construct deterministic features.
    df, nutrient_audit = engineer_features(raw)
    save_table(nutrient_audit, OUTPUT_DIR, "nutrient_quality.csv")
    print("Valid NOVA labels:", df.nova_target.notna().sum())
    print("Valid Nutri-Score grades:", df.nutri_target.notna().sum())
    print("Products retained but excluded from modeling for review:", int(df.needs_review.sum()))
    save_table(df[["product_id", "code", "needs_review", "flag_satfat_gt_fat", "flag_sugars_gt_carbs"]], OUTPUT_DIR, "product_quality_flags.csv")

    # 3. Split the two prediction tasks independently by duplicate groups.
    df, summaries = assign_splits(df, seed=SEED)
    for task, table in zip(["nova", "nutriscore"], summaries):
        save_table(table, OUTPUT_DIR, task + "_split_counts.csv")
    save_table(df[["product_id", "code", "family_group", "nova_split", "nutri_split", "needs_review"]], OUTPUT_DIR, "split_assignments.csv")
    group_sizes = df.groupby("family_group").size().rename("products").sort_values(ascending=False).reset_index()
    save_table(group_sizes, OUTPUT_DIR, "family_group_sizes.csv")
    train = df.loc[df.nova_target.notna() & df.nova_split.eq("train")]
    validation = df.loc[df.nova_target.notna() & df.nova_split.eq("validation")]
    test = df.loc[df.nova_target.notna() & df.nova_split.eq("test")]

    # 4. Select nutrients using training coverage; examine correlations and PCA.
    nutrients, profile = select_nutrients(train)
    save_table(profile, OUTPUT_DIR, "nova_feature_selection.csv")
    exploratory_analysis(train, validation, OUTPUT_DIR)

    # 5. Select a structured model by validation macro-F1.
    model, best_name, comparison = train_nova(train, validation, nutrients, OUTPUT_DIR)
    joblib.dump(model, OUTPUT_DIR / "models/nova_model.joblib")
    plot_model_results(model, validation, comparison, OUTPUT_DIR)

    # 6. Compare feature sets with model type, rows, and splits fixed.
    ablations = []
    for name, selected_nutrients, include_ingredients in [
        ("ingredients_plus_category", [], True),
        ("nutrients_plus_category", nutrients, False),
        ("combined_plus_category", nutrients, True),
        ("core_seven_plus_ingredients_and_category", CORE_NUTRIENTS, True),
    ]:
        candidate = model if name == "combined_plus_category" else make_pipeline(best_name, selected_nutrients, include_ingredients)
        if candidate is not model:
            candidate.fit(train, train.nova_target.astype(int))
        ablations.append({"feature_set": name, "model": best_name, **nova_metrics(candidate, validation)})
    save_table(pd.DataFrame(ablations), OUTPUT_DIR, "nova_feature_comparison.csv")

    # Quantify validation performance after removing increasingly direct features.
    marker_features = [
        feature for feature in INGREDIENT_FEATURES
        if feature.startswith("marker_")
    ]
    feature_sets = [
        ("full_features", INGREDIENT_FEATURES),
        (
            "without_markers",
            [feature for feature in INGREDIENT_FEATURES
             if feature not in marker_features],
        ),
        (
            "without_markers_or_additives_count",
            [feature for feature in INGREDIENT_FEATURES
             if feature not in marker_features
             and feature != "additives_n_feature"],
        ),
        ("nutrients_only", []),
    ]
    leakage_rows = []
    for label, ingredient_columns in feature_sets:
        candidate = (
            model
            if label == "full_features"
            else make_pipeline(
                best_name,
                nutrients,
                ingredient_columns=ingredient_columns,
            )
        )
        if candidate is not model:
            candidate.fit(train, train.nova_target.astype(int))
        leakage_rows.append(
            {
                "feature_set": label,
                "model": best_name,
                **nova_metrics(candidate, validation),
            }
        )
    save_table(
        pd.DataFrame(leakage_rows), OUTPUT_DIR, "e2_leakage_ablation.csv"
    )

    # 7. Use all validation products for global permutation importance.
    features = INGREDIENT_FEATURES + nutrients + ["category"]
    importance = permutation_importance(model, validation[features], validation.nova_target.astype(int), scoring="f1_macro", n_repeats=5, random_state=SEED, n_jobs=1)
    importance_table = pd.DataFrame({"feature": features, "macro_f1_drop": importance.importances_mean, "standard_deviation": importance.importances_std})
    save_table(importance_table.sort_values("macro_f1_drop", ascending=False), OUTPUT_DIR, "feature_importance.csv")

    # 8. Complete nutritional grades without overwriting valid observed grades.
    grades = complete_nutriscore(df, OUTPUT_DIR, run_final_test)

    # 9. Evaluate predefined single-marker sensitivity scenarios.
    scenarios = ingredient_scenarios(model, validation, OUTPUT_DIR)

    # 10. Evaluate test data only after freezing methods and settings.
    threshold = optimal_nova4_threshold(model, validation)
    calibrated_weights = calibrate_additive_weights(model, validation)
    calibrated_weights = pd.concat(
        [
            calibrated_weights,
            pd.DataFrame(
                [
                    {
                        "parameter": "nova4_threshold",
                        "weight": threshold,
                        "eligible_products": len(validation),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    save_table(
        calibrated_weights, OUTPUT_DIR, "calibrated_sandbox_weights.csv"
    )

    if run_final_test:
        save_table(
            final_test_metrics(model, test, threshold),
            OUTPUT_DIR,
            "nova_final_test.csv",
        )
    else:
        (OUTPUT_DIR / "tables/nova_final_test.csv").unlink(missing_ok=True)
        print("Final tests have not been run. Test labels are hidden from Tableau exports.")

    # 11. Export model predictions, original nutrition, scenarios, and provenance.
    tableau = export_tableau(
        df, model, best_name, grades, scenarios, importance_table, OUTPUT_DIR,
        run_final_test,
    )
    raw_sample = pd.read_csv(
        DATA_DIR / "us_openfoodfacts_nutri_nova_202407_202606.csv",
        dtype={"code": "string"},
        nrows=5000,
        low_memory=False,
    )
    raw_sample.to_csv(DATA_DIR / "sample_5000_us_food.csv", index=False)
    tableau.head(5000).to_csv(
        DATA_DIR / "sample_tableau_mock_data.csv", index=False
    )
    print("Refreshed source and Tableau samples (up to 5,000 rows each).")
    print("Selected model:", best_name)
    print("Scenario products:", scenarios.product_id.nunique())
    print("Results saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
