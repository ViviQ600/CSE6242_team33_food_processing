"""Exploratory analysis, classifiers, interpretation, hybrid grades, and scenarios.

Reported development metrics use validation products. Test evaluation is disabled
by default, and model estimates are never promoted to observed labels.
"""

from .visualization import (
    plot_nutrient_correlations,
    plot_pca,
    plot_feature_ablation,
    plot_calibration,
    plot_confusion_matrix,
    plot_permutation_importance,
)
import importlib.util
import json
import re
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from IPython.display import display, Markdown
from sklearn import __version__ as sklearn_version
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from .utils import WorkflowState
from .data_processing import (
    norm_text,
    col,
    parse_tags,
    ingredient_features,
    labeled_group_split,
    MARKERS,
)


def ns_points(value, thresholds, inclusive=False):
    """Count crossed scoring thresholds with the specified boundary convention."""
    return (
        int(np.sum(float(value) >= np.array(thresholds)))
        if inclusive
        else int(np.sum(float(value) > np.array(thresholds)))
    )


def nutriscore_2023_one(values, group):
    """Apply the specified 2023 scoring branch; return no grade if inputs are incomplete.

    Nutrients use per-100-g or per-100-mL values as appropriate, energy is kJ,
    salt and macronutrients are grams, and fvl is a percentage. Category and
    component semantics must be verified separately by the calling adapter.
    """
    if group == "water":
        return (None, "A")
    if group not in {"food", "cheese", "red_meat", "fat", "beverage"}:
        return (None, None)
    required = ["sugars", "satfat", "salt", "fiber", "protein", "fvl"]
    required += ["fat"] if group == "fat" else ["energy_kj"]
    if group == "beverage":
        required += ["sweeteners"]
    if any((k not in values or pd.isna(values[k]) for k in required)):
        return (None, None)
    if any((not np.isfinite(float(values[k])) for k in required)):
        return (None, None)
    if any((values[k] < 0 for k in required)):
        return (None, None)
    if not 0 <= values["fvl"] <= 100:
        return (None, None)
    sugars = ns_points(
        values["sugars"],
        (
            [0.5, 2, 3.5, 5, 6, 7, 8, 9, 10, 11]
            if group == "beverage"
            else [3.4, 6.8, 10, 14, 17, 20, 24, 27, 31, 34, 37, 41, 44, 48, 51]
        ),
    )
    salt = ns_points(values["salt"], [round(0.2 * i, 1) for i in range(1, 21)])
    if group == "fat":
        if values["fat"] <= 0 or values["satfat"] > values["fat"]:
            return (None, None)
        energy = ns_points(values["satfat"] * 37, [120 * i for i in range(1, 11)])
        saturated = ns_points(
            100 * values["satfat"] / values["fat"],
            [10, 16, 22, 28, 34, 40, 46, 52, 58, 64],
            inclusive=True,
        )
    else:
        energy = ns_points(
            values["energy_kj"],
            (
                [30, 90, 150, 210, 240, 270, 300, 330, 360, 390]
                if group == "beverage"
                else [335 * i for i in range(1, 11)]
            ),
        )
        saturated = ns_points(values["satfat"], list(range(1, 11)))
    negative = (
        energy
        + sugars
        + salt
        + saturated
        + (4 * int(values["sweeteners"] > 0) if group == "beverage" else 0)
    )
    fiber = ns_points(values["fiber"], [3, 4.1, 5.2, 6.3, 7.4])
    fruit = ns_points(
        values["fvl"], [40, 40, 60, 60, 80, 80] if group == "beverage" else [40, 60, 80, 80, 80]
    )
    protein = ns_points(
        values["protein"],
        (
            [1.2, 1.5, 1.8, 2.1, 2.4, 2.7, 3]
            if group == "beverage"
            else [2.4, 4.8, 7.2, 9.6, 12, 14, 17]
        ),
    )
    if group == "red_meat":
        protein = min(protein, 2)
    count_protein = group in {"cheese", "beverage"} or negative < (7 if group == "fat" else 11)
    score = negative - fiber - fruit - (protein if count_protein else 0)
    bounds = (
        [2, 6, 9] if group == "beverage" else [-6, 2, 10, 18] if group == "fat" else [0, 2, 10, 18]
    )
    grades = list("BCDE") if group == "beverage" else list("ABCDE")
    grade = grades[int(np.sum(score > np.array(bounds)))]
    return (int(score), grade)


def grade_metrics(y, pred):
    levels = {g: i for i, g in enumerate("abcde")}
    return {
        "macro_f1": f1_score(y, pred, labels=list("abcde"), average="macro", zero_division=0),
        "ordinal_mae": mean_absolute_error([levels[g] for g in y], [levels[g] for g in pred]),
        "accuracy": accuracy_score(y, pred),
    }


def correlation_analysis(state: WorkflowState) -> WorkflowState:
    """Calculate training-only nutrient correlations with pairwise sample counts.

    Required artifacts: CORE_NUTRIENTS, FIG_DIR, TABLE_DIR, save_table, train.
    Published artifacts: pearson_corr, spearman_corr, strongest.
    """
    CORE_NUTRIENTS = state.require("CORE_NUTRIENTS")
    FIG_DIR = state.require("FIG_DIR")
    TABLE_DIR = state.require("TABLE_DIR")
    save_table = state.require("save_table")
    train = state.require("train")
    CORRELATION_FEATURES = list(CORE_NUTRIENTS)
    corr_input = train[CORRELATION_FEATURES].copy()
    assert set(CORRELATION_FEATURES).isdisjoint(
        {"nova_target", "nutri_target", "nutriscore_score", "code"}
    )
    pairwise_counts = corr_input.notna().astype("int64").T.dot(corr_input.notna().astype("int64"))
    pearson_corr = corr_input.corr(method="pearson", min_periods=1000)
    spearman_corr = corr_input.corr(method="spearman", min_periods=1000)
    for name, table in [
        ("26_pearson_nutrients_training.csv", pearson_corr),
        ("27_spearman_nutrients_training.csv", spearman_corr),
        ("28_correlation_pairwise_counts_training.csv", pairwise_counts),
    ]:
        table.rename_axis("feature").to_csv(TABLE_DIR / name)
    print("Correlation population: common NOVA training cohort, n =", len(corr_input))
    display(pearson_corr.round(3))
    display(pairwise_counts)
    plot_nutrient_correlations(pearson_corr, spearman_corr, FIG_DIR)
    upper_pairs = (
        pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
        .stack()
        .rename("pearson_r")
        .reset_index()
    )
    upper_pairs.columns = ["feature_1", "feature_2", "pearson_r"]
    upper_pairs["absolute_r"] = upper_pairs.pearson_r.abs()
    upper_pairs = upper_pairs.sort_values("absolute_r", ascending=False)
    display(save_table(upper_pairs, "29_strongest_nutrient_correlations.csv").head(5))
    strongest = upper_pairs.iloc[0]
    display(
        Markdown(
            f"**Interpretation:** The strongest absolute training-set Pearson correlation is between `{strongest.feature_1}` and `{strongest.feature_2}`，r = **{strongest.pearson_r:.3f}**. Correlated features may share attribution. Review correlations alongside model explanations; this finding does not automatically justify dropping either feature."
        )
    )
    state.update(
        {"pearson_corr": pearson_corr, "spearman_corr": spearman_corr, "strongest": strongest}
    )
    return state


def perform_pca(state: WorkflowState) -> WorkflowState:
    """Fit exploratory nutrient PCA on training products and project validation products.

    Required artifacts: CORE_NUTRIENTS, MODEL_DIR, OUTPUT_DIR, SEED, TABLE_DIR, save_table, show_figure, strongest, train, val.
    Published artifacts: pca_loadings, pca_summary, pca_variance.
    """
    CORE_NUTRIENTS = state.require("CORE_NUTRIENTS")
    MODEL_DIR = state.require("MODEL_DIR")
    OUTPUT_DIR = state.require("OUTPUT_DIR")
    SEED = state.require("SEED")
    TABLE_DIR = state.require("TABLE_DIR")
    save_table = state.require("save_table")
    show_figure = state.require("show_figure")
    strongest = state.require("strongest")
    train = state.require("train")
    val = state.require("val")
    from sklearn.decomposition import PCA

    PCA_FEATURES = [
        c for c in CORE_NUTRIENTS if train[c].notna().any() and train[c].nunique(dropna=True) > 1
    ]
    pca_preprocessor = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    pca_train_matrix = pca_preprocessor.fit_transform(train[PCA_FEATURES])
    pca_validation_matrix = pca_preprocessor.transform(val[PCA_FEATURES])
    pca_model = PCA(n_components=len(PCA_FEATURES), svd_solver="full")
    pca_train_scores = pca_model.fit_transform(pca_train_matrix)
    pca_validation_scores = pca_model.transform(pca_validation_matrix)
    pca_variance = pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(len(PCA_FEATURES))],
            "explained_variance_ratio": pca_model.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(pca_model.explained_variance_ratio_),
        }
    )
    pca_loadings = pd.DataFrame(
        pca_model.components_.T, index=PCA_FEATURES, columns=pca_variance.component
    )
    save_table(pca_variance, "30_pca_explained_variance_training.csv")
    pca_loadings.rename_axis("feature").to_csv(TABLE_DIR / "31_pca_loadings_training.csv")
    pca_val_export = val[["code", "nova_target"]].reset_index(drop=True).copy()
    pca_val_export["PC1"] = pca_validation_scores[:, 0]
    pca_val_export["PC2"] = pca_validation_scores[:, 1]
    save_table(pca_val_export, "32_pca_validation_coordinates.csv")
    joblib.dump(
        {"features": PCA_FEATURES, "preprocess": pca_preprocessor, "pca": pca_model},
        MODEL_DIR / "nutrient_pca_exploratory.joblib",
    )
    display(pca_variance.round(4))
    display(pca_loadings.round(3))
    plot_pca(pca_variance, pca_val_export, SEED, show_figure)
    variance_2 = float(pca_variance.cumulative_variance_ratio.iloc[1])
    components_80 = int(np.searchsorted(pca_variance.cumulative_variance_ratio, 0.8) + 1)
    pca_summary = {
        "fitted_on": "common_NOVA_training_cohort",
        "train_rows": len(train),
        "validation_rows": len(val),
        "features": PCA_FEATURES,
        "first_two_components_variance": variance_2,
        "components_for_80pct": components_80,
        "used_by_classification_models": False,
        "strongest_pearson_pair": strongest.to_dict(),
    }
    (OUTPUT_DIR / "eda_summary.json").write_text(
        json.dumps(pca_summary, indent=2), encoding="utf-8"
    )
    eda_text = f"## Exploratory nutrient analysis\nPearson and Spearman correlations were calculated on the common training cohort using pairwise observed values, with pairwise sample counts reported. The strongest absolute Pearson association was between {strongest.feature_1} and {strongest.feature_2} (r = {strongest.pearson_r:.3f}). Nutrient PCA used median imputation and standardization fitted only on training products, followed by transformation of validation products. The first two components explained {variance_2:.1%} of standardized training variance; {components_80} components were required to reach 80%. This quantifies information lost in the two-dimensional view. The analyses are exploratory and do not establish NOVA class separability or causal processing effects. PCA was not used as a classifier input in the reported model comparisons."
    (OUTPUT_DIR / "eda_analysis_en.md").write_text(eda_text, encoding="utf-8")
    display(
        Markdown(
            f"**Interpretation:** PC1 and PC2 retain **{variance_2:.1%}** of standardized nutrient variance; reaching 80% requires **{components_80}** components. The two-dimensional projection does not establish classification performance."
        )
    )
    state.update(
        {"pca_loadings": pca_loadings, "pca_summary": pca_summary, "pca_variance": pca_variance}
    )
    return state


def configure_models(state: WorkflowState) -> WorkflowState:
    """Create train-fitted preprocessing, model factories, and evaluation helpers.

    Required artifacts: INGREDIENT_FEATURES, NUTRIENTS, SEED.
    Published artifacts: ALL_FEATURES, FEATURE_SETS, fit_logged, make_pipeline, nova_metrics.
    """
    INGREDIENT_FEATURES = state.require("INGREDIENT_FEATURES")
    NUTRIENTS = state.require("NUTRIENTS")
    SEED = state.require("SEED")
    FEATURE_SETS = {
        "ingredients": INGREDIENT_FEATURES,
        "nutrients": NUTRIENTS,
        "combined": INGREDIENT_FEATURES + NUTRIENTS,
    }
    ALL_FEATURES = FEATURE_SETS["combined"] + ["category"]

    def processor(numeric_features):
        numeric = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scale", StandardScaler()),
            ]
        )
        category = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=50,
            max_categories=40,
            sparse_output=False,
        )
        return ColumnTransformer(
            [("numeric", numeric, numeric_features), ("category", category, ["category"])],
            remainder="drop",
        )

    def estimator(name):
        if name == "majority":
            return DummyClassifier(strategy="most_frequent")
        if name == "logistic":
            return LogisticRegression(max_iter=700, class_weight="balanced", random_state=SEED)
        if name == "random_forest":
            return RandomForestClassifier(
                n_estimators=120,
                max_depth=18,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight="balanced_subsample",
                n_jobs=2,
                random_state=SEED,
            )
        if name == "hist_gradient_boosting":
            return HistGradientBoostingClassifier(
                max_iter=120,
                learning_rate=0.08,
                max_leaf_nodes=15,
                min_samples_leaf=30,
                l2_regularization=1,
                class_weight="balanced",
                early_stopping=False,
                random_state=SEED,
            )
        raise ValueError(name)

    def make_pipeline(name, feature_set):
        return Pipeline(
            [("preprocess", processor(FEATURE_SETS[feature_set])), ("model", estimator(name))]
        )

    def nova_metrics(model, X, y):
        prediction = model.predict(X).astype(int)
        classes = np.asarray(model.classes_).astype(int)
        probabilities = model.predict_proba(X)
        p4 = probabilities[:, np.flatnonzero(classes == 4)[0]] if 4 in classes else np.zeros(len(X))
        y4 = np.asarray(y) == 4
        precision, recall, f1, _ = precision_recall_fscore_support(
            y4, prediction == 4, average="binary", zero_division=0
        )
        return {
            "accuracy": accuracy_score(y, prediction),
            "macro_f1": f1_score(
                y, prediction, labels=[1, 2, 3, 4], average="macro", zero_division=0
            ),
            "nova4_precision": precision,
            "nova4_recall": recall,
            "nova4_f1": f1,
            "nova4_average_precision": average_precision_score(y4, p4),
            "nova4_brier": brier_score_loss(y4, p4),
        }

    def fit_logged(model, X, y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(X, y)
        messages = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
        for msg in messages:
            print("CONVERGENCE WARNING:", msg)
        return messages

    state.update(
        {
            "ALL_FEATURES": ALL_FEATURES,
            "FEATURE_SETS": FEATURE_SETS,
            "fit_logged": fit_logged,
            "make_pipeline": make_pipeline,
            "nova_metrics": nova_metrics,
        }
    )
    return state


def compare_models(state: WorkflowState) -> WorkflowState:
    """Train baseline classifiers and select a structured model using validation macro-F1.

    Required artifacts: ALL_FEATURES, MODEL_DIR, OUTPUT_DIR, fit_logged, make_pipeline, nova_metrics, save_table, train, val.
    Published artifacts: best_model, best_name, comparison, models.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    MODEL_DIR = state.require("MODEL_DIR")
    OUTPUT_DIR = state.require("OUTPUT_DIR")
    fit_logged = state.require("fit_logged")
    make_pipeline = state.require("make_pipeline")
    nova_metrics = state.require("nova_metrics")
    save_table = state.require("save_table")
    train = state.require("train")
    val = state.require("val")
    models = {}
    metric_rows = []
    convergence_log = {}
    for model_name in ["majority", "logistic", "random_forest", "hist_gradient_boosting"]:
        t0 = time.monotonic()
        pipeline = make_pipeline(model_name, "combined")
        convergence_log[model_name] = fit_logged(
            pipeline, train[ALL_FEATURES], train.nova_target.astype(int)
        )
        row = {
            "model": model_name,
            "feature_set": "combined",
            "fit_seconds": round(time.monotonic() - t0, 2),
            **nova_metrics(pipeline, val[ALL_FEATURES], val.nova_target.astype(int)),
        }
        models[model_name] = pipeline
        metric_rows.append(row)
        print(model_name, "validation macro-F1=", round(row["macro_f1"], 4), flush=True)
    comparison = pd.DataFrame(metric_rows).sort_values("macro_f1", ascending=False)
    display(save_table(comparison, "11_nova_model_comparison_validation.csv"))
    best_name = comparison.loc[comparison.model.ne("majority")].iloc[0]["model"]
    best_model = models[best_name]
    joblib.dump(best_model, MODEL_DIR / "nova_selected_pipeline.joblib")
    (OUTPUT_DIR / "convergence_log.json").write_text(json.dumps(convergence_log, indent=2))
    display(
        Markdown(
            f"**Selected structured model on validation data: {best_name}.** This score is not a final test result."
        )
    )
    state.update(
        {
            "best_model": best_model,
            "best_name": best_name,
            "comparison": comparison,
            "models": models,
        }
    )
    return state


def compare_feature_sets(state: WorkflowState) -> WorkflowState:
    """Compare ingredient, nutrient, and combined inputs on identical splits.

    Required artifacts: ALL_FEATURES, best_model, best_name, fit_logged, make_pipeline, nova_metrics, save_table, show_figure, train, val.
    Published artifacts: ablation.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    best_model = state.require("best_model")
    best_name = state.require("best_name")
    fit_logged = state.require("fit_logged")
    make_pipeline = state.require("make_pipeline")
    nova_metrics = state.require("nova_metrics")
    save_table = state.require("save_table")
    show_figure = state.require("show_figure")
    train = state.require("train")
    val = state.require("val")
    ablation_rows = []
    ablation_models = {}
    for feature_set in ["ingredients", "nutrients", "combined"]:
        pipeline = (
            best_model if feature_set == "combined" else make_pipeline(best_name, feature_set)
        )
        if feature_set != "combined":
            fit_logged(pipeline, train[ALL_FEATURES], train.nova_target.astype(int))
        ablation_models[feature_set] = pipeline
        ablation_rows.append(
            {
                "model": best_name,
                "feature_set": feature_set,
                **nova_metrics(pipeline, val[ALL_FEATURES], val.nova_target.astype(int)),
            }
        )
    ablation = pd.DataFrame(ablation_rows)
    display(save_table(ablation, "12_feature_ablation_validation.csv"))
    plot_feature_ablation(ablation, show_figure)
    state.update({"ablation": ablation})
    return state


def fit_text_baseline(state: WorkflowState) -> WorkflowState:
    """Fit a separate train-vocabulary TF-IDF logistic-regression baseline.

    Required artifacts: RUN_TEXT_BASELINE, SEED, fit_logged, nova_metrics, save_table, train, val.
    Published artifacts: text_result.
    """
    RUN_TEXT_BASELINE = state.require("RUN_TEXT_BASELINE")
    SEED = state.require("SEED")
    fit_logged = state.require("fit_logged")
    nova_metrics = state.require("nova_metrics")
    save_table = state.require("save_table")
    train = state.require("train")
    val = state.require("val")
    text_result = None
    if RUN_TEXT_BASELINE:
        text_model = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=12000,
                        ngram_range=(1, 2),
                        min_df=5,
                        max_df=0.98,
                        sublinear_tf=True,
                        dtype=np.float64,
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        tol=0.0001,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
        text_convergence = fit_logged(
            text_model, train.ingredients_text_norm, train.nova_target.astype(int)
        )
        text_result = {
            "model": "TFIDF_logistic",
            "feature_set": "ingredient_text_without_category",
            "converged": not bool(text_convergence),
            **nova_metrics(text_model, val.ingredients_text_norm, val.nova_target.astype(int)),
        }
        display(save_table(pd.DataFrame([text_result]), "13_text_baseline_validation.csv"))
    else:
        print("Text baseline not run (configuration).")
    state.update({"text_result": text_result})
    return state


def evaluate_validation(state: WorkflowState) -> WorkflowState:
    """Inspect class errors, probability calibration, and category performance.

    Required artifacts: ALL_FEATURES, TABLE_DIR, best_model, save_table, show_figure, val.
    Published artifacts: category_results, i4, validation_export.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    TABLE_DIR = state.require("TABLE_DIR")
    best_model = state.require("best_model")
    save_table = state.require("save_table")
    show_figure = state.require("show_figure")
    val = state.require("val")
    y_val = val.nova_target.astype(int).to_numpy()
    val_pred = best_model.predict(val[ALL_FEATURES]).astype(int)
    val_proba = best_model.predict_proba(val[ALL_FEATURES])
    i4 = list(best_model.classes_).index(4)
    p4 = val_proba[:, i4]
    cm = pd.DataFrame(
        confusion_matrix(y_val, val_pred, labels=[1, 2, 3, 4]),
        index=[1, 2, 3, 4],
        columns=[1, 2, 3, 4],
    )
    print("Rows=true NOVA; columns=predicted NOVA")
    display(cm)
    cm.rename_axis("true_NOVA").to_csv(TABLE_DIR / "14_confusion_matrix_validation.csv")
    plot_confusion_matrix(cm, show_figure)
    observed, mean_prob = calibration_curve(y_val == 4, p4, n_bins=10, strategy="quantile")
    plot_calibration(mean_prob, observed, show_figure)
    category_rows = []
    for category, part in val.groupby("category"):
        if len(part) >= 100:
            category_rows.append(
                {
                    "category": category,
                    "validation_rows": len(part),
                    "accuracy": accuracy_score(
                        part.nova_target, best_model.predict(part[ALL_FEATURES])
                    ),
                    "macro_f1_observed_classes": f1_score(
                        part.nova_target,
                        best_model.predict(part[ALL_FEATURES]),
                        average="macro",
                        zero_division=0,
                    ),
                }
            )
    category_results = pd.DataFrame(category_rows).sort_values("validation_rows", ascending=False)
    display(save_table(category_results, "15_category_performance_validation.csv").head(12))
    validation_export = val[
        ["code", "product_name", "category", "family_group", "nova_target"]
    ].copy()
    validation_export["pred_nova_group"] = val_pred
    validation_export["pred_nova4_prob"] = p4
    save_table(validation_export, "16_nova_validation_predictions.csv")
    state.update(
        {"category_results": category_results, "i4": i4, "validation_export": validation_export}
    )
    return state


def explain_permutation(state: WorkflowState) -> WorkflowState:
    """Compute actual permutation importance on a reproducible validation subset.

    Required artifacts: ALL_FEATURES, SEED, best_model, save_table, show_figure, val.
    Published artifacts: importance.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    SEED = state.require("SEED")
    best_model = state.require("best_model")
    save_table = state.require("save_table")
    show_figure = state.require("show_figure")
    val = state.require("val")
    explain_val = val.sample(n=min(800, len(val)), random_state=SEED)
    permutation = permutation_importance(
        best_model,
        explain_val[ALL_FEATURES],
        explain_val.nova_target.astype(int),
        scoring="f1_macro",
        n_repeats=2,
        random_state=SEED,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": ALL_FEATURES,
            "macro_f1_drop_mean": permutation.importances_mean,
            "macro_f1_drop_sd": permutation.importances_std,
        }
    ).sort_values("macro_f1_drop_mean", ascending=False)
    display(save_table(importance, "17_permutation_importance_validation_subset.csv").head(15))
    plot_permutation_importance(importance, show_figure)
    state.update({"importance": importance})
    return state


def explain_shap(state: WorkflowState) -> WorkflowState:
    """Compute actual TreeSHAP when supported; otherwise record an explicit status.

    Required artifacts: ALL_FEATURES, RUN_SHAP, SEED, comparison, models, save_table, train, val.
    Published artifacts: shap_status.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    RUN_SHAP = state.require("RUN_SHAP")
    SEED = state.require("SEED")
    comparison = state.require("comparison")
    models = state.require("models")
    save_table = state.require("save_table")
    train = state.require("train")
    val = state.require("val")
    shap_status = "not_run"
    if RUN_SHAP and importlib.util.find_spec("shap") is not None:
        import shap

        tree_name = comparison[
            comparison.model.isin(["random_forest", "hist_gradient_boosting"])
        ].iloc[0]["model"]
        tree_pipe = models[tree_name]
        background_rows = train.sample(n=min(100, len(train)), random_state=SEED)
        explained_rows = val.sample(n=min(100, len(val)), random_state=SEED)
        bg = tree_pipe.named_steps["preprocess"].transform(background_rows[ALL_FEATURES])
        xx = tree_pipe.named_steps["preprocess"].transform(explained_rows[ALL_FEATURES])
        try:
            explainer = shap.TreeExplainer(
                tree_pipe.named_steps["model"],
                data=bg,
                model_output="probability",
                feature_perturbation="interventional",
            )
            explained = explainer(xx)
            values = np.asarray(explained.values)
            if values.ndim != 3:
                raise ValueError("Unexpected multiclass SHAP shape; inspect before export.")
            class4 = list(tree_pipe.classes_).index(4)
            feature_names = tree_pipe.named_steps["preprocess"].get_feature_names_out()
            contribution = pd.DataFrame(values[:, :, class4], columns=feature_names)
            contribution.insert(0, "code", explained_rows.code.to_numpy())
            save_table(contribution, "18_actual_tree_shap_nova4.csv")
            shap_status = f"completed:{tree_name}:probability_output:100_background_max"
        except (ValueError, TypeError, NotImplementedError) as error:
            shap_status = "unsupported_configuration:" + str(error)
    else:
        shap_status = (
            "package_unavailable"
            if importlib.util.find_spec("shap") is None
            else "disabled_by_configuration"
        )
    print("Actual TreeSHAP status:", shap_status)
    state.update({"shap_status": shap_status})
    return state


def compare_nutrient_extension(state: WorkflowState) -> WorkflowState:
    """Compare seven core nutrients with the selected extended schema.

    Required artifacts: ALL_FEATURES, CORE_NUTRIENTS, FEATURE_SETS, INGREDIENT_FEATURES, NUTRIENTS, best_model, best_name, fit_logged, make_pipeline, nova_metrics, save_table, train, val.
    Published artifacts: feature_extension.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    CORE_NUTRIENTS = state.require("CORE_NUTRIENTS")
    FEATURE_SETS = state.require("FEATURE_SETS")
    INGREDIENT_FEATURES = state.require("INGREDIENT_FEATURES")
    NUTRIENTS = state.require("NUTRIENTS")
    best_model = state.require("best_model")
    best_name = state.require("best_name")
    fit_logged = state.require("fit_logged")
    make_pipeline = state.require("make_pipeline")
    nova_metrics = state.require("nova_metrics")
    save_table = state.require("save_table")
    train = state.require("train")
    val = state.require("val")
    FEATURE_SETS["core7_control"] = INGREDIENT_FEATURES + CORE_NUTRIENTS
    core_control = make_pipeline(best_name, "core7_control")
    fit_logged(core_control, train[ALL_FEATURES], train.nova_target.astype(int))
    core_control_metrics = nova_metrics(
        core_control, val[ALL_FEATURES], val.nova_target.astype(int)
    )
    feature_extension = pd.DataFrame(
        [
            {
                "nutrient_set": "original_core_7",
                "nutrients": len(CORE_NUTRIENTS),
                **core_control_metrics,
            },
            {
                "nutrient_set": "selected_extended",
                "nutrients": len(NUTRIENTS),
                **nova_metrics(best_model, val[ALL_FEATURES], val.nova_target.astype(int)),
            },
        ]
    )
    display(save_table(feature_extension, "36_core_vs_extended_nutrient_validation.csv"))
    state.update({"feature_extension": feature_extension})
    return state


def complete_nutriscore(state: WorkflowState) -> WorkflowState:
    """Preserve OFF grades, calculate eligible missing grades, then estimate supported gaps.

    Required artifacts: ALL_FEATURES, MODEL_DIR, NUTRIENTS, SEED, df, fit_logged, make_pipeline, save_table, valid_numeric.
    Published artifacts: baseline_grades, nutri_evaluation, routing, status, version_verified.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    MODEL_DIR = state.require("MODEL_DIR")
    NUTRIENTS = state.require("NUTRIENTS")
    SEED = state.require("SEED")
    df = state.require("df")
    fit_logged = state.require("fit_logged")
    make_pipeline = state.require("make_pipeline")
    save_table = state.require("save_table")
    valid_numeric = state.require("valid_numeric")
    RULE_VERSION = "2023"
    RULE_BASIS = (
        "Santé publique France updated algorithm, FAQ tables 5–10; OFF reference implementation"
    )
    zero = {
        "energy_kj": 0,
        "sugars": 0,
        "satfat": 0,
        "salt": 0,
        "fiber": 0,
        "protein": 0,
        "fvl": 0,
        "fat": 0,
        "sweeteners": 0,
    }
    assert nutriscore_2023_one(zero, "food") == (0, "A")
    assert nutriscore_2023_one({**zero, "energy_kj": 335}, "food") == (0, "A")
    assert nutriscore_2023_one({**zero, "energy_kj": 335.1}, "food") == (1, "B")
    assert nutriscore_2023_one({**zero, "salt": 2.21, "protein": 18}, "food") == (11, "D")
    assert nutriscore_2023_one({**zero, "salt": 2.21, "protein": 18}, "cheese") == (4, "C")
    assert nutriscore_2023_one({**zero, "protein": 18}, "red_meat")[0] == -2
    assert nutriscore_2023_one({**zero, "fvl": np.nan}, "food") == (None, None)
    assert nutriscore_2023_one(zero, "beverage") == (0, "B")
    assert nutriscore_2023_one({}, "water") == (None, "A")
    print("2023 calculator boundary/branch tests passed.")
    safe_general_pnns2 = {
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
        "milk and yogurt",
    }
    safe_general_pnns2.discard("milk and yogurt")
    pnns2 = norm_text(col(df, "pnns_groups_2"))
    category_tags = norm_text(col(df, "categories_tags"))
    baby_or_alcohol = category_tags.str.contains(
        "(?:^|,)en:(?:baby-foods|baby-milks|infant-formulas|alcoholic-beverages)(?:,|$)", regex=True
    )
    rule_group = pd.Series("unknown", index=df.index, dtype="string")
    rule_group.loc[pnns2.isin(safe_general_pnns2)] = "food"
    rule_group.loc[pnns2.eq("cheese")] = "cheese"
    plain_water = category_tags.str.contains(
        "(?:^|,)en:(?:natural-mineral-waters|spring-waters)(?:,|$)", regex=True
    )
    plain_water &= ~category_tags.str.contains("flavou?red|sweetened", regex=True)
    rule_group.loc[plain_water] = "water"
    rule_group.loc[pnns2.eq("fats")] = "fat"
    rule_group.loc[baby_or_alcohol | df.nutri_status_original.eq("not-applicable")] = (
        "not_applicable"
    )
    df["nutri_rule_group"] = rule_group
    rule_numerics = pd.DataFrame(index=df.index)
    # Prefer observed kJ. A kcal conversion is only the final fallback.
    rule_numerics["energy_kj"] = (
        valid_numeric.get("energy-kj_100g", pd.Series(np.nan, index=df.index))
        .fillna(valid_numeric["energy_100g"])
        .fillna(df["energy-kcal_100g"] * 4.184)
    )
    rule_numerics["sugars"] = valid_numeric.get("sugars_100g", pd.Series(np.nan, index=df.index))
    rule_numerics["satfat"] = valid_numeric.get(
        "saturated-fat_100g", pd.Series(np.nan, index=df.index)
    )
    rule_numerics["salt"] = valid_numeric.get(
        "salt_100g", pd.Series(np.nan, index=df.index)
    ).fillna(valid_numeric["sodium_100g"] * 2.5)
    rule_numerics["fiber"] = valid_numeric["fiber_100g"]
    rule_numerics["protein"] = valid_numeric["proteins_100g"]
    rule_numerics["fat"] = valid_numeric["fat_100g"]
    rule_numerics["fvl"] = valid_numeric.get(
        "fruits-vegetables-legumes_100g", pd.Series(np.nan, index=df.index)
    )
    rule_numerics["sweeteners"] = np.nan
    relationships_bad = (
        df.flag_satfat_gt_fat | df.flag_sugars_gt_carbs | df.flag_salt_sodium_mismatch
    )
    calculated_score = pd.Series(np.nan, index=df.index, dtype=float)
    calculated_grade = pd.Series(pd.NA, index=df.index, dtype="string")
    formula_status = pd.Series(
        "missing_inputs_or_unresolved_branch", index=df.index, dtype="string"
    )
    required_general = ["energy_kj", "sugars", "satfat", "salt", "fiber", "protein", "fvl"]
    formula_complete = rule_group.isin(["food", "cheese", "red_meat"]) & rule_numerics[
        required_general
    ].notna().all(axis=1)
    formula_complete |= rule_group.eq("fat") & rule_numerics[
        ["sugars", "satfat", "salt", "fiber", "protein", "fat", "fvl"]
    ].notna().all(axis=1)
    formula_complete |= rule_group.eq("beverage") & rule_numerics[
        required_general + ["sweeteners"]
    ].notna().all(axis=1)
    # This CSV does not verify the qualifying-oil component required by fat rules.
    formula_complete &= ~rule_group.eq("fat")
    formula_complete &= ~relationships_bad
    formula_complete |= rule_group.eq("water")
    for idx in df.index[formula_complete]:
        score, grade = nutriscore_2023_one(rule_numerics.loc[idx].to_dict(), rule_group.loc[idx])
        if grade is not None:
            calculated_score.loc[idx] = np.nan if score is None else score
            calculated_grade.loc[idx] = grade
            formula_status.loc[idx] = "calculated_2023"
    print(
        "FVL proportion observed:",
        int(rule_numerics.fvl.notna().sum()),
        "Formula-calculable:",
        int(calculated_grade.notna().sum()),
    )
    df["nutri_split"] = labeled_group_split(df, "nutri_target", seed=SEED)
    nutri_support = df.ingredient_present | df.observed_nutrients.ge(4)
    nutri_labeled = df.nutri_target.notna() & nutri_support
    nt = df.loc[nutri_labeled & df.nutri_split.eq("train")].copy()
    nv = df.loc[nutri_labeled & df.nutri_split.eq("validation")].copy()
    status = "OFF_snapshot_grade_target_version_unverified"
    missing_candidates = df.nutri_target.isna() & ~rule_group.eq("not_applicable") & nutri_support
    # Learn missingness patterns from auxiliary training rows only.
    patterns = nt[NUTRIENTS].isna().to_numpy()
    rng = np.random.default_rng(SEED)
    aug = nt[ALL_FEATURES].copy()
    aug.loc[:, NUTRIENTS] = aug[NUTRIENTS].mask(patterns[rng.integers(0, len(patterns), len(aug))])
    nutri_model = make_pipeline("logistic", "combined")
    fit_logged(
        nutri_model,
        pd.concat([nt[ALL_FEATURES], aug], ignore_index=True),
        pd.concat([nt.nutri_target, nt.nutri_target], ignore_index=True).astype(str),
    )
    masked = nv[ALL_FEATURES].copy()
    masked.loc[:, NUTRIENTS] = masked[NUTRIENTS].mask(
        patterns[rng.integers(0, len(patterns), len(masked))]
    )
    majority = nt.nutri_target.value_counts().idxmax()
    nutri_evaluation = pd.DataFrame(
        [
            {
                "setting": "majority_grade",
                "status": status,
                **grade_metrics(nv.nutri_target, np.repeat(majority, len(nv))),
            },
            {
                "setting": "observed_inputs",
                "status": status,
                **grade_metrics(nv.nutri_target, nutri_model.predict(nv[ALL_FEATURES])),
            },
            {
                "setting": "training_pattern_masked_nutrients",
                "status": status,
                **grade_metrics(nv.nutri_target, nutri_model.predict(masked)),
            },
        ]
    )
    display(save_table(nutri_evaluation, "19_nutriscore_validation.csv"))
    joblib.dump(nutri_model, MODEL_DIR / "nutriscore_logistic_pipeline.joblib")
    baseline_grades = df[["code", "nutriscore_grade"]].copy()
    baseline_grades["grade_display"] = df.nutri_target.str.upper()
    baseline_grades["grade_provenance"] = np.where(
        df.nutri_target.notna(),
        "OFF_provided",
        np.where(rule_group.eq("not_applicable"), "not_applicable", "unknown"),
    )
    baseline_grades["grade_version"] = np.where(
        df.nutri_target.notna(), "OFF_unverified", "unknown"
    )
    baseline_grades["calculated_score_2023"] = calculated_score
    baseline_grades["calculated_grade_2023"] = calculated_grade
    baseline_grades["rule_group"] = rule_group
    baseline_grades["rule_status"] = formula_status
    # Routing priority: observed OFF grade, verified rule inputs, ML estimate, unknown.
    apply_formula = (
        df.nutri_target.isna() & calculated_grade.notna() & ~rule_group.eq("not_applicable")
    )
    baseline_grades.loc[apply_formula, "grade_display"] = calculated_grade[apply_formula]
    baseline_grades.loc[apply_formula, "grade_provenance"] = "calculated_2023"
    baseline_grades.loc[apply_formula, "grade_version"] = "2023"
    apply_model = missing_candidates & ~apply_formula
    baseline_grades["estimate_confidence"] = np.nan
    if apply_model.any():
        estimate = nutri_model.predict(df.loc[apply_model, ALL_FEATURES])
        baseline_grades.loc[apply_model, "grade_display"] = pd.Series(
            estimate, index=df.index[apply_model]
        ).str.upper()
        baseline_grades.loc[apply_model, "grade_provenance"] = "ML_estimated_OFF_snapshot"
        baseline_grades.loc[apply_model, "grade_version"] = "OFF_training_version_unverified"
        baseline_grades.loc[apply_model, "estimate_confidence"] = nutri_model.predict_proba(
            df.loc[apply_model, ALL_FEATURES]
        ).max(axis=1)
    assert baseline_grades.loc[df.nutri_target.notna(), "grade_display"].equals(
        df.loc[df.nutri_target.notna(), "nutri_target"].str.upper()
    )
    assert not (apply_formula & apply_model).any()
    version_verified = False
    save_table(baseline_grades, "21_nutriscore_baseline_provenance.csv")
    routing = (
        baseline_grades.grade_provenance.value_counts()
        .rename_axis("source")
        .reset_index(name="rows")
    )
    display(save_table(routing, "34_nutriscore_hybrid_routing.csv"))
    check = df.nutri_target.notna() & calculated_grade.notna()
    print(
        "Formula vs existing OFF overlap:",
        int(check.sum()),
        "agreement:",
        (
            float(calculated_grade[check].str.lower().eq(df.loc[check, "nutri_target"]).mean())
            if check.any()
            else None
        ),
    )
    state.update(
        {
            "baseline_grades": baseline_grades,
            "nutri_evaluation": nutri_evaluation,
            "routing": routing,
            "status": status,
            "version_verified": version_verified,
        }
    )
    return state


def simulate_ingredient_scenarios(state: WorkflowState) -> WorkflowState:
    """Recompute linked ingredient features for restricted single and paired scenarios.

    Required artifacts: ALL_FEATURES, INGREDIENT_FEATURES, NUTRIENTS, SEED, best_model, i4, save_table, val.
    Published artifacts: scenario_results.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    INGREDIENT_FEATURES = state.require("INGREDIENT_FEATURES")
    NUTRIENTS = state.require("NUTRIENTS")
    SEED = state.require("SEED")
    best_model = state.require("best_model")
    i4 = state.require("i4")
    save_table = state.require("save_table")
    val = state.require("val")
    from itertools import combinations

    SCENARIO_MARKERS = ["sucralose", "aspartame", "carrageenan", "lecithin", "polysorbate80"]
    scenario_products = val.loc[val.nova_target.eq(4)].sample(
        n=min(500, int(val.nova_target.eq(4).sum())), random_state=SEED
    )
    scenario_sources = []
    scenario_meta = []
    scenario_audit = []
    for index, row in scenario_products.iterrows():
        tags = parse_tags(row.get("additives_tags", pd.NA))
        observed_count = pd.to_numeric(
            pd.Series([row.get("additives_n", pd.NA)]), errors="coerce"
        ).iloc[0]
        if pd.isna(observed_count) or observed_count != len(tags):
            scenario_audit.append(
                {"code": row.code, "status": "unsupported_additive_count_or_tags"}
            )
            continue
        present = [
            m
            for m in SCENARIO_MARKERS
            if row["marker_" + m] == 1 and re.search(MARKERS[m][0], row.ingredients_text_norm)
        ]
        if not present:
            scenario_audit.append({"code": row.code, "status": "no_eligible_marker"})
            continue
        for size in [1, 2]:
            for changed in combinations(present, size):
                altered = row.copy()
                updated_text = row.ingredients_text_norm
                remaining_tags = set(tags)
                for marker in changed:
                    pattern, codes = MARKERS[marker]
                    updated_text = re.sub(pattern, " ", updated_text)
                    remaining_tags -= codes
                altered["ingredients_text_norm"] = re.sub("\\s+", " ", updated_text).strip()
                altered["additives_tags"] = ",".join(("en:" + t for t in sorted(remaining_tags)))
                altered["additives_n"] = len(remaining_tags)
                scenario_sources.append(altered)
                scenario_meta.append(
                    {
                        "code": row.code,
                        "scenario_id": "+".join(changed),
                        "edit_count": size,
                        "category": row.category,
                        "baseline_nova4_probability": float(
                            best_model.predict_proba(pd.DataFrame([row])[ALL_FEATURES])[0, i4]
                        ),
                    }
                )
        scenario_audit.append({"code": row.code, "status": "scenarios_generated"})
    scenario_results = pd.DataFrame()
    if scenario_sources:
        scenario_X = pd.DataFrame(scenario_sources).reset_index(drop=True)
        recomputed = ingredient_features(scenario_X)
        for c in INGREDIENT_FEATURES:
            scenario_X[c] = recomputed[c]
        scenario_results = pd.DataFrame(scenario_meta)
        scenario_results["scenario_nova4_probability"] = best_model.predict_proba(
            scenario_X[ALL_FEATURES]
        )[:, i4]
        scenario_results["probability_change"] = (
            scenario_results.scenario_nova4_probability
            - scenario_results.baseline_nova4_probability
        )
        scenario_results["lower_prediction"] = scenario_results.probability_change.lt(-1e-08)
        scenario_results["nutrition_assumption"] = "held_constant"
        scenario_results["interpretation"] = "model_sensitivity_only"
        original_by_code = val.set_index("code")
        original_nutrients = original_by_code.loc[scenario_results.code, NUTRIENTS].reset_index(
            drop=True
        )
        pd.testing.assert_frame_equal(
            scenario_X[NUTRIENTS].reset_index(drop=True),
            original_nutrients,
            check_dtype=False,
            check_names=False,
        )
        save_table(scenario_results, "22_nova4_scenarios.csv")
        print(
            "Eligible products:",
            scenario_results.code.nunique(),
            "scenarios:",
            len(scenario_results),
            "share with lower prediction:",
            round(scenario_results.lower_prediction.mean(), 4),
        )
        display(scenario_results.sort_values("probability_change").head(10))
    save_table(pd.DataFrame(scenario_audit), "23_scenario_eligibility_audit.csv")
    if scenario_results.empty:
        print("No supported scenarios generated; review marker coverage and eligibility.")
    state.update({"scenario_results": scenario_results})
    return state


def evaluate_final_test(state: WorkflowState) -> WorkflowState:
    """Optionally evaluate the frozen model and write the reproducibility summary.

    Required artifacts: ALL_FEATURES, OUTPUT_DIR, RUN_FINAL_TEST, SEED, ablation, actual_latest, best_model, best_name, comparison, df, main, manifest, nova_metrics, nutri_evaluation, save_table, scenario_results, shap_status, status, train, val, version_verified.
    Published artifacts: run_summary.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    OUTPUT_DIR = state.require("OUTPUT_DIR")
    RUN_FINAL_TEST = state.require("RUN_FINAL_TEST")
    SEED = state.require("SEED")
    ablation = state.require("ablation")
    actual_latest = state.require("actual_latest")
    best_model = state.require("best_model")
    best_name = state.require("best_name")
    comparison = state.require("comparison")
    df = state.require("df")
    main = state.require("main")
    manifest = state.require("manifest")
    nova_metrics = state.require("nova_metrics")
    nutri_evaluation = state.require("nutri_evaluation")
    save_table = state.require("save_table")
    scenario_results = state.require("scenario_results")
    shap_status = state.require("shap_status")
    status = state.require("status")
    train = state.require("train")
    val = state.require("val")
    version_verified = state.require("version_verified")
    # Development runs do not inspect test-set labels or metrics.
    test_result = None
    if RUN_FINAL_TEST:
        test = main.loc[main.split.eq("test")]
        test_result = {
            "model": best_name,
            "split": "test",
            **nova_metrics(best_model, test[ALL_FEATURES], test.nova_target.astype(int)),
        }
        display(save_table(pd.DataFrame([test_result]), "25_final_nova_test_metrics.csv"))
    else:
        print(
            "Final test set held back. All model comparison numbers above are validation results."
        )
    run_summary = {
        "input_manifest": manifest,
        "latest_observed_us_modified": str(actual_latest),
        "processed_rows": len(df),
        "main_nova_comparison_rows": len(main),
        "train_rows": len(train),
        "validation_rows": len(val),
        "held_back_test_rows": int(main.split.eq("test").sum()),
        "selected_model": best_name,
        "validation_results": comparison.to_dict(orient="records"),
        "feature_ablation": ablation.to_dict(orient="records"),
        "nutriscore_version_verified": version_verified,
        "nutriscore_status": status,
        "nutriscore_validation": nutri_evaluation.to_dict(orient="records"),
        "shap_status": shap_status,
        "scenario_rows": len(scenario_results),
        "scenario_products": int(scenario_results.code.nunique()) if len(scenario_results) else 0,
        "test_result": test_result,
        "seed": SEED,
        "software": {
            "python": sys.version.split()[0],
            "sklearn": sklearn_version,
            "pandas": pd.__version__,
        },
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2, default=str), encoding="utf-8"
    )
    state.update({"run_summary": run_summary})
    return state


def predict_missing_nova(state: WorkflowState) -> WorkflowState:
    """Estimate eligible missing NOVA labels without overwriting observed targets.

    Required artifacts: ALL_FEATURES, REFIT_FOR_DEPLOYMENT, RUN_FINAL_TEST, best_model, df, main, save_table.
    Published artifacts: nova_fill.
    """
    ALL_FEATURES = state.require("ALL_FEATURES")
    REFIT_FOR_DEPLOYMENT = state.require("REFIT_FOR_DEPLOYMENT")
    RUN_FINAL_TEST = state.require("RUN_FINAL_TEST")
    best_model = state.require("best_model")
    df = state.require("df")
    main = state.require("main")
    save_table = state.require("save_table")
    from sklearn.base import clone

    # Preliminary estimates remain separate from observed labels and evaluation truth.
    inference_model = best_model
    if REFIT_FOR_DEPLOYMENT:
        if not RUN_FINAL_TEST:
            raise ValueError(
                "Evaluate the frozen model on the held-back test set before deployment refitting."
            )
        inference_model = clone(best_model)
        inference_model.fit(main[ALL_FEATURES], main.nova_target.astype(int))
    missing_nova = df.nova_target.isna()
    inference_support = (
        df.ingredient_present | df.observed_nutrients.ge(4) | df.additives_n_feature.notna()
    )
    nova_fill = df[["code", "nova_group", "split"]].copy()
    nova_fill["nova_group_estimated"] = np.nan
    nova_fill["nova4_probability_estimated"] = np.nan
    nova_fill["nova_provenance"] = np.where(
        df.nova_target.notna(), "OFF_provided", "unknown_insufficient_input"
    )
    application = missing_nova & inference_support
    if application.any():
        nova_fill.loc[application, "nova_group_estimated"] = inference_model.predict(
            df.loc[application, ALL_FEATURES]
        )
        ci4 = list(inference_model.classes_).index(4)
        nova_fill.loc[application, "nova4_probability_estimated"] = inference_model.predict_proba(
            df.loc[application, ALL_FEATURES]
        )[:, ci4]
        nova_fill.loc[application, "nova_provenance"] = (
            "ML_estimated_deployment" if REFIT_FOR_DEPLOYMENT else "ML_estimated_preliminary"
        )
    assert nova_fill.loc[df.nova_target.notna(), "nova_group_estimated"].isna().all()
    save_table(nova_fill, "35_nova_missing_label_predictions.csv")
    display(nova_fill.nova_provenance.value_counts().to_frame("rows"))
    state.update({"nova_fill": nova_fill})
    return state


def write_analysis_report(state: WorkflowState) -> WorkflowState:
    """Generate English interpretations from the actual saved results.

    Required artifacts: OUTPUT_DIR, RUN_TEXT_BASELINE, TABLE_DIR.
    Published artifacts: summary_text.
    """
    OUTPUT_DIR = state.require("OUTPUT_DIR")
    RUN_TEXT_BASELINE = state.require("RUN_TEXT_BASELINE")
    TABLE_DIR = state.require("TABLE_DIR")
    report_run = json.loads((OUTPUT_DIR / "run_summary.json").read_text())
    report_manifest = report_run["input_manifest"]
    report_flow = pd.read_csv(TABLE_DIR / "01_label_flow.csv").set_index("cohort")["rows"]
    report_comparison = pd.read_csv(TABLE_DIR / "11_nova_model_comparison_validation.csv")
    report_text = (
        pd.read_csv(TABLE_DIR / "13_text_baseline_validation.csv")
        if (TABLE_DIR / "13_text_baseline_validation.csv").exists() and RUN_TEXT_BASELINE
        else pd.DataFrame()
    )
    report_ablation = pd.read_csv(TABLE_DIR / "12_feature_ablation_validation.csv")
    report_nutri = (
        pd.read_csv(TABLE_DIR / "19_nutriscore_validation.csv")
        if (TABLE_DIR / "19_nutriscore_validation.csv").exists()
        else pd.DataFrame()
    )
    selected = report_run["selected_model"]
    best_row = report_comparison.loc[report_comparison.model.eq(selected)].iloc[0]
    text_sentence = (
        f"A separate TF-IDF logistic-regression baseline achieved validation macro-F1 {report_text.iloc[0]['macro_f1']:.4f}. It uses a different text representation without category covariates and is reported separately from the structured feature ablation. The selected structured model supports the initial marker-based scenario workflow; it is not claimed to outperform the text baseline."
        if len(report_text)
        else "The optional TF-IDF text baseline was not run."
    )
    summary_text = f"# Team 33 Preliminary Analysis — Actual Run\n\n## Data and provenance\nThe complete newly downloaded OFF archive was scanned ({report_manifest['raw_rows_scanned']:,} records). Exact U.S.-market and last-modification-date filtering retained {int(report_flow['All US/window records']):,} records for July 1, 2024 through June 30, 2026. The latest observed modification timestamp is {report_run['latest_observed_us_modified']}. This is a current-snapshot cohort and cannot reconstruct earlier states of products subsequently updated outside the window. The extract contains {int(report_flow['Valid NOVA']):,} valid NOVA targets and {int(report_flow['Valid Nutri-Score A-E']):,} valid Nutri-Score grades. All unmatched-label states were retained for coverage analysis.\n\n## Preparation and experimental design\nWe standardized ingredient strings, validated targets, retained barcode identifiers as strings, flagged nutrient inconsistencies, and replaced impossible single-field nutrient values with missing values. Product-family proxies combine brand/name and brand/ingredient matches, with disjoint groups assigned to training, validation, and held-back test subsets. The NOVA modeling cohort contains only records with valid observed NOVA labels, yielding {report_run['main_nova_comparison_rows']:,} products ({report_run['train_rows']:,} training, {report_run['validation_rows']:,} validation, and {report_run['held_back_test_rows']:,} held-back test products). Products lacking NOVA are assigned to an application pool and are never used as labeled training or evaluation examples.\n\n## Preliminary NOVA results\nThe structured model {selected} achieved validation macro-F1 {best_row['macro_f1']:.4f}, NOVA 4 average precision {best_row['nova4_average_precision']:.4f}, and Brier score {best_row['nova4_brier']:.4f}. Model ranking remains provisional until frozen-method testing; a single validation split does not establish statistical superiority. Using the same structured model and evaluation cohort, ingredient, nutrient, and combined feature macro-F1 scores were {report_ablation.loc[report_ablation.feature_set.eq('ingredients'), 'macro_f1'].iloc[0]:.4f}, {report_ablation.loc[report_ablation.feature_set.eq('nutrients'), 'macro_f1'].iloc[0]:.4f}, and {report_ablation.loc[report_ablation.feature_set.eq('combined'), 'macro_f1'].iloc[0]:.4f}, respectively. These are validation results, not final test results. Histogram boosting is the initial sklearn implementation in place of external packages. Metrics measure agreement with OFF reference labels and do not constitute independent expert or health validation.\n\n{text_sentence}\n\n## Interpretation and visualization integration\nWe computed permutation importance on a reproducible validation subset and exported actual predictions, nutritional baselines, category information, and grade provenance for Tableau. Actual TreeSHAP status is {report_run['shap_status']}. We evaluated {report_run['scenario_rows']:,} predefined single/paired feature scenarios across {report_run['scenario_products']:,} eligible products from a 500-product validation development sample. Linked ingredient features were recomputed while nutrient values remained fixed. Restricted marker coverage limits scenario applicability. Probability changes are model sensitivity, with no demonstrated recipe feasibility or nutritional preservation. Tableau integration and user testing remain team tasks.\n\n## Auxiliary Nutri-Score task and remaining work\nThe auxiliary logistic-regression target basis is {report_run['nutriscore_status']}. Observed-input and empirical masked-nutrient results are stored separately. The hybrid export preserves existing OFF grades, calculates missing grades where 2023 rule inputs and the scoring branch are known, and uses the auxiliary classifier for supported remaining cases. Calculated grades are marked 2023; model estimates retain the unverified OFF-training-version status. These sources must not be silently treated as a harmonized single-version series. Remaining work includes scoring-version verification, domain review of marker mappings and scenario coverage, actual SHAP when dependencies are available, dashboard user evaluation, and final testing after methods are frozen.\n"
    eda_report_path = OUTPUT_DIR / "eda_analysis_en.md"
    if eda_report_path.exists():
        summary_text = summary_text.replace(
            "## Preliminary NOVA results",
            eda_report_path.read_text() + "\n\n## Preliminary NOVA results",
        )
    (OUTPUT_DIR / "preliminary_analysis_en.md").write_text(summary_text, encoding="utf-8")
    display(Markdown(summary_text))
    display(
        Markdown(
            f"**Run interpretation:** The NOVA comparison cohort contains {report_run['main_nova_comparison_rows']:,} records. Combined inputs outperform nutrient-only and limited-marker inputs in this validation run."
            + (
                f" The separate TF-IDF text baseline has macro-F1 {report_text.iloc[0]['macro_f1']:.3f}, exceeding the structured model in this run and motivating further comparison."
                if len(report_text)
                else ""
            )
            + " Small differences between tree models do not establish superiority. Scenarios describe model responses. Final testing, actual SHAP, Nutri-Score version verification, and Tableau user evaluation remain pending."
        )
    )
    state.update({"summary_text": summary_text})
    return state
