import numpy as np
import pandas as pd

from src.analysis import CORE_NUTRIENTS, assign_splits, ingredient_features
from src.modeling import (
    final_test_metrics,
    ingredient_scenarios,
    optimal_nova4_threshold,
)
from src.visualization import calibrate_additive_weights


class ProbabilityModel:
    classes_ = np.array([1, 2, 3, 4])

    def predict_proba(self, frame):
        sugar = frame["sugars_100g"].fillna(0).to_numpy()
        additives = frame["additives_n_feature"].fillna(0).to_numpy()
        p4 = 1 / (1 + np.exp(-(sugar * 0.03 + additives * 0.2 - 1)))
        return np.column_stack(
            [(1 - p4) * 0.3, (1 - p4) * 0.2, (1 - p4) * 0.5, p4]
        )


def test_validation_calibration_and_final_test_metrics():
    model = ProbabilityModel()
    validation = pd.DataFrame(
        {
            "nova_target": [4, 4, 1, 1],
            "sugars_100g": [25.0, 20.0, 1.0, 2.0],
            "additives_n_feature": [2.0, 1.0, 0.0, 0.0],
            "marker_lecithin": [1.0, 0.0, 0.0, 0.0],
            "marker_carrageenan": [0.0, 1.0, 0.0, 0.0],
            "marker_polysorbate80": [0.0] * 4,
            "marker_xanthan": [0.0] * 4,
            "marker_guar": [0.0] * 4,
            "marker_sucralose": [1.0, 0.0, 0.0, 0.0],
            "marker_aspartame": [0.0] * 4,
            "marker_acesulfame": [0.0] * 4,
            "marker_saccharin": [0.0] * 4,
            "marker_benzoate": [0.0] * 4,
            "marker_sorbate": [0.0] * 4,
            "marker_nitrite": [0.0] * 4,
            "marker_msg": [0.0] * 4,
            "marker_artificial_color": [0.0] * 4,
        }
    )

    weights = calibrate_additive_weights(model, validation)
    threshold = optimal_nova4_threshold(model, validation)
    test_metrics = final_test_metrics(model, validation, threshold)

    assert weights.parameter.tolist() == [
        "emulsifier_weight",
        "sweetener_weight",
        "preservative_weight",
        "flavor_enhancer_weight",
        "artificial_color_weight",
    ]
    assert weights.loc[0, "eligible_products"] == 2
    assert 0 <= threshold <= 1
    assert {"macro_f1", "nova4_average_precision", "nova4_brier"} <= set(
        test_metrics.metric
    )
    assert set(test_metrics.loc[test_metrics.metric.eq("f1"), "class"]) == {
        1, 2, 3, 4
    }


def test_ingredient_scenarios_include_paired_edits_and_preserve_nutrients(tmp_path):
    row = {
        "product_id": "p1",
        "nova_target": 4,
        "needs_review": False,
        "ingredients_text_norm": (
            "sugar sucralose carrageenan lecithin water salt flour oil"
        ),
        "additives_tags": "en:e955,en:e407,en:e322",
        "additives_n": 3,
        "additives_n_feature": 3.0,
        "sugars_100g": 20.0,
        "sodium_100g": 0.4,
        "saturated-fat_100g": 1.5,
        "energy-kcal_100g": 200.0,
        "fat_100g": 4.0,
        "fiber_100g": 3.0,
        "proteins_100g": 5.0,
    }
    validation = pd.DataFrame([row])
    validation[ingredient_features(validation).columns] = ingredient_features(
        validation
    )
    original_nutrients = validation[CORE_NUTRIENTS].copy()

    scenarios = ingredient_scenarios(ProbabilityModel(), validation, tmp_path)

    assert set(scenarios.scenario_id) == {
        "remove_sucralose",
        "remove_carrageenan",
        "remove_lecithin",
        "remove_sucralose_lecithin",
        "remove_carrageenan_lecithin",
        "remove_sucralose_carrageenan",
    }
    pd.testing.assert_frame_equal(
        validation[CORE_NUTRIENTS], original_nutrients, check_exact=True
    )


def test_assign_splits_keeps_family_groups_out_of_other_splits():
    rows = []
    for family in range(20):
        for duplicate in range(4):
            rows.append(
                {
                    "product_id": f"{family}-{duplicate}",
                    "ingredients_text_norm": (
                        f"recipe family {family} wheat flour water yeast salt "
                        "vegetable oil whole grains"
                    ),
                    "product_name": f"product {family} variant {duplicate}",
                    "brands": None,
                    "nova_target": family % 4 + 1,
                    "nutri_target": "abcde"[family % 5],
                    "needs_review": False,
                    "additives_tags": None,
                    "additives_n": np.nan,
                }
            )
    frame = pd.DataFrame(rows)

    split, _ = assign_splits(frame, seed=13)

    for split_column in ("nova_split", "nutri_split"):
        assert (
            split.groupby("family_group")[split_column].nunique().max() == 1
        )
