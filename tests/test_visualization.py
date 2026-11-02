import numpy as np
import pandas as pd

from src.analysis import INGREDIENT_FEATURES, ingredient_features, tableau_additive_features
from src.visualization import TABLEAU_EXPORT_COLUMNS, export_tableau


class ProbabilityModel:
    classes_ = np.array([1, 2, 3, 4])

    def predict_proba(self, frame):
        sugar = pd.to_numeric(frame["sugars_100g"], errors="coerce").fillna(0)
        additives = pd.to_numeric(
            frame["additives_n_feature"], errors="coerce"
        ).fillna(0)
        p4 = 1 / (1 + np.exp(-(sugar * 0.03 + additives * 0.2 - 1)))
        return np.column_stack(
            [(1 - p4) * 0.3, (1 - p4) * 0.2, (1 - p4) * 0.5, p4]
        )

    def predict(self, frame):
        return np.where(self.predict_proba(frame)[:, 3] > 0.5, 4, 1)


def test_tableau_additive_features_use_ingredient_and_tag_evidence():
    frame = pd.DataFrame(
        {
            "ingredients_text": ["E-471 mono-and diglycerides", None, None],
            "additives_en": [None, None, None],
            "additives_tags": ["en:e471,en:e955", "en:e211", None],
            "additives_n": [2, 1, 0],
        }
    )

    features = tableau_additive_features(frame)

    assert features.loc[0, "has_emulsifier"] == 1
    assert features.loc[0, "has_sweetener"] == 1
    assert features.loc[1, "has_preservative"] == 1
    assert features.loc[1, "has_other_additive"] == 0
    assert pd.isna(features.loc[2, "has_emulsifier"])
    assert features.loc[2, "has_other_additive"] == 0


def test_tableau_export_keeps_workbook_schema_and_hides_test_grade(tmp_path):
    frame = pd.DataFrame(
        {
            "product_id": ["1", "2"],
            "code": ["100110296241329314942455330", "2"],
            "product_name": ["Product one", None],
            "url": ["https://example.test/1", "https://example.test/2"],
            "category": ["snacks", "unknown"],
            "ingredients_text_norm": ["sugar", "sucralose"],
            "ingredients_text": ["sugar", "sucralose"],
            "additives_tags": [None, "en:e955"],
            "additives_en": [None, "E955 - Sucralose"],
            "additives_n": [0, 1],
            "nutriscore_score": [4, 8],
            "nutri_split": ["test", "inference"],
            "sugars_100g": [25.0, 10.0],
            "sodium_100g": [0.2, 0.1],
            "saturated-fat_100g": [1.0, 2.0],
            "fiber_100g": [2.0, 1.0],
            "proteins_100g": [3.0, 4.0],
        }
    )
    frame[INGREDIENT_FEATURES] = ingredient_features(frame)
    grades = pd.DataFrame(
        {
            "product_id": ["1", "2"],
            "grade_display": ["B", "D"],
            "grade_source": ["OFF_observed", "OFF_observed"],
            "nutri_split": ["test", "inference"],
            "formula_score_2023": [np.nan, np.nan],
        }
    )

    exported = export_tableau(
        frame,
        ProbabilityModel(),
        "test",
        grades,
        pd.DataFrame(),
        pd.DataFrame(),
        tmp_path,
        run_final_test=False,
    )

    assert exported.columns.tolist() == TABLEAU_EXPORT_COLUMNS
    assert len(exported) == 2
    assert exported.code.tolist() == ["100110296241329314942455330", "2"]
    assert pd.isna(exported.loc[0, "nutriscore_grade_final"])
    assert exported.loc[0, "nutriscore_provenance"] == "hidden_final_test"
    assert exported.loc[1, "nutriscore_grade_final"] == "D"
    assert (tmp_path / "tables" / "tableau_products.csv").is_file()
