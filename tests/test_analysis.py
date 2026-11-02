import numpy as np
import pandas as pd

from src.analysis import NUTRIENT_CANDIDATES, load_processed_data, select_nutrients
from src.data_processing import clean_data


def test_load_processed_data_combines_and_audits_both_sources(tmp_path):
    labeled_path = tmp_path / "us_openfoodfacts_nutri_nova_202407_202606.csv"
    unlabeled_path = tmp_path / "us_openfoodfacts_unlabeled_202407_202606.csv"
    labeled = pd.DataFrame(
        {
            "code": ["001", "001"],
            "nova_group": ["4", "4"],
            "nutriscore_grade": ["d", "d"],
            "ingredients_text": ["sugar, oil", "sugar, oil"],
        }
    )
    unlabeled = pd.DataFrame(
        {
            "code": ["002"],
            "nova_group": [None],
            "nutriscore_grade": ["unknown"],
            "ingredients_text": ["wheat flour"],
        }
    )
    labeled.to_csv(labeled_path, index=False)
    unlabeled.to_csv(unlabeled_path, index=False)

    combined, audit = load_processed_data(tmp_path, clean_function=clean_data)

    assert len(combined) == 2
    assert combined.code.tolist() == ["001", "002"]
    assert combined.input_file.tolist() == [
        "us_openfoodfacts_nutri_nova_202407_202606.csv",
        "us_openfoodfacts_unlabeled_202407_202606.csv",
    ]
    assert audit.after_exact_deduplication.tolist() == [1, 1]


def test_select_nutrients_accepts_sparse_current_coverage():
    training = pd.DataFrame(
        {
            column: pd.Series(range(1, 13), dtype=float).reindex(range(40))
            for column in NUTRIENT_CANDIDATES
        }
    )
    training["trans-fat_100g"] = pd.Series(range(1, 9), dtype=float).reindex(
        range(40)
    )

    selected, profile = select_nutrients(training)

    assert len(selected) >= 4
    assert "trans-fat_100g" not in selected
    assert profile.loc[profile.feature.eq("sugars_100g"), "selected"].item()
