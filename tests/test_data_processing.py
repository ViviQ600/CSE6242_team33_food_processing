"""Unit tests for data_processing module"""

import pandas as pd
import pytest

from src.data_processing import clean_data, load_raw_data

class TestDataProcessing:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            "A": [1, 2, 2, 4],
            "B": [5, 6, 6, 8],
            "C": [9, 10, 11, 12],
        })

    def test_clean_data_removes_duplicates(self, sample_data):
        cleaned = clean_data(sample_data)
        assert len(cleaned) == len(sample_data.drop_duplicates())

    def test_load_raw_data_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.csv")
