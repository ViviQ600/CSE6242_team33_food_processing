"""Unit tests for analysis module"""

import numpy as np
import pandas as pd
import pytest

from src.analysis import compute_descriptive_statistics, correlation_analysis

class TestAnalysis:
    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        return pd.DataFrame({
            "X": np.random.randn(100),
            "Y": np.random.randn(100),
            "Z": np.random.randn(100),
        })

    def test_descriptive_statistics(self, sample_data):
        stats = compute_descriptive_statistics(sample_data)
        assert not stats.empty

    def test_correlation_analysis(self, sample_data):
        corr = correlation_analysis(sample_data)
        assert corr.shape == (3, 3)
        assert np.allclose(np.diag(corr), np.ones(3))
