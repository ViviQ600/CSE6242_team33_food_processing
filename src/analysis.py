"""Analysis Module

Contains functions for descriptive statistics and basic modeling.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_descriptive_statistics(df):
    """Compute summary statistics for a dataframe."""
    return df.describe()

def perform_pca(df, n_components=2):
    """Apply PCA to numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)
    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(scaled)
    return transformed, pca, pca.explained_variance_ratio_

def correlation_analysis(df):
    """Compute correlation matrix for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()
