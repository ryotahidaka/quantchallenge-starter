# This script handles data loading and preprocessing of csv data files.
# Normalization is performed for both dense and sparse features.
    # - Dense features (A-N) are normalized
    # - Sparse features (O-P) are normalized
# Sparse features are imputed.
# Note: time features should not be included in normalization or imputation.

import pandas as pd
from sklearn.impute import KNNImputer

def load_data(file_path):
    """Load data from a CSV file.
    
    Args:
        file_path (str): The path to the CSV file.
    """
    return pd.read_csv(file_path)

def preprocess_data(data, start_index=1, end_index=15):
    """Preprocess the data by handling missing values and normalizing.
    
    Args:
        data (pd.DataFrame): The input data frame.
        start_index (int): The starting index for dense features (A).
        end_index (int): The ending index for dense features (N).
    """
    # Handle missing values (if any)
    data = data.fillna(data.mean())
    
    # Normalize features (A-N)
    feature_cols = data.columns[start_index:end_index]  # A to N
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    return data

def preprocess_sparse_data(data, imputer: KNNImputer, start_index=0, end_index=2):
    """Preprocess sparse data by normalizing and imputing.
    
    Args:
        data (pd.DataFrame): The input data frame.
        imputer (KNNImputer): The KNN imputer instance.
        start_index (int): The starting index for sparse features (O).
        end_index (int): The ending index for sparse features (P).
    """
    feature_cols = data.columns[start_index:end_index]  # O to P
    
    # Normalize
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    # Impute missing values
    data[feature_cols] = imputer.fit_transform(data[feature_cols])

    return data