import pandas as pd
from sklearn.impute import KNNImputer

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data, start_index=1, end_index=15):
    """Preprocess the data by handling missing values and normalizing."""
    # Handle missing values (if any)
    data = data.fillna(data.mean())
    
    # Normalize features (A-N)
    feature_cols = data.columns[start_index:end_index]  # A to N
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    return data

def preprocess_sparse_data(data, imputer: KNNImputer, start_index=0, end_index=2):
    """Preprocess sparse data by normalizing and imputing."""
    feature_cols = data.columns[start_index:end_index]  # O to P
    
    # Normalize
    data[feature_cols] = (data[feature_cols] - data[feature_cols].mean()) / data[feature_cols].std()

    # Impute missing values
    data[feature_cols] = imputer.fit_transform(data[feature_cols])

    return data