# Test the model using Test Data
import pandas as pd
from ml_models import train_model_pca
from data_preparation import load_data, preprocess_data
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score

def pca_predict(model, pca, test_data):
    feature_cols = test_data.columns[1:16]  # time and A to N

    test_data_pca = pca.transform(test_data[feature_cols])  # Use transform, NOT fit_transform
    predictions = model.predict(test_data_pca)
    return predictions


if __name__ == "__main__":
    # Load and preprocess training and test data 
    # Training data: time,A,B,C,D,E,F,G,H,I,J,K,L,M,N,Y1,Y2
    print("Loading and preprocessing data...")
    train_data = load_data("train.csv")
    train_data = preprocess_data(train_data)

    # Train model and evaluate
    print("Training model...")
    model, pca = train_model_pca(train_data)
    print("Model trained. Now processing test data...")

    # Test data: id,time,A,B,C,D,E,F,G,H,I,J,K,L,M,N
    test_data = load_data("test.csv")
    test_data = preprocess_data(test_data, start_index=2, end_index=16)  # Adjusted for test data

    # Make predictions
    print("Making predictions...")
    predictions = pca_predict(model, pca, test_data)
    print(predictions)
    
    # Save predictions to CSV
    # Required Columns:
    # id - Unique (one-indexed) identifier for each prediction (integer)
    # Y1 - Your numeric prediction value for Y1 (float)
    # Y2 - Your numeric prediction value for Y2 (float)
    pd.DataFrame(predictions, columns=['Y1', 'Y2']).to_csv("predictions.csv", index_label='id')