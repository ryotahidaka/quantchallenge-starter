# Final model training script using Elastic Net regression.
# This script loads the training and test data, preprocesses it, trains the model, and
# makes predictions on the test set.
# It also includes functionality to find the best hyperparameters for the Elastic Net model.
# The model's performance is evaluated using R^2 score.
# PCA can be optionally applied for dimensionality reduction. - currently set to 0 as it tended to perform worse (no PCA).
# The script saves the predictions to a CSV file.
# Additionally, it compares the R^2 score of the model's predictions with a friend's model to ensure that we haven't overfitted.

# Models we have considered:
    # 1. Ridge Regression
    # 2. Lasso Regression
    # 3. Elastic Net - Choose this as final model
    # 4. Support Vector Machine (SVM)
    # 5. Random Forest Regressor
    # 6. XGBoost

# Data description:
    # Columns: 17 columns (time, A-N, Y1, Y2) and 2 sparse columns (O-P)
    # Rows: 80,000 rows of data
    # Features: Time column and columns A through P represent real market features that have been anonymized for the purposes of this competition
    # Targets: Y1 and Y2 are the target variables you need to predict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer

from data_preparation import load_data, preprocess_data, preprocess_sparse_data


def train_model_elastic_net(data, sparse_data, pca_dimensions=0):
    """Train an Elastic Net model to predict Y1 and Y2.
    
    Example outputs:
    Validation Mean Squared Error without PCA: 0.264553 | R^2 Score: 0.705588
    R^2 Score compared to friend's model: 0.6132634126750129

    Validation Mean Squared Error without PCA: 0.260566 | R^2 Score: 0.710055
    R^2 Score Y1: 0.7758236279667183, R^2 Score Y2: 0.6442855826779028

    Args:
        data (pd.DataFrame): The input data frame with dense features and targets.
        sparse_data (pd.DataFrame): The input data frame with sparse features.
        pca_dimensions (int): Number of PCA dimensions to reduce to. If 0, PCA is not applied.
    Returns:
        model: The trained Elastic Net model.
        pca: The PCA instance if applied, else None.
    """
    feature_cols = data.columns[0:15]  # A to N
    feature_cols_sparse = sparse_data.columns[0:2]  # O to P
    target_cols = ['Y1', 'Y2']
    
    X = data[feature_cols]
    X_sparse = sparse_data[feature_cols_sparse]
    y = data[target_cols]

    # Append sparse data features to X and y
    X = pd.concat([X, X_sparse], axis=1)
    
    # Use PCA if provided
    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None)

    model = ElasticNet(alpha=0.01, l1_ratio=0.015)
    model.fit(X_train, y_train)
    # print(X_val)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    r2_x1, r2_x2 = r2_score(y_val['Y1'], y_pred[:,0]), r2_score(y_val['Y2'], y_pred[:,1])
    if pca_dimensions > 0:
        print(f'Validation Mean Squared Error with PCA-{pca_dimensions}: {mse:.6f} | R^2 Score: {r2:.6f}')
        print(f'R^2 Score Y1: {r2_x1}, R^2 Score Y2: {r2_x2}')
    else:
        print(f'Validation Mean Squared Error without PCA: {mse:.6f} | R^2 Score: {r2:.6f}')
        print(f'R^2 Score Y1: {r2_x1}, R^2 Score Y2: {r2_x2}')
    
    return model, pca if pca_dimensions > 0 else None


def find_best_parameters_for_elastic_net(data, sparse_data):
    """
    Find the best alpha and l1_ratio parameters for Elastic Net using cross-validation.
    The function runs multiple iterations to average out randomness in train-test splits.
    It prints the average MSE and R^2 scores for each combination of parameters.
    Finally, it returns the best alpha and l1_ratio based on the highest average R^2 score for Y1.

    Example output:
    For Y1: Best alpha: 0.03593813663804626, Best l1_ratio: 0.1, 
        Best MSE: 0.25489049648824 R^2 Score: 0.6902946858692584 R^2 Score for Y1, Y2: 0.7628473330753816, 0.6177420386631366

    For Y2: Best alpha: 0.0031622776601683794, Best l1_ratio: 0.06, 
        Best MSE: 0.26046565555464046 R^2 Score: 0.7030421686088779 R^2 Score for Y1, Y2: 0.7822381674083239, 0.6238461698094308

    Args:
        data (pd.DataFrame): The input data frame with dense features and targets.
        sparse_data (pd.DataFrame): The input data frame with sparse features.
    Returns:
        best_avg_alpha (float): The best alpha parameter found.
        best_avg_l1_ratio (float): The best l1_ratio parameter found.
    """
    feature_cols = data.columns[0:15]  # A to N
    feature_cols_sparse = sparse_data.columns[0:2]  # O to P
    target_cols = ['Y1', 'Y2']
    
    X = data[feature_cols]
    X_sparse = sparse_data[feature_cols_sparse]
    y = data[target_cols]
    # Append sparse data features to X and y
    X = pd.concat([X, X_sparse], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    best_avg_mse = float('inf')
    best_avg_alpha = None
    best_avg_l1_ratio = None
    best_avg_r2 = -float('inf')
    best_avg_r2_for_y1, best_avg_r2_for_y2 = -float('inf'), -float('inf')

    for alpha in np.logspace(-3, -1, 10):
        for l1_ratio in np.linspace(0.01, 0.1, 10):
            mse_list = []
            r2_list = []
            r2_y1_list = []
            r2_y2_list = []
            for _ in range(20):  # Run multiple times to average out randomness
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=None)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_val)
                mse = mean_squared_error(y_val, y_pred)
                r2 = r2_score(y_val, y_pred)
                mse_list.append(mse)
                r2_list.append(r2)

                r2_y1, r2_y2 = r2_score(y_val['Y1'], y_pred[:,0]), r2_score(y_val['Y2'], y_pred[:,1])
                r2_y1_list.append(r2_y1)
                r2_y2_list.append(r2_y2)

            avg_mse = np.mean(mse_list)
            avg_r2 = np.mean(r2_list)
            avg_r2_for_y1, avg_r2_for_y2 = np.mean(r2_y1_list), np.mean(r2_y2_list)
            if best_avg_r2_for_y1 < avg_r2_for_y1:
                best_avg_mse = avg_mse
                best_avg_alpha = alpha
                best_avg_l1_ratio = l1_ratio
                best_avg_r2 = avg_r2
                best_avg_r2_for_y1, best_avg_r2_for_y2 = avg_r2_for_y1, avg_r2_for_y2

            print(f'Alpha: {round(alpha, 3)}, L1 Ratio: {round(l1_ratio, 3)}, Avg MSE: {round(avg_mse, 3)}, \
                  Avg R^2: {round(avg_r2, 3)}, Avg R^2 for Y1, Y2: {round(avg_r2_for_y1, 3)}, {round(avg_r2_for_y2, 3)}')

    print(f'Best alpha: {best_avg_alpha}, Best l1_ratio: {best_avg_l1_ratio}, Best MSE: {best_avg_mse}', f'R^2 Score: {best_avg_r2}', f'R^2 Score for Y1, Y2: {best_avg_r2_for_y1}, {best_avg_r2_for_y2}')
    return best_avg_alpha, best_avg_l1_ratio

def model_predict(model, pca, test_data, sparse_test_data):
    """Make predictions on the test data using the trained model and PCA (if provided).
    
    Args:
        model: The trained regression model.
        pca: The PCA instance (if used), else None.
        test_data (pd.DataFrame): The preprocessed test data.
        sparse_test_data (pd.DataFrame): The preprocessed sparse test data.
    Returns:
        predictions (np.ndarray): The predicted values for Y1 and Y2.
    """
    # id,time,A,B,C,D,E,F,G,H,I,J,K,L,M,N
    # Drop the first column (id)
    test_data = test_data.iloc[:, 1:]

    # Concatenate dense and sparse features
    X = pd.concat([test_data, sparse_test_data], axis=1)

    if pca is not None:
        # Apply PCA transformation
        X = pca.transform(X)  # Use transform, NOT fit_transform
    else:
        print("No PCA applied during prediction.")
    
    # print(X)
    predictions = model.predict(X)
    return predictions


if __name__ == "__main__":
    PCA_DIMENSIONS = 0
    MODEL = train_model_elastic_net
    IMPUTER = KNNImputer(n_neighbors=5)

    print("Loading and preprocessing training data...")
    train_data = load_data("train.csv")
    train_data = preprocess_data(train_data)

    print("Loading and preprocessing sparse columns data...")
    sparse_data = load_data("train_new.csv")
    sparse_data = preprocess_sparse_data(sparse_data, IMPUTER)

    # print("Finding best parameters for Elastic Net...")
    # best_alpha, best_l1_ratio = find_best_parameters_for_elastic_net(train_data, sparse_data)
    # print(f"Best parameters found: alpha={best_alpha}, l1_ratio={best_l1_ratio}")

    print("Training model...")
    model, pca = MODEL(train_data, sparse_data, PCA_DIMENSIONS)

    print("Loading and preprocessing test data...")
    test_data = load_data("test.csv")
    test_data = preprocess_data(test_data, start_index=2, end_index=16)  # skip id

    test_sparse_data = load_data("test_new.csv")
    test_sparse_data = preprocess_sparse_data(test_sparse_data, IMPUTER)

    print("Making predictions...")
    predictions = model_predict(model, pca, test_data, test_sparse_data)

    # Save predictions to CSV
    output_df = pd.DataFrame(predictions, columns=['Y1', 'Y2'])
    output_df.index += 1  # Make index 1-based as per 'id' requirement
    output_df.index.name = 'id'
    output_df.to_csv("predictions.csv")

    print("Predictions saved to predictions.csv")

    # Compare R^2 score with a friend's model
    friend_predictions = pd.read_csv("friend_predictions.csv")
    friend_y = friend_predictions[['Y1', 'Y2']].values
    r2 = r2_score(friend_y, predictions)
    print(f"R^2 Score compared to friend's model: {r2}")
