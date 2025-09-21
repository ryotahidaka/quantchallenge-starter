# Columns: 17 columns (time, A-N, Y1, Y2)
# Rows: 80,000 rows of data
# Features: Time column and columns A through N represent real market features that have been anonymized for the purposes of this competition
# Targets: Y1 and Y2 are the target variables you need to predict

import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.impute import KNNImputer
import joblib
import xgboost as xgb

from data_preparation import load_data, preprocess_data, preprocess_sparse_data

# Possible Models to try:
# 1. Ridge Regression
# 2. Lasso Regression
# 3. Elastic Net
# 4. Support Vector Machine (SVM)
# 5. Random Forest Regressor
# 6. XGBoost

def train_model_ridge_regression(data, pca_dimensions=0):
    """Train a Ridge Regression model to predict Y1 and Y2.
    Validation Mean Squared Error without PCA: 0.256433 | R^2 Score: 0.714656
    R^2 Score compared to friend's model: 0.3582698073248509
    """
    feature_cols = data.columns[0:15]  # time and A to N
    print(f"Feature columns: {feature_cols}")
    target_cols = ['Y1', 'Y2']
    
    X = data[feature_cols]
    y = data[target_cols]

    # Use PCA if provided
    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    if pca_dimensions > 0:
        print(f'Validation Mean Squared Error with PCA-{pca_dimensions}: {mse:.6f} | R^2 Score: {r2:.6f}')
    else:
        print(f'Validation Mean Squared Error without PCA: {mse:.6f} | R^2 Score: {r2:.6f}')

    return model, pca if pca_dimensions > 0 else pca_dimensions

def train_model_lasso_regression(data, pca_dimensions=0):
    """Train a Lasso Regression model to predict Y1 and Y2.
    Validation Mean Squared Error without PCA: 0.278431 | R^2 Score: 0.690119
    R^2 Score compared to friend's model: 0.7382371391881472
    """    
    feature_cols = data.columns[0:15]  # time and A to N
    print(f"Feature columns: {feature_cols}")
    target_cols = ['Y1', 'Y2']
    
    X = data[feature_cols]
    y = data[target_cols]

    # Fit PCA
    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Lasso(alpha=0.1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    if pca_dimensions > 0:
        print(f'Validation Mean Squared Error with PCA-{pca_dimensions}: {mse:.6f} | R^2 Score: {r2:.6f}')
    else:
        print(f'Validation Mean Squared Error without PCA: {mse:.6f} | R^2 Score: {r2:.6f}')

    return model, pca if pca_dimensions > 0 else pca_dimensions

def train_model_svm(data, pca_dimensions=0):
    """Train a Support Vector Machine (SVM) model to predict Y1 and Y2 separately.
    Validation Mean Squared Error with SVM: 
    """

    feature_cols = data.columns[0:15]  # time and A to N
    print(f"Feature columns: {feature_cols}")
    target_cols = ['Y1', 'Y2']

    X = data[feature_cols]
    y = data[target_cols]

    # Fit PCA
    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model_y1 = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    model_y2 = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)

    model_y1.fit(X_train, y_train['Y1'])
    model_y2.fit(X_train, y_train['Y2'])

    y1_pred = model_y1.predict(X_val)
    y2_pred = model_y2.predict(X_val)

    y_pred = np.column_stack((y1_pred, y2_pred))

    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f'Validation Mean Squared Error with SVM: {mse} | R^2 Score: {r2}')
    model = (model_y1, model_y2)

    return model, pca if pca_dimensions > 0 else pca_dimensions

def train_model_xgboost(data, pca_dimensions=0):
    """Train an XGBoost model to predict Y1 and Y2.
    Validation Mean Squared Error with XGBoost: 0.19615958377518639 R^2 Score with XGBoost: 0.7824620499143545
    R^2 Score compared to friend's model: -7.183067664002874
    """
    feature_cols = data.columns[0:15]  # time and A to N
    print(f"Feature columns: {feature_cols}")
    target_cols = ['Y1', 'Y2']
    
    X = data[feature_cols]
    y = data[target_cols]

    # Fit PCA
    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    print(f'Validation Mean Squared Error with XGBoost: {mse}', f"R^2 Score with XGBoost: {r2_score(y_val, y_pred)}")

    return model, pca if pca_dimensions > 0 else pca_dimensions

def train_model_random_forest(data, pca_dimensions=0):
    """Train a model using PCA for dimensionality reduction.
    Validation Mean Squared Error with PCA-0: 0.231279 | R^2 Score: 0.742925
    R^2 Score compared to friend's model: -1.2322581150289484
    """
    feature_cols = data.columns[0:15]  # time and A to N
    print(f"Feature columns: {feature_cols}")
    target_cols = ['Y1', 'Y2']

    X = data[feature_cols]
    y = data[target_cols]

    # Fit PCA
    if pca_dimensions > 0:
        pca = PCA(n_components=pca_dimensions)
        X = pca.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    print(f'Validation Mean Squared Error with PCA-{pca_dimensions}: {mse:.6f} | R^2 Score: {r2:.6f}')

    return model, pca if pca_dimensions > 0 else pca_dimensions


def train_model_elastic_net(data, sparse_data, pca_dimensions=0):
    """Train an Elastic Net model to predict Y1 and Y2.
    Validation Mean Squared Error without PCA: 0.264553 | R^2 Score: 0.705588
    R^2 Score compared to friend's model: 0.6132634126750129

    Validation Mean Squared Error without PCA: 0.260566 | R^2 Score: 0.710055
    R^2 Score Y1: 0.7758236279667183, R^2 Score Y2: 0.6442855826779028
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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ElasticNet(alpha=0.01, l1_ratio=0.015)
    model.fit(X_train, y_train)
    print(X_val)
    
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
    best_avg_r2_for_y1, best_avg_r2_for_y2 = None, None

    for alpha in np.logspace(-2, -1, 5):
        for l1_ratio in np.linspace(0.005, 0.015, 5):
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
            if best_avg_r2 < avg_r2:
                best_avg_mse = avg_mse
                best_avg_alpha = alpha
                best_avg_l1_ratio = l1_ratio
                best_avg_r2 = avg_r2
                best_avg_r2_for_y1, best_avg_r2_for_y2 = avg_r2_for_y1, avg_r2_for_y2

            print(f'Alpha: {alpha}, L1 Ratio: {l1_ratio}, Avg MSE: {avg_mse}, Avg R^2: {avg_r2}, Avg R^2 for Y1, Y2: {avg_r2_for_y1}, {avg_r2_for_y2}')

    print(f'Best alpha: {best_avg_alpha}, Best l1_ratio: {best_avg_l1_ratio}, Best MSE: {best_avg_mse}', f'R^2 Score: {best_avg_r2}', f'R^2 Score for Y1, Y2: {best_avg_r2_for_y1}, {best_avg_r2_for_y2}')
    return best_avg_alpha, best_avg_l1_ratio

def pca_predict(model, pca, test_data, sparse_test_data):
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
    
    print(X)
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
    predictions = pca_predict(model, pca, test_data, test_sparse_data)

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
