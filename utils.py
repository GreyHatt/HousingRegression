import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split as sklearn_train_test_split,
    GridSearchCV
)
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def load_data():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def train_test_split(df, test_size=0.2, random_state=13):
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    return sklearn_train_test_split(X, y, test_size=test_size, random_state=random_state)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2

def grid_search_cv(model, param_grid, X_train, y_train, scoring='neg_mean_squared_error', cv=5):
    grid_search = GridSearchCV(model, param_grid, scoring=scoring, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def comparison_graph(df):
    df = pd.DataFrame(df)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.bar(df['Model'], df['MSE'], color='skyblue')
    plt.title('MSE (lower is better)')
    plt.ylabel('MSE')
    plt.subplot(1,2,2)
    plt.bar(df['Model'], df['R2'], color='lightgreen')
    plt.title('R2 Score (higher is better)')
    plt.ylabel('R2')
    plt.tight_layout()
    plt.savefig("/app/output/comparison_plot.png")