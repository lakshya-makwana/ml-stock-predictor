from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False


FEATURE_COLUMNS = [
    'Lag_1',
    'Lag_2',
    'Lag_5',
    'MA_10',
    'MA_20',
    'MA_50',
    'Volatility',
    'RSI',
    'MACD',
    'Volume_Change'
]


def prepare_data(df):

    X = df[FEATURE_COLUMNS]
    y = df['Target']

    split_index = int(len(df) * 0.8)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    print("\nModel Performance")
    print("-" * 30)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions))
    print("Recall:", recall_score(y_test, predictions))
    print("F1 Score:", f1_score(y_test, predictions))

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, predictions))

    return predictions


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def train_logistic_regression(X_train, y_train):

    model = LogisticRegression()

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):

    if not XGB_AVAILABLE:
        print("XGBoost not installed.")
        return None

    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def save_model(model, filename="saved_model.pkl"):

    joblib.dump(model, filename)

    print(f"Model saved as {filename}")