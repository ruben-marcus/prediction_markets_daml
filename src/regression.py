from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pathlib import Path

from data_merge import model_df

# freatures from data_merge.py
FEATURES = [
    "prob_lag1",
    "prob_lag2",
    "prob_change",
    "prob_change_lag1",
    "prob_change_lag2",
    "vix",
    "gold",
    "silver",
    "vix_ret",
    "gold_ret",
    "silver_ret",
    "vix_lag1",
    "gold_lag1",
    "silver_lag1",
    "vix_ret_lag1",
    "gold_ret_lag1",
    "silver_ret_lag1",
]
TARGET = "target"


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """build model matrix with market fixed effects via one-hot encoding"""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "market"]).reset_index(drop=True)

    missing = [col for col in FEATURES + [TARGET, "market", "date"] if col not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = data[FEATURES].copy()
    market_dummies = pd.get_dummies(data["market"], prefix="market", drop_first=True, dtype=float)
    X = pd.concat([X, market_dummies], axis=1)
    y = data[TARGET].astype(float)

    return X.astype(float), y, data


def train_test_split_time_ordered(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """chronological split to reduce look-ahead bias"""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("not enough rows for the requested train/test split")

    X_train = X.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_train = y.iloc[:split_idx].copy()
    y_test = y.iloc[split_idx:].copy()
    meta_train = meta.iloc[:split_idx].copy()
    meta_test = meta.iloc[split_idx:].copy()

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def print_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n{label} metrics")
    print("-" * (len(label) + 8))
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R^2:  {r2:.6f}")


def print_coefficients(model: LinearRegression, columns: pd.Index) -> None:
    coef_table = pd.DataFrame({
        "feature": columns,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    }).sort_values("abs_coefficient", ascending=False)

    print("\nTop coefficients by absolute size")
    print("-" * 33)
    print(coef_table[["feature", "coefficient"]].to_string(index=False))
    print(f"\nIntercept: {model.intercept_:.6f}")


def main() -> None:
    X, y, meta = prepare_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_time_ordered(X, y, meta)

    model = LinearRegression()
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"Rows used: {len(X):,}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Feature count: {X.shape[1]}")

    print_metrics(y_train, train_pred, "Train")
    print_metrics(y_test, test_pred, "Test")
    print_coefficients(model, X.columns)

    predictions = meta_test[["date", "market", "prob", TARGET]].copy()
    predictions["prediction"] = test_pred
    predictions["residual"] = predictions[TARGET] - predictions["prediction"]
    
    output_path = Path(__file__).resolve().parent.parent / "models" / "regression_predictions.csv"
    predictions.to_csv(output_path, index=False)
    print("\nSaved out-of-sample predictions to regression_predictions.csv")


if __name__ == "__main__":
    main()
