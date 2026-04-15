from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_merge import model_df

# same core features as regression.py
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
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "market"]).reset_index(drop=True)

    required_cols = FEATURES + [TARGET, "market", "date", "prob"]
    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    X = data[FEATURES].copy()

    market_dummies = pd.get_dummies(
        data["market"],
        prefix="market",
        drop_first=True,
        dtype=float,
    )
    X = pd.concat([X, market_dummies], axis=1)

    y = data[TARGET].astype(float)

    return X.astype(float), y, data


def train_test_split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """chronological split by date to reduce look-ahead bias"""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    unique_dates = np.array(sorted(meta["date"].unique()))
    split_idx = int(len(unique_dates) * (1 - test_size))

    if split_idx <= 0 or split_idx >= len(unique_dates):
        raise ValueError(
            "not enough unique dates for the requested train/test split")

    split_date = unique_dates[split_idx]

    train_mask = meta["date"] < split_date
    test_mask = meta["date"] >= split_date

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        raise ValueError("train/test split produced an empty partition")

    X_train = X.loc[train_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_train = y.loc[train_mask].copy()
    y_test = y.loc[test_mask].copy()
    meta_train = meta.loc[train_mask].copy()
    meta_test = meta.loc[test_mask].copy()

    return X_train, X_test, y_train, y_test, meta_train, meta_test


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """fraction of times the model gets the sign of the move correct"""
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def print_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction = directional_accuracy(y_true, y_pred)

    print(f"\n{label} metries")
    print("-" * (len(label) + 8))
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2: {r2:.6f}")
    print(f"direction accuracy: {direction:.4f}")


def print_features_importance(model: XGBRegressor, columns: pd.Index, top_n: int = 20) -> None:
    importance = pd.DataFrame({
        "feature": columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\ntop {top_n} features by importance")
    print("-" * 35)
    print(importance.head(top_n).to_string(index=False))


def main() -> None:
    X, y, meta = prepare_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_by_date(
        X, y, meta)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    print(f"rows used: {len(X):,}")
    print(f"train rows: {len(X_train):,}")
    print(f"test rows: {len(X_test):,}")
    print(f"feature count: {X.shape[1]}")

    print_metrics(y_train, train_pred, "train")
    print_metrics(y_test, test_pred, "test")
    print_features_importance(model, X.columns)

    predictions = meta_test[["date", "market", "prob", TARGET]].copy()
    predictions["prediction"] = test_pred
    predictions["predicted_next_prob"] = (
        predictions["prob"] + predictions["prediction"]).clip(0, 1)
    predictions["actual_next_prob"] = (
        predictions["prob"] + predictions[TARGET]).clip(0, 1)
    predictions["residual"] = predictions[TARGET] - predictions["prediction"]

    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "xgb_predictions.csv"
    predictions.to_csv(output_path, index=False)

    print(f"\nsaved out-of-sample predictions to {output_path.name}")


if __name__ == "__main__":
    main()
