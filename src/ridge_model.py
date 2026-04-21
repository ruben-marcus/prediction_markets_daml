from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data_merge import (
    BASELINE_COL,
    FEATURES,
    RESIDUAL_TARGET,
    SUPPORTED_TARGET_HORIZONS,
    TARGET,
    build_model_df,
)
ALPHAS = np.logspace(-4, 4, 25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit RidgeCV regression for a selected prediction horizon."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=SUPPORTED_TARGET_HORIZONS,
        default=1,
        help="Target horizon in days.",
    )
    return parser.parse_args()


def prepare_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """build model matrix with market fixed effects via one-hot encoding"""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "market"]).reset_index(drop=True)

    missing = [col for col in FEATURES +
               [TARGET, RESIDUAL_TARGET, "market", "date"] if col not in data.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    X = data[FEATURES].copy()
    market_dummies = pd.get_dummies(
        data["market"], prefix="market", drop_first=True, dtype=float)
    X = pd.concat([X, market_dummies], axis=1)
    y = data[RESIDUAL_TARGET].astype(float)

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


def baseline_predictions(
    meta_train: pd.DataFrame,
    meta_eval: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "Baseline: zero change": np.zeros(len(meta_eval)),
        "Baseline: previous 1d change": meta_eval["prob_change"].to_numpy(),
        "Baseline: market expanding mean": meta_eval[BASELINE_COL].to_numpy(),
    }


def print_baselines(
    y_true: pd.Series,
    baselines: dict[str, np.ndarray],
) -> None:
    for label, pred in baselines.items():
        print_metrics(y_true, pred, label)


def print_coefficients(
    model: RidgeCV,
    columns: pd.Index,
    top_n: int = 25,
) -> None:
    coef_table = pd.DataFrame({
        "feature": columns,
        "coefficient": model.coef_,
        "abs_coefficient": np.abs(model.coef_),
    }).sort_values("abs_coefficient", ascending=False)

    print(f"\nTop {top_n} coefficients by absolute size")
    print("-" * 41)
    print(coef_table[["feature", "coefficient"]].head(top_n).to_string(index=False))
    print(f"\nIntercept: {model.intercept_:.6f}")
    print(f"Selected alpha: {model.alpha_:.6f}")


def main() -> None:
    args = parse_args()
    model_df = build_model_df(args.horizon)
    X, y, meta = prepare_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_time_ordered(
        X, y, meta)

    pipeline = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS),
    )
    pipeline.fit(X_train, y_train)

    ridge = pipeline.named_steps["ridgecv"]
    train_pred_residual = pipeline.predict(X_train)
    test_pred_residual = pipeline.predict(X_test)
    train_pred = train_pred_residual + meta_train[BASELINE_COL].to_numpy()
    test_pred = test_pred_residual + meta_test[BASELINE_COL].to_numpy()
    test_baselines = baseline_predictions(meta_train, meta_test)

    print(f"Rows used: {len(X):,}")
    print(f"Train rows: {len(X_train):,}")
    print(f"Test rows: {len(X_test):,}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Target horizon: {args.horizon} day(s)")

    print_metrics(meta_train[TARGET], train_pred, "Train")
    print_metrics(meta_test[TARGET], test_pred, "Test")
    print_baselines(meta_test[TARGET], test_baselines)
    print_coefficients(ridge, X.columns)

    predictions = meta_test[
        ["date", "market", "prob", "target_horizon_days", TARGET, BASELINE_COL]
    ].copy()
    predictions["prediction"] = test_pred
    predictions["residual_prediction"] = test_pred_residual
    for label, baseline_pred in test_baselines.items():
        col = label.removeprefix("Baseline: ").replace(" ", "_")
        predictions[col] = baseline_pred
    predictions["residual"] = predictions[TARGET] - predictions["prediction"]

    output_path = Path(__file__).resolve().parent.parent / \
        "models" / f"ridge_predictions_{args.horizon}d.csv"
    predictions.to_csv(output_path, index=False)
    print(f"\nSaved out-of-sample predictions to {output_path.name}")


if __name__ == "__main__":
    main()
