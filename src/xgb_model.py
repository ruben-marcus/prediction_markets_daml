from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_merge import (
    BASELINE_COL,
    FEATURES,
    RESIDUAL_TARGET,
    SUPPORTED_TARGET_HORIZONS,
    TARGET,
    build_model_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit XGBoost for a selected prediction horizon."
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
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "market"]).reset_index(drop=True)

    required_cols = FEATURES + [TARGET, RESIDUAL_TARGET, "market", "date", "prob"]
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

    y = data[RESIDUAL_TARGET].astype(float)

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


def train_validation_split_by_date(
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
    validation_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """hold out the tail of the training window for early stopping"""
    return train_test_split_by_date(X, y, meta, test_size=validation_size)


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    """fraction of times the model gets the sign of the move correct"""
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def print_metrics(y_true: pd.Series, y_pred: np.ndarray, label: str) -> None:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    direction = directional_accuracy(y_true, y_pred)

    print(f"\n{label} metrics")
    print("-" * (len(label) + 8))
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R^2: {r2:.6f}")
    print(f"direction accuracy: {direction:.4f}")


def baseline_predictions(
    meta_train: pd.DataFrame,
    meta_eval: pd.DataFrame,
) -> dict[str, np.ndarray]:
    return {
        "baseline: zero change": np.zeros(len(meta_eval)),
        "baseline: previous 1d change": meta_eval["prob_change"].to_numpy(),
        "baseline: market expanding mean": meta_eval[BASELINE_COL].to_numpy(),
    }


def print_baselines(
    y_true: pd.Series,
    baselines: dict[str, np.ndarray],
) -> None:
    for label, pred in baselines.items():
        print_metrics(y_true, pred, label)


def print_features_importance(model: XGBRegressor, columns: pd.Index, top_n: int = 20) -> None:
    importance = pd.DataFrame({
        "feature": columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print(f"\ntop {top_n} features by importance")
    print("-" * 35)
    print(importance.head(top_n).to_string(index=False))


def main() -> None:
    args = parse_args()
    model_df = build_model_df(args.horizon)
    X, y, meta = prepare_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_by_date(
        X, y, meta)
    X_fit, X_valid, y_fit, y_valid, meta_fit, meta_valid = train_validation_split_by_date(
        X_train, y_train, meta_train)

    model = XGBRegressor(
        n_estimators=2000,
        max_depth=2,
        learning_rate=0.02,
        min_child_weight=8,
        gamma=0.001,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.05,
        reg_lambda=5.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        early_stopping_rounds=50,
        random_state=42,
    )

    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    train_pred_residual = model.predict(X_fit)
    valid_pred_residual = model.predict(X_valid)
    test_pred_residual = model.predict(X_test)
    train_pred = train_pred_residual + meta_fit[BASELINE_COL].to_numpy()
    valid_pred = valid_pred_residual + meta_valid[BASELINE_COL].to_numpy()
    test_pred = test_pred_residual + meta_test[BASELINE_COL].to_numpy()

    print(f"rows used: {len(X):,}")
    print(f"fit rows: {len(X_fit):,}")
    print(f"validation rows: {len(X_valid):,}")
    print(f"test rows: {len(X_test):,}")
    print(f"feature count: {X.shape[1]}")
    print(f"target horizon: {args.horizon} day(s)")
    if hasattr(model, "best_iteration"):
        print(f"best boosting round: {model.best_iteration + 1}")

    print_metrics(meta_fit[TARGET], train_pred, "fit")
    print_metrics(meta_valid[TARGET], valid_pred, "validation")
    print_metrics(meta_test[TARGET], test_pred, "test")
    print_baselines(meta_test[TARGET], baseline_predictions(meta_train, meta_test))
    print_features_importance(model, X.columns)

    predictions = meta_test[
        ["date", "market", "prob", "target_horizon_days", TARGET, BASELINE_COL]
    ].copy()
    predictions["prediction"] = test_pred
    predictions["residual_prediction"] = test_pred_residual
    for label, baseline_pred in baseline_predictions(meta_train, meta_test).items():
        col = label.removeprefix("baseline: ").replace(" ", "_")
        predictions[col] = baseline_pred
    predictions["predicted_next_prob"] = (
        predictions["prob"] + predictions["prediction"]).clip(0, 1)
    predictions["actual_next_prob"] = (
        predictions["prob"] + predictions[TARGET]).clip(0, 1)
    predictions["residual"] = predictions[TARGET] - predictions["prediction"]

    output_dir = Path(__file__).resolve().parent.parent / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"xgb_predictions_{args.horizon}d.csv"
    predictions.to_csv(output_path, index=False)

    print(f"\nsaved out-of-sample predictions to {output_path.name}")


if __name__ == "__main__":
    main()
