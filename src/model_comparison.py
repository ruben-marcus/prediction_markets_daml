from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from data_merge import BASELINE_COL, SUPPORTED_TARGET_HORIZONS, TARGET, build_model_df
from regression import (
    baseline_predictions as regression_baselines,
    prepare_data as prepare_linear_data,
    train_test_split_time_ordered,
)
from ridge_model import ALPHAS
from xgb_model import (
    baseline_predictions as xgb_baselines,
    prepare_data as prepare_xgb_data,
    train_test_split_by_date,
    train_validation_split_by_date,
)


def metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "direction_accuracy": float((np.sign(y_true) == np.sign(y_pred)).mean()),
    }


def add_metrics(
    row: dict[str, object],
    prefix: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> None:
    for metric_name, value in metrics(y_true, y_pred).items():
        row[f"{prefix}_{metric_name}"] = value


def clean_baseline_name(label: str) -> str:
    return (
        label.lower()
        .removeprefix("baseline: ")
        .replace("1d", "one_day")
        .replace(" ", "_")
    )


def run_ols(horizon: int) -> dict[str, object]:
    model_df = build_model_df(horizon)
    X, y, meta = prepare_linear_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_time_ordered(
        X, y, meta
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_pred = model.predict(X_test) + meta_test[BASELINE_COL].to_numpy()

    row: dict[str, object] = {
        "model": "ols",
        "horizon_days": horizon,
        "rows": len(X),
        "train_rows": len(X_train),
        "validation_rows": 0,
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "selected_alpha": np.nan,
        "best_iteration": np.nan,
    }
    add_metrics(row, "model_test", meta_test[TARGET], test_pred)

    for label, pred in regression_baselines(meta_train, meta_test).items():
        add_metrics(row, clean_baseline_name(label), meta_test[TARGET], pred)

    return row


def run_ridge(horizon: int) -> dict[str, object]:
    model_df = build_model_df(horizon)
    X, y, meta = prepare_linear_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_time_ordered(
        X, y, meta
    )

    pipeline = make_pipeline(
        StandardScaler(),
        RidgeCV(alphas=ALPHAS),
    )
    pipeline.fit(X_train, y_train)
    ridge = pipeline.named_steps["ridgecv"]
    test_pred = pipeline.predict(X_test) + meta_test[BASELINE_COL].to_numpy()

    row: dict[str, object] = {
        "model": "ridgecv",
        "horizon_days": horizon,
        "rows": len(X),
        "train_rows": len(X_train),
        "validation_rows": 0,
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "selected_alpha": float(ridge.alpha_),
        "best_iteration": np.nan,
    }
    add_metrics(row, "model_test", meta_test[TARGET], test_pred)

    for label, pred in regression_baselines(meta_train, meta_test).items():
        add_metrics(row, clean_baseline_name(label), meta_test[TARGET], pred)

    return row


def run_xgb(horizon: int) -> dict[str, object]:
    model_df = build_model_df(horizon)
    X, y, meta = prepare_xgb_data(model_df)
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split_by_date(
        X, y, meta
    )
    X_fit, X_valid, y_fit, y_valid, _, _ = train_validation_split_by_date(
        X_train, y_train, meta_train
    )

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
    test_pred = model.predict(X_test) + meta_test[BASELINE_COL].to_numpy()

    row: dict[str, object] = {
        "model": "xgb",
        "horizon_days": horizon,
        "rows": len(X),
        "train_rows": len(X_fit),
        "validation_rows": len(X_valid),
        "test_rows": len(X_test),
        "feature_count": X.shape[1],
        "selected_alpha": np.nan,
        "best_iteration": int(model.best_iteration + 1),
    }
    add_metrics(row, "model_test", meta_test[TARGET], test_pred)

    for label, pred in xgb_baselines(meta_train, meta_test).items():
        add_metrics(row, clean_baseline_name(label), meta_test[TARGET], pred)

    return row


def main() -> None:
    rows = []
    runners = (run_ols, run_ridge, run_xgb)

    for horizon in SUPPORTED_TARGET_HORIZONS:
        for runner in runners:
            print(f"Running {runner.__name__.removeprefix('run_')} at {horizon}d...")
            rows.append(runner(horizon))

    comparison = pd.DataFrame(rows).sort_values(["horizon_days", "model"])

    output_path = Path(__file__).resolve().parent.parent / \
        "models" / "model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output_path, index=False)

    display_cols = [
        "model",
        "horizon_days",
        "model_test_rmse",
        "model_test_mae",
        "model_test_r2",
        "zero_change_rmse",
        "market_expanding_mean_rmse",
    ]
    print("\nModel comparison")
    print("-" * 16)
    print(comparison[display_cols].to_string(index=False))
    print(f"\nSaved comparison to {output_path.name}")


if __name__ == "__main__":
    main()
