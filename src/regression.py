from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data_merge import model_df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("date")
    df = pd.get_dummies(df, columns=["market"], drop_first=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    base_features = [
        "vix",
        "gold",
        "silver",
        "vix_ret",
        "gold_ret",
        "silver_ret",
        "prob_lag1",
        "prob_change_lag1",
    ]

    market_dummies = [col for col in df.columns if col.startswith("market_")]
    return base_features + market_dummies


def time_split(df: pd.DataFrame, split_ratio: float = 0.8):
    split_idx = int(len(df) * split_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> tuple[float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_pred, y_true)[0, 1]
    return rmse, corr


def main():
    df = prepare_data(model_df)
    feature_cols = get_feature_columns(df)

    train, test = time_split(df, split_ratio=0.8)

    X_train = train[feature_cols]
    y_train = train["target"]
    X_test = test[feature_cols]
    y_test = test["target"]

    # Baseline linear regression
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse, corr = evaluate_regression(y_test, preds)

    print(f"Train rows: {len(train)}")
    print(f"Test rows: {len(test)}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"Prediction correlation: {corr:.6f}")

    raw_coef = pd.Series(model.coef_, index=feature_cols).sort_values()
    print("\nRaw coefficients:")
    print(raw_coef)

    # Standardized regression for comparable feature importance
    scaled_model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LinearRegression())
    ])
    scaled_model.fit(X_train, y_train)

    scaled_coef = pd.Series(
        scaled_model.named_steps["lr"].coef_,
        index=feature_cols
    ).sort_values()

    print("\nStandardized coefficients:")
    print(scaled_coef)

    # Scatter plot: actual vs predicted
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, preds, alpha=0.7)
    plt.xlabel("Actual Change")
    plt.ylabel("Predicted Change")
    plt.title("Prediction Market Forecast: Actual vs Predicted")
    plt.tight_layout()
    plt.show()

    # Bar chart: standardized coefficients
    plt.figure(figsize=(10, 6))
    scaled_coef.plot(kind="barh")
    plt.xlabel("Standardized Coefficient")
    plt.title("Feature Importance (Standardized Linear Regression)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()