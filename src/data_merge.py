import pandas as pd

from polymarket_data_fetch import polymarket_df
from macro_data_fetch import macro_df

TARGET_HORIZON_DAYS = 1
SUPPORTED_TARGET_HORIZONS = (1, 3, 7)
TARGET = "target"
RESIDUAL_TARGET = "residual_target"
BASELINE_COL = "market_expanding_mean"
MACRO_COLS = [col for col in macro_df.columns if col != "date"]
CORE_MACRO_FEATURES = [
    "vix",
    "vix_ret_lag1",
    "sp500_ret_lag1",
    "ten_year_yield_ret_lag1",
    "dollar_index_ret_lag1",
    "oil_ret_lag1",
    "gold_ret_lag1",
    "bitcoin_ret_lag1",
]
PROB_FEATURES = [
    "prob_lag1",
    "prob_lag2",
    "prob_lag3",
    "prob_change",
    "prob_change_lag1",
    "prob_change_lag2",
    "prob_change_lag3",
    "prob_change_3d_mean",
    "prob_change_7d_mean",
    "prob_change_7d_std",
]
MARKET_SPECIFIC_FEATURES = [
    BASELINE_COL,
    "days_to_resolution",
    "market_age_days",
    "prob_uncertainty",
    "abs_prob_change_7d_mean",
]

panel_df = polymarket_df.merge(macro_df, on="date", how="left")
panel_df = panel_df.sort_values(["market", "date"]).reset_index(drop=True)

# for dates with no macro data (eg weekends)
panel_df[MACRO_COLS] = panel_df.groupby("market")[MACRO_COLS].ffill()

market_first_date = panel_df.groupby("market")["date"].transform("min")
market_last_date = panel_df.groupby("market")["date"].transform("max")
panel_df["resolution_date"] = panel_df["resolution_date"].fillna(market_last_date)
panel_df["days_to_resolution"] = (
    panel_df["resolution_date"] - panel_df["date"]
).dt.days.clip(lower=0)
panel_df["market_age_days"] = (panel_df["date"] - market_first_date).dt.days
panel_df["prob_uncertainty"] = panel_df["prob"] * (1 - panel_df["prob"])

# probability lags
panel_df["prob_lag1"] = panel_df.groupby("market")["prob"].shift(1)
panel_df["prob_lag2"] = panel_df.groupby("market")["prob"].shift(2)
panel_df["prob_lag3"] = panel_df.groupby("market")["prob"].shift(3)
panel_df["prob_change"] = panel_df.groupby("market")["prob"].diff()
panel_df["prob_change_lag1"] = panel_df.groupby("market")["prob_change"].shift(1)
panel_df["prob_change_lag2"] = panel_df.groupby("market")["prob_change"].shift(2)
panel_df["prob_change_lag3"] = panel_df.groupby("market")["prob_change"].shift(3)
panel_df["prob_change_3d_mean"] = (
    panel_df.groupby("market")["prob_change"]
    .transform(lambda s: s.shift(1).rolling(3).mean())
)
panel_df["prob_change_7d_mean"] = (
    panel_df.groupby("market")["prob_change"]
    .transform(lambda s: s.shift(1).rolling(7).mean())
)
panel_df["prob_change_7d_std"] = (
    panel_df.groupby("market")["prob_change"]
    .transform(lambda s: s.shift(1).rolling(7).std())
)
panel_df["abs_prob_change_7d_mean"] = (
    panel_df.groupby("market")["prob_change"]
    .transform(lambda s: s.abs().shift(1).rolling(7).mean())
)

# macro lags
MACRO_LAG_FEATURES = []
for col in MACRO_COLS:
    lag_col = f"{col}_lag1"
    panel_df[lag_col] = panel_df.groupby("market")[col].shift(1)
    MACRO_LAG_FEATURES.append(lag_col)

AVAILABLE_CORE_MACRO_FEATURES = [
    col for col in CORE_MACRO_FEATURES if col in panel_df.columns
]
FEATURES = PROB_FEATURES + MARKET_SPECIFIC_FEATURES + AVAILABLE_CORE_MACRO_FEATURES


def build_model_df(target_horizon_days: int = TARGET_HORIZON_DAYS) -> pd.DataFrame:
    if target_horizon_days not in SUPPORTED_TARGET_HORIZONS:
        raise ValueError(
            f"target_horizon_days must be one of {SUPPORTED_TARGET_HORIZONS}"
        )

    data = panel_df.copy()
    data["target_horizon_days"] = target_horizon_days
    data[TARGET] = (
        data.groupby("market")["prob"].shift(-target_horizon_days)
        - data["prob"]
    )
    data[BASELINE_COL] = (
        data.groupby("market")[TARGET]
        .transform(lambda s: s.shift(target_horizon_days).expanding().mean())
        .fillna(0.0)
    )
    data[RESIDUAL_TARGET] = data[TARGET] - data[BASELINE_COL]

    return data.dropna().reset_index(drop=True)


model_df = build_model_df()
