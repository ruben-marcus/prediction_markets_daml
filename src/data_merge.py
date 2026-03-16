from polymarket_data_fetch import polymarket_df
from macro_data_fetch import macro_df

MACRO_COLS = ["vix", "gold", "silver", "vix_ret", "gold_ret", "silver_ret"]

panel_df = polymarket_df.merge(macro_df, on="date", how="left")
panel_df = panel_df.sort_values(["market", "date"]).reset_index(drop=True)

# for dates with no macro data (eg weekends)
panel_df[MACRO_COLS] = panel_df.groupby("market")[MACRO_COLS].ffill()

# one-day-ahead target
panel_df["target"] = panel_df.groupby("market")["prob"].shift(-1) - panel_df["prob"]

# probability lags
panel_df["prob_lag1"] = panel_df.groupby("market")["prob"].shift(1)
panel_df["prob_lag2"] = panel_df.groupby("market")["prob"].shift(2)
panel_df["prob_change"] = panel_df.groupby("market")["prob"].diff()
panel_df["prob_change_lag1"] = panel_df.groupby("market")["prob_change"].shift(1)
panel_df["prob_change_lag2"] = panel_df.groupby("market")["prob_change"].shift(2)

# macro lags
panel_df["vix_lag1"] = panel_df.groupby("market")["vix"].shift(1)
panel_df["gold_lag1"] = panel_df.groupby("market")["gold"].shift(1)
panel_df["silver_lag1"] = panel_df.groupby("market")["silver"].shift(1)

panel_df["vix_ret_lag1"] = panel_df.groupby("market")["vix_ret"].shift(1)
panel_df["gold_ret_lag1"] = panel_df.groupby("market")["gold_ret"].shift(1)
panel_df["silver_ret_lag1"] = panel_df.groupby("market")["silver_ret"].shift(1)

model_df = panel_df.dropna().reset_index(drop=True)