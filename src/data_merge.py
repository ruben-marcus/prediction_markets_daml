from polymarket_data_fetch import polymarket_df
from macro_data_fetch import macro_df

panel_df = polymarket_df.merge(macro_df, on="date", how="left")
panel_df = panel_df.sort_values(["market", "date"])

panel_df[["vix", "gold", "silver", "vix_ret", "gold_ret", "silver_ret"]] = (
    panel_df.groupby("market")[
        ["vix", "gold", "silver", "vix_ret", "gold_ret", "silver_ret"]].ffill()
)

# print(panel_df.head(10))

panel_df["target"] = panel_df.groupby(
    "market")["prob"].shift(-1) - panel_df["prob"]

panel_df["prob_lag1"] = panel_df.groupby("market")["prob"].shift(1)
panel_df["prob_lag2"] = panel_df.groupby("market")["prob"].shift(2)

panel_df["prob_change"] = panel_df.groupby("market")["prob"].diff()
panel_df["prob_change_lag1"] = panel_df.groupby(
    "market")["prob_change"].shift(1)

panel_df["vix_lag1"] = panel_df["vix"].shift(1)
panel_df["gold_lag1"] = panel_df["gold"].shift(1)
panel_df["silver_lag1"] = panel_df["silver"].shift(1)

model_df = panel_df.dropna()
# print(model_df.shape)