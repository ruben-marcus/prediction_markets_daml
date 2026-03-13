import yfinance as yf
import pandas as pd


def get_macro_data(start="2023-01-01", end=None):
    tickers = {
        "vix": "^VIX",
        "gold": "GC=F",
        "silver": "SI=F",
    }

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    macro = pd.DataFrame(index=raw.index)
    for name, ticker in tickers.items():
        macro[name] = raw[ticker]["Close"]

    macro = macro.reset_index().rename(columns={"Date": "date"})
    macro["date"] = pd.to_datetime(macro["date"]).dt.floor("D")
    macro = macro.sort_values("date")

    macro["vix_ret"] = macro["vix"].pct_change(fill_method=None)
    macro["gold_ret"] = macro["gold"].pct_change(fill_method=None)
    macro["silver_ret"] = macro["silver"].pct_change(fill_method=None)

    return macro


macro_df = get_macro_data(start="2023-01-01")
