import yfinance as yf
import pandas as pd


def get_macro_data(start="2023-01-01", end=None):
    tickers = {
        "vix": "^VIX",
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "ten_year_yield": "^TNX",
        "dollar_index": "DX-Y.NYB",
        "oil": "CL=F",
        "gold": "GC=F",
        "silver": "SI=F",
        "bitcoin": "BTC-USD",
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
        try:
            macro[name] = raw[ticker]["Close"]
        except KeyError:
            print(f"Skipping macro ticker {ticker}: no close data returned")

    macro = macro.reset_index().rename(columns={"Date": "date"})
    macro["date"] = pd.to_datetime(macro["date"]).dt.floor("D")
    macro = macro.sort_values("date")

    price_cols = [col for col in macro.columns if col != "date"]
    for col in price_cols:
        macro[f"{col}_ret"] = macro[col].pct_change(fill_method=None)

    return macro


macro_df = get_macro_data(start="2023-01-01")
