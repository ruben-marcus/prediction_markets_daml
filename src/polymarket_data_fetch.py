import json
import pandas as pd
import requests

MARKETS = {
    "recession_2026": "us-recession-by-end-of-2026",
    "fed_hike_2026": "fed-rate-hike-in-2026",
    "china_taiwan": "will-china-invade-taiwan-before-2027",
    "ukraine_ceasefire": "russia-x-ukraine-ceasefire-before-2027",
}

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"


def get_market(slug):
    r = requests.get(f"{GAMMA}/markets/slug/{slug}", timeout=30)
    r.raise_for_status()
    return r.json()


def parse_pm_field(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return json.loads(value)
    return []


def get_yes_token_id(market):
    outcomes = parse_pm_field(market.get("outcomes"))
    clob_token_ids = parse_pm_field(market.get("clobTokenIds"))

    if not outcomes:
        raise ValueError(f"No outcomes found for {market.get('slug')}")
    if not clob_token_ids:
        raise ValueError(f"No clobTokenIds found for {market.get('slug')}")
    if len(outcomes) != len(clob_token_ids):
        raise ValueError(
            f"Length mismatch for {market.get('slug')}: "
            f"{len(outcomes)} outcomes vs {len(clob_token_ids)} token ids"
        )

    for outcome, token_id in zip(outcomes, clob_token_ids):
        if str(outcome).strip().lower() == "yes":
            return token_id

    raise ValueError(f"Could not find YES outcome for {market.get('slug')}")


def get_price_history(asset_id, interval="max", fidelity=1440):
    r = requests.get(
        f"{CLOB}/prices-history",
        params={
            "market": asset_id,
            "interval": interval,
            "fidelity": fidelity
        },
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()["history"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
    df = df.rename(columns={"p": "prob"})
    df = df.sort_values("timestamp")

    # convert to daily close
    df["date"] = df["timestamp"].dt.tz_convert(None).dt.floor("D")
    daily = df.groupby("date", as_index=False)["prob"].last()

    return daily


data = []

for name, slug in MARKETS.items():
    try:
        market = get_market(slug)
        outcomes = parse_pm_field(market.get("outcomes"))
        clob_ids = parse_pm_field(market.get("clobTokenIds"))

        yes_token = get_yes_token_id(market)

        df = get_price_history(yes_token)

        if df.empty:
            continue

        df["market"] = name
        data.append(df)

    except Exception as e:
        print(f"Failed for {name}: {e}")

if not data:
    raise ValueError("No market data was fetched successfully.")

polymarket_df = pd.concat(data, ignore_index=True)
# print(polymarket_df.head())
