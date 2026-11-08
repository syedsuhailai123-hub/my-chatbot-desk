import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from forward_test import indicators

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "signal.json")
ASSETS = {"GOLD": "GC=F", "BTC": "BTC-USD"}
HEADERS = {"User-Agent": "Mozilla/5.0 (signal page)"}
MAX_AGE_HOURS = 3.0


def fetch(sym):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"interval": "1h", "range": "60d"}, headers=HEADERS, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = [(int(t), o, h, l, c) for t, o, h, l, c in
            zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"])
            if None not in (o, h, l, c)]
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close"])
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return df[df["t"] + 3600 <= time.time()].reset_index(drop=True)


def gold_market_open(now):
    """Approximate: gold is closed from Friday 22:00 to Sunday 22:00 UTC."""
    wd, h = now.weekday(), now.hour
    if wd == 5:
        return False
    if wd == 6 and h < 22:
        return False
    if wd == 4 and h >= 22:
        return False
    return True


def side_checks(row, side):
    if side == 1:
        items = [
            ("Trend: EMA20 above EMA50", f"{row.EMA20:.2f} vs {row.EMA50:.2f}", row.EMA20 > row.EMA50),
            ("RSI between 50 and 70", f"{row.RSI:.1f}", 50 <= row.RSI <= 70),
            ("MACD above its signal line", f"{row.MACD:.3f} vs {row.MACDS:.3f}", row.MACD > row.MACDS),
        ]
    else:
        items = [
            ("Trend: EMA20 below EMA50", f"{row.EMA20:.2f} vs {row.EMA50:.2f}", row.EMA20 < row.EMA50),
            ("RSI between 30 and 50", f"{row.RSI:.1f}", 30 <= row.RSI <= 50),
            ("MACD below its signal line", f"{row.MACD:.3f} vs {row.MACDS:.3f}", row.MACD < row.MACDS),
        ]
    return [{"name": n, "value": v, "ok": bool(ok)} for n, v, ok in items]


def build(name, sym, now):
    df = fetch(sym)
    if len(df) < 60:
        return {"error": f"Only {len(df)} completed candles returned."}
    d = indicators(df)
    row = d.iloc[-1]
    bar_open = int(row["t"])
    age = (now.timestamp() - bar_open) / 3600
    market = gold_market_open(now) if name == "GOLD" else True
    sig = int(row["Signal"])
    side = "LONG" if sig == 1 else "SHORT" if sig == -1 else "NONE"
    history = []
    for r in d.tail(500).itertuples():
        if r.Signal != 0:
            history.append({
                "time_utc": datetime.fromtimestamp(int(r.t), timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "side": "LONG" if r.Signal == 1 else "SHORT",
                "close": round(float(r.Close), 2),
            })
    return {
        "last_bar_utc": datetime.fromtimestamp(bar_open, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "close": round(float(row["Close"]), 2),
        "candle_age_hours": round(age, 1),
        "fresh": age <= MAX_AGE_HOURS,
        "market_open": market,
        "signal": side,
        "values": {
            "ema20": round(float(row["EMA20"]), 2), "ema50": round(float(row["EMA50"]), 2),
            "rsi": round(float(row["RSI"]), 1), "atr": round(float(row["ATR"]), 2),
        },
        "long_checks": side_checks(row, 1),
        "short_checks": side_checks(row, -1),
        "history": history[-10:][::-1],
    }


def main():
    now = datetime.now(timezone.utc)
    assets = {name: build(name, sym, now) for name, sym in ASSETS.items()}
    out = {
        "updated_utc": now.strftime("%Y-%m-%d %H:%M UTC"),
        "rule": "Frozen gold rule: EMA20/50 trend, RSI band, MACD cross, on completed hourly candles.",
        "max_candle_age_hours": MAX_AGE_HOURS,
        "assets": assets,
    }
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("signal.json written:", {k: v.get("signal") for k, v in assets.items()})


if __name__ == "__main__":
    main()
