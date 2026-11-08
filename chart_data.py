import json
import os
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from forward_test import indicators

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
OUT = os.path.join(DATA, "chart.json")
PRED = os.path.join(DATA, "predictions.json")
ASSETS = {"GOLD": "GC=F", "BTC": "BTC-USD"}
HEADERS = {"User-Agent": "Mozilla/5.0 (chart research)"}
SHOW = 240


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


def baseline_up(df, h):
    c = df["Close"].to_numpy()
    if len(c) <= h:
        return None
    return round(float((c[h:] > c[:-h]).mean()) * 100, 1)


def grade(preds, frames, now):
    out = []
    for p in preds:
        rec = dict(p)
        due = p["made_ts"] + p["horizon_h"] * 3600
        rec["due_utc"] = datetime.fromtimestamp(due, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        status, result = "pending", None
        df = frames.get(p["asset"])
        if df is not None and now >= due:
            after = df[df["t"] >= due]
            if len(after):
                result = float(after.iloc[0]["Close"])
                if p["direction"] == "UP":
                    hit = result > p["price_at_call"]
                else:
                    hit = result < p["price_at_call"]
                status = "hit" if hit else "miss"
        rec["status"] = status
        rec["result_price"] = None if result is None else round(result, 4)
        out.append(rec)
    return out


def main():
    now = time.time()
    frames, assets_out = {}, {}
    for name, sym in ASSETS.items():
        df = fetch(sym)
        frames[name] = df
        d = indicators(df)
        tail = d.tail(SHOW)
        bars = [{"t": int(r.t), "o": float(r.Open), "h": float(r.High), "l": float(r.Low),
                 "c": float(r.Close),
                 "e20": None if pd.isna(r.EMA20) else float(r.EMA20),
                 "e50": None if pd.isna(r.EMA50) else float(r.EMA50)}
                for r in tail.itertuples()]
        signals = [{"t": int(r.t), "side": "LONG" if r.Signal == 1 else "SHORT"}
                   for r in tail.itertuples() if r.Signal != 0]
        last = d.iloc[-1]
        assets_out[name] = {
            "bars": bars,
            "signals": signals,
            "last_close": round(float(last["Close"]), 2),
            "last_bar_utc": datetime.fromtimestamp(int(last["t"]), timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        }

    preds = json.load(open(PRED, encoding="utf-8")) if os.path.exists(PRED) else []
    graded = grade(preds, frames, now)
    horizons = sorted({p["horizon_h"] for p in preds} | {24})
    base = {name: {str(h): baseline_up(frames[name], h) for h in horizons} for name in ASSETS}
    done = [g for g in graded if g["status"] in ("hit", "miss")]
    hits = sum(1 for g in done if g["status"] == "hit")
    summary = {
        "total": len(graded),
        "graded": len(done),
        "pending": len(graded) - len(done),
        "hit_rate_pct": round(hits / len(done) * 100, 1) if done else None,
        "baseline_up_pct": base,
        "note": ("Each prediction is graded on the first completed hourly candle at or after its "
                 "due time. Compare the hit rate with the baseline. 100 or more graded "
                 "predictions are needed before drawing any conclusion."),
    }
    out = {
        "updated_utc": datetime.fromtimestamp(now, timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "assets": assets_out,
        "predictions": graded[-100:][::-1],
        "summary": summary,
    }
    os.makedirs(DATA, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("chart.json written.", summary["total"], "predictions,", summary["graded"], "graded.")


if __name__ == "__main__":
    main()
