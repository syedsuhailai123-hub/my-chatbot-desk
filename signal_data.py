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
LOG = os.path.join(DATA, "candle_log.json")
DEMO = os.path.join(DATA, "demo_status.json")
ASSETS = {"GOLD": "GC=F", "BTC": "BTC-USD"}
HEADERS = {"User-Agent": "Mozilla/5.0 (signal page)"}
MAX_AGE_HOURS = 3.0
LOG_KEEP = 800
BACKFILL_CANDLES = 24


def fmt_utc(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


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


def gold_market_open(ts):
    dt = datetime.fromtimestamp(ts, timezone.utc)
    wd, h = dt.weekday(), dt.hour
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


def demo_orders_by_candle():
    """Maps candle open time (epoch seconds) to the demo order placed on that candle."""
    if not os.path.exists(DEMO):
        return {}
    with open(DEMO, encoding="utf-8") as f:
        orders = json.load(f).get("recent_orders", [])
    out = {}
    for o in orders:
        sc = o.get("signal_candle")
        if not sc:
            continue
        try:
            ts = pd.Timestamp(sc)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        except Exception:
            continue
        out[int(ts.floor("h").timestamp())] = o
    return out


def decide(sig, fresh, market, order, live):
    if order is not None:
        if order.get("filled"):
            return ("DEMO ORDER PLACED",
                    f"Filled at {order.get('fill_price')}. Stop {order.get('stop')}, target {order.get('target')}.")
        return "DEMO ORDER NOT FILLED", "The order was sent but did not fill."
    if not market:
        return "NO TRADE", "Market was closed for this candle."
    if sig == 0:
        return "NO TRADE", "No signal: not every condition passed."
    if not live:
        return ("UNKNOWN",
                "The rule fired on this older candle, but it was not recorded live, "
                "so whether the bot acted is unknown.")
    if not fresh:
        return "NO TRADE", "Signal too old to act on."
    return ("NO DEMO ORDER",
            "The rule fired on a fresh candle, but no demo order is recorded. Run publish_demo.py "
            "after the bot runs, or the bot was not started with --trade.")


def records_for(name, d, now, demo):
    recs = []
    for i in range(max(60, len(d) - BACKFILL_CANDLES), len(d)):
        row = d.iloc[i]
        bar_open = int(row["t"])
        live = i == len(d) - 1
        age = (now - bar_open) / 3600
        fresh = age <= MAX_AGE_HOURS
        market = gold_market_open(bar_open) if name == "GOLD" else True
        sig = int(row["Signal"])
        order = demo.get(bar_open)
        decision, reason = decide(sig, fresh, market, order, live)
        recs.append({
            "asset": name,
            "open_ts": bar_open,
            "candle_open_utc": fmt_utc(bar_open),
            "candle_close_utc": fmt_utc(bar_open + 3600),
            "close": round(float(row["Close"]), 2),
            "signal": "LONG" if sig == 1 else "SHORT" if sig == -1 else "NONE",
            "decision": decision,
            "reason": reason,
            "live": live,
            "demo_order": None if order is None else {
                "side": order.get("side"), "units": order.get("units"),
                "stop": order.get("stop"), "target": order.get("target"),
                "filled": order.get("filled"), "fill_price": order.get("fill_price"),
            },
        })
    return recs


def signal_block(name, d, now):
    row = d.iloc[-1]
    bar_open = int(row["t"])
    age = (now - bar_open) / 3600
    sig = int(row["Signal"])
    side = "LONG" if sig == 1 else "SHORT" if sig == -1 else "NONE"
    market = gold_market_open(bar_open) if name == "GOLD" else True
    history = []
    for r in d.tail(500).itertuples():
        if r.Signal != 0:
            history.append({
                "time_utc": fmt_utc(int(r.t)),
                "side": "LONG" if r.Signal == 1 else "SHORT",
                "close": round(float(r.Close), 2),
            })
    return {
        "last_bar_utc": fmt_utc(bar_open),
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
    now = time.time()
    stamp = fmt_utc(now)
    demo = demo_orders_by_candle()
    assets, new_records = {}, []
    for name, sym in ASSETS.items():
        df = fetch(sym)
        if len(df) < 60:
            assets[name] = {"error": f"Only {len(df)} completed candles returned."}
            continue
        d = indicators(df)
        assets[name] = signal_block(name, d, now)
        new_records += records_for(name, d, now, demo)

    # PAPER_LINK: mark candles where the paper gold account entered
    FWD = os.path.join(DATA, "forward_trades.jsonl")
    FWD_SUM = os.path.join(DATA, "forward_summary.json")
    paper = {}
    def _ts(text):
        return int(pd.Timestamp(text.replace(" UTC", ":00+00:00")).timestamp())
    if os.path.exists(FWD):
        with open(FWD, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    t = json.loads(line)
                    paper[_ts(t["entry_utc"])] = {"kind": "closed", **t}
    if os.path.exists(FWD_SUM):
        with open(FWD_SUM, encoding="utf-8") as f:
            op = json.load(f).get("open_position")
        if op and op.get("opened_utc"):
            paper.setdefault(_ts(op["opened_utc"]), {"kind": "open", **op})
    for r in new_records:
        entry = paper.get(r["open_ts"] + 3600)
        if r["asset"] != "GOLD" or entry is None or r["demo_order"] is not None:
            continue
        if entry["kind"] == "closed":
            detail = (f"Paper {entry['side']} entered at {entry['entry']}, closed by "
                      f"{entry['reason']}, P&L {entry['pnl']:,.2f}.")
        else:
            detail = (f"Paper {entry['side']} entered at {entry['entry']}, still open "
                      f"(stop {entry['stop']}, target {entry['target']}).")
        r["decision"] = "PAPER TRADE TAKEN"
        r["reason"] = detail + " Entry was at the open of the next candle."
        r["paper"] = True
    index = {}
    if os.path.exists(LOG):
        with open(LOG, encoding="utf-8") as f:
            for r in json.load(f).get("records", []):
                index[(r["asset"], r["open_ts"])] = r
    for r in new_records:
        key = (r["asset"], r["open_ts"])
        old = index.get(key)
        # Keep an earlier live decision unless a demo order now exists for that candle
        if old and old.get("live") and not r["live"] and r["demo_order"] is None and not r.get("paper"):
            continue
        index[key] = r
    merged = sorted(index.values(), key=lambda r: r["open_ts"], reverse=True)[:LOG_KEEP]

    os.makedirs(DATA, exist_ok=True)
    with open(LOG, "w", encoding="utf-8") as f:
        json.dump({"updated_utc": stamp, "records": merged}, f, indent=2)
    out = {
        "updated_utc": stamp,
        "rule": "Frozen gold rule: EMA20/50 trend, RSI band, MACD cross, on completed hourly candles.",
        "max_candle_age_hours": MAX_AGE_HOURS,
        "assets": assets,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("signal.json and candle_log.json written. Records:", len(merged))


if __name__ == "__main__":
    main()
