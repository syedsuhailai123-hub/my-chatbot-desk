import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
CONFIG_FILE = os.path.join(DATA, "forward_config.json")
TRADES_FILE = os.path.join(DATA, "forward_trades.jsonl")
SUMMARY_FILE = os.path.join(DATA, "forward_summary.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (forward research)"}

# FROZEN RULES. Do not edit after the test starts.
RULES = {
    "symbol": "GC=F",
    "bar_seconds": 3600,
    "start_cash": 10000.0,
    "risk_per_trade": 0.005,
    "stop_atr": 1.5,
    "reward_risk": 2.0,
    "cost_per_side": 0.10,
}
PASS_MIN_TRADES = 40
PASS_MIN_PF = 1.2


def fmt(ts):
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg, cfg["rules"] == RULES
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    cfg = {
        "forward_start_utc": fmt(int(now.timestamp())),
        "forward_start_ts": int(now.timestamp()),
        "rules": RULES,
        "criteria": {"min_trades": PASS_MIN_TRADES, "min_profit_factor": PASS_MIN_PF},
    }
    os.makedirs(DATA, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return cfg, True


def fetch_bars():
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{RULES['symbol']}"
    r = requests.get(url, params={"interval": "1h", "range": "730d"},
                     headers=HEADERS, timeout=60)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    rows = []
    for t, o, h, l, c in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"]):
        if None in (o, h, l, c):
            continue
        rows.append((int(t), float(o), float(h), float(l), float(c)))
    df = pd.DataFrame(rows, columns=["t", "Open", "High", "Low", "Close"])
    df = df.drop_duplicates("t").sort_values("t").reset_index(drop=True)
    now = datetime.now(timezone.utc).timestamp()
    # Keep only completed hourly bars
    return df[df["t"] + RULES["bar_seconds"] <= now].reset_index(drop=True)


def indicators(df):
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    change = d["Close"].diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    ag = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    al = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    d["RSI"] = 100 - 100 / (1 + rs)
    d.loc[(al == 0) & (ag > 0), "RSI"] = 100
    d.loc[(al == 0) & (ag == 0), "RSI"] = 50
    d["MACD"] = (d["Close"].ewm(span=12, adjust=False).mean()
                 - d["Close"].ewm(span=26, adjust=False).mean())
    d["MACDS"] = d["MACD"].ewm(span=9, adjust=False).mean()
    pc = d["Close"].shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - pc).abs(),
                    (d["Low"] - pc).abs()], axis=1).max(axis=1)
    d["ATR"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    long_c = (d["EMA20"] > d["EMA50"]) & (d["RSI"] >= 50) & (d["RSI"] <= 70) & (d["MACD"] > d["MACDS"])
    short_c = (d["EMA20"] < d["EMA50"]) & (d["RSI"] >= 30) & (d["RSI"] <= 50) & (d["MACD"] < d["MACDS"])
    d["Signal"] = 0
    d.loc[long_c, "Signal"] = 1
    d.loc[short_c, "Signal"] = -1
    return d


def _record(pos, ex, bar, reason, cash):
    side = pos["side"]
    actual = ex - side * RULES["cost_per_side"]
    pnl = side * (actual - pos["entry"]) * pos["units"]
    cash += pnl
    rec = {
        "entry_utc": fmt(pos["t"]),
        "exit_utc": fmt(int(bar["t"])),
        "side": "LONG" if side == 1 else "SHORT",
        "entry": round(pos["entry"], 2),
        "exit": round(actual, 2),
        "reason": reason,
        "pnl": round(pnl, 2),
        "R": round(pnl / pos["risk"], 3),
        "cash_after": round(cash, 2),
    }
    return cash, rec


def simulate(d, start_idx):
    cash = RULES["start_cash"]
    pos = None
    trades = []
    for i in range(start_idx, len(d)):
        prev = d.iloc[i - 1]
        cur = d.iloc[i]
        had_pos = pos is not None
        o, h, l = float(cur["Open"]), float(cur["High"]), float(cur["Low"])

        if pos is not None:
            side, stop, tgt = pos["side"], pos["stop"], pos["target"]
            ex, reason = None, None
            if side == 1:
                if o <= stop:
                    ex, reason = o, "Stop gap"
                elif o >= tgt:
                    ex, reason = tgt, "Target gap"
                elif l <= stop:
                    ex, reason = stop, "Stop loss"
                elif h >= tgt:
                    ex, reason = tgt, "Take profit"
            else:
                if o >= stop:
                    ex, reason = o, "Stop gap"
                elif o <= tgt:
                    ex, reason = tgt, "Target gap"
                elif h >= stop:
                    ex, reason = stop, "Stop loss"
                elif l <= tgt:
                    ex, reason = tgt, "Take profit"
            if ex is not None:
                cash, rec = _record(pos, ex, cur, reason, cash)
                trades.append(rec)
                pos = None

        if pos is None and cash > 0 and not had_pos:
            sig = int(prev["Signal"])
            atr = prev["ATR"]
            if sig in (1, -1) and pd.notna(atr) and atr > 0:
                sd = RULES["stop_atr"] * float(atr)
                risk = cash * RULES["risk_per_trade"]
                units = risk / (sd + 2 * RULES["cost_per_side"])
                entry = o + sig * RULES["cost_per_side"]
                if sig == 1:
                    stop, tgt = o - sd, o + sd * RULES["reward_risk"]
                    hit_stop, hit_tgt = l <= stop, h >= tgt
                else:
                    stop, tgt = o + sd, o - sd * RULES["reward_risk"]
                    hit_stop, hit_tgt = h >= stop, l <= tgt
                pos = {"side": sig, "entry": entry, "stop": stop, "target": tgt,
                       "units": units, "risk": risk, "t": int(cur["t"])}
                ex, reason = None, None
                if hit_stop:
                    ex, reason = stop, "Entry-candle stop"
                elif hit_tgt:
                    ex, reason = tgt, "Entry-candle target"
                if ex is not None:
                    cash, rec = _record(pos, ex, cur, reason, cash)
                    trades.append(rec)
                    pos = None
    return trades, cash, pos


def summarize(cfg, d, start_idx, trades, cash, pos, rules_ok):
    n = len(trades)
    pnls = [t["pnl"] for t in trades]
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = -sum(p for p in pnls if p < 0)
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else None
    win_rate = round(sum(1 for p in pnls if p > 0) / n * 100, 1) if n else None
    avg_r = round(sum(t["R"] for t in trades) / n, 3) if n else None

    eq = [RULES["start_cash"]] + [t["cash_after"] for t in trades]
    peak, mdd = eq[0], 0.0
    for v in eq:
        peak = max(peak, v)
        mdd = min(mdd, (v - peak) / peak * 100)

    net = round(cash - RULES["start_cash"], 2)
    if start_idx < len(d):
        bh = round((float(d.iloc[-1]["Close"]) / float(d.iloc[start_idx]["Open"]) - 1) * 100, 2)
    else:
        bh = None

    min_n = cfg["criteria"]["min_trades"]
    min_pf = cfg["criteria"]["min_profit_factor"]
    if not rules_ok:
        status = "INVALID: rules were changed after the test started"
    elif n < min_n:
        status = f"WAITING: {n} of {min_n} trades needed"
    else:
        pf_ok = pf is None or pf >= min_pf
        status = "PASS" if (net > 0 and pf_ok) else "FAIL"

    open_txt = None
    if pos is not None:
        open_txt = {
            "side": "LONG" if pos["side"] == 1 else "SHORT",
            "entry": round(pos["entry"], 2),
            "stop": round(pos["stop"], 2),
            "target": round(pos["target"], 2),
            "opened_utc": fmt(pos["t"]),
        }
    return {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "forward_start_utc": cfg["forward_start_utc"],
        "rules_unchanged": rules_ok,
        "status": status,
        "net_pnl_usd": net,
        "return_pct": round(net / RULES["start_cash"] * 100, 2),
        "closed_trades": n,
        "win_rate_pct": win_rate,
        "profit_factor": pf,
        "avg_R": avg_r,
        "max_drawdown_pct": round(mdd, 2),
        "buy_and_hold_since_start_pct": bh,
        "open_position": open_txt,
        "criteria": {"min_trades": min_n, "min_profit_factor": min_pf, "min_net_pnl": 0},
    }


def main():
    cfg, rules_ok = load_config()
    df = fetch_bars()
    if len(df) < 100:
        raise SystemExit(f"Only {len(df)} completed bars returned. Stopping.")
    d = indicators(df)
    start_idx = int(np.searchsorted(d["t"].to_numpy(), cfg["forward_start_ts"]))
    if start_idx < 60:
        raise SystemExit("Forward start is too early in the data. Stopping.")

    trades, cash, pos = [], RULES["start_cash"], None
    if start_idx < len(d):
        trades, cash, pos = simulate(d, start_idx)

    os.makedirs(DATA, exist_ok=True)
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary = summarize(cfg, d, start_idx, trades, cash, pos, rules_ok)
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Forward test started {cfg['forward_start_utc']}")
    print(f"Status: {summary['status']}")
    print(f"Closed trades: {summary['closed_trades']} | net P&L: ${summary['net_pnl_usd']:,.2f}")
    print(f"Profit factor: {summary['profit_factor']} | win rate: {summary['win_rate_pct']}%")
    print(f"Buy and hold since start: {summary['buy_and_hold_since_start_pct']}%")
    print(f"Open position: {summary['open_position']}")


if __name__ == "__main__":
    main()
