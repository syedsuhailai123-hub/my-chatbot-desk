import json
import os
from datetime import datetime, timezone

import requests

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
STATE_FILE = os.path.join(DATA, "paper_state.json")
SUMMARY_FILE = os.path.join(DATA, "paper_summary.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (paper research)"}

START_EQUITY = 10000.0
RISK_PER_TRADE = 0.01
STOP_ATR = 1.5
TARGET_ATR = 3.0
MIN_SAMPLE = 30
PAUSE_BELOW_R = -0.10
RESUME_ABOVE_R = 0.10

SYMBOLS = {"BTC": "BTC-USD", "GOLD": "GC=F"}
BAR_SECONDS = {"1h": 3600, "4h": 14400, "1d": 86400}

SLOTS = [
    {"id": "BTC-1h-trend", "asset": "BTC", "tf": "1h", "strategy": "trend", "max_hold": 48},
    {"id": "BTC-4h-breakout", "asset": "BTC", "tf": "4h", "strategy": "breakout", "max_hold": 30},
    {"id": "BTC-1d-trend", "asset": "BTC", "tf": "1d", "strategy": "trend", "max_hold": 20},
    {"id": "GOLD-1h-reversion", "asset": "GOLD", "tf": "1h", "strategy": "reversion", "max_hold": 24},
    {"id": "GOLD-1d-breakout", "asset": "GOLD", "tf": "1d", "strategy": "breakout", "max_hold": 15},
]

def fetch_bars(symbol, interval, rng):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    r = requests.get(url, params={"interval": interval, "range": rng},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    q = res["indicators"]["quote"][0]
    bars = []
    for t, o, h, l, c in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"]):
        if None in (o, h, l, c):
            continue
        bars.append({"t": int(t), "o": float(o), "h": float(h), "l": float(l), "c": float(c)})
    return bars

def resample(bars, hours):
    size = hours * 3600
    out, cur = [], None
    for b in bars:
        key = b["t"] // size * size
        if cur is None or cur["t"] != key:
            if cur:
                out.append(cur)
            cur = {"t": key, "o": b["o"], "h": b["h"], "l": b["l"], "c": b["c"]}
        else:
            cur["h"] = max(cur["h"], b["h"])
            cur["l"] = min(cur["l"], b["l"])
            cur["c"] = b["c"]
    if cur:
        out.append(cur)
    return out

def get_series(asset, tf):
    """Returns only completed bars, so the newest unfinished candle is never used."""
    now = datetime.now(timezone.utc).timestamp()
    if tf == "1h":
        bars = fetch_bars(SYMBOLS[asset], "1h", "60d")
    elif tf == "4h":
        bars = resample(fetch_bars(SYMBOLS[asset], "1h", "60d"), 4)
    else:
        bars = fetch_bars(SYMBOLS[asset], "1d", "2y")
    secs = BAR_SECONDS[tf]
    return [b for b in bars if b["t"] + secs <= now]

def ema_series(vals, n):
    k = 2 / (n + 1)
    out, e = [], None
    for v in vals:
        e = v if e is None else v * k + e * (1 - k)
        out.append(e)
    return out

def rsi_series(closes, n=14):
    out = [None] * len(closes)
    if len(closes) <= n:
        return out
    gains, losses = [], []
    for i in range(1, n + 1):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0))
        losses.append(max(-d, 0))
    ag, al = sum(gains) / n, sum(losses) / n
    out[n] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(n + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (n - 1) + max(d, 0)) / n
        al = (al * (n - 1) + max(-d, 0)) / n
        out[i] = 100 if al == 0 else 100 - 100 / (1 + ag / al)
    return out

def atr_series(bars, n=14):
    out = [None] * len(bars)
    trs = [0.0]
    for i in range(1, len(bars)):
        h, l, pc = bars[i]["h"], bars[i]["l"], bars[i - 1]["c"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(bars) <= n:
        return out
    a = sum(trs[1:n + 1]) / n
    out[n] = a
    for i in range(n + 1, len(bars)):
        a = (a * (n - 1) + trs[i]) / n
        out[i] = a
    return out

def signal_at(bars, ind, i, strategy):
    """Decides using only bars up to index i. Returns 1 (long), -1 (short), or 0."""
    if i < 30:
        return 0
    c = bars[i]["c"]
    if strategy == "trend":
        f, s = ind["ema9"], ind["ema21"]
        if f[i - 1] <= s[i - 1] and f[i] > s[i]:
            return 1
        if f[i - 1] >= s[i - 1] and f[i] < s[i]:
            return -1
        return 0
    if strategy == "breakout":
        hi = max(b["h"] for b in bars[i - 20:i])
        lo = min(b["l"] for b in bars[i - 20:i])
        if c > hi:
            return 1
        if c < lo:
            return -1
        return 0
    if strategy == "reversion":
        r0, r1 = ind["rsi"][i - 1], ind["rsi"][i]
        if r0 is None or r1 is None:
            return 0
        if r0 < 30 <= r1:
            return 1
        if r0 > 70 >= r1:
            return -1
        return 0
    return 0

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"equity": START_EQUITY, "slots": {}}
    for slot in SLOTS:
        state["slots"].setdefault(slot["id"], {
            "open": None, "last_signal_t": None, "trades": [],
            "mode": "active", "mode_note": "",
        })
    return state

def save_state(state):
    os.makedirs(DATA, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

def fmt_time(t):
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def close_trade(state, st_, pos, exit_price, reason, exit_t, slot_id):
    risk = STOP_ATR * pos["atr"]
    R = (exit_price - pos["entry"]) * pos["side"] / risk if risk > 0 else 0.0
    rec = {
        "slot": slot_id,
        "side": "LONG" if pos["side"] == 1 else "SHORT",
        "entry": round(pos["entry"], 4),
        "exit": round(exit_price, 4),
        "opened": pos["opened"],
        "closed": fmt_time(exit_t),
        "reason": reason,
        "R": round(R, 3),
        "shadow": pos["shadow"],
    }
    if not pos["shadow"]:
        pnl = state["equity"] * RISK_PER_TRADE * R
        state["equity"] += pnl
        rec["pnl_usd"] = round(pnl, 2)
    st_["trades"].append(rec)
    st_["open"] = None
    print(f"  CLOSED {slot_id}: {rec['side']} {reason}, R {rec['R']:+.2f}"
          + (" (shadow)" if pos["shadow"] else ""))

def manage(state, st_, bars, slot_id):
    pos = st_["open"]
    if not pos:
        return
    for b in bars:
        if b["t"] <= pos["last_t"]:
            continue
        pos["last_t"] = b["t"]
        pos["bars_held"] += 1
        side = pos["side"]
        hit_stop = b["l"] <= pos["stop"] if side == 1 else b["h"] >= pos["stop"]
        hit_target = b["h"] >= pos["target"] if side == 1 else b["l"] <= pos["target"]
        if hit_stop:
            exit_price, reason = pos["stop"], "stop"
        elif hit_target:
            exit_price, reason = pos["target"], "target"
        elif pos["bars_held"] >= pos["max_hold"]:
            exit_price, reason = b["c"], "time"
        else:
            continue
        close_trade(state, st_, pos, exit_price, reason, b["t"], slot_id)
        return

def open_new(st_, slot, bars, ind, slot_id):
    if st_["open"]:
        return
    i = len(bars) - 1
    t = bars[i]["t"]
    if st_["last_signal_t"] == t:
        return
    sig = signal_at(bars, ind, i, slot["strategy"])
    if sig == 0:
        return
    atr = ind["atr"][i]
    if atr is None or atr == 0:
        return
    st_["last_signal_t"] = t
    entry = bars[i]["c"]
    risk = STOP_ATR * atr
    st_["open"] = {
        "side": sig, "entry": entry, "entry_t": t, "last_t": t, "atr": atr,
        "stop": entry - sig * risk,
        "target": entry + sig * TARGET_ATR * atr,
        "max_hold": slot["max_hold"], "bars_held": 0,
        "opened": fmt_time(t),
        "shadow": st_["mode"] == "paused",
    }
    side = "LONG" if sig == 1 else "SHORT"
    tag = " (shadow, slot paused)" if st_["open"]["shadow"] else ""
    print(f"  OPENED {slot_id}: {side} at {entry:,.2f}{tag}")

def update_mode(st_):
    """Runs after every trade. Judges the strategy on all its recent signals."""
    recent = st_["trades"][-MIN_SAMPLE:]
    if len(recent) < MIN_SAMPLE:
        st_["mode_note"] = f"Learning: {len(recent)}/{MIN_SAMPLE} trades needed before any mode change."
        return
    avg = sum(t["R"] for t in recent) / len(recent)
    if st_["mode"] == "active" and avg < PAUSE_BELOW_R:
        st_["mode"] = "paused"
        st_["mode_note"] = f"Paused: last {MIN_SAMPLE} trades averaged {avg:+.2f}R."
    elif st_["mode"] == "paused" and avg > RESUME_ABOVE_R:
        st_["mode"] = "active"
        st_["mode_note"] = f"Resumed: last {MIN_SAMPLE} trades averaged {avg:+.2f}R."
    else:
        st_["mode_note"] = f"Last {MIN_SAMPLE} trades averaged {avg:+.2f}R. Mode unchanged."

def run():
    state = load_state()
    for slot in SLOTS:
        slot_id = slot["id"]
        st_ = state["slots"][slot_id]
        print(f"{slot_id}:")
        try:
            bars = get_series(slot["asset"], slot["tf"])
        except Exception as e:
            print(f"  data error: {e}")
            continue
        if len(bars) < 60:
            print("  not enough completed bars yet")
            continue
        closes = [b["c"] for b in bars]
        ind = {
            "ema9": ema_series(closes, 9),
            "ema21": ema_series(closes, 21),
            "rsi": rsi_series(closes),
            "atr": atr_series(bars),
        }
        before = len(st_["trades"])
        manage(state, st_, bars, slot_id)
        open_new(st_, slot, bars, ind, slot_id)
        if len(st_["trades"]) != before:
            update_mode(st_)
        else:
            update_mode(st_)
    save_state(state)
    write_summary(state)
    report(state)

def report(state):
    print(f"\nPaper equity: ${state['equity']:,.2f} (start ${START_EQUITY:,.2f})")
    for slot in SLOTS:
        st_ = state["slots"][slot["id"]]
        trades = st_["trades"]
        n = len(trades)
        avg = sum(t["R"] for t in trades) / n if n else 0.0
        win = sum(1 for t in trades if t["R"] > 0) / n * 100 if n else 0.0
        pos = st_["open"]
        pos_txt = "none"
        if pos:
            pos_txt = (("LONG" if pos["side"] == 1 else "SHORT") +
                       f" from {pos['entry']:,.2f}" + (" (shadow)" if pos["shadow"] else ""))
        print(f"{slot['id']:<20} mode {st_['mode']:<7} trades {n:<4} "
              f"win {win:5.1f}%  avg {avg:+.2f}R  open {pos_txt}")
        print(f"{'':20} {st_['mode_note']}")

def write_summary(state):
    slots = {}
    for slot in SLOTS:
        st_ = state["slots"][slot["id"]]
        trades = st_["trades"]
        n = len(trades)
        slots[slot["id"]] = {
            "mode": st_["mode"],
            "mode_note": st_["mode_note"],
            "trades": n,
            "win_rate_pct": round(sum(1 for t in trades if t["R"] > 0) / n * 100, 1) if n else None,
            "avg_R": round(sum(t["R"] for t in trades) / n, 3) if n else None,
            "open": st_["open"] and {
                "side": "LONG" if st_["open"]["side"] == 1 else "SHORT",
                "entry": round(st_["open"]["entry"], 2),
                "stop": round(st_["open"]["stop"], 2),
                "target": round(st_["open"]["target"], 2),
                "shadow": st_["open"]["shadow"],
            },
        }
    all_trades = []
    for st_ in state["slots"].values():
        all_trades += st_["trades"]
    all_trades.sort(key=lambda t: t["closed"], reverse=True)
    summary = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "equity": round(state["equity"], 2),
        "slots": slots,
        "recent_trades": all_trades[:20],
    }
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    run()
