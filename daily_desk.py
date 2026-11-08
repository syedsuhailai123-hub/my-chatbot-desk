import argparse
import csv
import json
import os
import re
from datetime import date, datetime, timezone

import requests

from daily_data import load_daily
from news import get_news

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
SERIES_FILE = os.path.join(DATA, "daily_closes.csv")
HEADLINES_FILE = os.path.join(DATA, "headlines.jsonl")
IDEAS_FILE = os.path.join(DATA, "ideas.jsonl")
RAW_FILE = os.path.join(DATA, "model_raw.txt")
STATUS_FILE = os.path.join(DATA, "status.json")
ASSETS = ("BTC", "GOLD")
DIRECTIONS = ("UP", "DOWN")
HORIZON = 5
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

SYSTEM = (
    "You are a research assistant for a PAPER trading desk. Use ONLY the data given. "
    "Suggest at most 3 ideas about the next 5 trading days. Each idea must be one line "
    "in this format, where you fill in the parts with real choices:\n"
    "IDEA | BTC | UP | one sentence reason from the data\n"
    "Write BTC or GOLD for the asset and UP or DOWN for the direction, choosing ONE of "
    "each. Do not write the words 'or' between choices. Only mention news that appears "
    "in the headlines. If nothing is clearly supported, reply with only: NONE. "
    "Do not give investment advice."
)

def load_series():
    series = {a: {} for a in ASSETS}
    if os.path.exists(SERIES_FILE):
        with open(SERIES_FILE, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                series[r["asset"]][r["date"]] = float(r["close"])
    return series

def sorted_series(series, asset):
    return sorted(series[asset].items())

def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def collect():
    os.makedirs(DATA, exist_ok=True)
    series = load_series()
    for name in ASSETS:
        for d, close in load_daily(name, years=2):
            series[name][d.isoformat()] = close
    with open(SERIES_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["asset", "date", "close"])
        for name in ASSETS:
            for d, c in sorted_series(series, name):
                w.writerow([name, d, c])

    items = get_news("world", 10) + get_news("finance", 10)
    headlines = [i["title"] for i in items]
    today = date.today().isoformat()
    entry = {"collected_on": today, "headlines": headlines}
    old = [h for h in read_jsonl(HEADLINES_FILE) if h["collected_on"] != today]
    with open(HEADLINES_FILE, "w", encoding="utf-8") as f:
        for h in old + [entry]:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")
    return series, headlines

CLEAN_HEADLINES = True

def clean_text(s):
    """Remove line/paragraph separators and non-ASCII punctuation that break some encoders."""
    s = str(s).replace("\u2028", " ").replace("\u2029", " ")
    return s.encode("ascii", "ignore").decode("ascii").strip()

def build_prompt(series, headlines):
    lines = []
    for name in ASSETS:
        s = sorted_series(series, name)[-20:]
        lines.append(f"{name} last 20 daily closes: " +
                     ", ".join(f"{d} {c:,.2f}" for d, c in s))
    if headlines:
        lines.append("Headlines collected today (the ONLY news you may mention):")
        lines += [f"- {clean_text(h)}" for h in headlines[:15]]
    return "\n".join(lines)

def parse_ideas(text):
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.replace("*", "").split("|")]
        if len(parts) < 4:
            continue
        asset, direction, reason = parts[1].upper(), parts[2].upper(), parts[3]
        if asset not in ASSETS or direction not in DIRECTIONS or not reason:
            continue
        out.append({"asset": asset, "direction": direction, "reason": reason[:250]})
    return out[:3]

def safe_ascii(s):
    """Replace every non-ASCII character so no HTTP library can fail on encoding."""
    s = str(s)
    for bad, good in {"\u2028": " ", "\u2029": " ", "\u2019": "'", "\u2018": "'",
                      "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-",
                      "\u2026": "..."}.items():
        s = s.replace(bad, good)
    return s.encode("ascii", "replace").decode("ascii")

def ask_llm(system, user):
    # TRACE_LOGGED: all text is cleaned before any network call
    system, user = safe_ascii(system), safe_ascii(user)
    # KEY_CLEANED: remove hidden characters that break HTTP headers
    key = safe_ascii(os.environ.get("GROQ_API_KEY", "")).strip()
    if key:
        payload = {
            "model": GROQ_MODEL, "temperature": 0.2, "max_tokens": 300,
            "messages": [{"role": "system", "content": safe_ascii(system)},
                         {"role": "user", "content": safe_ascii(user)}],
        }
        r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"},
                          data=json.dumps(payload).encode("ascii"),
                          timeout=120)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"], "groq"
    r = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "stream": False, "keep_alive": "60m",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "options": {"num_ctx": 2048, "num_predict": 300, "temperature": 0.2},
    }, timeout=300)
    r.raise_for_status()
    return r.json()["message"]["content"], "ollama"

def make_ideas(series, headlines):
    today = date.today().isoformat()
    if any(i["made_on"] == today for i in read_jsonl(IDEAS_FILE)):
        return "ideas already made today"
    try:
        raw, backend = ask_llm(SYSTEM, build_prompt(series, headlines))
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"error: {type(e).__name__}: {str(e)[:120]}"
    with open(RAW_FILE, "a", encoding="utf-8") as f:
        f.write(f"--- {today} ({backend}) ---\n{raw}\n\n")
    ideas = parse_ideas(raw)
    as_of = {name: sorted_series(series, name)[-1][0] for name in ASSETS if series[name]}
    with open(IDEAS_FILE, "a", encoding="utf-8") as f:
        for idea in ideas:
            d = as_of.get(idea["asset"])
            if d is None:
                continue
            rec = {"made_on": today, "as_of": d, "asset": idea["asset"],
                   "direction": idea["direction"], "reason": idea["reason"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"Saved idea: {idea['asset']} {idea['direction']}: {idea['reason']}")
    if not ideas:
        print("No valid ideas in the answer.")
    return f"ok ({backend}, {len(ideas)} ideas)"

def grade_stats(series):
    ideas = read_jsonl(IDEAS_FILE)
    hits, rets, pending = 0, [], 0
    for idea in ideas:
        s = sorted_series(series, idea["asset"])
        dates = [d for d, _ in s]
        if idea["as_of"] not in dates:
            continue
        i = dates.index(idea["as_of"])
        if i + HORIZON >= len(s):
            pending += 1
            continue
        ret = s[i + HORIZON][1] / s[i][1] - 1
        rets.append(ret)
        if (idea["direction"] == "UP" and ret > 0) or (idea["direction"] == "DOWN" and ret < 0):
            hits += 1
    baseline = {}
    for name in ASSETS:
        s = sorted_series(series, name)
        w = [s[i + HORIZON][1] > s[i][1] for i in range(len(s) - HORIZON)]
        if w:
            baseline[name] = round(sum(w) / len(w) * 100, 1)
    return {
        "made": len(ideas), "graded": len(rets), "pending": pending,
        "hit_rate": round(hits / len(rets) * 100, 1) if rets else None,
        "avg_return_pct": round(sum(rets) / len(rets) * 100, 2) if rets else None,
        "baseline_up_pct": baseline,
    }

def write_status(series, headlines, ai_status):
    prices = {}
    for name in ASSETS:
        s = sorted_series(series, name)
        if len(s) >= 2:
            prices[name] = {"date": s[-1][0], "close": round(s[-1][1], 2),
                            "change_1d_pct": round((s[-1][1] / s[-2][1] - 1) * 100, 2)}
    status = {
        "updated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ai_status": ai_status,
        "prices": prices,
        "headlines": headlines[:15],
        "ideas": read_jsonl(IDEAS_FILE)[-10:][::-1],
        "grade": grade_stats(series),
    }
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(status, f, indent=2, ensure_ascii=False)
    return status

def run_daily():
    series, headlines = collect()
    ai_status = make_ideas(series, headlines)
    series = load_series()
    status = write_status(series, headlines, ai_status)
    g = status["grade"]
    print(f"Prices: " + ", ".join(f"{k} {v['close']:,.2f}" for k, v in status["prices"].items()))
    print(f"AI: {ai_status}")
    print(f"Ideas made {g['made']} · graded {g['graded']} · pending {g['pending']}")
    if g["graded"] < 100:
        print("Fewer than 100 graded ideas. Do not draw conclusions yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily data, AI ideas, and grading.")
    parser.add_argument("command", choices=["daily", "grade"])
    args = parser.parse_args()
    if args.command == "daily":
        run_daily()
    else:
        series = load_series()
        print(json.dumps(grade_stats(series), indent=2))
