import argparse
import csv
import json
import os
from datetime import date

import requests

from daily_data import load_daily
from news import get_news

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
SERIES_FILE = os.path.join(DATA, "daily_closes.csv")
HEADLINES_FILE = os.path.join(DATA, "headlines.jsonl")
IDEAS_FILE = os.path.join(DATA, "ideas.jsonl")
RAW_FILE = os.path.join(DATA, "model_raw.txt")
OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "llama3.2:3b"
ASSETS = ("BTC", "GOLD")
DIRECTIONS = ("UP", "DOWN")
HORIZON = 5  # trading days until an idea is graded

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
    entry = {"collected_on": date.today().isoformat(), "headlines": headlines}
    old = [h for h in read_jsonl(HEADLINES_FILE)
           if h["collected_on"] != entry["collected_on"]]
    with open(HEADLINES_FILE, "w", encoding="utf-8") as f:
        for h in old + [entry]:
            f.write(json.dumps(h, ensure_ascii=False) + "\n")

    print("Collected prices and headlines.")
    for name in ASSETS:
        s = sorted_series(series, name)
        if s:
            print(f"  {name}: last close {s[-1][1]:,.2f} on {s[-1][0]} ({len(s)} days stored)")
    print(f"  Headlines saved: {len(headlines)}")

def build_prompt(series):
    lines = []
    for name in ASSETS:
        s = sorted_series(series, name)[-20:]
        lines.append(f"{name} last 20 daily closes: " +
                     ", ".join(f"{d} {c:,.2f}" for d, c in s))
    headlines = read_jsonl(HEADLINES_FILE)
    if headlines:
        lines.append("Headlines collected today (the ONLY news you may mention):")
        lines += [f"- {h}" for h in headlines[-1]["headlines"][:15]]
    return "\n".join(lines)

def parse_ideas(text):
    """Strict: accepts only 'IDEA | BTC | UP | reason' with an exact asset and direction."""
    out = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.replace("*", "").split("|")]
        if len(parts) < 4:
            continue
        asset, direction, reason = parts[1].upper(), parts[2].upper(), parts[3]
        if asset not in ASSETS or direction not in DIRECTIONS:
            continue  # rejects 'BTC or GOLD' and 'UP or DOWN' template text
        if not reason:
            continue
        out.append({"asset": asset, "direction": direction, "reason": reason[:250]})
    return out[:3]

def make_ideas():
    os.makedirs(DATA, exist_ok=True)
    today = date.today().isoformat()
    if any(i["made_on"] == today for i in read_jsonl(IDEAS_FILE)):
        print("Ideas were already made today. Run grade to see results.")
        return
    series = load_series()
    system = (
        "You are a research assistant for a PAPER trading desk. Use ONLY the data given. "
        "Suggest at most 3 ideas about the next 5 trading days. Each idea must be one line "
        "in this format, where you fill in the parts with real choices:\n"
        "IDEA | BTC | UP | one sentence reason from the data\n"
        "Write BTC or GOLD for the asset and UP or DOWN for the direction, choosing ONE of "
        "each. Do not write the words 'or' between choices. Only mention news that appears "
        "in the headlines. If nothing is clearly supported, reply with only: NONE. "
        "Do not give investment advice."
    )
    r = requests.post(OLLAMA_URL, json={
        "model": MODEL, "stream": False, "keep_alive": "60m",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": build_prompt(series)}],
        "options": {"num_ctx": 2048, "num_predict": 300, "temperature": 0.2},
    }, timeout=300)
    r.raise_for_status()
    raw = r.json()["message"]["content"]

    with open(RAW_FILE, "a", encoding="utf-8") as f:
        f.write(f"--- {today} ---\n{raw}\n\n")
    print("Model answer (saved to data/model_raw.txt):")
    print(raw)
    print()

    ideas = parse_ideas(raw)
    as_of = {name: sorted_series(series, name)[-1][0]
             for name in ASSETS if series[name]}
    with open(IDEAS_FILE, "a", encoding="utf-8") as f:
        for idea in ideas:
            d = as_of.get(idea["asset"])
            if d is None:
                continue
            rec = {"made_on": today, "as_of": d, "asset": idea["asset"],
                   "direction": idea["direction"], "reason": idea["reason"]}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  Saved idea: {idea['asset']} {idea['direction']}: {idea['reason']}")
    if not ideas:
        print("  No valid ideas. The answer did not follow the required format, or it said NONE.")

def grade():
    series = load_series()
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

    print(f"Ideas made: {len(ideas)} · graded: {len(rets)} · still pending: {pending}")
    if rets:
        print(f"Hit rate: {hits / len(rets) * 100:.1f}% · average {HORIZON}-day return "
              f"of the ideas: {sum(rets) / len(rets) * 100:+.2f}%")
    print(f"\nBaseline: how often price rose over {HORIZON} days in the stored history")
    for name in ASSETS:
        s = sorted_series(series, name)
        windows = [s[i + HORIZON][1] > s[i][1] for i in range(len(s) - HORIZON)]
        if windows:
            print(f"  {name}: {sum(windows) / len(windows) * 100:.1f}% of {HORIZON}-day windows went up")
    if len(rets) < 100:
        print("\nFewer than 100 graded ideas. Do not draw conclusions yet.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily data, AI ideas, and grading.")
    parser.add_argument("command", choices=["collect", "ideas", "grade", "daily"])
    args = parser.parse_args()
    if args.command in ("collect", "daily"):
        collect()
    if args.command in ("ideas", "daily"):
        make_ideas()
    if args.command == "grade":
        grade()