import requests
from datetime import datetime, timezone

HEADERS = {"User-Agent": "Mozilla/5.0 (personal research tool)"}
ASSETS = {"BTC": "BTC-USD", "GOLD": "GC=F"}

def load_daily(name, years=10):
    """Returns a list of (date, close) for daily bars, oldest first."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ASSETS[name]}"
    r = requests.get(url, params={"interval": "1d", "range": f"{years}y"},
                     headers=HEADERS, timeout=30)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    rows = []
    for t, c in zip(stamps, closes):
        if c is None or c <= 0:
            continue
        rows.append((datetime.fromtimestamp(t, timezone.utc).date(), float(c)))
    today = datetime.now(timezone.utc).date()
    if rows and rows[-1][0] >= today:
        rows.pop()  # today's bar is still forming, so it is dropped
    return rows

def check_data(rows):
    warnings = []
    if len(rows) < 300:
        warnings.append(f"Only {len(rows)} daily bars. At least 300 are needed for the 200-day average.")
    dates = [d for d, _ in rows]
    if any(b <= a for a, b in zip(dates, dates[1:])):
        warnings.append("Dates are not strictly increasing. The data may be corrupted.")
    gaps = [(a, b) for a, b in zip(dates, dates[1:]) if (b - a).days > 5]
    if gaps:
        warnings.append(f"{len(gaps)} gap(s) longer than 5 days. The first starts at {gaps[0][0]}.")
    moves = [abs(c2 / c1 - 1) for (_, c1), (_, c2) in zip(rows, rows[1:])]
    if moves and max(moves) > 0.5:
        warnings.append(f"A single-day move of {max(moves):.0%} was found. Check for bad data.")
    return warnings