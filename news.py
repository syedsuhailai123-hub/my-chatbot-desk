import html
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import xml.etree.ElementTree as ET
import requests
import streamlit as st

FEEDS = {
    "world": [
        ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml"),
        ("BBC Top Stories", "https://feeds.bbci.co.uk/news/rss.xml"),
    ],
    "finance": [
        ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
    ],
}
PANEL_LABELS = {"home": "HOME", "world": "WORLD NEWS", "finance": "FINANCE"}
HEADERS = {"User-Agent": "Mozilla/5.0 (personal-chatbot RSS reader)"}
CACHE_SECONDS = 600

NEWS_QUERY = re.compile(
    r"\b(news|headlines?|markets?|stocks?|finance|financial|economy|economic|"
    r"world|inflation|crypto|happening)\b", re.I)
OPEN_CMD = re.compile(
    r"^\s*(please\s+)?(jarvis[,\s]+|twin[,\s]+)?(open|show|display|switch to|go to|bring up|pull up)\b",
    re.I)
CLOSE_CMD = re.compile(r"^\s*(please\s+)?(close|hide|go home|back to home)\b", re.I)
REFRESH_CMD = re.compile(r"^\s*(please\s+)?(refresh|reload)\b", re.I)
FINANCE_WORDS = re.compile(
    r"\b(finance|financial|market|markets|stock|stocks|business|money|economy)\b", re.I)
WORLD_WORDS = re.compile(r"\b(world|global|international|news|headlines)\b", re.I)
HOME_WORDS = re.compile(r"\bhome\b", re.I)

def parse_command(text):
    """Returns a command dict if the message is a screen command, otherwise None."""
    if REFRESH_CMD.search(text):
        return {"action": "refresh"}
    if CLOSE_CMD.search(text):
        return {"action": "home"}
    if OPEN_CMD.search(text):
        if FINANCE_WORDS.search(text):
            return {"action": "open", "panel": "finance"}
        if WORLD_WORDS.search(text):
            return {"action": "open", "panel": "world"}
        if HOME_WORDS.search(text):
            return {"action": "home"}
    return None

def _when(text):
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)

@st.cache_data(ttl=CACHE_SECONDS, show_spinner=False)
def fetch_feed(source, url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except Exception:
        return []
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not title or not link.startswith(("http://", "https://")):
            continue
        items.append({
            "title": title,
            "link": link,
            "source": source,
            "time": _when(item.findtext("pubDate")),
        })
    return items

def get_news(category, limit=12):
    seen, items = set(), []
    for source, url in FEEDS[category]:
        for item in fetch_feed(source, url):
            if item["link"] not in seen:
                seen.add(item["link"])
                items.append(item)
    items.sort(key=lambda x: x["time"], reverse=True)
    return items[:limit]

def ago(dt):
    if dt.year < 2000:
        return "time unknown"
    secs = (datetime.now(timezone.utc) - dt).total_seconds()
    if secs < 3600:
        return f"{max(1, int(secs // 60))}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"

def cards_html(items):
    if not items:
        return '<div class="empty">No headlines available right now. Check your internet connection.</div>'
    out = []
    for i, it in enumerate(items):
        out.append(
            f'<a class="card" style="animation-delay:{i * 0.07:.2f}s" '
            f'href="{html.escape(it["link"], quote=True)}" target="_blank" rel="noopener noreferrer">'
            f'<div class="meta"><span class="live"></span>{html.escape(it["source"])} · {ago(it["time"])}</div>'
            f'<div class="head">{html.escape(it["title"])}</div></a>'
        )
    return "".join(out)

def ticker_html(items):
    if not items:
        return ""
    text = "   ///   ".join(html.escape(i["title"]) for i in items)
    return f'<div class="ticker"><div class="track">{text}   ///   {text}</div></div>'

def headlines_context(limit=8):
    lines = []
    for cat in ("world", "finance"):
        for it in get_news(cat, limit):
            lines.append(f"[{cat}] {it['title']} ({it['source']}, {ago(it['time'])})")
    return "\n".join(lines) or "No live headlines could be loaded right now."

NEWS_CSS = """
<style>
.mini {
  border: 1px solid #00e5ff; border-radius: 4px; padding: 12px;
  background: rgba(0,229,255,.04); box-shadow: 0 0 22px rgba(0,229,255,.18);
  position: relative; overflow: hidden; margin: 14px 0;
}
.mini::after {
  content: ""; position: absolute; left: 0; right: 0; height: 40px; top: -40px;
  background: linear-gradient(transparent, rgba(0,229,255,.14), transparent);
  animation: scan 4s linear infinite; pointer-events: none;
}
@keyframes scan { to { top: 100%; } }
.mini-title {
  font-weight: bold; letter-spacing: 3px; color: #e9fdff;
  text-shadow: 0 0 8px #00e5ff; font-size: 14px; margin-bottom: 8px;
}
.live {
  display: inline-block; width: 7px; height: 7px; border-radius: 50%;
  background: #ff005c; margin-right: 6px; animation: blink 1s infinite;
}
@keyframes blink { 50% { opacity: .2; } }
.ticker {
  overflow: hidden; white-space: nowrap; color: #7fdfff; font-size: 13px;
  border-top: 1px solid #00e5ff55; border-bottom: 1px solid #00e5ff55;
  padding: 6px 0; margin-bottom: 10px;
}
.track { display: inline-block; padding-left: 100%; animation: marquee 45s linear infinite; }
@keyframes marquee { to { transform: translateX(-100%); } }
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 10px; }
.card {
  display: block; text-decoration: none; color: #d9fbff;
  border: 1px solid rgba(0,229,255,.35); border-left: 3px solid #00e5ff;
  padding: 9px 10px; border-radius: 4px; background: rgba(0,0,0,.25);
  opacity: 0; animation: slideIn .5s ease forwards;
  transition: transform .15s ease, box-shadow .15s ease;
}
.card:hover { transform: translateY(-2px); box-shadow: 0 0 14px rgba(0,229,255,.4); }
@keyframes slideIn {
  from { opacity: 0; transform: translateY(12px); }
  to { opacity: 1; transform: none; }
}
.meta { font-size: 11px; color: #7fdfff; letter-spacing: 1px; margin-bottom: 5px; }
.head { font-size: 13px; line-height: 1.35; color: #e9fdff; }
.empty { color: #7fdfff; font-size: 13px; }
</style>
"""