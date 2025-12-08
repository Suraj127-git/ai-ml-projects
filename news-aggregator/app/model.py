import os
import sqlite3
import feedparser
from transformers import pipeline

_sum = None

def db_path():
    base = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(base, "../data/news.db"))

def ensure_db():
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS articles (id INTEGER PRIMARY KEY AUTOINCREMENT, ts DATETIME DEFAULT CURRENT_TIMESTAMP, title TEXT, summary TEXT, link TEXT)")
    conn.commit()
    conn.close()

def load_model():
    global _sum
    _sum = pipeline("summarization")

def ingest(rss_url="https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml"):
    ensure_db()
    d = feedparser.parse(rss_url)
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    for e in d.entries[:10]:
        s = _sum(e.get('summary', ''), max_length=120, min_length=30, do_sample=False)[0]['summary_text']
        cur.execute("INSERT INTO articles (title, summary, link) VALUES (?, ?, ?)", (e.get('title',''), s, e.get('link','')))
    conn.commit()
    conn.close()

def list_articles(limit=20):
    conn = sqlite3.connect(db_path())
    cur = conn.cursor()
    cur.execute("SELECT title, summary, link FROM articles ORDER BY id DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return rows
