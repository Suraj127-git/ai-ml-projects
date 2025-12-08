from fastapi import FastAPI
from app.schemas import ArticleResponse
from app.model import load_model, ensure_db, ingest, list_articles

app = FastAPI(title="News Aggregator + Summarizer")

@app.on_event("startup")
async def startup():
    load_model()
    ensure_db()

@app.get("/")
def root():
    return {"message": "News Aggregator"}

@app.post("/ingest")
def ingest_api(url: str | None = None):
    ingest(url or "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml")
    return {"status": "ok"}

@app.get("/articles")
def articles(limit: int = 20):
    rows = list_articles(limit)
    return {"articles": [ArticleResponse(title=r[0], summary=r[1], link=r[2]) for r in rows]}
