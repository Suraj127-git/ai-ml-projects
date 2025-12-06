from pydantic import BaseModel

class ArticleResponse(BaseModel):
    title: str
    summary: str
    link: str
