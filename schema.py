from pydantic import BaseModel


class PostGet(BaseModel):
    """
    Схема ответа для рекомендованного поста
    """
    id: int
    text: str
    topic: str

    class Config:
        from_attributes = True