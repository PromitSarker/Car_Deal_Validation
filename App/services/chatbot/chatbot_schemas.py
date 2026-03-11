from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
    language: str = Field(
        default="English",
        description="Language in which the chatbot should respond (e.g. French, German, Japanese)"
    )
class ChatResponse(BaseModel):
    reply: str
