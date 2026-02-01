from pydantic import BaseModel, Field
from typing import List, Optional


class ChatRequest(BaseModel):
    message: str
class ChatResponse(BaseModel):
    reply: str