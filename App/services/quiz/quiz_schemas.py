from typing import Dict
from pydantic import BaseModel, Field

class QuizQuestion(BaseModel):
    question: str
    options: Dict[str, str]  # A, B, C, D as keys
    correct_answer: str
    explanation: str

class QuizRequest(BaseModel):
    user_input: str = Field(..., description="Topic to generate quiz questions about")
    language: str = Field(
        default="English",
        description="Language in which quiz questions will be generated (e.g. French, German, Japanese)"
    )
