from typing import Dict
from pydantic import BaseModel

class QuizQuestion(BaseModel):
    question: str
    options: Dict[str, str]  # A, B, C, D as keys
    correct_answer: str
    explanation: str

class QuizRequest(BaseModel):
    user_input: str
    language:str