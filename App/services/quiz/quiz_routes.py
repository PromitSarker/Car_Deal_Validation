import httpx
from fastapi import APIRouter, HTTPException, Query
from App.services.quiz.quiz_schemas import QuizQuestion
from App.core.config import settings
from typing import List
import json 
from enum import Enum
from App.services.quiz.quiz_schemas import QuizQuestion, QuizRequest

router = APIRouter(prefix="/quiz", tags=["quiz"])

class SupportedLanguage(str, Enum):
    english = "English"
    spanish = "Spanish"
    arabic  = "Arabic"
    mandarin = "Mandarin"
    hindi = "Hindi"
     

QUIZ_GENERATION_PROMPT = """
You are an expert car sales trainer. Generate multiple-choice quiz questions to test consumer knowledge about car buying, dealership practices, financing, trade-ins, GAP Logic, VSC Logic, Lease Audit, APR or warranties and others. Always provide new, unique questions that are not commonly found online. 
MAKE SURE YOU PROVIDE THE ANSWERS CORRECTLY.

Respond only with a valid JSON array. Do not use markdown formatting like ```json.

IMPORTANT: You MUST generate exactly 2 quiz questions. 
Return them strictly as a JSON array of length 2, no more and no less. 


Each object in the array should have the following structure:
- question: A string representing the quiz question.
- options: An object with keys "A", "B", "C", and "D", and values being the corresponding answer texts.
- correct_answer: One of the keys "A", "B", "C", or "D", corresponding to the correct option.
- explanation: A detailed at least 5 lined explanation explaining why the correct answer is correct. IN CHILD FRIENDLY ANALOGY.

The entire output should be in the language requested by the user.
Example format:

[
  {
    "question": "What is the purpose of a vehicle history report when buying a used car?",
    "options": {
      "A": "To determine the car's fuel efficiency",
      "B": "To check for past accidents and title issues",
      "C": "To find out the original price of the car",
      "D": "To estimate the car's future resale value"
    },
    "correct_answer": "B",
    "explanation": "A vehicle history report provides important information about a used car, including any past accidents, title issues, service history, and whether the car has been reported as stolen."
  }
]
"""


# Add this at the top, outside the function, as a module-level cache
generated_questions_cache = set()

MAX_RETRIES = 3

@router.post("/generate", response_model=List[QuizQuestion])
async def generate_quiz_questions(
    body: QuizRequest,
    count: int = 2
):
    user_input = body.user_input
    language = body.language

    if not settings.GROQ_API_KEY or not settings.GROQ_URL:
        raise HTTPException(status_code=500, detail="LLM API settings not configured")

    collected = []

    for attempt in range(MAX_RETRIES):
        needed = count - len(collected)
        if needed <= 0:
            break

        # Ask for more than needed to cover duplicates
        requested_count = needed + 3  

        system_prompt = QUIZ_GENERATION_PROMPT + f"\n\nFocus on this topic: {user_input}"
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Generate exactly {requested_count} unique quiz question(s) in {language}. "
                    f"Return only a JSON array of length {requested_count}."
                ),
            },
        ]

        payload = {
            "model": settings.GROQ_MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1500,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    settings.GROQ_URL,
                    headers={
                        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()

                parsed = response.json()
                raw_output = parsed["choices"][0]["message"]["content"]

                cleaned = raw_output.strip().strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[len("json"):].strip()

                data = json.loads(cleaned)
                if isinstance(data, dict):
                    data = [data]

                # Filter new ones into collected
                for q in data:
                    question_text = q.get("question")
                    if question_text and question_text not in generated_questions_cache:
                        generated_questions_cache.add(question_text)
                        collected.append(q)
                        if len(collected) == count:
                            break

        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise HTTPException(status_code=502, detail=f"Quiz generation failed: {e}")

    if len(collected) < count:
        raise HTTPException(
            status_code=500,
            detail=f"Could not generate {count} unique questions after {MAX_RETRIES} attempts."
        )

    # Ensure exactly count
    return [QuizQuestion(**q) for q in collected[:count]]
