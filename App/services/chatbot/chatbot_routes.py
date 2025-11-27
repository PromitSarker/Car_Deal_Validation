import os, httpx
from cachetools import LRUCache
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from App.services.chatbot.chatbot_schemas import ChatRequest, ChatResponse
from App.core.config import settings

router = APIRouter(prefix="/concierge", tags=["concierge"])

MAX_THREADS = 10000
memory: LRUCache[str, List[dict]] = LRUCache(maxsize=MAX_THREADS)

# Buyer scenario detection keywords
SCENARIO_KEYWORDS = {
    "first_time": ["first time", "new buyer", "never bought", "first car"],
    "esl_immigrant": ["esl", "immigrant", "english second", "translate", "not native"],
    "luxury": ["luxury", "premium", "high-end", "bmw", "mercedes", "audi", "lexus"],
    "negative_equity": ["negative equity", "upside down", "owe more", "underwater"],
    "refinance": ["refinance", "refi", "lower payment", "better rate"],
    "lease": ["lease", "leasing", "money factor", "residual", "mileage"]
}

# Dealer tactic detection
DEALER_TACTICS = {
    "expiring_deal": ["expires today", "today only", "limited time"],
    "everyone_buys": ["everyone buys", "most people get", "standard package"],
    "monthly_payment": ["just $", "only $", "per month", "monthly"],
    "financing_requirement": ["only if you finance", "must finance with us", "financing discount"],
    "non_removable_fee": ["can't remove", "mandatory fee", "required fee"],
    "arbitration_clause": ["arbitration", "waive right to sue", "binding arbitration"]
}

# Red flag terms
RED_FLAGS = [
    "processing fee", "admin fee", "doc fee", "protection package required",
    "pre-installed", "dealer markup", "market adjustment", "propack",
    "family guarantee", "finance certificate", "additional fee"
]

# Regional considerations
REGIONAL_KEYWORDS = {
    "california": ["california", "ca", "los angeles", "san francisco", "san diego"],
    "texas": ["texas", "tx", "houston", "dallas", "austin"],
    "canada": ["canada", "ontario", "toronto", "vancouver", "quebec"],
    "eu": ["europe", "eu", "germany", "france", "uk", "spain", "italy"]
}

def detect_buyer_scenario(message: str) -> str:
    """Detect the buyer scenario based on message content"""
    message_lower = message.lower()
    
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        if any(keyword in message_lower for keyword in keywords):
            return scenario
    
    return "standard"

def detect_dealer_tactic(message: str) -> Optional[str]:
    """Detect if user is reporting a dealer tactic"""
    message_lower = message.lower()
    
    for tactic, keywords in DEALER_TACTICS.items():
        if any(keyword in message_lower for keyword in keywords):
            return tactic
    
    return None

def detect_red_flags(message: str) -> List[str]:
    """Detect red flag terms in the message"""
    message_lower = message.lower()
    detected_flags = []
    
    for flag in RED_FLAGS:
        if flag in message_lower:
            detected_flags.append(flag)
    
    return detected_flags

def detect_region(message: str) -> Optional[str]:
    """Detect if user mentions a specific region"""
    message_lower = message.lower()
    
    for region, keywords in REGIONAL_KEYWORDS.items():
        if any(keyword in message_lower for keyword in keywords):
            return region
    
    return None

def build_scenario_context(scenario: str, region: Optional[str] = None) -> str:
    """Build context prompt based on detected scenario"""
    scenario_contexts = {
        "first_time": "The user appears to be a first-time buyer. Provide clear explanations of basic terms like APR, GAP insurance, and add-ons. Be reassuring and educational.",
        "esl_immigrant": "The user may be non-native English speaker. Use simple, clear language. Avoid complex financial jargon. Be culturally sensitive.",
        "luxury": "The user is discussing luxury/high-ticket items. Focus on value framing rather than just price. Highlight prestige aspects but caution against unfair upsells.",
        "negative_equity": "The user has negative equity concerns. Focus on mathematical clarity about payoffs and trade-in realities. Warn against rolling extra debt into new loans.",
        "refinance": "The user is interested in refinancing. Provide side-by-side comparisons of total loan costs. Watch for dealer refinancing gimmicks.",
        "lease": "The user is considering leasing. Explain residual values, money factors, and mileage limits. Watch for mandatory wear-and-tear add-ons.",
        "standard": "The user is a standard buyer. Provide balanced advice that protects their interests while maintaining reasonable negotiation stance."
    }
    
    base_context = scenario_contexts.get(scenario, scenario_contexts["standard"])
    
    # Add regional considerations
    if region == "california":
        base_context += " Note: California has banned certain add-ons and requires full price disclosure."
    elif region == "texas":
        base_context += " Watch for 'finance certificates' which are common in Texas - these often hide higher rates."
    elif region == "canada":
        base_context += " Note: Canadian provinces have stricter rules on deposits and all-in pricing requirements."
    elif region == "eu":
        base_context += " Note: EU markets have strong consumer protection laws requiring plain-language contracts."
    
    return base_context

@router.post("", response_model=ChatResponse)
async def concierge(
    req: ChatRequest,
    thread_id: str = Query(..., description="unique conversation id")
):
    api_key = settings.GROQ_API_KEY
    if not api_key:
        raise HTTPException(500, "GROQ_API_KEY not set")

    # Check if user is asking for an explanation
    explanation_keywords = ["what is", "meaning of", "explain", "define"]
    is_explanation_request = any(keyword in req.message.lower() for keyword in explanation_keywords)
    
    if is_explanation_request:
        # Handle explanation requests
        base_prompt = (
            "You are a negotiation expert. The user is asking for an explanation of a negotiation term or concept. "
            "Provide a child-like, clear, concise definition and a practical example of how it's used in negotiations. "
            "Keep your response focused and educational."
        )
        messages = [{"role": "system", "content": base_prompt}]
        messages.append({"role": "user", "content": req.message})
    else:
        # Enhanced roleplay mode with scenario detection
        scenario = detect_buyer_scenario(req.message)
        region = detect_region(req.message)
        dealer_tactic = detect_dealer_tactic(req.message)
        red_flags = detect_red_flags(req.message)
        
        scenario_context = build_scenario_context(scenario, region)
        
        base_prompt = (
            "You are playing the role of a dealer in a negotiation scenario. "
            "Your name is SmartDealer. You are negotiating a business deal, selling, or service agreement. "
            "Stay in character and respond as a real dealer would, but incorporate subtle coaching elements. "
            "Be resistant but slightly reasonable. Push back on prices and terms but be open to compromise. "
            f"{scenario_context} "
            "Your responses should indirectly teach negotiation techniques through the conversation. "
            "Keep responses concise (1-2 sentences typically)."
        )
        
        # Add specific guidance if dealer tactic is detected
        if dealer_tactic:
            tactic_responses = {
                "expiring_deal": "Subtly suggest that good deals don't need artificial time pressure.",
                "everyone_buys": "Hint that popularity doesn't equal necessity - encourage itemized review.",
                "monthly_payment": "Gently steer conversation toward total cost rather than monthly payments.",
                "financing_requirement": "Suggest comparing the 'discount' against potentially higher rates.",
                "non_removable_fee": "Encourage asking for legal justification of mandatory fees.",
                "arbitration_clause": "Hint at the importance of understanding dispute resolution options."
            }
            base_prompt += f" The user seems to be encountering a '{dealer_tactic}' tactic. {tactic_responses.get(dealer_tactic, '')}"
        
        # Add red flag warnings if detected
        if red_flags:
            base_prompt += f" The user mentioned these potential red flags: {', '.join(red_flags)}. Gently alert them to question these items."
        
        # Server-side memory
        history = memory.get(thread_id, [])
        if not history:
            history = [{"role": "system", "content": base_prompt}]
            # Start with scenario-appropriate opening
            opening_lines = {
                "first_time": "Hello! I see you're new to this process. I'm here to help you understand your options.",
                "esl_immigrant": "Welcome! I'll make sure we communicate clearly about your needs.",
                "luxury": "Good day! I specialize in premium vehicles and ensuring you get exceptional value.",
                "negative_equity": "I understand you have concerns about your current vehicle's value. Let's discuss options.",
                "refinance": "Interested in refinancing? Let me show you how we can potentially improve your terms.",
                "lease": "Leasing can be a great option! Let me explain how it works for your situation.",
                "standard": "Hello, I'm SmartDealer. I noticed you're interested in our offerings. How can I assist today?"
            }
            history.append({"role": "assistant", "content": opening_lines.get(scenario, opening_lines["standard"])})
        
        # Update system prompt if scenario has changed
        if history and history[0]["role"] == "system":
            history[0]["content"] = base_prompt
        
        history.append({"role": "user", "content": req.message})
        messages = history

    payload = {
        "model": settings.GROQ_MODEL,
        "messages": messages,
        "max_tokens": 550,  # Slightly increased for more nuanced responses
        "temperature": 0.7 if not is_explanation_request else 0.3
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            r = await client.post(
                settings.GROQ_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            r.raise_for_status()
            reply = r.json()["choices"][0]["message"]["content"].strip()

            # Save to memory if it's a roleplay conversation
            if not is_explanation_request:
                history.append({"role": "assistant", "content": reply})
                memory[thread_id] = history
                
            return ChatResponse(reply=reply)
        except Exception as e:
            raise HTTPException(502, f"Groq error: {e}")