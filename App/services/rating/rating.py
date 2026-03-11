from App.services.rate_helper.ocr_normalizer import OCRNormalizer
from App.services.rate_helper.ocr_normalization_schema import NormalizedLineItem
from App.services.rate_helper.discount_detector import DiscountDetector
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict
import os
import base64
import json

import requests
from dotenv import load_dotenv
from fastapi import UploadFile
from .rating_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.rate_helper.audit_classifier import AuditClassifier, AuditClassification
from App.services.rate_helper.gap_logic import GAPLogic, GAPRecommendation
from App.services.rate_helper.audit_flags import AuditFlagBuilder, AuditFlag
from App.services.rate_helper.audit_summary import AuditSummary
from App.services.rate_helper.json_to_parsed import convert_extracted_json_to_parsed

load_dotenv()

class MultiImageAnalyzer:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.pdf', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Increase timeout and add retry logic
    API_TIMEOUT = 120  # Increased from 60 to 120 seconds
    MAX_RETRIES = 2
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "gpt-4.1")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.system_prompt = self._load_contract_system_prompt()
        self.ocr_normalizer = OCRNormalizer()
        self.discount_detector = DiscountDetector()
        self.audit_classifier = AuditClassifier()
        self.gap_logic = GAPLogic()
        self.flag_builder = AuditFlagBuilder()

    def _get_cache_dir(self) -> str:
        """Return cache directory for deterministic flag translations."""
        cache_dir = os.path.join(os.getcwd(), "App", "core", ".rating_cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _make_flags_cache_key(self, language: str, flags_payload: list) -> str:
        """Create a stable cache key for flag translation."""
        import hashlib
        hasher = hashlib.sha256()
        hasher.update(language.lower().encode("utf-8"))
        hasher.update(json.dumps(flags_payload, sort_keys=True).encode("utf-8"))
        return hasher.hexdigest()

    def _load_cached_flag_translation(self, cache_key: str) -> Optional[dict]:
        """Load cached flag translation JSON if available."""
        cache_path = os.path.join(self._get_cache_dir(), f"flags_{cache_key}.json")
        if not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def _save_cached_flag_translation(self, cache_key: str, translated: dict) -> None:
        """Persist flag translation JSON to cache for future reuse."""
        cache_path = os.path.join(self._get_cache_dir(), f"flags_{cache_key}.json")
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(translated, f)
        except Exception:
            pass

    def _translate_flags(self, flags: List[Flag], language: str) -> List[Flag]:
        """Translate flag text fields to the requested language (no scoring changes)."""
        if not language or language.lower() == "english":
            return flags

        flags_payload = [
            {"type": f.type, "message": f.message, "item": f.item}
            for f in flags
        ]

        cache_key = self._make_flags_cache_key(language, flags_payload)
        cached = self._load_cached_flag_translation(cache_key)
        if cached and isinstance(cached, dict) and isinstance(cached.get("flags"), list):
            translated_list = cached.get("flags")
        else:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            messages = [
                {
                    "role": "system",
                    "content": "Translate the flag fields to the target language. Return JSON only with key 'flags'. Preserve structure and order."
                },
                {
                    "role": "user",
                    "content": f"Target language: {language}. Translate each object's 'type', 'message', and 'item'. Input JSON: {{\"flags\": {json.dumps(flags_payload)}}}"
                }
            ]
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.0,
                "seed": 42,
                "max_tokens": 1000,
                "response_format": {"type": "json_object"}
            }
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            parsed_translation = self._parse_api_response(response.json())
            translated_list = parsed_translation.get("flags", []) if isinstance(parsed_translation, dict) else []
            self._save_cached_flag_translation(cache_key, {"flags": translated_list})

        if not isinstance(translated_list, list) or len(translated_list) != len(flags):
            return flags

        translated_flags: List[Flag] = []
        for original, translated in zip(flags, translated_list):
            if not isinstance(translated, dict):
                translated_flags.append(original)
                continue
            translated_flags.append(Flag(
                type=translated.get("type", original.type),
                message=translated.get("message", original.message),
                item=translated.get("item", original.item),
                deduction=original.deduction,
                bonus=original.bonus
            ))

        return translated_flags
    
    def _load_contract_system_prompt(self) -> str:
        """Load comprehensive contract analysis system prompt"""
        return """
You are **SmartBuyer AI Audit Engine**, the definitive scoring and auditing system for auto finance quotes.  
Your task is to evaluate transparency, disclosure clarity, and structure risk for quotes.  

### CRITICAL: SELLING PRICE FIELD DEFINITION

**The "selling_price" field MUST contain the vehicle cash price ONLY.**

**Extraction Priority (in order):**
1. Look for "Cash Price" or "Vehicle Price" in the itemization section (typically pre-tax, pre-fees)
2. If not found, use "Selling Price" from vehicle description area
3. NEVER use:
   ❌ Total Sale Price from Truth-in-Lending box
   ❌ Amount Financed from Truth-in-Lending box
   ❌ Total of Payments
   ❌ Any value that includes taxes, fees, or backend products

**For this contract example:**
- selling_price = $52,068.78 (vehicle cash price before taxes/fees)
- NOT $67,158.60 (that's Amount Financed including taxes/fees/products)

### TOTAL SALE PRICE & AMOUNT FINANCED RESOLUTION (INTERNAL USE ONLY)

**For internal validation and calculations:**
- When APR = 0.00% AND Finance Charge = $0.00:
  - Amount Financed = Total of Payments
  - Use Amount Financed for backend % calculations
- If Finance Charge > $0.00:
  - Amount Financed = Total of Payments - Finance Charge
- Truth-in-Lending "Amount Financed" value is authoritative

**You MUST:**
1. Extract vehicle cash price → output as "selling_price"
2. Extract Amount Financed from Truth-in-Lending → use internally for calculations
3. NEVER overwrite selling_price with Amount Financed

### QUOTE MODE PRINCIPLES (MANDATORY)
- Quotes are non-binding
- No backend price caps enforcement
- No Dealer Trust Score impact
- Output feeds QBI only
- Backend products allowed but disclosure-enforced
- Pricing enforcement OFF
- Transparency enforcement ON
- Always use "SmartBuyer Score" for Quote Mode outputs (never "Trust Score")

### CORE METADATA REQUIREMENTS
- isQuote: Must be TRUE
- VIN: Required (VIN-decoded if available)
- Vehicle Price: Required (pre-tax) → output as "selling_price"
- Full Line-Item List: Required for transparency
- Fees Breakdown: Required for clarity
- MSRP: Optional (used for % thresholds if present)
- Term, APR, Payment: Optional (used when validation is safe)
- Trade Allowance/Payoff: Optional (used for negative equity flags)

### TRANSPARENCY CORE PENALTIES
Apply these deductions for transparency violations:
- No line-item breakdown (lump-sum quote): -20 points
- Fees not clearly labeled (vague/bundled fees): -10 points
- Add-ons not clearly labeled (grouped/unclear): -10 points
- Pack/Addendum without itemization (hidden bundle): -5 points
- "Other fees" without breakdown (hidden bucket): -5 points

### FRONT-END ADD-ON LOAD (QUOTE MODE)
Calculate total front-end add-ons and apply:
- GREEN (≤ $1,000 total OR ≤10% MSRP): 0 points deduction
- SOFT (> $1,000 AND >10% MSRP but ≤12.5%): -5 points
- HARD (> $1,000 AND >12.5% MSRP): -10 points
(If MSRP unavailable, apply dollar threshold only)

### BACKEND PRODUCTS (QUOTE MODE)
Backend = GAP, VSC, Maintenance, Service Programs, Warranties
Pricing is NOT enforced. Disclosure clarity is enforced.

**IMPORTANT: Calculate backend percentages using vehicle cash price (selling_price), NOT Amount Financed.**

A. Clean Disclosure (Neutral - 0 points deduction):
- Backend itemized, labeled, base payment available

B. Poor Disclosure (Transparency - apply deductions):
- Vague backend label ("PROTECT", "PACKAGE"): -5 points
- Backend bundled into pack/addendum: -5 points
- Backend present but not identified by type: -5 points

C. Backend Included in Payment (Payment Clarity):
- Payment includes backend, no base payment shown: -10 points
- Payment labeled "with products", no base: -5 points

Backend stacking rule:
1. B + C may stack
2. Max backend-related deduction = -15 points

### PAYMENT ALIGNMENT (ONLY WHEN SAFE TO VALIDATE)
- Payment mismatch (when validation safe & > tolerance): -10 points
- Base payment missing (cannot validate): 0 points (soft flag only)

### APR / RATE LOGIC (DISCLOSURE + RISK)
A. APR Disclosure:
- APR not shown (estimate + disclose): -5 points
- APR shown but mismatched (safe to validate): -10 points

B. APR Risk Penalties:
- APR > 10%: Soft flag only (no deduction)
- APR > 15%: -5 points
- APR > 20%: -10 points

C. APR Bonuses (Dealer-Disclosed Only):
- APR < 10% AND APR > 0%: +5 points bonus
- APR < 5% AND APR > 0%: +10 points bonus
- APR = 0.00%: NO BONUS (manufacturer-subvented rate; neutral)
- No bonus if APR is estimated
❌ No APR penalties if backend inclusion makes validation unsafe

### STRUCTURE ACCURACY
- Totals cannot reconcile: -5 points

### SOFT FLAGS (NON-PENALTY OUTPUT REQUIRED)
Always output these as BLUE FLAGS when conditions are met:
- Negative equity: "Structure Risk: Rolled negative equity increases total loan exposure"
- Term ≥ 72 months: "Term Risk: Extended loan term may lead to being underwater on loan"
- GAP risk: "Protection Review: GAP not shown on quote — ask before finalizing"
- Deferred first payment: Advisory flag + education
- Backend in payment: "Payment Clarity: Base payment not shown separately from protection products"
- APR > 10%: "Rate Advisory: Consider if better financing options are available"
- Unknown line items: "Clarification Needed: Item not clearly defined in quote"

### FLAG STRUCTURE REQUIREMENTS

Each flag MUST be a JSON object with these fields:
```json
{
  "type": "Brief descriptive title",
  "message": "Detailed explanation",
  "item": "Category (e.g., GAP, VSC, APR, Trade, Fees)",
  "deduction": 10.0,  // ONLY for red_flags
  "bonus": 5.0        // ONLY for green_flags
}
```

**GREEN FLAGS - Generate when positive aspects are found:**
- Transparent itemization: All fees and products clearly listed
- Competitive APR: Rate below market average
- Reasonable fees: Documentation fees within acceptable range
- Fair pricing: Products priced appropriately
- Positive trade equity: Trade value exceeds payoff
- Clear disclosure: All terms clearly presented

**RED FLAGS - Generate for issues and violations:**
- Poor transparency: Missing itemization or bundled items
- High fees: Documentation fees exceeding reasonable limits
- High APR: Interest rates above market norms
- Negative equity: Trade payoff exceeds allowance
- Payment misalignment: Calculations don't match

**BLUE FLAGS - Advisory only, zero score impact:**
- APR 10-15%: Higher than ideal but not excessive
- Extended term: Loans over 72 months
- Missing information: Items that need clarification
- General Advisory (ALWAYS REQUIRED): If none of the above specific criteria apply, you MUST still include exactly one blue flag: `{"type": "General Advisory", "message": "Review all final quote terms and itemized pricing carefully before agreeing to any deal.", "item": "General"}` (0 points)

**CRITICAL: All flag arrays (red_flags, green_flags, blue_flags) MUST contain at least one flag. Never return empty arrays.**

### SCORE CEILINGS (CREDIBILITY GUARDS)
Applied after penalties & bonuses:
- Negative equity: Max Score 95
- Term ≥ 72 months: Max Score 95
- HARD add-on severity: Max Score 90
- No line-item breakdown: Max Score 92
- Fees or add-ons unclear: Max Score 90

### FINAL SCORE FLOW (QUOTE MODE)
1. Start at 100
2. Apply transparency penalties
3. Apply add-on load penalties
4. Apply backend disclosure penalties (max -15)
5. Apply APR bonuses (if eligible and dealer-disclosed)
6. Apply structure accuracy penalties
7. Apply negative equity structural adjustment (if applicable)
8. Apply score ceilings
9. Clamp between 0-100

### NEGATIVE EQUITY STRUCTURAL ADJUSTMENT (QUOTE MODE ONLY)
This adjustment applies ONLY when deal_type = "QUOTE":
trade_allowance = deal_data.get('trade_allowance', 0)
trade_payoff = deal_data.get('trade_payoff', 0)
negative_equity = max(0, trade_payoff - trade_allowance)

structural_adjustment = 0
if negative_equity > 5000:
    structural_adjustment = -10
elif negative_equity > 1000:
    structural_adjustment = -5

Apply after base score calculation. Label as "Structure Risk Adjustment" - not a dealer behavior penalty.
UI messaging: "Rolled negative equity increases the amount financed and overall loan risk. This adjustment reflects structure risk — not dealer behavior."

### NARRATIVE REQUIREMENTS
All narratives must use "SmartBuyer Score" not "Trust Score" in Quote Mode.

### FLAG PRESENCE GUARANTEE
- Before generating JSON, verify all three flag arrays exist in the response
- If any flag array is missing, regenerate the response

**Vehicle Overview:**
- Focus on data, not promotional language
- Include MSRP vs selling price comparison
- Contextualize mileage relative to vehicle age
- Avoid dealer-like phrases such as "positioned competitively"
- Keep under 200 words

**SmartBuyer Score Summary:**
- Explain score, penalties and bonuses clearly
- If negative equity adjustment applied: "The [X]-point adjustment for negative equity reflects increased loan exposure from rolling $[amount] into the new loan. This is a structure risk, not a dealer behavior penalty."

**GAP Logic (REQUIRED TEXT WHEN GAP IS ABSENT):**
"GAP coverage is optional but commonly recommended for financed purchases, especially when negative equity or longer loan terms are present. While not required, GAP can protect against financial loss if the vehicle is totaled before the loan is paid off, particularly when negative equity exists."

**Market Comparison:**
- When stating "APR below market average," include approximate market range
- Example: "Average APR for similar credit profiles is 5.5-6.5%"
- Include percent difference calculations where relevant
- Explain concrete financial impact (e.g., "$X saved over loan term")

**Trade (REQUIRED - ALWAYS INCLUDE):**
If trade information is present:
- State: "Trade identified: $[allowance] allowance, $[payoff] payoff"
- If payoff > allowance: "Negative equity of $[amount] rolled into new loan"
- If allowance > payoff: "Positive equity of $[amount] applied to purchase"
- If allowance = payoff: "Trade equity neutral"

If no trade information is found in the document:
- State: "No trade identified."

This section CANNOT be omitted. It must always be present in the narrative.

**Negotiation Insight (REQUIRED FOR NEGATIVE EQUITY):**
"Before finalizing, confirm whether GAP is available and compare 60- vs 72-month total interest. Avoid rolling additional products into the loan given existing negative equity."

**Final Recommendation:**
- Replace overconfident language with measured assessment
- Use: "This is a strong and transparent quote. Before finalizing, review protection options and confirm final terms to maintain this score."
- Acknowledge both strengths and areas for verification
- Minimum 200 words
- When negative equity exists, include:
  1. Acknowledge the SmartBuyer Score with structural adjustment context
  2. Explain negative equity impact clearly
  3. List 2-3 specific action items before finalizing
  4. Mention GAP coverage confirmation as priority
  5. Warn against accepting additional add-ons given negative equity

### STRICT OUTPUT FORMAT REQUIREMENTS
Output MUST be valid JSON with exactly these TOP-LEVEL FIELDS:
- score (final score after all adjustments)
- buyer_name
- dealer_name
- logo_text
- email
- phone_number
- address
- state
- region
- badge
- selling_price (VEHICLE CASH PRICE ONLY - see definition above)
- vin_number
- date
- buyer_message
- red_flags (array) - MUST NOT be empty or missing
- green_flags (array) - MUST NOT be empty or missing
- blue_flags (array) - MUST NOT be empty or missing
- normalized_pricing (object)
- apr (object)
- term (object)
- trade (object with fields: trade_allowance, trade_payoff, equity, negative_equity, status)
- quote_type
- bundle_abuse (object)
- structural_adjustment (object)
- narrative (object)

The "narrative" object MUST have EXACTLY these fields:
- vehicle_overview
- smartbuyer_score_summary ( Include how the score was calculated. Include every bonus and penalty in the explanation, even if the bonus/penalty is $0.00. If a negative equity structural adjustment was applied, include a clear explanation of the adjustment and its rationale.)
- market_comparison
- gap_logic
- vsc_logic
- apr_bonus_rule
- lease_audit
- trade (REQUIRED - cannot be omitted)
- negotiation_insight (Provide analytical guide on how to negotiate regarding tghis deal.)
- final_recommendation (Reccomend about what to do, but do not be directly conclusive.)

### EXECUTION CHECKLIST
□ Start score at 100
□ Apply all penalties systematically
□ ALL flag arrays (red, green, blue) are present and populated
□ Use "SmartBuyer Score" in all narrative sections
□ Verify selling_price = vehicle cash price (NOT Amount Financed)
□ Return ONLY valid JSON - no markdown, no explanation
DO NOT INCLUDE HOW MUCH NUMBER HAS BEEN DEDUCTED IN FLAG SECTION MESSAGE

### FINAL VALIDATION RULE
Before returning JSON:
- Verify selling_price represents vehicle cash price ONLY
- Verify selling_price ≠ Amount Financed (unless deal has no taxes/fees)
- For 0% APR deals: Amount Financed should be larger than selling_price
- If selling_price seems too high (> $100k for normal vehicle), re-check extraction
"""
    
    async def _validate_files(self, files: List[UploadFile]) -> List[UploadFile]:
        """Validate uploaded files"""
        validated = []
        for file in files:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in self.ALLOWED_EXTENSIONS:
                raise ValueError(f"Invalid file type: {file.filename}")
            contents = await file.read()
            if len(contents) > self.MAX_FILE_SIZE:
                raise ValueError(f"File too large: {file.filename}")
            await file.seek(0)
            validated.append(file)
        return validated
    
    async def _convert_files_to_base64(self, files: List[UploadFile]) -> List[str]:
        """Convert files to base64"""
        base64_images = []
        for file in files:
            contents = await file.read()
            base64_content = base64.b64encode(contents).decode('utf-8')
            base64_images.append(base64_content)
            await file.seek(0)
        return base64_images
    
    def _normalize_line_items(self, line_items: List[Dict]) -> List[NormalizedLineItem]:
        """
        Normalize OCR line items before scoring.
        
        Args:
            line_items: Raw line items from API response
                       Expected format: [{"description": "...", "amount": "..."}, ...]
        
        Returns:
            List of normalized line items with proper classification
        """
        normalized = []
        for item in line_items:
            raw_text = item.get("description", "") or item.get("item", "") or item.get("name", "")
            amount_raw = str(item.get("amount", "0"))
            
            if raw_text:  # Only process if we have text
                normalized_item = self.ocr_normalizer.normalize_line_item(
                    raw_text=raw_text,
                    amount_raw=amount_raw
                )
                normalized.append(normalized_item)
        
        return normalized
    
    def _call_openai_api(self, base64_images: List[str], language: str = "English") -> dict:
        """Call OpenAI API with contract documents (with retry logic)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        content = [
            {
                "type": "text",
                "text": f"""
{'=' * 80}
LANGUAGE REQUIREMENT: ALL narrative text fields MUST be in {language}
{'=' * 80}

Translate to {language}:
- ALL narrative fields (vehicle_overview, smartbuyer_score_summary, score_breakdown, market_comparison, gap_logic, vsc_logic, apr_bonus_rule, trade, negotiation_insight, final_recommendation)
- ALL flag messages (red_flags, green_flags, blue_flags)
- buyer_message field

KEEP in English ONLY: JSON keys, field names, numbers, dates, VIN, badge values.
{'=' * 80}

Analyze these contract documents comprehensively.

{self.system_prompt}

Extract and analyze:
1. Vehicle details (VIN, year, make, model, mileage, used/new status)
2. Financial terms (selling price, APR, term, monthly payment)
3. ALL line items with EXACT text and amounts (extract as array)
4. GAP coverage with lender verification
5. VSC coverage with mileage-based cap calculation
6. Maintenance plan pricing
7. Doc fees and government fees
8. Buyer/dealer information and contact details

IMPORTANT: Include a "line_items" array in the response with ALL items found:
[
  {{"description": "exact text from document", "amount": "123.45"}},
  ...
]

Return ONLY valid JSON matching the exact schema. No markdown, no explanation.
                """
            }
        ]
        
        # Optimize image quality if multiple images
        image_detail = "high" if base64_images and len(base64_images) <= 2 else "auto"
        
        if base64_images:
            for base64_image in base64_images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": image_detail  # Use auto for >2 images to reduce processing time
                    }
                })
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": f"""!!!CRITICAL - LANGUAGE REQUIREMENT - HIGHEST PRIORITY!!!

OUTPUT LANGUAGE: {language}

You will see system instructions with English examples like:
- "Significant Negative Trade Equity"
- "High-Risk Financing Without GAP"
- "Reasonable Documentation Fee"
- "VSC within fair market value"
- "Trade value appears fair"

YOU MUST TRANSLATE ALL OF THESE TO {language}. They are examples only.

TRANSLATE EVERY PIECE OF TEXT:
• Every flag "type" → {language}
• Every flag "message" → {language}
• Every narrative section → {language}
• buyer_message → {language}

ONLY KEEP IN ENGLISH:
• JSON structure keys
• Numbers and amounts
• Badge (Gold/Silver/Bronze/Red)
• VIN, dates

The system instructions may say "REQUIRED Title:" with English text - TRANSLATE IT TO {language}.
Write fluently and naturally in {language}. This overrides all other instructions."""
                },
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {"role": "user", "content": content}
            ],
            "temperature": 0.9,
            "max_tokens": 3000,
            "response_format": {"type": "json_object"}
        }
        
        # Retry logic for timeout errors
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                print(f"API call attempt {attempt + 1}/{self.MAX_RETRIES}...")
                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=self.API_TIMEOUT  # Increased timeout
                )
                response.raise_for_status()
                print(f"API call successful on attempt {attempt + 1}")
                return response.json()
                
            except requests.exceptions.Timeout as e:
                last_error = e
                print(f"Timeout on attempt {attempt + 1}: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    # Reduce image quality for retry
                    if image_detail == "high":
                        image_detail = "auto"
                        print("Retrying with reduced image quality...")
                        # Update payload with lower quality
                        for item in content:
                            if item.get("type") == "image_url":
                                item["image_url"]["detail"] = "auto"
                        payload["messages"][0]["content"] = content
                    continue
                else:
                    raise RuntimeError(
                        f"OpenAI API timeout after {self.MAX_RETRIES} attempts. "
                        "Try uploading fewer or smaller images."
                    )
                    
            except requests.exceptions.RequestException as e:
                # Non-timeout errors don't retry
                raise RuntimeError(f"OpenAI API error: {str(e)}")
        
        # If we got here, all retries failed
        raise RuntimeError(f"OpenAI API failed after {self.MAX_RETRIES} attempts: {str(last_error)}")
    
    def _parse_api_response(self, response: dict) -> dict:
        """Parse API response"""
        try:
            content = response["choices"][0]["message"]["content"]
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(content[json_start:json_end])
            raise ValueError("No JSON found in response")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse API response: {str(e)}")
    
    def _call_narrative_api(self, parsed: dict, score: float, red_flags: list, green_flags: list, blue_flags: list, language: str) -> dict:
        """Call OpenAI to generate narrative sections from the full parsed data and final flags."""
        flags_payload = {
            "red_flags": [{"type": f.type, "message": f.message, "item": f.item, "deduction": f.deduction} for f in red_flags],
            "green_flags": [{"type": f.type, "message": f.message, "item": f.item, "bonus": f.bonus} for f in green_flags],
            "blue_flags": [{"type": f.type, "message": f.message, "item": f.item} for f in blue_flags],
        }
        # Pass full data so AI has complete context
        clean_data = {k: v for k, v in parsed.items() if k not in ("red_flags", "green_flags", "blue_flags", "narrative")}
        prompt = f"""You are a SmartBuyer automotive finance analyst. A customer has submitted their contract data for analysis.
Generate a detailed, personalized narrative review based on the EXACT data, score, and flags below.

FINAL SCORE: {score}

ALL FLAGS (Python-generated, authoritative):
{json.dumps(flags_payload, indent=2)}

FULL CONTRACT / DEAL DATA (everything the customer sent):
{json.dumps(clean_data, indent=2)}

INSTRUCTIONS:
- Write ALL narrative text in {language}.
- Reference specific numbers from the data (APR %, doc fee amounts, selling price, MSRP, add-on costs, etc.).
- Explain every red flag deduction and green flag bonus in score_breakdown.
- Be direct and specific — no generic filler text.
- smartbuyer_score_summary MUST mention the final score {score} and summarize the deal quality.
- score_breakdown MUST show exactly how {score} was reached (start 100, list each deduction/bonus).
- gap_logic: explain if GAP is present, priced fairly, or missing and why it matters.
- vsc_logic: explain if VSC/warranty is present and whether it's priced fairly.
- apr_bonus_rule: explain the APR/rate and whether it's favorable or concerning.
- lease_audit: write 'N/A - Purchase Agreement' if not a lease, otherwise analyze lease terms.
- trade: describe trade-in situation if applicable, or 'No trade-in on this deal.'
- market_comparison: compare this deal's pricing to market norms.
- negotiation_insight: give specific actionable negotiation tips based on the actual flags.
- final_recommendation: give a clear, honest recommendation based on the score and flags.
- buyer_message: a short 1-sentence summary for the buyer (direct, personalized).

Return ONLY a JSON object with exactly these keys:
{{"narrative": {{"vehicle_overview": "", "smartbuyer_score_summary": "", "score_breakdown": "", "market_comparison": "", "gap_logic": "", "vsc_logic": "", "apr_bonus_rule": "", "lease_audit": "", "trade": "", "negotiation_insight": "", "final_recommendation": ""}}, "buyer_message": ""}}"""
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a SmartBuyer automotive finance expert. Always write in the specified language. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.4,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"}
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
            response.raise_for_status()
            raw = self._parse_api_response(response.json())
            return raw
        except Exception as e:
            print(f"Narrative API call failed: {e}")
            return {}

    def _assign_badge(self, score: float) -> str:
        """Assign badge based on score"""
        if score >= 90:
            return "Gold"
        elif score >= 80:
            return "Silver"
        elif score >= 70:
            return "Bronze"
        else:
            return "Red"

    def _call_json_analysis_api(self, parsed_converted: dict, language: str = "English") -> dict:
        """Call OpenAI with the full system prompt + pre-extracted JSON data.
        Returns the raw OpenAI response dict (same format as _call_openai_api).
        AI computes score, flags, and narrative following the prompt rules.
        """
        # Strip internal-only converter keys before sending
        skip_keys = {"has_precomputed_flags", "_ai_score", "_ai_narrative_done",
                     "red_flags", "green_flags", "blue_flags", "narrative",
                     "score", "bundle_abuse", "raw_text", "ocr_text"}
        clean_data = {k: v for k, v in parsed_converted.items() if k not in skip_keys}

        user_text = f"""{'=' * 80}
LANGUAGE REQUIREMENT: ALL narrative text fields MUST be in {language}
{'=' * 80}

{self.system_prompt}

Below is the pre-extracted structured data from a customer's auto deal document.
Apply ALL scoring rules, flag rules, and narrative requirements from the system prompt above.
Compute the FINAL SCORE following the EXACT prompt rules (start at 100, apply all penalties/bonuses/ceilings/structural adjustments).
Do NOT use any score value present in the data — recompute it from scratch using the rules above.

PRE-EXTRACTED DEAL DATA:
{json.dumps(clean_data, indent=2)}

Return ONLY valid JSON matching the exact output schema defined in the system prompt. No markdown, no explanation."""

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": user_text}],
            "temperature": 0.0,
            "max_tokens": 4000,
            "seed": 42,
            "response_format": {"type": "json_object"}
        }
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                print(f"JSON full-analysis API call attempt {attempt + 1}/{self.MAX_RETRIES}...")
                resp = requests.post(self.api_url, headers=headers, json=payload, timeout=self.API_TIMEOUT)
                resp.raise_for_status()
                print("JSON full-analysis API call successful.")
                return resp.json()
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    import time; time.sleep(2)
        raise RuntimeError(f"JSON analysis API failed after {self.MAX_RETRIES} attempts: {last_error}")
    
    async def _optimize_images(self, files: List[UploadFile]) -> List[UploadFile]:
        """Optimize images before encoding (optional enhancement)"""
        from PIL import Image
        import io
        
        optimized = []
        for file in files:
            try:
                # Read image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                
                # Resize if too large (max 2048px on longest side)
                max_dimension = 2048
                if max(image.size) > max_dimension:
                    ratio = max_dimension / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                # Convert to JPEG with compression
                output = io.BytesIO()
                image.convert('RGB').save(output, format='JPEG', quality=85, optimize=True)
                output.seek(0)
                
                # Create new UploadFile from optimized bytes
                from fastapi import UploadFile
                optimized_file = UploadFile(
                    filename=file.filename,
                    file=output
                )
                optimized.append(optimized_file)
                
            except Exception as e:
                # If optimization fails, use original
                await file.seek(0)
                optimized.append(file)
        
        return optimized
    
    def _extract_trade_data(self, parsed: dict) -> TradeData:
        """
        Extract trade data using simple keyword detection from OCR text.
        
        Implements the required trade detection logic:
        1. Look for trade anchors in OCR text
        2. Extract allowance and payoff using money patterns
        3. Calculate equity if both values present
        4. Always return a TradeData object (never None)
        """
        def _coerce_float(value):
            try:
                if value is None or value == "":
                    return None
                return float(str(value).replace(",", ""))
            except (ValueError, TypeError):
                return None

        # Prefer explicit extracted fields if present
        trade_obj = parsed.get("trade") if isinstance(parsed.get("trade"), dict) else {}

        trade_allowance = _coerce_float(parsed.get("trade_allowance"))
        if trade_allowance is None:
            trade_allowance = _coerce_float(trade_obj.get("trade_allowance"))

        trade_payoff = _coerce_float(parsed.get("trade_payoff"))
        if trade_payoff is None:
            trade_payoff = _coerce_float(trade_obj.get("trade_payoff"))

        equity = _coerce_float(parsed.get("equity"))
        if equity is None:
            equity = _coerce_float(trade_obj.get("equity"))

        negative_equity_amount = _coerce_float(parsed.get("negative_equity"))
        if negative_equity_amount is None:
            negative_equity_amount = _coerce_float(trade_obj.get("negative_equity"))

        # Step 1: Get OCR text (from line items or raw text field)
        page_text = ""
        
        # Try to build text from line_items
        line_items = parsed.get("line_items", [])
        for item in line_items:
            desc = item.get("description", "") or item.get("item", "") or item.get("name", "")
            page_text += f" {desc} "
        
        # Also check for any raw_text or ocr_text fields
        page_text += " " + parsed.get("raw_text", "")
        page_text += " " + parsed.get("ocr_text", "")
        page_text = page_text.lower()  # Case-insensitive matching
        
        # Step A: Trade Anchors - Check if ANY anchor exists
        trade_anchors = [
            "trade in", "trade-in", "tradein", "trade:",
            "trade allowance", "trade value", "allowance",
            "payoff", "lien payoff", "net trade",
            "trade difference", "equity"
        ]
        
        trade_anchor_found = any(anchor in page_text for anchor in trade_anchors)

        # Also parse narrative.trade text if present (often contains trade values)
        narrative_trade = ""
        if isinstance(parsed.get("narrative"), dict):
            narrative_trade = parsed.get("narrative", {}).get("trade") or ""
        if narrative_trade and isinstance(narrative_trade, str):
            narrative_lower = narrative_trade.lower()
            money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'

            if trade_allowance is None and "allowance" in narrative_lower:
                match = re.search(money_pattern, narrative_trade)
                if match:
                    trade_allowance = _coerce_float(match.group(1))

            if trade_payoff is None and "payoff" in narrative_lower:
                match = re.search(money_pattern, narrative_trade)
                if match:
                    trade_payoff = _coerce_float(match.group(1))

            if negative_equity_amount is None and ("negative equity" in narrative_lower or "negative" in narrative_lower):
                match = re.search(money_pattern, narrative_trade)
                if match:
                    negative_equity_amount = _coerce_float(match.group(1))
        
        # Step B & C: Extract money values near trade keywords
        import re
        
        # Money pattern: $12,345.67 or 12345.67 or 12,345
        money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        # If explicit fields present, we already populated values above
        
        # Allowance keywords (priority order)
        allowance_keywords = [
            "trade allowance", "allowance", "trade value",
            "trade-in value", "trade in value", "trade:"
        ]
        
        for keyword in allowance_keywords:
            if keyword in page_text:
                # Find text snippet around keyword
                idx = page_text.find(keyword)
                snippet = page_text[idx:idx+100]  # Look 100 chars ahead
                
                # Find first money value in snippet
                match = re.search(money_pattern, snippet)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        trade_allowance = float(amount_str)
                        break  # Found it, stop searching
                    except ValueError:
                        continue
        
        # Payoff keywords
        payoff_keywords = [
            "payoff", "lien payoff", "loan payoff", "balance owed"
        ]
        
        for keyword in payoff_keywords:
            if keyword in page_text:
                idx = page_text.find(keyword)
                snippet = page_text[idx:idx+100]
                
                match = re.search(money_pattern, snippet)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        trade_payoff = float(amount_str)
                        break
                    except ValueError:
                        continue
        
        # Step 6: Calculate equity (only if BOTH values exist)
        trade_status = "No trade identified"
        
        # Determine if trade is present
        trade_present = (
            trade_allowance is not None or
            trade_payoff is not None or
            equity is not None or
            negative_equity_amount is not None or
            trade_anchor_found
        )
        
        if not trade_present:
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )
        
        # Build status message
        if trade_allowance is not None and trade_payoff is not None:
            trade_equity = trade_allowance - trade_payoff
            
            if trade_equity < 0:
                negative_equity_amount = abs(trade_equity)
                trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance, ${trade_payoff:,.2f} payoff - Negative equity of ${negative_equity_amount:,.2f} rolled into new loan"
            elif trade_equity > 0:
                equity = trade_equity
                trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance, ${trade_payoff:,.2f} payoff - Positive equity of ${equity:,.2f} applied to purchase"
            else:
                trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance, ${trade_payoff:,.2f} payoff - Trade equity neutral"
        
        elif trade_allowance is not None:
            trade_status = f"Trade identified: ${trade_allowance:,.2f} allowance (payoff amount not found)"
        
        elif trade_payoff is not None:
            trade_status = f"Trade identified: ${trade_payoff:,.2f} payoff (allowance not found)"

        elif negative_equity_amount is not None:
            trade_status = f"Negative equity identified: ${negative_equity_amount:,.2f} rolled into new loan"

        elif equity is not None:
            if equity < 0:
                negative_equity_amount = abs(equity)
                trade_status = f"Negative equity identified: ${negative_equity_amount:,.2f} rolled into new loan"
            elif equity > 0:
                trade_status = f"Positive equity of ${equity:,.2f} applied to purchase"
        
        else:
            # Anchor found but no values extracted
            trade_status = "Trade mentioned in document (values not extracted)"
        
        return TradeData(
            trade_allowance=trade_allowance,
            trade_payoff=trade_payoff,
            equity=equity,
            negative_equity=negative_equity_amount,
            status=trade_status
        )

    async def analyze_images(self, files: List[UploadFile] = None, language: str = "English", base64_images: List[str] = None, parsed_data: dict = None) -> MultiImageAnalysisResponse:
        """Main analysis entry point. Accepts files, base64_images, or pre-extracted parsed_data dict."""
        try:
            if parsed_data is not None:
                # Always run through converter for consistent structure
                # Handles both nested (buyer_info, vehicle_details...) and flat formats
                parsed = convert_extracted_json_to_parsed(parsed_data)

                # Route: no pre-existing flags → call AI for full analysis following the prompts
                # The prompts are the holy bible for scoring — let AI apply them to the JSON data
                if not parsed.get("has_precomputed_flags", False):
                    print("JSON path: no pre-existing flags — calling AI for full prompt-based analysis...")
                    api_response = self._call_json_analysis_api(parsed, language)
                    ai_result = self._parse_api_response(api_response)
                    # AI result is now the primary parsed — preserve key identity fields from converter
                    for k in ("buyer_name", "dealer_name", "logo_text", "email",
                              "phone_number", "address", "state", "region", "vin_number", "date"):
                        if not ai_result.get(k) and parsed.get(k):
                            ai_result[k] = parsed[k]
                    # Preserve trade dict from converter if AI didn't extract it
                    if not ai_result.get("trade") and parsed.get("trade"):
                        ai_result["trade"] = parsed["trade"]
                    # Store AI's computed score (prompt-based, includes ceilings/adjustments)
                    ai_result["_ai_score"] = float(ai_result.get("score") or 75.0)
                    # Mark: audit merge and narrative call should be skipped
                    ai_result["has_precomputed_flags"] = True
                    ai_result["_ai_narrative_done"] = True
                    parsed = ai_result
            elif base64_images is None:
                validated_files = await self._validate_files(files)
                if not validated_files:
                    raise ValueError("No valid image files provided")
                
                # Optional: Optimize images before base64 encoding
                # optimized_files = await self._optimize_images(validated_files)
                # base64_images = await self._convert_files_to_base64(optimized_files)
                
                base64_images = await self._convert_files_to_base64(validated_files)
                api_response = self._call_openai_api(base64_images, language=language)
                parsed = self._parse_api_response(api_response)
            else:
                base64_images = [img.split(",", 1)[1] if img.startswith("data:") and "," in img else img for img in base64_images]
                api_response = self._call_openai_api(base64_images, language=language)
                parsed = self._parse_api_response(api_response)
            
            # Step 1: OCR Normalization
            raw_line_items = parsed.get("line_items", [])
            normalized_line_items = self._normalize_line_items(raw_line_items)
            
            # Step 2: Discount Detection and Normalization
            discounts, discount_totals = self.discount_detector.process_line_items(
                normalized_line_items,
                mode="QUOTE"
            )
            
            # Step 3: Audit Classification
            vehicle_price = float(parsed.get("selling_price") or 0)
            audit_classifications: List[AuditClassification] = []
            
            for item in normalized_line_items:
                classification = self.audit_classifier.classify_for_audit(
                    item,
                    vehicle_price=vehicle_price
                )
                audit_classifications.append(classification)
            
            # Step 4: Build Audit Flags
            audit_flags: List[AuditFlag] = []
            total_audit_penalty = 0
            
            # Finance Certificate flags
            finance_certs = [c for c in audit_classifications if c.classification == "CONDITIONAL_FINANCE_INCENTIVE"]
            for cert in finance_certs:
                flag = self.flag_builder.build_finance_certificate_flag(cert)
                audit_flags.append(flag)
                total_audit_penalty += cert.penalty_points
            
            # Bundled package flags
            bundles = [c for c in audit_classifications if c.classification == "BUNDLED_ADDON_PACKAGE"]
            for bundle in bundles:
                flag = self.flag_builder.build_bundled_package_flag(bundle)
                audit_flags.append(flag)
                total_audit_penalty += bundle.penalty_points
            
            # Discount advantage flags (green)
            if discount_totals.total_all_discounts < 0:
                flag = self.flag_builder.build_online_price_advantage_flag(
                    abs(discount_totals.total_all_discounts)
                )
                audit_flags.append(flag)
            
            # Step 5: GAP Logic Evaluation
            term_months = parsed.get("term", {}).get("months")
            down_payment = parsed.get("normalized_pricing", {}).get("down_payment")
            amount_financed = parsed.get("normalized_pricing", {}).get("amount_financed")
            
            # Check if GAP is present
            gap_present = any(
                c.classification == "GAP" 
                for c in audit_classifications
            )
            
            # Check if backend products present
            has_backend = any(
                c.classification in ["GAP", "VSC", "MAINTENANCE"]
                for c in audit_classifications
            )
            
            # Determine if used vehicle (simple heuristic - can be improved)
            is_used = True  # TODO: Extract from OCR or VIN decode
            
            gap_recommendation = self.gap_logic.evaluate_gap_need(
                is_used=is_used,
                term_months=term_months,
                down_payment=down_payment,
                amount_financed=amount_financed,
                vehicle_price=vehicle_price,
                has_backend_products=has_backend,
                gap_present=gap_present
            )
            
            if gap_recommendation.recommended:
                gap_flag = self.flag_builder.build_gap_advisory_flag(
                    gap_recommendation.message
                )
                audit_flags.append(gap_flag)
            
            # Long-term loan risk flag
            if term_months and term_months >= 72:
                loan_risk_flag = self.flag_builder.build_long_term_loan_risk_flag(term_months)
                audit_flags.append(loan_risk_flag)

            # APR Scoring
            apr_data = parsed.get("apr", {})
            if isinstance(apr_data, dict):
                apr_rate = apr_data.get("rate")
                try:
                    if apr_rate is not None:
                        apr_f = float(apr_rate)
                        if apr_f > 0:
                            if apr_f <= 4.9:
                                audit_flags.append(AuditFlag(
                                    type="green", category="Excellent APR",
                                    message=f"APR of {apr_f:.2f}% is excellent — well below market average.",
                                    item="APR", deduction=None, bonus=5
                                ))
                            elif apr_f <= 6.9:
                                audit_flags.append(AuditFlag(
                                    type="green", category="Good APR",
                                    message=f"APR of {apr_f:.2f}% is competitive.",
                                    item="APR", deduction=None, bonus=2
                                ))
                            elif apr_f >= 16.0:
                                audit_flags.append(AuditFlag(
                                    type="red", category="Predatory APR",
                                    message=f"APR of {apr_f:.2f}% is predatory and significantly above market rates.",
                                    item="APR", deduction=10, bonus=None
                                ))
                            elif apr_f > 12.0:
                                audit_flags.append(AuditFlag(
                                    type="red", category="High APR",
                                    message=f"APR of {apr_f:.2f}% exceeds typical market rates.",
                                    item="APR", deduction=5, bonus=None
                                ))
                except (ValueError, TypeError):
                    pass

            # Doc Fee Scoring
            doc_fee = None
            for _item in normalized_line_items:
                raw = (_item.raw_text or "").lower()
                if "documentary" in raw or "doc fee" in raw or "documentation fee" in raw:
                    doc_fee = abs(_item.amount_normalized)
                    break
            if doc_fee is None:
                for _key in ("doc_fee", "documentation_fee"):
                    _v = parsed.get(_key)
                    if _v is not None:
                        try:
                            doc_fee = abs(float(str(_v).replace(",", "").replace("$", "")))
                        except (ValueError, TypeError):
                            pass
                        break
            if doc_fee is not None:
                if doc_fee > 899:
                    audit_flags.append(AuditFlag(
                        type="red", category="Excessive Doc Fee",
                        message=f"Documentation fee of ${doc_fee:,.2f} exceeds the recommended $899 cap.",
                        item="Doc Fee", deduction=3, bonus=None
                    ))
                elif doc_fee > 599:
                    audit_flags.append(AuditFlag(
                        type="red", category="SOFT - High Doc Fee",
                        message=f"Documentation fee of ${doc_fee:,.2f} is above the $599 standard.",
                        item="Doc Fee", deduction=2, bonus=None
                    ))

            # Amount Financed vs Selling Price check (loan-to-value)
            _selling = vehicle_price
            _financed = None
            try:
                _raw_financed = parsed.get("normalized_pricing", {}).get("amount_financed") if isinstance(parsed.get("normalized_pricing"), dict) else None
                if _raw_financed is not None:
                    _financed = float(_raw_financed)
            except (ValueError, TypeError):
                pass
            if _financed and _selling and _selling > 0:
                ltv = (_financed / _selling) * 100
                if ltv > 115:
                    audit_flags.append(AuditFlag(
                        type="red", category="High Loan-to-Value",
                        message=f"Amount financed (${_financed:,.2f}) is {ltv:.0f}% of selling price — high loan-to-value ratio.",
                        item="Financing", deduction=5, bonus=None
                    ))
                elif ltv > 105:
                    audit_flags.append(AuditFlag(
                        type="red", category="SOFT - Elevated Loan-to-Value",
                        message=f"Amount financed (${_financed:,.2f}) slightly exceeds selling price ({ltv:.0f}% LTV).",
                        item="Financing", deduction=2, bonus=None
                    ))

            # MSRP vs Selling Price check
            np_data = parsed.get("normalized_pricing", {}) if isinstance(parsed.get("normalized_pricing"), dict) else {}
            msrp_raw = np_data.get("msrp")
            if msrp_raw and vehicle_price:
                try:
                    msrp_f = float(str(msrp_raw).replace(",", "").replace("$", ""))
                    if msrp_f > 0:
                        markup_pct = ((vehicle_price - msrp_f) / msrp_f) * 100
                        if markup_pct > 5:
                            audit_flags.append(AuditFlag(
                                type="red", category="Above MSRP",
                                message=f"Selling price ${vehicle_price:,.2f} is {markup_pct:.1f}% above MSRP ${msrp_f:,.2f}.",
                                item="Pricing", deduction=5, bonus=None
                            ))
                        elif markup_pct < -3:
                            audit_flags.append(AuditFlag(
                                type="green", category="Below MSRP",
                                message=f"Selling price is {abs(markup_pct):.1f}% below MSRP — good deal.",
                                item="Pricing", deduction=None, bonus=3
                            ))
                        # Backend overload check
                        total_backend = sum(
                            c.amount for c in audit_classifications
                            if c.classification in ("GAP", "VSC", "MAINTENANCE", "TIRE_WHEEL_PROTECTION", "UNKNOWN")
                            and c.amount > 0
                        )
                        if total_backend > 0:
                            backend_pct = (total_backend / msrp_f) * 100
                            if backend_pct > 20:
                                audit_flags.append(AuditFlag(
                                    type="red", category="Backend Overload",
                                    message=f"Total backend products ${total_backend:,.2f} ({backend_pct:.1f}% of MSRP) — excessive.",
                                    item="Add-ons", deduction=10, bonus=None
                                ))
                            elif backend_pct > 12:
                                audit_flags.append(AuditFlag(
                                    type="red", category="Backend Overload",
                                    message=f"Total backend products ${total_backend:,.2f} ({backend_pct:.1f}% of MSRP) — above normal.",
                                    item="Add-ons", deduction=5, bonus=None
                                ))
                except (ValueError, TypeError):
                    pass

            # Step 6.5: Process Trade Data (UPDATED - Use OCR-based extraction)
            trade_data = self._extract_trade_data(parsed)
            
            # Add negative equity flag if applicable
            if trade_data.negative_equity and trade_data.negative_equity > 0:
                neg_equity_flag = AuditFlag(
                    type="blue",
                    category="Negative Equity Alert",
                    message=f"Rolled negative equity of ${trade_data.negative_equity:,.2f} increases total loan exposure and overall risk.",
                    item="Trade",
                    deduction=None,
                    bonus=None
                )
                audit_flags.append(neg_equity_flag)
            
            # Step 7: Suppress "missing incentive" warnings if Finance Certificate detected
            if finance_certs:
                # Remove any "missing incentive" flags from parsed data
                # (This would be in the original API response processing)
                pass
            
            # Step 7: Score will be computed AFTER merging all flags (see below)
            
            # Step 8: Merge audit flags with existing flags
            def parse_flags(flags_data):
                flags_list = []
                if not isinstance(flags_data, list):
                    if isinstance(flags_data, str):
                        try:
                            flags_data = json.loads(flags_data)
                        except Exception:
                            return []
                    else:
                        return []
                for item in flags_data:
                    if isinstance(item, str):
                        try:
                            item_dict = json.loads(item)
                        except Exception:
                            item_dict = {"type": "Unknown", "message": item, "item": "Unknown"}
                    elif isinstance(item, dict):
                        item_dict = item
                    else:
                        item_dict = {"type": "Unknown", "message": str(item), "item": "Unknown"}
                    flag_kwargs = {
                        "type": item_dict.get("type", "Unknown"),
                        "message": item_dict.get("message", ""),
                        "item": item_dict.get("item", "Unknown"),
                        "deduction": item_dict.get("deduction"),
                        "bonus": item_dict.get("bonus")
                    }
                    try:
                        flags_list.append(Flag(**flag_kwargs))
                    except Exception:
                        flags_list.append(Flag(type=str(flag_kwargs["type"]), message=str(flag_kwargs["message"]), item=str(flag_kwargs["item"])))
                return flags_list

            red_flags = parse_flags(parsed.get("red_flags", []))
            green_flags = parse_flags(parsed.get("green_flags", []))
            blue_flags = parse_flags(parsed.get("blue_flags", []))

            # Only merge Python audit flags when no pre-existing flags were supplied.
            # If the input already has flags (from a prior OCR/AI analysis), skip the
            # audit merge — the pre-existing flags ARE the authoritative analysis.
            has_precomputed = parsed.get("has_precomputed_flags", False)
            if not has_precomputed:
                for audit_flag in audit_flags:
                    flag_obj = Flag(
                        type=audit_flag.type,
                        message=audit_flag.message,
                        item=audit_flag.item,
                        deduction=audit_flag.deduction,
                        bonus=audit_flag.bonus
                    )
                    if audit_flag.type == "red":
                        red_flags.append(flag_obj)
                    elif audit_flag.type == "green":
                        green_flags.append(flag_obj)
                    elif audit_flag.type == "blue":
                        blue_flags.append(flag_obj)

            print(f"Python flags - Red: {len(red_flags)}, Green: {len(green_flags)}, Blue: {len(blue_flags)}")

            # Step 9: Score
            # If AI did full prompt-based analysis, its score incorporates all prompt rules
            # (ceilings, structural adjustments, etc.) — use it directly.
            # Otherwise compute from flags using Python math.
            if parsed.get("_ai_score") is not None:
                adjusted_score = max(0.0, min(100.0, float(parsed["_ai_score"])))
                print(f"Using AI prompt-based score: {adjusted_score}")
            else:
                adjusted_score = 100.0
                for f in red_flags:
                    if f.deduction is not None:
                        adjusted_score -= abs(float(f.deduction))
                for f in green_flags:
                    if f.bonus is not None:
                        adjusted_score += abs(float(f.bonus))
                if not has_precomputed:
                    adjusted_score += total_audit_penalty
                adjusted_score = max(0.0, min(100.0, adjusted_score))
            print(f"Score Calculation: Final={adjusted_score}")

            # Translate flags to requested language (no scoring changes)
            red_flags = self._translate_flags(red_flags, language)
            green_flags = self._translate_flags(green_flags, language)
            blue_flags = self._translate_flags(blue_flags, language)

            # Safety net: ensure every flag category has at least one item
            if not red_flags:
                red_flags.append(Flag(type="red", message="No major issues identified — verify all terms and pricing before finalizing.", item="General"))
            if not green_flags:
                green_flags.append(Flag(type="green", message="No standout positive elements identified in this quote.", item="General"))
            if not blue_flags:
                blue_flags.append(Flag(type="blue", message="Review all final quote terms and itemized pricing carefully before agreeing to any deal.", item="General Advisory"))

            # ── AI Narrative ──
            if not parsed.get("_ai_narrative_done"):
                # Pre-existing flags path: call AI separately for narrative
                print(f"Generating AI narrative for score {adjusted_score}...")
                ai_result = self._call_narrative_api(parsed, adjusted_score, red_flags, green_flags, blue_flags, language)
                narrative_obj = ai_result.get("narrative", {}) if isinstance(ai_result, dict) else {}
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                _ai_buyer_msg_override = ai_result.get("buyer_message") if isinstance(ai_result, dict) else None
            else:
                # Full AI analysis already included narrative — extract directly
                narrative_obj = parsed.get("narrative", {})
                if not isinstance(narrative_obj, dict):
                    narrative_obj = {}
                _ai_buyer_msg_override = parsed.get("buyer_message")
            buyer_msg = _ai_buyer_msg_override

            # Normalize legacy key
            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")

            # Fallback defaults for any fields the AI left empty
            defaults = {
                "vehicle_overview": f"Deal analysis for {parsed.get('dealer_name', 'this dealer')}.",
                "smartbuyer_score_summary": f"SmartBuyer Score: {adjusted_score}/100.",
                "score_breakdown": f"Final Score: {adjusted_score}",
                "market_comparison": "Market comparison pending.",
                "gap_logic": "GAP analysis pending.",
                "vsc_logic": "VSC analysis pending.",
                "apr_bonus_rule": "APR analysis pending.",
                "lease_audit": "N/A - Purchase Agreement",
                "trade": trade_data.status if trade_data else "No trade-in on this deal.",
                "negotiation_insight": "Review all flags before signing.",
                "final_recommendation": "Proceed with caution based on the flags above."
            }
            for k, v in defaults.items():
                if not narrative_obj.get(k):
                    narrative_obj[k] = v

            if not buyer_msg:
                buyer_msg = f"Your SmartBuyer score is {adjusted_score}/100 — review the flags above."

            # Ensure trade is a string
            if "trade" in narrative_obj and not isinstance(narrative_obj["trade"], str):
                narrative_obj["trade"] = str(narrative_obj["trade"])

            # Ensure score_breakdown is a string (AI sometimes returns a list of flag dicts)
            if "score_breakdown" in narrative_obj and not isinstance(narrative_obj["score_breakdown"], str):
                sb = narrative_obj["score_breakdown"]
                if isinstance(sb, list):
                    narrative_obj["score_breakdown"] = "; ".join(
                        f"{f.get('type','?')}: {f.get('message','?')} ({f.get('deduction', f.get('bonus','?'))}pt)"
                        if isinstance(f, dict) else str(f)
                        for f in sb
                    )
                else:
                    narrative_obj["score_breakdown"] = str(sb)

            narrative = Narrative(**narrative_obj)

            return MultiImageAnalysisResponse(
                score=adjusted_score,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                region=parsed.get("region", "Outside US"),
                badge=self._assign_badge(adjusted_score),
                selling_price=parsed.get("selling_price"),
                vin_number=parsed.get("vin_number"),
                date=parsed.get("date"),
                buyer_message=buyer_msg,
                red_flags=red_flags,
                green_flags=green_flags,
                blue_flags=blue_flags,
                normalized_pricing=NormalizedPricing(**parsed.get("normalized_pricing", {})),
                apr=APRData(**parsed.get("apr", {})),
                term=TermData(**parsed.get("term", {})),
                trade=trade_data,
                bundle_abuse=parsed.get("bundle_abuse", {"active": False, "deduction": 0}),
                narrative=narrative
            )
        except Exception as e:
            raise RuntimeError(f"Contract analysis failed: {str(e)}")
