from App.services.rate_helper.ocr_normalizer import OCRNormalizer
from App.services.rate_helper.ocr_normalization_schema import NormalizedLineItem
from App.services.rate_helper.discount_detector import DiscountDetector
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict
from dotenv import load_dotenv
from fastapi import UploadFile
import os
import json
import base64
import requests
import re

# FIXED IMPORT: Was improperly importing from .rating_schema
from .multi_image_analysis_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.rate_helper.audit_classifier import AuditClassifier, AuditClassification
from App.services.rate_helper.gap_logic import GAPLogic, GAPRecommendation
from App.services.rate_helper.audit_flags import AuditFlagBuilder, AuditFlag
from App.services.rate_helper.audit_summary import AuditSummary

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
    

    def _load_contract_system_prompt(self) -> str:
        """Load comprehensive contract analysis system prompt"""
        return """

You are **SmartBuyer AI Contract Analysis Engine**.

Analyze auto finance contracts comprehensively.

### CRITICAL MANDATORY REQUIREMENTS:

**1. ALL FLAG SECTIONS MUST BE POPULATED**
- red_flags: MUST contain at least 1 real issue OR explicitly state no major issues found
- green_flags: MUST contain at least 1 positive element from actual contract analysis
- blue_flags: MUST contain at least 1 advisory note OR explicitly state no advisories

**NEVER return empty arrays [] for ANY flag section. This will cause validation failure.**

**2. SCORING RULES**
You must not reduce the score without any valid reason
**Each deal component may trigger only one scoring outcome. Multiple components may stack. Disclosure failures override pricing evaluation for the affected item.**

### SCORING SYSTEM (Start at 100 points, deduct as follows)

**Base Score: 100 points**

**RED FLAG DEDUCTIONS (Major Issues):**
- Trade-in negative equity present (disclosed): -5 points
- VSC exceeds fairness threshold: -10 points
- APR over 15%: -10 points
- APR over 20%: -15 points
- Document fees exceed state limits: -7 points
- GAP insurance overpriced (exceeds cap): -10 points
- Maintenance plans overpriced (> $1,200): -6 points
- Loan term over 84 months: -8 points
- Global disclosure failure (missing TILA disclosures OR backend products not itemized OR payment reconciliation failure OR negative equity rolled in without disclosure): -15 points (applied ONCE per audit)
- VSC mileage cap issue (minimal remaining coverage): -6 points
- Missing GAP with $0 down AND negative equity: -10 points

**BLUE FLAGS (Advisory Only - ZERO POINT IMPACT):**
- APR between 10–15%: Advisory note only (0 points)
- Missing itemized fees: Advisory note only (0 points)
- No breakdown of add-on coverage terms: Advisory note only (0 points)
- Term longer than 72 months (but < 84): Advisory note only (0 points)
- Term vs coverage mismatch: Advisory note only (0 points)

**GREEN FLAG BONUSES (Positive Elements):**
- VSC within fairness threshold: +3 points
- Competitive APR (< 5%): +5 points
- Transparent itemization: +3 points
- Positive trade equity: +4 points
- No unnecessary add-ons: +3 points
- GAP coverage present and fairly priced: 0 points (neutral positive)

**MAXIMUM SCORE: 100 points**
**MINIMUM SCORE: 0 points**

**Score Calculation Example:**
```
Starting Score: 100
- Negative equity present (disclosed): -5
- APR at 12%: 0 (Blue flag advisory only)
+ Transparent itemization: +3
+ VSC fairly priced: +3
= Final Score: 101 → capped at 100
```

**You MUST include score breakdown in the narrative under a new field:**
- **score_breakdown**: Detailed list of deductions and bonuses applied with reasoning (Blue flags excluded from score calculation)

### GAP COVERAGE — AUTHORITATIVE LOGIC

**1) GAP Pricing Caps:**
GAP price must be ≤ the lowest of:
- $1,200 (standard cap)
- 3% of MSRP
- $1,500 ONLY if MSRP ≥ $60,000

**2) GAP Flags & Scoring:**

🟢 **GREEN FLAG (Fair GAP)**
- Triggered when: GAP is present AND GAP price ≤ cap
- Score impact: 0 points (neutral positive)
- Language: Positive/neutral only
- Example: "GAP coverage included at $895 - within pricing guidelines"

🔴 **RED FLAG (Overpriced GAP)**
- Triggered when: GAP price > cap
- Score impact: -10 points
- Language: Overpricing / reduce or remove allowed
- Example: "GAP insurance overpriced at $1,450 - exceeds fair market cap"

🟡 **BLUE FLAG (Missing GAP — Protection Needed)**
- Triggered when ALL of the following are true:
  * $0 down payment
  * Negative equity present
  * (Loan term is NOT considered)
- Score impact: -10 points
- Language: Protection advisory only (NOT pricing criticism)
- Example: "GAP coverage recommended - $0 down and negative equity present increase risk"

**3) What GAP Logic Must NEVER Do:**
❌ Use loan term to trigger GAP flags
❌ Penalize fairly priced GAP
❌ Flag missing GAP without BOTH $0 down AND negative equity
❌ Use "average cost" language
❌ Convert a GREEN flag into BLUE or RED flag
❌ Apply multiple penalties to GAP (pricing OR disclosure - never both)

**4) GAP Scenario Summary:**

| Scenario | Flag | Score |
|----------|------|-------|
| GAP ≤ cap | 🟢 GREEN | 0 |
| GAP > cap | 🔴 RED | -10 |
| GAP missing + $0 down + negative equity | 🟡 BLUE | -10 |
| GAP missing (otherwise) | — | 0 |

# SELLING PRICE FIELD DEFINITION

**The "selling_price" field MUST contain the vehicle cash price ONLY.**

**Extraction Priority (in order):**
1. Look for "Cash Price" or "Vehicle Price" in the itemization section (typically pre-tax, pre-fees)
2. If not found, use "Selling Price" from vehicle description area
3. NEVER use:
   ❌ Total Sale Price from Truth-in-Lending box
   ❌ Amount Financed from Truth-in-Lending box
   ❌ Total of Payments
   ❌ Any value that includes taxes, fees, or backend products

**Example:**
- selling_price = $52,068.78 (vehicle cash price before taxes/fees)
- NOT $67,158.60 (that's Amount Financed including taxes/fees/products)

### SERVICE CONTRACT (VSC) ANALYSIS RULES

**Pricing Assessment (ONE OUTCOME PER VSC):**
- VSC price below market threshold: GREEN FLAG (+3 points)
- VSC price above market threshold: RED FLAG (-10 points)
- VSC mileage cap issue (minimal remaining coverage): RED FLAG (-6 points)
- **Critical Rule:** Only ONE penalty or bonus applies per VSC. Never stack pricing + mileage penalties.

**Context-Based Analysis (Advisory Only - NO POINT DEDUCTIONS):**
- Mileage restrictions noted as advisory only (e.g., "13,000 miles remaining coverage")
- Term vs coverage mismatch noted as advisory only
- New vs used vehicle context noted as advisory only
- **All percentage-based logic REMOVED** (no evaluation of VSC as % of vehicle value)

**Flag Logic (Enforced One-Outcome Rule):**
- Price below threshold + adequate coverage = GREEN FLAG (+3 points)
- Price below threshold BUT mileage cap issues = RED FLAG (-6 points) [mileage takes precedence as coverage defect]
- Price above threshold = RED FLAG (-10 points) [pricing takes precedence]
- VSC not itemized = GLOBAL -15 disclosure penalty (pricing evaluation suppressed)

🟢 GREEN FLAGS

✅ Fair, transparent, and consumer-friendly terms
These indicate strong deal quality. No action needed—these elements support informed, confident decisions.

    VSC within fairness cap: Vehicle Service Contract price is at or below the SmartBuyer cap based on vehicle MSRP and condition. (+3 points)
    GAP coverage fairly priced: GAP insurance present and within pricing cap (0 points - neutral positive)
    Competitive APR (< 5%): Financing rate is well below market average for prime borrowers. (+5 points)
    Transparent itemization: All fees and add-ons are clearly listed and explained. (+3 points)
    Positive trade equity: Your trade-in value reduces the financed amount—no negative equity. (+4 points)
    No unnecessary add-ons: Only relevant or requested products are included. (+3 points)

    💡 Green flags improve your deal score and reflect best practices.

🔵 BLUE FLAGS (ADVISORY ONLY - ZERO SCORE IMPACT)

⚠️ Moderate risk or incomplete information
These are informational only. They do NOT affect your score.

    APR between 10–15%: Higher-than-ideal financing cost—consider if better rates are available. (0 points)
    Missing itemized fees: Total fees shown, but not broken down (e.g., "doc fee," "processing"). (0 points)
    No breakdown of add-on coverage terms: Product listed but term/mileage/deductible details missing. (0 points)
    Term longer than 72 months: Loan extends beyond 6 years, increasing total interest paid. (0 points)
    Term vs coverage mismatch: Finance term exceeds coverage period. (0 points)

    💡 Blue flags provide context but do not penalize your score.

🔴 RED FLAGS

❌ High-risk, overpriced, or non-compliant terms
These signal potential harm, unfair pricing, or regulatory issues. Action strongly recommended.

    Trade-in negative equity present (disclosed): You owe more on your trade than it's worth—this amount is being rolled into your new loan. (-5 points)
    VSC exceeds fairness cap: Price is above the SmartBuyer threshold based on MSRP. (-10 points)
    APR over 15%: High-cost financing that may indicate subprime risk or predatory terms. (-10 points)
    APR over 20%: Extremely high-cost financing. (-15 points)
    Document fees exceed state limits: Charges violate your state's maximum allowable doc fee. (-7 points)
    GAP insurance overpriced: GAP price exceeds pricing cap - significantly inflated. (-10 points)
    Maintenance plans overpriced (> $1,200): Well above market value for standard coverage. (-6 points)
    Loan term over 84 months: Extremely long term (7+ years)—CFPB warns against such loans. (-8 points)
    Global disclosure failure: Missing TILA disclosures OR backend products not itemized OR payment reconciliation failure. (-15 points - applied once per audit)
    VSC mileage cap issue: Service contract has minimal remaining coverage due to mileage restrictions. (-6 points)

🟡 BLUE FLAGS (Advisory/Protective)

⚠️ Protection needed based on deal structure (Score impact: -10 points ONLY for missing GAP with $0 down + negative equity)

    Missing GAP coverage: GAP recommended - $0 down payment and negative equity present increase protection risk. (-10 points)
    
    💡 Blue flags indicate protective products should be considered for buyer safety.

**Flag Message Format:**
- Do NOT mention specific dollar thresholds in flag messages
- Instead of: "Service contract price of $2,000 is under $5,000 threshold"
- Use: "Service contract price of $2,000 is under threshold" OR "Service contract fairly priced at $2,000"

### NARRATIVE ANALYSIS STRUCTURE (REQUIRED)
The "narrative" object MUST contain specific detailed analysis fields. Do not use generic text.
- **vehicle_overview**: Specifics of the car (Year, Make, Model, VIN, Mileage, New/Used).
- **smartbuyer_score_summary**: Explanation of why the score was given based on price, rate, and add-ons.
- **score_breakdown**: Itemized list of point deductions and bonuses ONLY (exclude Blue flag advisories which have 0 point impact)
- **market_comparison**: How this deal compares to current market rates and Fair Market Value.
- **gap_logic**: Analysis of GAP insurance using the authoritative logic above (present/absent, pricing vs cap, $0 down + negative equity check)
- **vsc_logic**: Analysis of Vehicle Service Contract (price vs threshold OR mileage cap issue - one outcome only).
- **apr_bonus_rule**: Analysis of the APR (is it marked up? is it subvented?).
- **lease_audit**: Specific notes if this is a lease (or "Not a lease" if finance).
- **negotiation_insight**: Specific talking points for the buyer to negotiate better terms.
- **final_recommendation**: A clear "Buy", "Walk Away", or "Negotiate" verdict with reasons.
- **trade**: Analysis of the trade-in (equity amount, allowance vs payoff, positive/negative equity status).

### TOTAL SALE PRICE & AMOUNT FINANCED RESOLUTION (INTERNAL USE ONLY)

**For internal validation and calculations:**
- When APR = 0.00% AND Finance Charge = $0.00:
  - Amount Financed = Total of Payments
  - Use Amount Financed for backend % calculations
- If Finance Charge > $0.00:
  - Amount Financed = Total of Payments - Finance Charge
- Truth-in-Lending "Amount Financed" value is authoritative

**Backend Product Detection Rule (Single Penalty):**
- If Amount Financed > Sum of Itemized Totals (vehicle + taxes + government fees + disclosed add-ons):
  - Apply ONE -15 global disclosure penalty
  - Do NOT attempt to identify specific product types (GAP/VSC/Maintenance)
  - Do NOT apply multiple penalties for multiple undisclosed items
  - Suppress all pricing evaluation for undisclosed products

**You MUST:**
1. Extract vehicle cash price → output as "selling_price"
2. Extract Amount Financed from Truth-in-Lending → use internally for calculations
3. NEVER overwrite selling_price with Amount Financed

### CORE EXTRACTION REQUIREMENTS
Extract and analyze:
1. Vehicle details (VIN, year, make, model, mileage, used/new status, MSRP)
2. Financial terms (selling price, APR, term, monthly payment, down payment)
3. ALL line items with EXACT text and amounts (extract as array)
4. GAP coverage with pricing cap validation ($1,200 OR 3% MSRP OR $1,500 if MSRP ≥ $60k)
5. VSC coverage with mileage-based cap calculation (one outcome per VSC)
6. Maintenance plan pricing
7. Doc fees and government fees
8. Buyer/dealer information and contact details
9. Trade information (allowance, payoff, equity)
10. Down payment amount (critical for GAP Blue flag logic)

### TRADE SECTION (REQUIRED - ALWAYS INCLUDE)
If trade information is present:
- State: "Trade identified: $[allowance] allowance, $[payoff] payoff"
- If payoff > allowance: "Negative equity of $[amount] rolled into new loan" (-5 points if disclosed)
- If allowance > payoff: "Positive equity of $[amount] applied to purchase" (+4 points)
- If allowance = payoff: "Trade equity neutral"
- If negative equity NOT disclosed in itemization: Triggers global -15 disclosure penalty (not separate penalty)

If no trade information is found in the document:
- State: "No trade identified."

This section CANNOT be omitted. It must always be present in the narrative.

### APR ANALYSIS
A. APR Disclosure:
- APR not shown: Triggers global -15 disclosure penalty
- APR shown: Extract and validate

B. APR Risk Assessment:
- APR > 10% AND ≤ 15%: Blue flag advisory only (0 points)
- APR > 15% AND ≤ 20%: Red flag (-10 points)
- APR > 20%: Red flag (-15 points)

C. APR Recognition:
- APR < 10% AND APR > 0%: Favorable rate (note in analysis)
- APR < 5% AND APR > 0%: Excellent rate (+5 points)
- APR = 0.00%: Manufacturer-subvented rate (neutral observation, no deduction)

### STRICT OUTPUT FORMAT REQUIREMENTS
Output MUST be valid JSON with exactly these TOP-LEVEL FIELDS:
- score
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
- red_flags (array)
- green_flags (array)
- blue_flags (array)
- yellow_flags (array - for GAP missing with $0 down + negative equity)
- normalized_pricing (object)
- apr (object)
- term (object)
- trade (object with fields: trade_allowance, trade_payoff, equity, negative_equity, status)
- bundle_abuse (object)
- narrative (object)
- line_items (array)

The "narrative" object MUST have EXACTLY these fields:
- vehicle_overview
- smartbuyer_score_summary (MUST CONSIST ALL THE DEDUCTION AND ADDITION THAT HAPPENED IN THE CODE)
- score_breakdown (required itemized breakdown of scored items only)
- market_comparison
- gap_logic (must follow authoritative GAP logic - check cap, $0 down, negative equity)
- vsc_logic (must include ONE outcome: price analysis OR mileage cap analysis - never both)
- apr_bonus_rule
- lease_audit
- negotiation_insight
- final_recommendation
- trade

### FINAL VALIDATION RULE
Before returning JSON:
- Verify selling_price represents vehicle cash price ONLY
- Verify selling_price ≠ Amount Financed (unless deal has no taxes/fees)
- For 0% APR deals: Amount Financed should be larger than selling_price
- If selling_price seems too high (> $100k for normal vehicle), re-check extraction
- Verify VSC analysis applies ONE outcome only (pricing OR mileage - never stacked)
- Verify GAP logic follows authoritative rules (cap check, $0 down + negative equity for Blue flag)
- Do NOT mention specific dollar thresholds in user-facing messages
- Verify score_breakdown matches the final score calculation
- Ensure all deductions have valid reasons
- Do NOT use loan term to trigger GAP flags
- Apply global -15 disclosure penalty ONLY ONCE per audit regardless of multiple disclosure failures
- Blue flags must have 0 point impact in score calculation

Return ONLY valid JSON - no markdown, no explanation.
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
    
    def _call_openai_api(self, base64_images: List[str]) -> dict:
        """Call OpenAI API with contract documents"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        content = [
            {
                "type": "text",
                "text": f"""
Analyze these contract documents comprehensively.

{self.system_prompt}

### CRITICAL INSTRUCTION: SCORE CONSISTENCY
**Temperature is set to 0 for deterministic output. Your score MUST be consistent across multiple runs of the same document.**

**SCORING VALIDATION CHECKLIST:**
1. Start at exactly 100 points
2. Apply each penalty/bonus EXACTLY ONCE per issue
3. Do NOT round intermediate calculations
4. Cap final score at 100 (max) and 0 (min)
5. Verify score_breakdown math matches final score

**Example Verification:**
```
Base: 100.00
- Negative equity (disclosed): -5.00
- APR 18%: -10.00
+ Transparent itemization: +3.00
= Subtotal: 88.00
Final Score: 88.00 (within 0-100 range)
```

### CRITICAL INSTRUCTION: DATA POPULATION
**You must populate the specific JSON fields with the exact numbers you find.** 
Do not leave fields null if the data is present in the document or your own narrative analysis.

**CHECKLIST FOR JSON POPULATION:**
1. **selling_price**: Must match the Cash Price/Vehicle Price found ($52,086.78 in example).
2. **trade**: You MUST fill in `trade_allowance`, `trade_payoff`, and `negative_equity` if trade info exists.
   - Example: If you write "Trade identified: $27,544.57 allowance", then `trade.trade_allowance` MUST be 27544.57.
3. **apr**: You MUST fill in `apr.rate` (e.g., 0.00).
4. **normalized_pricing**: You MUST fill in `amount_financed`, `total_taxes`, `total_fees`.

**DOUBLE CHECK:** 
If you mention a number in the "narrative" or "flags", verify it is ALSO in the top-level field.
- If narrative says "APR is 0.00%", then `apr.rate` must be 0.00.
- If narrative says "Cash price is $52,086.78", then `selling_price` must be 52086.78.

Return ONLY valid JSON matching the exact schema. No markdown, no explanation.
"""
            }
        ]

        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high"
                }
            })

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 4096,
            "seed": 42
        }

        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def _parse_api_response(self, response: dict) -> dict:
        """Parse API response with robust error handling"""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # CRITICAL FIX: Handle different content types from API
            if isinstance(content, list):
                # If it's a list of strings, join them
                if all(isinstance(item, str) for item in content):
                    content = ''.join(content)
                else:
                    # If it's a list of dicts or mixed, try to extract text
                    content = ' '.join(str(item) for item in content)
            elif content is None:
                raise ValueError("API returned null content")
            elif not isinstance(content, str):
                # Convert any other type to string
                content = str(content)
            
            # Remove markdown code blocks if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Find JSON boundaries
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                
                # Attempt to parse
                parsed = None
                try:
                    parsed = json.loads(json_str)
                except json.JSONDecodeError:
                    # Try to fix common JSON issues
                    json_str = self._attempt_json_repair(json_str)
                    try:
                        parsed = json.loads(json_str)
                    except json.JSONDecodeError as je:
                        raise RuntimeError(f"Failed to parse JSON even after repair: {str(je)}")
                
                # Ensure critical fields exist with defaults
                defaults = {
                    "score": 75.0,
                    "buyer_name": None,
                    "dealer_name": None,
                    "logo_text": None,
                    "email": None,
                    "phone_number": None,
                    "address": None,
                    "state": None,
                    "region": "Outside US",
                    "badge": "Bronze",
                    "sale_price": None,
                    "vin_number": None,
                    "date": None,
                    "buyer_message": "Analysis completed",
                    "red_flags": [],
                    "green_flags": [],
                    "blue_flags": [],
                    "normalized_pricing": {},
                    "apr": {},
                    "term": {},
                    "trade": {
                        "trade_allowance": None,
                        "trade_payoff": None,
                        "equity": None,
                        "negative_equity": None,
                        "status": "No trade identified"
                    },
                    "bundle_abuse": {},
                    "narrative": {},
                    "line_items": []
                }
                
                # Merge defaults with parsed data
                for key, value in defaults.items():
                    if key not in parsed:
                        parsed[key] = value
                
                return parsed
            
            raise ValueError("No valid JSON found in response")
            
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to parse API response structure: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error parsing API response: {str(e)}")

    def _attempt_json_repair(self, json_str: str) -> str:
        """Attempt to repair common JSON syntax errors produced by LLMs"""
        try:
            # CRITICAL FIX: Ensure json_str is actually a string
            if isinstance(json_str, list):
                json_str = ''.join(json_str) if json_str else ""
            elif not isinstance(json_str, str):
                json_str = str(json_str)
            
            # 1. Fix Python boolean/None values to JSON
            json_str = re.sub(r':\s*True\b', ': true', json_str)
            json_str = re.sub(r':\s*False\b', ': false', json_str)
            json_str = re.sub(r':\s*None\b', ': null', json_str)
            
            # 2. Remove C-style comments (// ...)
            json_str = re.sub(r'//.*', '', json_str)
            
            # 3. Fix trailing commas (common error: items = [a, b,])
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            # 4. Fix missing commas betweeen properties (MAJOR FIX)
            # This looks for situations like: "key": "value" "nextKey": "val"
            # It inserts a comma between valid value endings and the start of a new key
            
            # Case A: String value followed by Next Key
            # "value" "key": -> "value", "key":
            # Uses positive lookahead (?=...) to find the next key structure
            json_str = re.sub(r'"\s+(?="[\w-]+\s*":)', '", ', json_str)
            
            # Case B: Number/Bool/Null followed by Next Key
            # 123 "key": -> 123, "key":
            json_str = re.sub(r'(?<=[0-9]|true|false|null)\s+(?="[\w-]+\s*":)', ', ', json_str)
            
            # Case C: Closing Brace/Bracket followed by Next Key
            # } "key": -> }, "key":
            json_str = re.sub(r'(?<=[\]}])\s+(?="[\w-]+\s*":)', ', ', json_str)
            
            # 5. Fix missing commas in string arrays (risky but necessary sometimes)
            # "item1" "item2" -> "item1", "item2"
            # Only applied if "item2" is NOT a key (no colon)
            # We skip this for now to avoid breaking sentences in descriptions
            
            # 6. Handle specific Newline cases (fallback)
            # "field": "val"\n"field2":
            json_str = re.sub(r'"\s*\n\s*"', '",\n"', json_str)
            
            return json_str
        except Exception:
            # If regex fails, return original to let standard error handling propagate
            return json_str
    
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
            
    def _normalize_line_items(self, line_items: List[Dict]) -> List[NormalizedLineItem]:
        """Normalize line items for analysis (placeholder)"""
        return [NormalizedLineItem(**item) for item in line_items if isinstance(item, dict)]

    async def _optimize_images(self, files: List[UploadFile]) -> List[UploadFile]:
        """Optimize images (placeholder)"""
        return files
    
    def _extract_trade_data(self, parsed: dict) -> 'TradeData':
        """
        Extract trade data using simple keyword detection from OCR text.
        """
        from .multi_image_analysis_schema import TradeData
        
        page_text = ""
        
        line_items = parsed.get("line_items", [])
        for item in line_items:
            desc = item.get("description", "") or item.get("item", "") or item.get("name", "")
            # CRITICAL FIX: Ensure desc is a string
            if isinstance(desc, list):
                desc = ' '.join(str(d) for d in desc)
            elif not isinstance(desc, str):
                desc = str(desc)
            page_text += f" {desc} "
        
        # CRITICAL FIX: Ensure raw_text and ocr_text are strings
        raw_text = parsed.get("raw_text", "")
        if isinstance(raw_text, list):
            raw_text = ' '.join(str(r) for r in raw_text)
        elif not isinstance(raw_text, str):
            raw_text = str(raw_text)
        
        ocr_text = parsed.get("ocr_text", "")
        if isinstance(ocr_text, list):
            ocr_text = ' '.join(str(o) for o in ocr_text)
        elif not isinstance(ocr_text, str):
            ocr_text = str(ocr_text)
        
        page_text += " " + raw_text
        page_text += " " + ocr_text
        page_text = page_text.lower()
        
        trade_anchors = [
            "trade in", "trade-in", "tradein", "trade:",
            "trade allowance", "trade value", "allowance",
            "payoff", "lien payoff", "net trade",
            "trade difference", "equity"
        ]
        
        trade_anchor_found = any(anchor in page_text for anchor in trade_anchors)
        
        if not trade_anchor_found:
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )
        
        money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        trade_allowance = None
        trade_payoff = None
        
        allowance_keywords = [
            "trade allowance", "allowance", "trade value",
            "trade-in value", "trade in value", "trade:"
        ]
        
        for keyword in allowance_keywords:
            if keyword in page_text:
                idx = page_text.find(keyword)
                snippet = page_text[idx:idx+100]
                
                match = re.search(money_pattern, snippet)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    try:
                        trade_allowance = float(amount_str)
                        break
                    except ValueError:
                        continue
        
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
        
        equity = None
        negative_equity_amount = None
        trade_status = "No trade identified"
        
        trade_present = (
            trade_allowance is not None or 
            trade_payoff is not None or
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
        
        else:
            trade_status = "Trade mentioned in document (values not extracted)"
        
        return TradeData(
            trade_allowance=trade_allowance,
            trade_payoff=trade_payoff,
            equity=equity,
            negative_equity=negative_equity_amount,
            status=trade_status
        )
    
    async def analyze_images(self, files: List[UploadFile]) -> 'MultiImageAnalysisResponse':
        """Main analysis entry point"""
        try:
            validated_files = await self._validate_files(files)
            if not validated_files:
                raise ValueError("No valid image files provided")
            
            base64_images = await self._convert_files_to_base64(validated_files)
            api_response = self._call_openai_api(base64_images)
            parsed = self._parse_api_response(api_response)
            
            # VALIDATE SCORE CONSISTENCY
            score_raw = parsed.get("score")
            score = float(score_raw) if score_raw is not None else 75.0
            
            # ENFORCE SCORE BOUNDS (critical for consistency)
            score = max(0.0, min(100.0, score))
            
            # VALIDATE SCORE BREAKDOWN MATCHES (add this validation)
            score_breakdown = parsed.get("narrative", {}).get("score_breakdown", "")
            if score_breakdown:
                # CRITICAL FIX: Ensure score_breakdown is a string
                if isinstance(score_breakdown, list):
                    score_breakdown = ' '.join(str(item) for item in score_breakdown)
                elif not isinstance(score_breakdown, str):
                    score_breakdown = str(score_breakdown)
                
                # Extract final score from breakdown
                import re
                final_match = re.search(r'Final Score:\s*(\d+(?:\.\d+)?)', score_breakdown)
                if final_match:
                    breakdown_score = float(final_match.group(1))
                    # Allow 0.5 point tolerance for rounding
                    if abs(breakdown_score - score) > 0.5:
                        # Log warning but don't fail
                        print(f"WARNING: Score mismatch - API returned {score}, breakdown shows {breakdown_score}")
                        # Use the breakdown score as authoritative
                        score = max(0.0, min(100.0, breakdown_score))
            
            # Safe flag parsing with validation
            def parse_flags(flags_data):
                if not flags_data:
                    return []
                parsed_flags = []
                for flag in flags_data:
                    if isinstance(flag, dict):
                        parsed_flags.append(Flag(**flag))
                    elif isinstance(flag, str):
                        # Handle string flags by creating a minimal Flag object
                        parsed_flags.append(Flag(
                            type="info",
                            message=flag,
                            item="General"
                        ))
                return parsed_flags
            
            red_flags = parse_flags(parsed.get("red_flags", []))
            green_flags = parse_flags(parsed.get("green_flags", []))
            blue_flags = parse_flags(parsed.get("blue_flags", []))
            
            # CRITICAL VALIDATION: All flag sections must be populated
            if not red_flags:
                raise ValueError("API failed to generate red_flags - all flag sections must be populated")
            if not green_flags:
                raise ValueError("API failed to generate green_flags - all flag sections must be populated")
            if not blue_flags:
                raise ValueError("API failed to generate blue_flags - all flag sections must be populated")
            
            # REMOVE DUPLICATE FLAG LOGIC - This causes inconsistency!
            # The VSC fair pricing logic below modifies flags AFTER API response
            # This can cause different results on each run
            
            # COMMENTED OUT - Let the AI handle VSC flags based on prompt
            # vsc_data = parsed.get("normalized_pricing", {}).get("vsc")
            # if vsc_data:
            #     vsc_price = vsc_data.get("amount", 0)
            #     fair_market_threshold = 5000
            #     
            #     if vsc_price > 0 and vsc_price <= fair_market_threshold:
            #         red_flags = [f for f in red_flags if "service contract" not in f.message.lower()]
            #         green_flags.append(Flag(
            #             type="green",
            #             message=f"Service contract priced at ${vsc_price:,.2f} - within fair market range",
            #             item="VSC"
            #         ))
            
            # Extract trade data
            trade_data = self._extract_trade_data(parsed)
            
            # REMOVE DUPLICATE NEGATIVE EQUITY FLAG - AI should handle this
            # if trade_data.negative_equity and trade_data.negative_equity > 0:
            #     blue_flags.append(Flag(
            #         type="blue",
            #         message=f"Rolled negative equity of ${trade_data.negative_equity:,.2f} increases amount financed.",
            #         item="Trade"
            #     ))

            # REMOVE DEFAULT FLAG INJECTION - This causes inconsistency!
            # DO NOT add default flags - let AI determine based on actual analysis
            # if not red_flags:
            #     red_flags.append(...)
            
            # Handle buyer_message explicitly to prevent NoneType error
            buyer_msg = parsed.get("buyer_message")
            if buyer_msg is None:
                buyer_msg = "Analysis completed"
            
            # Ensure narrative has all required fields with defaults
            narrative_data = parsed.get("narrative")
            if not isinstance(narrative_data, dict):
                narrative_data = {}
            
            # CRITICAL FIX: Ensure usage of narrative['trade'] is a string, not a dict
            if "trade" in narrative_data and not isinstance(narrative_data["trade"], str):
                if isinstance(narrative_data["trade"], dict):
                     trade_obj = narrative_data["trade"]
                     narrative_data["trade"] = trade_obj.get("status") or trade_obj.get("description") or str(trade_obj)
                else:
                     narrative_data["trade"] = str(narrative_data["trade"])
                
            narrative_defaults = {
                "vehicle_overview": "Vehicle details extracted.",
                "smartbuyer_score_summary": f"Score calculated at {score}",
                "score_breakdown": f"Score breakdown: Final Score: {score}",
                "market_comparison": "Standard market rates applied.",
                "gap_logic": "GAP coverage analysis applied.",
                "vsc_logic": "VSC analysis applied.",
                "apr_bonus_rule": "APR verified.",
                "lease_audit": "N/A",
                "negotiation_insight": "Review identified flags.",
                "final_recommendation": "Proceed with caution based on flags.",
                "trade": trade_data.status if trade_data else "No trade identified."
            }
            
            # Apply defaults for missing keys or None values
            for key, default_val in narrative_defaults.items():
                if key not in narrative_data or narrative_data[key] is None:
                    narrative_data[key] = default_val

            return MultiImageAnalysisResponse(
                score=score,
                buyer_name=parsed.get("buyer_name"),
                dealer_name=parsed.get("dealer_name"),
                logo_text=parsed.get("logo_text"),
                email=parsed.get("email"),
                phone_number=parsed.get("phone_number"),
                address=parsed.get("address"),
                state=parsed.get("state"),
                # CRITICAL FIX: handle explicit None for region
                region=parsed.get("region") or "Outside US",
                badge=self._assign_badge(score),
                sale_price=parsed.get("sale_price"),
                vin_number=parsed.get("vin_number"),
                date=parsed.get("date"),
                buyer_message=buyer_msg,
                red_flags=red_flags,
                green_flags=green_flags,
                blue_flags=blue_flags,
                trade=trade_data,
                normalized_pricing=parsed.get("normalized_pricing") or {},
                apr=parsed.get("apr") or {},
                term=parsed.get("term") or {},
                narrative=narrative_data,
                line_items=parsed.get("line_items", [])
            )
        except Exception as e:
            raise RuntimeError(f"Contract analysis failed: {str(e)}")
