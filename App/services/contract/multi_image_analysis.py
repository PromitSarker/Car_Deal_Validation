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
You are **SmartBuyer AI Contract Analysis Engine**. Analyze auto finance contracts comprehensively.

---

## CRITICAL MANDATORY REQUIREMENTS

### 1. ALL FLAG SECTIONS MUST BE POPULATED
- **red_flags**: MUST contain ≥1 real issue OR explicitly state no major issues found
- **green_flags**: MUST contain ≥1 positive element from actual contract analysis
- **blue_flags**: MUST contain ≥1 advisory note OR explicitly state no advisories
- **NEVER return empty arrays []** - causes validation failure

### 2. SCORING RULES
- Do not reduce score without valid reason
- Each deal component triggers only ONE scoring outcome
- Multiple components may stack
- Disclosure failures override pricing evaluation for affected items

---

## SCORING SYSTEM

**Base Score: 100 points**

### RED FLAG DEDUCTIONS (Major Issues)
- Trade-in negative equity (disclosed) ≤$10,000: **-5**
- Trade-in negative equity (disclosed) >$10,000: **-10**
- VSC exceeds fairness threshold: **-10**
- APR >15% and ≤20%: **-10**
- APR >20%: **-15**
- Document fees exceed state limits: **-7**
- GAP insurance overpriced (exceeds cap): **-10**
- Maintenance plans overpriced (>$1,200): **-6**
- Loan term >84 months: **-8**
- **Global disclosure failure** (missing TILA disclosures OR backend products not itemized OR payment reconciliation failure OR negative equity rolled in without disclosure): **-15 (applied ONCE per audit)**
- VSC mileage cap issue (minimal remaining coverage): **-6**
- Missing GAP with zero effective down payment AND negative equity ≤-$1,000: **-10**

### BLUE FLAGS (Advisory Only - ZERO POINT IMPACT)
- APR between 10–15%: **0 points**
- Missing itemized fees: **0 points**
- No breakdown of add-on coverage terms: **0 points**
- Term longer than 72 months (but <84): **0 points**
- Term vs coverage mismatch: **0 points**

### GREEN FLAG BONUSES (Positive Elements)
- VSC within fairness threshold: **+3**
- Competitive APR (<5%): **+5**
- Transparent itemization: **+3**
- Positive trade equity: **+5**
- No unnecessary add-ons: **+3**
- GAP coverage present and fairly priced: **+5**

**MAXIMUM SCORE: 100 | MINIMUM SCORE: 0**

**Score Calculation Example:**
```
Starting Score: 100
  Negative equity >$10,000 (disclosed): -10
  Missing GAP with $0 down + negative equity: -10
  Transparent itemization: +3
  Competitive APR: +5
= Final Score: 88
```

**MUST include detailed score_breakdown in narrative showing ALL deductions/bonuses with reasoning (exclude Blue flags - 0 point impact)**

---

## GAP COVERAGE — AUTHORITATIVE LOGIC

### 1) GAP Pricing Caps
GAP price must be ≤ lowest of:
- $1,200 (standard cap)
- 3% of MSRP
- $1,500 ONLY if MSRP ≥$60,000

### 2) Effective Down Payment Definition
`Effective_Down = Cash_Down + max(Trade_Equity, 0)`
*(Negative trade equity does NOT count toward down payment)*

### 3) GAP Flags & Scoring

**🟢 GREEN FLAG (Fair GAP)**
- **Trigger**: GAP present AND GAP price ≤ cap
- **Score**: +5 points
- **Language**: Positive/neutral only
- **Example**: "GAP coverage included at $895 - within pricing guidelines"

**🔴 RED FLAG (Overpriced GAP)**
- **Trigger**: GAP price > cap
- **Score**: -10 points
- **Language**: Overpricing/reduce or remove allowed
- **Example**: "GAP insurance overpriced at $1,450 - exceeds fair market cap"

**🔴 RED FLAG (High-Risk Financing Without GAP)**
- **Trigger**: ALL must be true:
  * GAP NOT present
  * Effective_Down = $0
  * Trade_Equity ≤ -$1,000
- **Score**: -10 points
- **Language**: Protection advisory focused on risk (NOT dealer fault)
- **Example**: "High-Risk Financing Without GAP: No GAP coverage included with zero effective down payment and more than $1,000 in negative equity, creating elevated total-loss risk."

### 4) GAP Logic Rules — NEVER DO THESE:
❌ Use loan term to trigger GAP flags
❌ Apply multiple penalties to GAP (only ONE outcome per deal)
❌ Flag missing GAP without BOTH $0 effective down AND ≥$1,000 negative equity
❌ Use "average cost" language
❌ Convert GREEN flag into BLUE or RED flag

### 5) GAP Scenario Summary
| Scenario | Flag | Score |
|----------|------|-------|
| GAP ≤ cap | 🟢 GREEN | +5 |
| GAP > cap | 🔴 RED | -10 |
| GAP missing + $0 effective down + ≤-$1,000 trade equity | 🔴 RED | -10 |
| GAP missing (otherwise) | — | 0 |

---

## SELLING PRICE FIELD DEFINITION

**"selling_price" MUST contain vehicle cash price ONLY**

### Extraction Priority (in order):
1. "Cash Price" or "Vehicle Price" in itemization section (pre-tax, pre-fees)
2. If not found, "Selling Price" from vehicle description area
3. **NEVER use:**
   - ❌ Total Sale Price from Truth-in-Lending
   - ❌ Amount Financed from Truth-in-Lending
   - ❌ Total of Payments
   - ❌ Any value including taxes/fees/backend products

**Example:**
- ✅ selling_price = $52,068.78 (vehicle cash price)
- ❌ NOT $67,158.60 (Amount Financed with taxes/fees/products)

---

## SERVICE CONTRACT (VSC) ANALYSIS RULES

### Pricing Assessment (ONE OUTCOME PER VSC)
- VSC price **below** market threshold: **GREEN FLAG (+3)**
- VSC price **above** market threshold: **RED FLAG (-10)**
- VSC mileage cap issue (minimal remaining coverage): **RED FLAG (-6)**
- **Critical Rule**: Only ONE penalty or bonus per VSC. Never stack pricing + mileage penalties.

### Context-Based Analysis (Advisory Only - NO POINT DEDUCTIONS)
- Mileage restrictions: advisory only
- Term vs coverage mismatch: advisory only
- New vs used vehicle context: advisory only
- **All percentage-based logic REMOVED** (no VSC as % of vehicle value)

### Flag Logic (One-Outcome Rule)
- Price below threshold + adequate coverage = **GREEN FLAG (+3)**
- Price below threshold BUT mileage cap issues = **RED FLAG (-6)** [mileage takes precedence]
- Price above threshold = **RED FLAG (-10)** [pricing takes precedence]
- VSC not itemized = **GLOBAL -15 disclosure penalty** (pricing evaluation suppressed)

---

## FLAG DEFINITIONS

### 🟢 GREEN FLAGS
✅ Fair, transparent, consumer-friendly terms

- **VSC within fairness cap**: Price ≤ SmartBuyer cap (+3)
- **GAP coverage fairly priced**: GAP present and within cap (+5)
- **Competitive APR (<5%)**: Well below market average (+5)
- **Transparent itemization**: All fees/add-ons clearly listed (+3)
- **Positive trade equity**: Trade value reduces financed amount (+5)
- **No unnecessary add-ons**: Only relevant products included (+3)

### 🔵 BLUE FLAGS (ZERO SCORE IMPACT)
⚠️ Informational only - does NOT affect score

- **APR 10–15%**: Higher cost—consider better rates (0)
- **Missing itemized fees**: Total shown but not broken down (0)
- **No add-on coverage breakdown**: Product listed but details missing (0)
- **Term >72 months**: Loan >6 years increases interest (0)
- **Term vs coverage mismatch**: Finance term exceeds coverage (0)

### 🔴 RED FLAGS
❌ High-risk, overpriced, or non-compliant terms

- **Significant Negative Trade Equity**: >$10,000 rolled into loan (-10)
- **Trade-in negative equity ≤$10,000 (disclosed)**: Amount rolled into new loan (-5)
- **VSC exceeds fairness cap**: Above SmartBuyer threshold (-10)
- **APR >15%**: High-cost financing/subprime risk (-10)
- **APR >20%**: Extremely high-cost financing (-15)
- **Document fees exceed state limits**: Violates max allowable fee (-7)
- **GAP insurance overpriced**: Exceeds pricing cap (-10)
- **Maintenance plans overpriced (>$1,200)**: Above market value (-6)
- **Loan term >84 months**: 7+ years—CFPB warns against (-8)
- **Global disclosure failure**: Missing TILA/itemization/payment reconciliation (-15, once per audit)
- **VSC mileage cap issue**: Minimal remaining coverage (-6)
- **High-Risk Financing Without GAP**: No GAP + $0 effective down + >$1,000 negative equity (-10)

### Flag Message Format Requirements
- Negative equity >$10,000: Use "Significant Negative Trade Equity" with amount exceeds $10,000
- Missing GAP risk: Use "High-Risk Financing Without GAP" with zero effective down + negative equity
- Do NOT mention specific dollar thresholds beyond $10,000 distinction
- Do NOT imply dealer misconduct/fault
- Each distinct issue = separate flag (never combine)

---

## NARRATIVE ANALYSIS STRUCTURE (REQUIRED)

The "narrative" object MUST be analytical, descriptive and contain these specific fields:

- **vehicle_overview**: A analytic overview of Year, Make, Model, VIN, Mileage, New/Used atleast 100 words
- **smartbuyer_score_summary**: A analytic overview of why score was given (price, rate, add-ons). Also include Score breakdown. Should have atleast 100 words
- **score_breakdown**: Itemized deductions/bonuses ONLY (exclude Blue flags)
- **market_comparison**: Deal vs current market rates and Fair Market Value. Should have atleast 100 words
- **gap_logic**: GAP analysis using authoritative logic (present/absent, pricing vs cap, $0 down + negative equity check). Should have atleast 100 words
- **vsc_logic**: VSC analysis (price vs threshold OR mileage cap - one outcome only). Should have atleast 100 words
- **apr_bonus_rule**: Detailed APR analysis (marked up? subvented?). Should have atleast 100 words
- **lease_audit**: Lease notes or "Not a lease"
- **negotiation_insight**: Specific buyer talking points. Make it analytical and detailed. Should have atleast 100 words
- **final_recommendation**: Do not be direct, recommend what steps should take in suggestive way. Should have atleast 100 words
- **trade**: Trade-in detailed analysis (equity amount, allowance vs payoff, status) in details. Should have atleast 100 words

---

## TOTAL SALE PRICE & AMOUNT FINANCED (INTERNAL USE)

### For internal validation/calculations:
- When APR = 0.00% AND Finance Charge = $0.00:
  * Amount Financed = Total of Payments
  * Use Amount Financed for backend % calculations
- If Finance Charge > $0.00:
  * Amount Financed = Total of Payments - Finance Charge
- Truth-in-Lending "Amount Financed" is authoritative

### Backend Product Detection (Single Penalty):
- If Amount Financed > Sum of Itemized Totals:
  * Apply ONE -15 global disclosure penalty
  * Do NOT identify specific product types
  * Do NOT apply multiple penalties
  * Suppress pricing evaluation for undisclosed products

### YOU MUST:
1. Extract vehicle cash price → output as "selling_price"
2. Extract Amount Financed from Truth-in-Lending → use internally
3. NEVER overwrite selling_price with Amount Financed

---

## CORE EXTRACTION REQUIREMENTS

Extract and analyze:
1. Vehicle details (VIN, year, make, model, mileage, used/new, MSRP)
2. Financial terms (selling price, APR, term, monthly payment, down payment)
3. ALL line items with EXACT text and amounts (array)
4. GAP coverage with pricing cap validation
5. VSC coverage with mileage-based cap (one outcome)
6. Maintenance plan pricing
7. Doc fees and government fees
8. Buyer/dealer information and contact details
9. Trade information (allowance, payoff, equity)
10. Down payment (critical for GAP logic)

---

## TRADE SECTION (REQUIRED - ALWAYS INCLUDE)

### If trade present:
- State: "Trade identified: $[allowance] allowance, $[payoff] payoff"
- If payoff > allowance: "Negative equity of $[amount] rolled into new loan" (-5 if disclosed)
- If allowance > payoff: "Positive equity of $[amount] applied to purchase" (+5)
- If allowance = payoff: "Trade equity neutral"
- If negative equity NOT disclosed: Global -15 disclosure penalty (not separate)

### If no trade:
- State: "No trade identified."

**This section CANNOT be omitted.**

---

## APR ANALYSIS

### A. APR Disclosure
- APR not shown: Global -15 disclosure penalty
- APR shown: Extract and validate

### B. APR Risk Assessment
- APR >10% AND ≤15%: Blue flag (0 points)
- APR >15% AND ≤20%: Red flag (-10)
- APR >20%: Red flag (-15)

### C. APR Recognition
- APR <10% AND >0%: Favorable rate (note)
- APR <5% AND >0%: Excellent rate (+5)
- APR = 0.00%: Manufacturer-subvented (neutral, no deduction)

---

## STRICT OUTPUT FORMAT REQUIREMENTS

**CRITICAL: Return ONLY valid, parseable JSON. No exceptions.**

### JSON FORMATTING RULES (MANDATORY):
1. Every property separated by comma (,)
2. Every array element separated by comma (,)
3. No trailing commas before } or ]
4. String values use double quotes "
5. Property names use double quotes "
6. Use lowercase: true, false, null (not True, False, None)
7. NO comments (// or /* */)
8. NO markdown code blocks
9. All braces/brackets properly matched
10. Escape quotes in strings with \"

### TOP-LEVEL FIELDS (REQUIRED):
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
- selling_price (VEHICLE CASH PRICE ONLY)
- vin_number
- date
- buyer_message
- red_flags (array)
- green_flags (array)
- blue_flags (array)
- yellow_flags (array)
- normalized_pricing (object)
- apr (object)
- term (object)
- trade (object: trade_allowance, trade_payoff, equity, negative_equity, status)
- bundle_abuse (object)
- narrative (object)
- line_items (array)

### Flag Object Structure (EXACT):
```json
{
  "type": "Short title (≤10 words)",
  "message": "Detailed explanation",
  "item": "Item name (e.g., VSC, GAP, APR, Trade)",
  "deduction": 10.0,  // ONLY red_flags
  "bonus": 5.0       // ONLY green_flags
}
```

**Field Requirements:**
- "type" (REQUIRED): Brief description
- "message" (REQUIRED): Detailed explanation but not long
- "item" (REQUIRED): Item name
- "deduction" (OPTIONAL): Only red_flags
- "bonus" (OPTIONAL): Only green_flags

**Example red flag:**
```json
{
  "type": "VSC exceeds fair market value",
  "message": "Extended warranty priced at $4,500 exceeds fair market threshold for this vehicle MSRP and condition.",
  "item": "VSC",
  "deduction": 10.0
}
```

---

## FINAL VALIDATION RULE

Before returning JSON:
- ✅ selling_price = vehicle cash price ONLY
- ✅ selling_price ≠ Amount Financed (unless no taxes/fees)
- ✅ For 0% APR: Amount Financed > selling_price
- ✅ If selling_price >$100k (normal vehicle), re-check
- ✅ VSC = ONE outcome only (pricing OR mileage)
- ✅ GAP logic follows authoritative rules
- ✅ Do NOT mention dollar thresholds in messages
- ✅ score_breakdown matches final score
- ✅ All deductions have valid reasons
- ✅ Do NOT use loan term for GAP flags
- ✅ Global -15 penalty ONCE per audit only
- ✅ Blue flags = 0 point impact

### CRITICAL JSON CHECK:
1. Arrays have commas: ["item1", "item2"]
2. Objects have commas: {"key1": "val1", "key2": "val2"}
3. No commas before } or ]
4. All quotes closed
5. Valid JSON (bracket/brace matching)

**Return ONLY valid JSON - no markdown, no explanation, no extra text.**
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
                except json.JSONDecodeError as je:
                    # Try to fix common JSON issues
                    print(f"[DEBUG] Initial JSON parse failed: {str(je)}")
                    print(f"[DEBUG] Error at line {je.lineno}, column {je.colno}, position {je.pos}")
                    print(f"[DEBUG] Malformed JSON (first 1000 chars): {json_str[:1000]}")
                    
                    # Show context around the error
                    if je.pos and je.pos > 0:
                        start = max(0, je.pos - 100)
                        end = min(len(json_str), je.pos + 100)
                        print(f"[DEBUG] Error context: ...{json_str[start:end]}...")
                    
                    # Attempt repair
                    json_str = self._attempt_json_repair(json_str)
                    try:
                        parsed = json.loads(json_str)
                        print("[DEBUG] JSON successfully repaired")
                    except json.JSONDecodeError as je2:
                        print(f"[DEBUG] Repair failed: {str(je2)}")
                        print(f"[DEBUG] Error at line {je2.lineno}, column {je2.colno}, position {je2.pos}")
                        print(f"[DEBUG] Repaired JSON (first 1000 chars): {json_str[:1000]}")
                        
                        # Show context around the error after repair
                        if je2.pos and je2.pos > 0:
                            start = max(0, je2.pos - 100)
                            end = min(len(json_str), je2.pos + 100)
                            print(f"[DEBUG] Error context after repair: ...{json_str[start:end]}...")
                        
                        # Try advanced repair as last resort
                        print("[DEBUG] Attempting advanced JSON repair...")
                        json_str = self._advanced_json_repair(json_str)
                        try:
                            parsed = json.loads(json_str)
                            print("[DEBUG] JSON successfully repaired with advanced method")
                        except json.JSONDecodeError as je3:
                            print(f"[DEBUG] Advanced repair also failed: {str(je3)}")
                            # Save the problematic JSON to a file for debugging
                            try:
                                with open('/tmp/failed_json_response.txt', 'w') as f:
                                    f.write(json_str)
                                print("[DEBUG] Full JSON saved to /tmp/failed_json_response.txt")
                            except:
                                pass
                            
                            raise RuntimeError(f"Failed to parse JSON even after repair: {str(je3)}")
                
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
                
                # Normalize flag field names
                parsed = self._normalize_flag_fields(parsed)
                
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
            
            # 3. Fix trailing commas BEFORE other fixes (common error: items = [a, b,])
            json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
            
            # 4. Fix missing commas between objects/arrays FIRST (most common in LLM output)
            # { ... } { ... } -> { ... }, { ... }
            json_str = re.sub(r'}\s+\{', '}, {', json_str)
            # ] [ -> ], [
            json_str = re.sub(r']\s+\[', '], [', json_str)
            # ] { -> ], {
            json_str = re.sub(r']\s+\{', '], {', json_str)
            # } [ -> }, [
            json_str = re.sub(r'}\s+\[', '}, [', json_str)
            
            # 5. Fix missing commas before keys (property separators)
            # These patterns handle: value "key": -> value, "key":
            
            # MULTI-PASS APPROACH: Apply multiple times to catch nested cases
            for _ in range(5):  # Increased from 3 to 5 passes for better coverage
                # String value followed by key (with or without quotes around key)
                json_str = re.sub(r'"(\s+)("[\w\-_]+"\s*:)', r'",\1\2', json_str)
                
                # Number followed by key (including decimals and scientific notation)
                json_str = re.sub(r'([0-9.eE+-]+)(\s+)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
                
                # Boolean/null followed by key
                json_str = re.sub(r'\b(true|false|null)(\s+)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
                
                # Closing brace followed by key
                json_str = re.sub(r'}(\s+)("[\w\-_]+"\s*:)', r'},\1\2', json_str)
                
                # Closing bracket followed by key
                json_str = re.sub(r'](\s+)("[\w\-_]+"\s*:)', r'],\1\2', json_str)
                
                # Handle array elements without commas: "value1" "value2" -> "value1", "value2"
                json_str = re.sub(r'"(\s+)"(?=[^:]*[,\]])', r'",\1"', json_str)
                
                # Handle object properties in arrays missing commas
                # {...} {...} inside arrays
                json_str = re.sub(r'(\{[^{}]*\})(\s+)(\{)', r'\1,\2\3', json_str)
            
            # 6. Handle newline-separated properties (common in LLM output)
            # Apply these after the space-based patterns
            json_str = re.sub(r'"(\s*\n\s*)("[\w\-_]+"\s*:)', r'",\1\2', json_str)
            json_str = re.sub(r'([0-9.eE+-]+)(\s*\n\s*)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
            json_str = re.sub(r'\b(true|false|null)(\s*\n\s*)("[\w\-_]+"\s*:)', r'\1,\2\3', json_str)
            json_str = re.sub(r'}(\s*\n\s*)("[\w\-_]+"\s*:)', r'},\1\2', json_str)
            json_str = re.sub(r'](\s*\n\s*)("[\w\-_]+"\s*:)', r'],\1\2', json_str)
            
            # 7. Fix missing commas after array/object elements within arrays
            # Pattern: ] "key" -> ], "key" (when inside array context)
            json_str = re.sub(r'](\s+)"(?=[^:]*:)', r'],\1"', json_str)
            json_str = re.sub(r'}(\s+)"(?=[^:]*:)', r'},\1"', json_str)
            
            # 8. Fix unclosed strings by ensuring even quotes (last resort)
            # Count quotes - if odd, try to close the last one
            quote_count = json_str.count('"')
            if quote_count % 2 != 0:
                # Find the last quote and check if it needs closing
                last_quote_idx = json_str.rfind('"')
                if last_quote_idx > 0 and last_quote_idx < len(json_str) - 1:
                    # Check if there's a comma or bracket/brace after
                    next_char = json_str[last_quote_idx + 1:].lstrip()
                    if next_char and next_char[0] not in [',', '}', ']']:
                        # Add closing quote before next structural element
                        for i, char in enumerate(next_char):
                            if char in [',', '}', ']', '\n']:
                                insert_pos = last_quote_idx + 1 + len(json_str[last_quote_idx + 1:]) - len(next_char) + i
                                json_str = json_str[:insert_pos] + '"' + json_str[insert_pos:]
                                break
            
            return json_str
        except Exception as e:
            print(f"[DEBUG] JSON repair exception: {str(e)}")
            # If regex fails, return original to let standard error handling propagate
            return json_str
    
    def _normalize_flag_fields(self, parsed: dict) -> dict:
        """
        Normalize flag field names to match the expected schema.
        Handles cases where AI returns 'title' instead of 'type', etc.
        """
        flag_arrays = ['red_flags', 'green_flags', 'blue_flags', 'yellow_flags']
        
        for flag_array_name in flag_arrays:
            if flag_array_name in parsed and isinstance(parsed[flag_array_name], list):
                normalized_flags = []
                for flag in parsed[flag_array_name]:
                    if isinstance(flag, dict):
                        # Map alternative field names to expected ones
                        normalized_flag = {}
                        
                        # Handle 'type' field (may come as 'title', 'name', 'type')
                        normalized_flag['type'] = (
                            flag.get('type') or 
                            flag.get('title') or 
                            flag.get('name') or 
                            'Issue Identified'
                        )
                        
                        # Handle 'message' field (may come as 'message', 'description', 'detail')
                        normalized_flag['message'] = (
                            flag.get('message') or 
                            flag.get('description') or 
                            flag.get('detail') or 
                            flag.get('details') or 
                            'No details provided'
                        )
                        
                        # Handle 'item' field (may come as 'item', 'category', 'subject')
                        normalized_flag['item'] = (
                            flag.get('item') or 
                            flag.get('category') or 
                            flag.get('subject') or 
                            'General'
                        )
                        
                        # Handle optional fields
                        if 'deduction' in flag:
                            normalized_flag['deduction'] = flag['deduction']
                        if 'bonus' in flag:
                            normalized_flag['bonus'] = flag['bonus']
                        
                        normalized_flags.append(normalized_flag)
                    else:
                        # If it's not a dict, skip it
                        continue
                
                parsed[flag_array_name] = normalized_flags
        
        return parsed
    
    def _advanced_json_repair(self, json_str: str) -> str:
        """
        Advanced JSON repair using character-by-character analysis.
        This is a fallback when regex-based repair fails.
        """
        try:
            result = []
            in_string = False
            escape_next = False
            depth = 0
            last_significant_char = None
            i = 0
            
            while i < len(json_str):
                char = json_str[i]
                
                # Handle escape sequences
                if escape_next:
                    result.append(char)
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\' and in_string:
                    escape_next = True
                    result.append(char)
                    i += 1
                    continue
                
                # Track string state
                if char == '"':
                    in_string = not in_string
                    result.append(char)
                    if not in_string:
                        last_significant_char = '"'
                    i += 1
                    continue
                
                # Skip whitespace tracking
                if char in ' \t\n\r':
                    result.append(char)
                    i += 1
                    continue
                
                # If we're in a string, just copy
                if in_string:
                    result.append(char)
                    i += 1
                    continue
                
                # Track structure depth
                if char in '{[':
                    depth += 1
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                if char in '}]':
                    depth -= 1
                    # Check if we need a comma before this closing bracket
                    # Look back for the last significant character
                    if last_significant_char and last_significant_char not in [',', '{', '[', ':']:
                        # We might be missing a comma, but closing brackets don't need one
                        pass
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                # Handle colons
                if char == ':':
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                # Handle commas
                if char == ',':
                    result.append(char)
                    last_significant_char = char
                    i += 1
                    continue
                
                # We're about to process a value or key
                # Check if we need a comma before it
                if last_significant_char and last_significant_char not in [',', '{', '[', ':']:
                    # Look ahead to see what we're processing
                    # If it starts with ", it's likely a key or string value
                    # If it's a number, boolean, null, it's a value
                    # We need a comma if the last char was a closing quote, bracket, or value
                    if last_significant_char in ['"', '}', ']'] or (isinstance(last_significant_char, str) and last_significant_char.isdigit()):
                        # Look ahead to confirm this is a new property
                        look_ahead = json_str[i:i+50].lstrip()
                        if look_ahead.startswith('"'):
                            # Check if there's a colon ahead (indicating a property)
                            if ':' in look_ahead[:look_ahead.find('"', 1) + 10] if '"' in look_ahead[1:] else False:
                                result.append(',')
                
                # Process the character
                result.append(char)
                
                # Track what kind of character this was
                if char.isdigit() or char in 'truefalsnl':  # part of true/false/null or number
                    # Look ahead to get the full value
                    value_start = i
                    while i < len(json_str) and json_str[i] not in ' \t\n\r,}]':
                        i += 1
                    value = json_str[value_start:i]
                    result.append(value[1:])  # We already added the first char
                    last_significant_char = value[-1]
                    continue
                
                last_significant_char = char
                i += 1
            
            return ''.join(result)
        except Exception as e:
            print(f"[DEBUG] Advanced JSON repair failed: {str(e)}")
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
