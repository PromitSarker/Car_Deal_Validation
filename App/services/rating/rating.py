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

load_dotenv()

class MultiImageAnalyzer:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.pdf', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Increase timeout and add retry logic
    API_TIMEOUT = 120  # Increased from 60 to 120 seconds
    MAX_RETRIES = 2
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "gpt-4.1-mini")
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
- smartbuyer_score_summary (NOT "trust_score_summary")
- market_comparison
- gap_logic
- vsc_logic
- apr_bonus_rule
- lease_audit
- trade (REQUIRED - cannot be omitted)
- negotiation_insight
- final_recommendation

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
    
    def _call_openai_api(self, base64_images: List[str]) -> dict:
        """Call OpenAI API with contract documents (with retry logic)"""
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
        image_detail = "high" if len(base64_images) <= 2 else "auto"
        
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
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 3000
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
        
        if not trade_anchor_found:
            # No trade present - return default
            return TradeData(
                trade_allowance=None,
                trade_payoff=None,
                equity=None,
                negative_equity=None,
                status="No trade identified"
            )
        
        # Step B & C: Extract money values near trade keywords
        import re
        
        # Money pattern: $12,345.67 or 12345.67 or 12,345
        money_pattern = r'\$?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
        
        trade_allowance = None
        trade_payoff = None
        
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
        equity = None
        negative_equity_amount = None
        trade_status = "No trade identified"
        
        # Determine if trade is present
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

    async def analyze_images(self, files: List[UploadFile]) -> MultiImageAnalysisResponse:
        """Main analysis entry point"""
        try:
            validated_files = await self._validate_files(files)
            if not validated_files:
                raise ValueError("No valid image files provided")
            
            # Optional: Optimize images before base64 encoding
            # optimized_files = await self._optimize_images(validated_files)
            # base64_images = await self._convert_files_to_base64(optimized_files)
            
            base64_images = await self._convert_files_to_base64(validated_files)
            api_response = self._call_openai_api(base64_images)
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
            vehicle_price = float(parsed.get("selling_price", 0))
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
            
            # Step 7: Apply audit penalties to score
            base_score = float(parsed.get("score", 75.0))
            adjusted_score = base_score + total_audit_penalty
            adjusted_score = max(0.0, min(100.0, adjusted_score))  # Clamp 0-100
            
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
            
            # Merge audit flags
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
            
            # Parse narrative - accept either smartbuyer_score_summary or legacy trust_score_summary
            narrative_obj = parsed.get("narrative", {})
            if isinstance(narrative_obj, str):
                try:
                    narrative_obj = json.loads(narrative_obj)
                except Exception:
                    narrative_obj = {}
            # Normalize legacy key
            if "trust_score_summary" in narrative_obj and "smartbuyer_score_summary" not in narrative_obj:
                narrative_obj["smartbuyer_score_summary"] = narrative_obj.pop("trust_score_summary")
            # Provide safe defaults for missing narrative fields
            defaults = {
                "vehicle_overview": "Contract analysis",
                "smartbuyer_score_summary": f"Score: {adjusted_score}",
                "market_comparison": "Market analysis pending",
                "gap_logic": "GAP analysis pending",
                "vsc_logic": "VSC analysis pending",
                "apr_bonus_rule": "APR analysis pending",
                "lease_audit": "N/A - Purchase Agreement",
                "trade": trade_data.status,  # Use extracted trade status
                "negotiation_insight": "Negotiation guidance pending",
                "final_recommendation": "Review all terms carefully before signing."
            }
            for k, v in defaults.items():
                narrative_obj.setdefault(k, v)

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
                buyer_message=parsed.get("buyer_message", "Contract analysis completed"),
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
