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
from App.services.contract.multi_image_analysis_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.rate_helper.audit_classifier import AuditClassifier, AuditClassification
from App.services.rate_helper.gap_logic import GAPLogic, GAPRecommendation
from App.services.rate_helper.audit_flags import AuditFlagBuilder, AuditFlag
from App.services.rate_helper.audit_summary import AuditSummary

load_dotenv()

# Fix: Rename class to match what lease_analysis_route.py expects
class LeaseAnalyzer:
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.pdf', '.tiff'}
    MAX_FILE_SIZE = 10 * 1024 * 1024
    
    # Increase timeout and add retry logic
    API_TIMEOUT = 120  # Increased from 60 to 120 seconds
    MAX_RETRIES = 2
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("GROQ_MODEL", "gpt-4.1-mini")
        self.api_url = "https://api.openai.com/v1/chat/completions"
        self.system_prompt = self._load_lease_system_prompt()
        self.ocr_normalizer = OCRNormalizer()
        self.discount_detector = DiscountDetector()
        self.audit_classifier = AuditClassifier()
        self.gap_logic = GAPLogic()
        self.flag_builder = AuditFlagBuilder()
    
    def _load_lease_system_prompt(self) -> str:
        """Load comprehensive lease analysis system prompt"""
        return """
You are SmartBuyer AI Lease Audit Engine.
Analyze the document image and return valid JSON only.

DOCUMENT TYPE DETECTION & MAPPING
First, determine if this is a LEASE AGREEMENT or a RETAIL INSTALLMENT CONTRACT (PURCHASE).

SCENARIO A: LEASE AGREEMENT
Map fields as follows:
- cap_cost: Agreed Upon Value, Gross Capitalized Cost, or Adjusted Capitalized Cost.
- residual_value: Residual Value.
- term: Lease term (months).
- apr: Calculate from Money Factor (MF * 2400) or extract disclosed Rent Charge rate.

SCENARIO B: RETAIL INSTALLMENT / PURCHASE CONTRACT (FALLBACK)
If this is a PURCHASE contract (not a lease), map purchase values:
- cap_cost: Amount Financed (total principal including negative equity).
- residual_value: 0
- term: Number of Payments or Term.
- apr: Annual Percentage Rate or APR.
- normalized_pricing.selling_price: Cash Price or Vehicle Price.

CRITICAL RULES
- cap_cost must be a number and never null if any price exists.
- trade section is required with allowance, payoff, equity, negative_equity, and status.
- Provide term.months and apr.rate when present.
- Extract ALL line_items with description and amount.

OUTPUT SCHEMA (STRICT JSON ONLY)
Return a valid JSON object with no markdown, no comments, and no trailing commas.

{
  "score": 75,
  "buyer_name": "string or null",
  "dealer_name": "string or null",
  "logo_text": "string or null",
  "email": "string or null",
  "phone_number": "string or null",
  "address": "string or null",
  "state": "string or null",
  "region": "string or null",
  "badge": "string",
  "cap_cost": 0,
  "residual_value": 0,
  "vin_number": "string",
  "date": "YYYY-MM-DD",
  "buyer_message": "string",
  "raw_text": "string",
  "red_flags": [],
  "green_flags": [],
  "blue_flags": [],
  "normalized_pricing": {
    "selling_price": 0,
    "msrp": null,
    "down_payment": null,
    "amount_financed": null,
    "total_fees": null,
    "total_taxes": null
  },
  "apr": {
    "rate": 0,
    "money_factor": null
  },
  "term": {
    "months": 0
  },
  "trade": {
    "trade_allowance": null,
    "trade_payoff": null,
    "equity": null,
    "negative_equity": null,
    "status": "string"
  },
  "bundle_abuse": { "active": false, "deduction": 0 },
  "line_items": [
    { "description": "string", "amount": 0 }
  ],
  "narrative": {
    "vehicle_overview": "string",
    "smartbuyer_score_summary": "string",
    "market_comparison": "string",
    "gap_logic": "string",
    "vsc_logic": "string",
    "apr_bonus_rule": "string",
    "lease_audit": "string",
    "trade": "string",
    "negotiation_insight": "string",
    "final_recommendation": "string"
  }
}
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
        """Call OpenAI API with lease documents (with retry logic)"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        content = [
            {
                "type": "text",
                "text": """
Analyze these lease documents comprehensively.
Return ONLY valid JSON matching the schema in the system prompt. No markdown, no explanations.
                """
            }
        ]
        
        image_detail = "high" if len(base64_images) <= 2 else "auto"
        for base64_image in base64_images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": image_detail
                }
            })
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": content}
            ],
            "temperature": 0,
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
        """Parse API response with fallback repair logic"""
        try:
            content = response["choices"][0]["message"]["content"]

            # Remove markdown fences if present
            if "```" in content:
                content = content.replace("```json", "").replace("```", "")

            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            if json_start < 0 or json_end <= json_start:
                raise ValueError("No JSON content found")

            clean_content = content[json_start:json_end].strip()

            # Remove JS-style comments if model added them
            import re
            clean_content = re.sub(r'//.*?(\r?\n)', r'\1', clean_content)
            clean_content = re.sub(r'/\*.*?\*/', '', clean_content, flags=re.S)

            # Remove trailing commas before } or ]
            clean_content = re.sub(r',\s*([}\]])', r'\1', clean_content)

            # Fix missing commas between objects/arrays and keys
            clean_content = re.sub(r'([}\]])\s*("(?=[^"]*"\s*:))', r'\1,\2', clean_content)
            # Fix missing commas between string values and next key
            clean_content = re.sub(r'(")\s*("(?=[^"]*"\s*:))', r'\1,\2', clean_content)

            return json.loads(clean_content)
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            raise RuntimeError(f"Lease analysis failed: {str(e)}")
    
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

    async def analyze_lease_images(self, files: List[UploadFile]) -> MultiImageAnalysisResponse:
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
            # Fix: Safe float conversion handling None/null values
            raw_cap_cost = parsed.get("cap_cost")
            vehicle_price = float(raw_cap_cost) if raw_cap_cost is not None else 0.0
            
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
            # Fix: Safe float conversion handling None/null values
            raw_score = parsed.get("score")
            base_score = float(raw_score) if raw_score is not None else 75.0
            
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
            
            # Ensure required narrative fields exist to prevent validation errors
            # effectively removing "default outputs" (filler text)
            required_narrative_fields = [
                "vehicle_overview", "smartbuyer_score_summary", "market_comparison",
                "gap_logic", "vsc_logic", "apr_bonus_rule", "lease_audit",
                "trade", "negotiation_insight", "final_recommendation"
            ]
            
            for field in required_narrative_fields:
                if field not in narrative_obj or narrative_obj[field] is None:
                    narrative_obj[field] = ""
            
            # Use calculated trade status only if AI completely failed to provide trade narrative
            if not narrative_obj.get("trade"):
                 narrative_obj["trade"] = trade_data.status

            narrative = Narrative(**narrative_obj)
            
            # Fix: Handle None values for required string fields
            buyer_msg = parsed.get("buyer_message")
            if not buyer_msg:
                buyer_msg = ""

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
                selling_price=parsed.get("cap_cost"),
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
            raise RuntimeError(f"Lease analysis failed: {str(e)}")
