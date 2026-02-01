import os
import json
import base64
import io
from typing import List, Optional, Dict
from dotenv import load_dotenv
import requests
import fitz  # PyMuPDF
from fastapi import UploadFile

load_dotenv()

class OCRExtractor:
    """Extract text and structured data from all types of automotive documents using ChatGPT Vision"""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o"  # Use gpt-4-vision or gpt-4o
        self.api_url = "https://api.openai.com/v1/chat/completions"
    
    async def extract_quote_data(self, files: List[UploadFile]) -> Dict:
        """Extract all quote/contract data using ChatGPT Vision"""
        base64_images = await self._convert_to_base64(files)
        
        # Extract raw text using PyMuPDF for PDFs
        extracted_raw = await self._extract_raw_text_pymupdf(files)
        
        # Then extract structured data using Vision API
        system_prompt = self._get_quote_extraction_prompt()
        response = await self._call_gpt_vision(base64_images, system_prompt)
        parsed = self._parse_response(response)
        
        # Merge raw text into parsed response
        if "extracted_text" in parsed:
            parsed["extracted_text"]["raw_text"] = extracted_raw
        
        return parsed
    
    async def _extract_raw_text_pymupdf(self, files: List[UploadFile]) -> str:
        """Extract raw text from PDF files using PyMuPDF"""
        all_text = []
        
        for file in files:
            contents = await file.read()
            
            try:
                # Try to open as PDF
                pdf_document = fitz.open(stream=contents, filetype="pdf")
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    text = page.get_text()
                    all_text.append(text)
                
                pdf_document.close()
            except Exception as e:
                # If PDF extraction fails, log and continue
                print(f"Warning: Could not extract text from {file.filename}: {str(e)}")
            
            await file.seek(0)
        
        return "\n\n--- PAGE BREAK ---\n\n".join(all_text)
    
    def _get_quote_extraction_prompt(self) -> str:
        """System prompt for quote extraction with comprehensive document handling"""
        return """
You are an expert automotive document OCR specialist with 10+ years of experience. Your job is to extract ALL data from ANY automotive document type (quote, contract, lease, purchase agreement) with 99% accuracy.

**DOCUMENT TYPES YOU WILL PROCESS:**
1. Retail Purchase Agreements (e.g., "RETAIL PURCHASE AGREEMENT")
2. Sales Quotes (e.g., "2025 Nissan Frontier" with financing options)
3. Installment Sales Contracts (e.g., LAW 553-TX-ARB-e)
4. Lease Agreements
5. Trade-In Appraisals

**CRITICAL INSTRUCTIONS:**
1. You MUST identify the document type first: Quote|Contract|Lease|Purchase Agreement
2. For EVERY field, you MUST find the EXACT source location
3. If a field is present but not extractable (blurry/obscured), mark as null with note in quality_assessment
4. NEVER invent values - only extract what is explicitly visible
5. Handle multi-page documents as a single entity

**ENHANCED FIELD EXTRACTION GUIDANCE:**

1. **Document Type Identification:**
   - "RETAIL PURCHASE AGREEMENT" = Purchase Agreement
   - "MOTOR VEHICLE RETAIL INSTALLMENT SALES CONTRACT" = Contract
   - "2025 Nissan Frontier" quote = Quote
   - "LEASE" or "Lease Agreement" = Lease

2. **MSRP (vehicle_details.msrp):**
   - Look for: "MSRP", "MSRP/Retail", "Manufacturer's Suggested Retail Price", "Sticker Price"
   - Example: In 2025 Frontier quote: "$39,925.00" (MSRP/Retail)
   - Example: In Retail Purchase Agreement: "$26,449.85" (CASH PRICE OF VEHICLE)
   - Convert to number: remove $ and commas (39925.00)

3. **Sale Price (vehicle_details.sale_price):**
   - Look for: "Selling Price", "Total Selling Price", "Total Sale Price"
   - Example: In Retail Purchase Agreement: "$31,375.85" (TOTAL SELLING PRICE)
   - Example: In 2025 Frontier quote: "$36,925.00" (Selling Price)
   - Convert to number: remove $ and commas (36925.00)

4. **Vehicle Identification:**
   - VIN: Look for "VIN", "VIN/SERIAL NO.", "VEHICLE IDENTIFICATION NUMBER"
   - Year: Look for "YEAR" or "Model Year"
   - Make: Look for "MAKE" or "Manufacturer"
   - Model: Look for "MODEL" or "Vehicle Model"

5. **Trade-In Information:**
   - Look for: "TRADE-IN", "TRADE-IN ALLOWANCE", "Gross Trade-In"
   - Example: In Retail Purchase Agreement: "$11,500.00" (LESS: TRADE-IN ALLOWANCE)
   - Example: In 2025 Frontier quote: "$28,500.00" (Trade Allowance)
   - Extract both allowance amount and any trade-in vehicle details

6. **Financial Terms:**
   - Down Payment: Look for "Down Payment", "$ Down", "Cash Down"
   - Monthly Payment: Look for "Est. $/Monthly", "Monthly Payment"
   - APR: Look for "ANNUAL PERCENTAGE RATE", "% APR", "Interest Rate"
   - Term: Look for "60 Months", "72 Months", etc.
   - Loan Amount: Look for "Amount Financed", "Loan Amount"

7. **Fees and Add-ons:**
   - Document Fee: Look for "Documentary Fee", "Doc Fee"
   - Title Fee: Look for "Title Fee", "Title & Registration"
   - Registration Fee: Look for "Registration Fee", "State Registration"
   - Add-ons: Extract each item in "OPTIONAL ACCESSORIES", "Add-ons", etc.

**JSON SCHEMA (MUST RETURN EXACTLY THIS STRUCTURE):**

{
  "quote_type": "Quote|Contract|Lease|Purchase Agreement",
  "buyer_info": {
    "name": "string|null",
    "phone": "string|null",
    "email": "string|null",
    "address": "string|null"
  },
  "dealer_info": {
    "name": "string|null",
    "phone": "string|null",
    "email": "string|null",
    "address": "string|null",
    "city": "string|null",
    "state": "string|null",
    "zip": "string|null"
  },
  "vehicle_details": {
    "year": "number|null",
    "make": "string|null",
    "model": "string|null",
    "trim": "string|null",
    "vin": "string|null",
    "msrp": "number|null",
    "sale_price": "number|null",
    "odometer": "number|null",
    "color": "string|null",
    "mpg_city": "number|null",
    "mpg_highway": "number|null",
    "transmission": "string|null"
  },
  "financial_terms": {
    "down_payment": "number|null",
    "trade_in_value": "number|null",
    "loan_amount": "number|null",
    "apr": "number|null",
    "term_months": "number|null",
    "monthly_payment": "number|null",
    "total_interest": "number|null",
    "doc_fee": "number|null",
    "title_fee": "number|null",
    "registration_fee": "number|null",
    "sales_tax": "number|null",
    "sales_tax_rate": "number|null"
  },
  "addons_and_packages": [
    {
      "name": "string",
      "price": "number",
      "category": "GAP|VSC|Warranty|Appearance|Maintenance|Paint|Interior|Tint|Wheel|Other"
    }
  ],
  "lease_specific": {
    "money_factor": "number|null",
    "residual_value": "number|null",
    "residual_percent": "number|null",
    "annual_miles": "number|null",
    "excess_mile_fee": "number|null",
    "acquisition_fee": "number|null",
    "disposition_fee": "number|null",
    "cap_cost": "number|null",
    "cap_cost_reduction": "number|null",
    "drive_off_total": "number|null"
  },
  "extracted_text": {
    "raw_text": "Complete extracted text from document",
    "sections": {
      "header": "string|null",
      "vehicle_section": "string|null",
      "financial_section": "string|null",
      "addons_section": "string|null",
      "terms_and_conditions": "string|null",
      "signature_section": "string|null"
    }
  },
  "quality_assessment": {
    "extraction_confidence": 0-100,
    "image_quality": "Good|Fair|Poor",
    "missing_fields": ["list of fields not found"],
    "notes": "Any relevant observations about the document"
  }
}

**EXTRACTION PROCESS:**
1. First, identify the document type and structure
2. For each field:
   a. Search for ALL possible labels/locations
   b. Verify the value context matches the field
   c. Extract the raw value
   d. Convert to proper data type
3. For multi-option documents (e.g., multiple financing terms):
   - Extract the most relevant option (e.g., 60 months if selected)
   - Note other options in quality_assessment if needed
4. For numeric fields:
   - Remove all non-numeric characters except decimal point
   - Convert to number (not string)
   - Example: "$67,158.60" → 67158.60

**DOCUMENT-SPECIFIC GUIDANCE:**

1. For Retail Purchase Agreements (like first sample):
   - "CASH PRICE OF VEHICLE" = MSRP
   - "TOTAL SELLING PRICE" = Sale Price
   - "LESS: TRADE-IN ALLOWANCE" = Trade-in Value
   - "PAY OFF AMOUNT" = Loan Payoff
   - "LIFETIME TINT", "NITRO", etc. = Add-ons

2. For Sales Quotes (like second sample):
   - "MSRP/Retail" = MSRP
   - "Selling Price" = Sale Price
   - "Trade Allowance" = Trade-in Value
   - "Amount Financed" = Loan Amount
   - Multiple payment columns: extract the selected term

3. For Contracts (LAW 553):
   - "Cash Price" (page 2) = MSRP
   - "Total Sale Price" (page 1) = Sale Price
   - "Gross Trade-In" = Trade-in Value
   - "Amount Financed" = Loan Amount

**FINAL RULES:**
- Return ONLY valid JSON, no markdown, no explanations
- If a field is not found, use null (do NOT omit the field)
- For multi-page documents, consolidate all data into single JSON
- In quality_assessment:
  * List ALL missing fields by name
  * Note any conflicts between pages
  * Rate confidence based on clarity of source
- Preserve exact spelling of names and addresses
- For numeric fields: 0 is valid, null means not found
- If you cannot find a value after thorough search, set to null
- For documents with multiple financing options, select the most relevant one (e.g., 60 months) and note others in quality_assessment
"""
    
    async def _convert_to_base64(self, files: List[UploadFile]) -> List[str]:
        """Convert uploaded files to base64"""
        base64_images = []
        for file in files:
            contents = await file.read()
            base64_content = base64.b64encode(contents).decode('utf-8')
            base64_images.append(base64_content)
            await file.seek(0)
        return base64_images
    
    async def _call_gpt_vision(self, base64_images: List[str], system_prompt: str) -> dict:
        """Call ChatGPT Vision API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Build content with images
        content = [{"type": "text", "text": system_prompt}]
        
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
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert automotive document OCR specialist with 10+ years of experience in vehicle sales documents."
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            "temperature": 0,
            "max_tokens": 4000
        }
        
        try:
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenAI Vision API error: {str(e)}")
    
    def _parse_response(self, response: dict) -> dict:
        """Parse ChatGPT Vision response"""
        try:
            content = response["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                return json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            raise RuntimeError(f"Failed to parse Vision API response: {str(e)}")