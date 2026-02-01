from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List
from .ocr_extractor import OCRExtractor

router = APIRouter(prefix="/api", tags=["extraction"])
extractor = OCRExtractor()

@router.post("/extract_quote_vision", operation_id="extract_quote_data_vision")
async def extract_quote_vision(files: List[UploadFile] = File(...)):
    """
    Extract quote/contract data using ChatGPT Vision OCR.
    
    Returns:
    - Buyer info (name, phone, email, address)
    - Dealer info (name, location, contact)
    - Vehicle details (VIN, year, make, model, MSRP, price)
    - Financial terms (APR, down payment, monthly payment, term)
    - Addons/packages (GAP, VSC, warranties, etc.)
    - Lease-specific data (money factor, residual, acquisition fees)
    - Raw extracted text and quality assessment
    
    Files: Multiple quote/contract images (jpg, png, pdf)
    """
    try:
        if not files:
            raise ValueError("No files provided")
        
        result = await extractor.extract_quote_data(files)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
