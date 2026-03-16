from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List
from .ocr_extractor import OCRExtractor
import traceback

router = APIRouter(prefix="/api", tags=["extraction"])
extractor = OCRExtractor()

@router.post("/extract_quote_vision", operation_id="extract_quote_data_vision")
async def extract_quote_vision(
    files: List[UploadFile] = File(..., description="Upload quote/contract images (jpg, png, pdf)")
):
    """
    Extract quote/contract data using ChatGPT Vision OCR.

    Returns:
    - Buyer info (name, phone, email, address)
    - Co-buyer info
    - Dealer info (name, location, contact)
    - Vehicle details (VIN, year, make, model, MSRP, sale price, condition, use purpose)
    - Trade-in details (year/make/model/VIN, allowance, payoff, net trade)
    - TILA disclosures (APR, Finance Charge, Amount Financed, Total of Payments, Total Sale Price)
    - Payment schedule (number, amount, due dates)
    - Full itemization of amount financed (lines 1-5 with all sub-items)
    - Complete fees breakdown (A through N/O rows)
    - Addons/packages (GAP, VSC, warranties, service contracts, etc.)
    - Lease-specific data (money factor, residual, acquisition fees)
    - Lender info and OCCC notice
    - Legal clauses (arbitration, returned payment, liability, etc.)
    - Signatures
    - Raw extracted text per page and verbatim sections
    - Quality assessment

    Files: Multiple quote/contract images (jpg, png, pdf)
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        result = await extractor.extract_quote_data(files)
        return result
    except RuntimeError as e:
        tb = traceback.format_exc()
        print(f"[extract_quote_vision] RuntimeError:\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[extract_quote_vision] Unexpected error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
