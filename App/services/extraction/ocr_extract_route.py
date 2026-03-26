from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List
from .ocr_extractor import OCRExtractor
from .gemini_extractor import GeminiExtractor
import traceback

router = APIRouter(prefix="/api", tags=["extraction"])
extractor = OCRExtractor()
gemini_extractor = GeminiExtractor()

# @router.post("/extract_quote_vision", operation_id="extract_quote_data_vision")
# async def extract_quote_vision(
#     files: List[UploadFile] = File(..., description="Upload quote/contract images (jpg, png, pdf)")
# ):
#     """
#     Extract quote/contract data using ChatGPT Vision OCR.
#     ...
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")
#
#     try:
#         result = await extractor.extract_quote_data(files)
#         return result
#     except Exception as e:
#         ...
#         raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")

@router.post("/extract_quote_vision", operation_id="extract_quote_data_vision")
async def extract_quote_vision(
    files: List[UploadFile] = File(..., description="Upload quote/contract images (jpg, png, pdf)")
):
    """
    Extract quote/contract data using Gemini 2.0 Flash.
    Now the primary extraction endpoint.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    try:
        result = await gemini_extractor.extract_quote_data(files)
        return result
    except RuntimeError as e:
        tb = traceback.format_exc()
        print(f"[extract_quote_vision_gemini] RuntimeError:\n{tb}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        tb = traceback.format_exc()
        print(f"[extract_quote_vision_gemini] Unexpected error:\n{tb}")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")
