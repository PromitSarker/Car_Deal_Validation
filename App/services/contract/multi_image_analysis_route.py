from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List
from .multi_image_analysis_schema import MultiImageAnalysisResponse
from .multi_image_analysis import MultiImageAnalyzer

router = APIRouter(prefix="/api", tags=["contract"])
analyzer = MultiImageAnalyzer()

@router.post("/contract_analyze", response_model=MultiImageAnalysisResponse)
async def analyze_multiple_images(files: List[UploadFile] = File(...)):
    """
    Analyze multiple contract image files and provide a comprehensive rating.
    
    - **files**: Multiple image files (jpg, png, gif, bmp, webp, pdf, tiff)
    - Accepts up to 10MB per file
    """
    try:
        result = await analyzer.analyze_images(files)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
