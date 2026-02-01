from fastapi import APIRouter, HTTPException, File, UploadFile
from typing import List
from .lease_analysis_schema import LeaseAnalysisResponse
from .lease_analysis import LeaseAnalyzer

router = APIRouter(prefix="/api", tags=["lease"])
analyzer = LeaseAnalyzer()

@router.post("/lease_analyze", response_model=LeaseAnalysisResponse)
async def analyze_lease_documents(files: List[UploadFile] = File(...)):
    """
    Analyze multiple lease document image files and provide comprehensive rating.
    
    - **files**: Multiple image files (jpg, png, gif, bmp, webp, pdf, tiff)
    - Accepts up to 10MB per file
    """
    try:
        result = await analyzer.analyze_lease_images(files)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
