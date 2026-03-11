from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import List
from .lease_analysis_schema import LeaseAnalysisResponse, LeaseJsonRequest
from .lease_analysis import LeaseAnalyzer

router = APIRouter(prefix="/api", tags=["lease"])
analyzer = LeaseAnalyzer()


@router.get("/lease_health")
async def health_check():
    """Health check endpoint for lease analyzer"""
    return {
        "status": "healthy",
        "service": "lease_analyzer",
        "version": "2.0",
        "features": ["two_step_analysis", "score_recalculation"]
    }


@router.post("/lease_analyze", response_model=LeaseAnalysisResponse)
async def analyze_lease_upload(
    files: List[UploadFile] = File(
        ...,
        description="Upload image files (jpg, png, gif, bmp, webp, pdf, tiff)"
    ),
    language: str = Form(default="English", description="Language for narrative parts")
):
    """
    Analyze multiple lease document image files (file upload) and provide comprehensive rating.

    - **files**: Multiple image files (jpg, png, gif, bmp, webp, pdf, tiff) — up to 10MB per file
    - **language**: Language for narrative parts (default: English)
    """
    try:
        result = await analyzer.analyze_lease_images(files, language=language)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail=f"Analysis timeout: The lease analysis is taking longer than expected. Please try again or use fewer images. Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")


@router.post("/lease_analyze/json", response_model=LeaseAnalysisResponse)
async def analyze_lease_json(request: LeaseJsonRequest):
    """
    Analyze lease document via JSON body.

    - **data**: Pre-extracted JSON data dict for lease analysis
    - **language**: Language for narrative parts (default: English)
    """
    try:
        result = await analyzer.analyze_lease_images(language=request.language, parsed_data=request.data)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        error_msg = str(e)
        if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail=f"Analysis timeout: The lease analysis is taking longer than expected. Please try again or use fewer images. Error: {error_msg}"
            )
        raise HTTPException(status_code=500, detail=f"Analysis failed: {error_msg}")
