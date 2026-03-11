from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from typing import List
from .multi_image_analysis_schema import MultiImageAnalysisResponse, ContractJsonRequest
from .multi_image_analysis import MultiImageAnalyzer

router = APIRouter(prefix="/api", tags=["contract"])
analyzer = MultiImageAnalyzer()


@router.post("/contract_analyze", response_model=MultiImageAnalysisResponse)
async def analyze_contract_upload(
    files: List[UploadFile] = File(
        ...,
        description="Upload image files (jpg, png, gif, bmp, webp, pdf, tiff)"
    ),
    language: str = Form(default="English", description="Language for narrative parts")
):
    """
    Analyze multiple contract image files (file upload) and provide a comprehensive rating.

    - **files**: Multiple image files (jpg, png, gif, bmp, webp, pdf, tiff) — up to 10MB per file
    - **language**: Language for narrative parts (default: English)
    """
    try:
        result = await analyzer.analyze_images(files, language=language)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.post("/contract_analyze/json", response_model=MultiImageAnalysisResponse)
async def analyze_contract_json(request: ContractJsonRequest):
    """
    Analyze contract via JSON body.

    - **data**: Pre-extracted JSON data dict for contract analysis
    - **language**: Language for narrative parts (default: English)
    """
    try:
        result = await analyzer.analyze_images(language=request.language, parsed_data=request.data)
        return result
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
