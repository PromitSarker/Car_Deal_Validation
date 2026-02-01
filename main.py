from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from App.services.extraction.extract_route import router as extraction_router
from App.services.extraction.document_extract_route import router as document_extract_router
from App.services.extraction.ocr_extract_route import router as ocr_extract_router
from App.services.rating.rating_route import router as rating_router
from App.services.chatbot.chatbot_routes import router as chatbot_router
from App.services.quiz.quiz_routes import router as quiz_router
from App.services.contract.multi_image_analysis_route import router as contract_router
from App.services.lease.lease_analysis_route import router as lease_router
from App.services.rate_helper.discount_schema import DiscountLineItem, DiscountTotals
from typing import List, Optional, Dict
from dotenv import load_dotenv
from fastapi import UploadFile
import os
import json
import base64
import requests
import re

from App.services.contract.multi_image_analysis_schema import (
    MultiImageAnalysisResponse, Flag, NormalizedPricing, 
    APRData, TermData, TradeData, Narrative
)
from App.services.rate_helper.audit_classifier import AuditClassifier, AuditClassification

app = FastAPI(
    title="Document-AI FastAPI", 
    version="1.0.0"
)

# Include all routers (each only ONCE)
app.include_router(extraction_router)
app.include_router(document_extract_router)
app.include_router(ocr_extract_router)
app.include_router(rating_router)
app.include_router(chatbot_router)
app.include_router(quiz_router)
app.include_router(contract_router)
app.include_router(lease_router)

# Add CORS middleware if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


