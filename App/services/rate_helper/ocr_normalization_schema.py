from typing import Optional, Literal
from pydantic import BaseModel, Field

# Normalized categories (exact list - no more, no less)
NormalizedCategory = Literal[
    "DISCOUNT_INCENTIVE",
    "VSC",
    "GAP",
    "MAINTENANCE",
    "TIRE_WHEEL_PROTECTION",
    "ADDON_PACKAGE",
    "DEALER_FEE",
    "GOV_FEE",
    "TAX",
    "MARKET_ADJUSTMENT",
    "AMBIGUOUS"
]

class NormalizedLineItem(BaseModel):
    """Normalized OCR line item output"""
    raw_text: str = Field(..., description="Original OCR text")
    amount_raw: str = Field(..., description="Raw amount as OCR'd")
    amount_normalized: float = Field(..., description="Normalized signed amount")
    normalized_category: NormalizedCategory = Field(..., description="Classified category")
    normalized_label: str = Field(..., description="Clean label for this item")
    matched_keyword: Optional[str] = Field(None, description="Keyword that triggered match")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Match confidence 0-1")

class KeywordPattern(BaseModel):
    """Single keyword pattern definition"""
    pattern: str = Field(..., description="Keyword or regex to match")
    normalized_category: NormalizedCategory
    normalized_label: str
    default_sign: Literal["+", "-", "force"] = Field(..., description="Sign enforcement rule")
    priority_rank: int = Field(..., ge=1, description="Priority for conflict resolution")