from typing import Optional, Literal
from pydantic import BaseModel, Field

# Discount types (exact classification)
DiscountType = Literal[
    "UNCONDITIONAL",
    "CONDITIONAL_FINANCE",
    "CONDITIONAL_LEASE",
    "DEALER_DISCOUNT",
    "AMBIGUOUS"
]

# Sign source tracking
SignSource = Literal["explicit", "inferred"]

# Mode awareness
AnalysisMode = Literal["QUOTE", "CONTRACT", "LEASE"]

class DiscountLineItem(BaseModel):
    """Normalized discount/rebate/incentive line item"""
    label_text: str = Field(..., description="Raw OCR text")
    amount_raw: str = Field(..., description="Raw amount as OCR'd")
    amount_normalized: float = Field(..., description="Normalized amount (always negative)")
    sign_source: SignSource = Field(..., description="Whether sign was explicit or inferred")
    discount_type: DiscountType = Field(..., description="Classification of discount")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Classification confidence 0-1")
    mode: AnalysisMode = Field(..., description="Analysis mode context")
    matched_keyword: Optional[str] = Field(None, description="Keyword that triggered classification")

class DiscountTotals(BaseModel):
    """Aggregated discount totals for math validation"""
    total_unconditional: float = Field(0.0, description="Sum of unconditional discounts")
    total_conditional_finance: float = Field(0.0, description="Sum of finance-conditional discounts")
    total_conditional_lease: float = Field(0.0, description="Sum of lease-conditional discounts")
    total_dealer_discount: float = Field(0.0, description="Sum of dealer discounts")
    total_all_discounts: float = Field(0.0, description="Sum of all discounts (negative)")
    count: int = Field(0, description="Total number of discount line items")