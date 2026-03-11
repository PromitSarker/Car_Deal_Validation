from pydantic import BaseModel, Field
from typing import List, Optional

class Flag(BaseModel):
    type: str = Field(..., description="Flag type in 10 words or less")
    message: str = Field(..., description="Detailed explanation")
    deduction: Optional[int] = Field(default=None, description="Points deducted (red flags)")
    bonus: Optional[int] = Field(default=None, description="Points added (green flags)")
    item: str = Field(..., description="Item name")

class ContractRatings(BaseModel):
    document_clarity: float
    contract_terms_completeness: float
    risk_assessment: float
    financial_transparency: float
    legal_compliance: float

class NormalizedPricing(BaseModel):
    gap_cap: Optional[float] = None
    vsc_cap: Optional[float] = None
    bundle_total: Optional[float] = None

class APRData(BaseModel):
    listed: Optional[float] = None
    bonus: int = 0
    source: Optional[str] = None

class TermData(BaseModel):
    months: Optional[int] = None
    risk_deduction: int = 0

class TradeData(BaseModel):
    """Trade-in information with equity calculation"""
    trade_allowance: Optional[float] = Field(default=None, description="Trade-in allowance/value")
    trade_payoff: Optional[float] = Field(default=None, description="Remaining loan payoff on trade")
    equity: Optional[float] = Field(default=None, description="Positive equity (allowance - payoff) or None")
    negative_equity: Optional[float] = Field(default=None, description="Negative equity (payoff - allowance) or None")
    status: str = Field(default="No trade identified", description="Trade status message")

class Narrative(BaseModel):
    vehicle_overview: str
    smartbuyer_score_summary: str
    score_breakdown: Optional[str] = None
    market_comparison: str
    gap_logic: str
    vsc_logic: str
    apr_bonus_rule: str
    lease_audit: str
    trade: str = Field(default="No trade identified.", description="Trade section - always required")
    negotiation_insight: str
    final_recommendation: str

class RatingJsonRequest(BaseModel):
    data: dict = Field(..., description="Pre-extracted JSON data for rating analysis")
    language: str = Field(default="English", description="Language for narrative parts")

class MultiImageAnalysisResponse(BaseModel):
    score: float = Field(..., description="Overall score 0-95")
    discount_incentive: Optional[float] = None
    buyer_name: Optional[str] = None
    dealer_name: Optional[str] = None
    logo_text: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    address: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None
    badge: str = Field(..., description="Gold|Silver|Bronze|Red")
    selling_price: Optional[float] = None
    vin_number: Optional[str] = None
    date: Optional[str] = None
    buyer_message: str
    red_flags: List[Flag] = Field(default_factory=list)
    green_flags: List[Flag] = Field(default_factory=list)
    blue_flags: List[Flag] = Field(default_factory=list)
    normalized_pricing: NormalizedPricing
    apr: APRData
    term: TermData
    trade: TradeData = Field(default_factory=lambda: TradeData())
    quote_type: str = "Audit"
    bundle_abuse: dict = Field(default_factory=dict)
    narrative: Narrative
