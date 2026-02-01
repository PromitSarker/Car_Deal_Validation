from typing import Optional, Dict
from pydantic import BaseModel

class GAPRecommendation(BaseModel):
    """GAP coverage recommendation logic"""
    recommended: bool
    is_advisory: bool = True  # Always advisory, never penalty
    reason: str
    message: str

class GAPLogic:
    """
    GAP recommendation logic (buyer-protection focused).
    
    Rules:
    - Used vehicle + Term ≥ 72 months + Down ≤ $1,000 + Backend add-ons present
    - SOFT ADVISORY only (no penalty, no dealer judgment)
    """
    
    @staticmethod
    def evaluate_gap_need(
        is_used: bool,
        term_months: Optional[int],
        down_payment: Optional[float],
        amount_financed: Optional[float],
        vehicle_price: Optional[float],
        has_backend_products: bool,
        gap_present: bool
    ) -> GAPRecommendation:
        """
        Evaluate if GAP should be recommended.
        
        Returns:
            GAPRecommendation with advisory messaging
        """
        # Default: No recommendation
        if gap_present:
            return GAPRecommendation(
                recommended=False,
                is_advisory=False,
                reason="GAP already included",
                message=""
            )
        
        # Check conditions for recommendation
        conditions_met = []
        
        if is_used:
            conditions_met.append("used vehicle")
        
        if term_months and term_months >= 72:
            conditions_met.append("extended loan term (≥72 months)")
        
        if down_payment is not None and down_payment <= 1000:
            conditions_met.append("minimal down payment")
        
        if amount_financed and vehicle_price and amount_financed >= vehicle_price:
            conditions_met.append("financing full vehicle price or more")
        
        if has_backend_products:
            conditions_met.append("backend products included")
        
        # Recommend if 3+ conditions met
        if len(conditions_met) >= 3:
            reason = ", ".join(conditions_met)
            return GAPRecommendation(
                recommended=True,
                is_advisory=True,
                reason=reason,
                message=(
                    "GAP coverage is recommended due to loan structure. "
                    "GAP can protect against financial loss if the vehicle is totaled before "
                    "the loan is paid off, especially with extended terms or negative equity. "
                    "This is a buyer-protection recommendation, not a dealer requirement."
                )
            )
        
        return GAPRecommendation(
            recommended=False,
            is_advisory=False,
            reason="Conditions not met for recommendation",
            message=""
        )