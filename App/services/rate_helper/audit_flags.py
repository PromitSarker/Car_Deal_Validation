from typing import List, Dict, Optional
from pydantic import BaseModel
from .audit_classifier import AuditClassification

class AuditFlag(BaseModel):
    """Structured audit flag for output"""
    type: str  # "red", "green", "blue"
    category: str  # e.g., "Conditional Finance Incentive", "Overpriced Add-On Package"
    message: str
    item: str
    deduction: Optional[int] = None
    bonus: Optional[int] = None

class AuditFlagBuilder:
    """Build structured flags for audit output"""
    
    @staticmethod
    def build_finance_certificate_flag(
        classification: AuditClassification
    ) -> AuditFlag:
        """Build flag for Finance Certificate detection"""
        return AuditFlag(
            type="blue",  # Transparency issue (soft flag)
            category="Conditional Finance Incentive (Finance Certificate)",
            message=classification.flag_message,
            item="Finance Certificate",
            deduction=classification.penalty_points
        )
    
    @staticmethod
    def build_bundled_package_flag(
        classification: AuditClassification
    ) -> AuditFlag:
        """Build flag for bundled package detection"""
        flag_type = "red" if classification.is_overpriced else "blue"
        category = "Overpriced Add-On Package (Propack Plus)" if classification.is_overpriced else "Bundled Package Disclosure"
        
        return AuditFlag(
            type=flag_type,
            category=category,
            message=classification.flag_message,
            item="Bundled Package",
            deduction=classification.penalty_points if classification.penalty_points < 0 else None
        )
    
    @staticmethod
    def build_gap_advisory_flag(
        gap_message: str
    ) -> AuditFlag:
        """Build advisory flag for GAP recommendation"""
        return AuditFlag(
            type="blue",
            category="GAP Recommended (Advisory)",
            message=gap_message,
            item="GAP Coverage"
        )
    
    @staticmethod
    def build_online_price_advantage_flag(
        discount_amount: float
    ) -> AuditFlag:
        """Build green flag for online price advantage"""
        return AuditFlag(
            type="green",
            category="Online Price Advantage",
            message=f"Discount of ${discount_amount:,.0f} applied for online/cash purchase.",
            item="Pricing"
        )
    
    @staticmethod
    def build_long_term_loan_risk_flag(
        term_months: int
    ) -> AuditFlag:
        """Build blue flag for long-term loan risk"""
        return AuditFlag(
            type="blue",
            category="Long-Term Loan Risk",
            message=(
                f"Extended loan term ({term_months} months) may lead to being underwater on loan. "
                "Consider shorter term if financially feasible."
            ),
            item="Loan Structure"
        )