from typing import List, Dict
from .ocr_normalization_schema import KeywordPattern

class OCRKeywordDictionary:
    """Source of truth for OCR term normalization"""
    
    @staticmethod
    def get_patterns() -> List[KeywordPattern]:
        """
        Returns keyword patterns in PRIORITY ORDER.
        First match wins within priority tier.
        """
        return [
            # PRIORITY 1: Discounts / Incentives (MUST be checked first)
            KeywordPattern(
                pattern="rebate",
                normalized_category="DISCOUNT_INCENTIVE",
                normalized_label="Rebate",
                default_sign="-",
                priority_rank=1
            ),
            KeywordPattern(
                pattern="finance certificate",
                normalized_category="DISCOUNT_INCENTIVE",
                normalized_label="Finance Certificate",
                default_sign="-",
                priority_rank=1
            ),
            KeywordPattern(
                pattern="incentive",
                normalized_category="DISCOUNT_INCENTIVE",
                normalized_label="Incentive",
                default_sign="-",
                priority_rank=1
            ),
            KeywordPattern(
                pattern="discount",
                normalized_category="DISCOUNT_INCENTIVE",
                normalized_label="Discount",
                default_sign="-",
                priority_rank=1
            ),
            KeywordPattern(
                pattern="cash back",
                normalized_category="DISCOUNT_INCENTIVE",
                normalized_label="Cash Back",
                default_sign="-",
                priority_rank=1
            ),
            KeywordPattern(
                pattern="manufacturer rebate",
                normalized_category="DISCOUNT_INCENTIVE",
                normalized_label="Manufacturer Rebate",
                default_sign="-",
                priority_rank=1
            ),
            
            # PRIORITY 2: GAP
            KeywordPattern(
                pattern="gap",
                normalized_category="GAP",
                normalized_label="GAP Insurance",
                default_sign="+",
                priority_rank=2
            ),
            KeywordPattern(
                pattern="guaranteed asset protection",
                normalized_category="GAP",
                normalized_label="GAP Insurance",
                default_sign="+",
                priority_rank=2
            ),
            KeywordPattern(
                pattern="debt cancellation agreement",
                normalized_category="GAP",
                normalized_label="GAP Insurance",
                default_sign="+",
                priority_rank=2
            ),
            KeywordPattern(
                pattern="dca",
                normalized_category="GAP",
                normalized_label="GAP Insurance",
                default_sign="+",
                priority_rank=2
            ),
            KeywordPattern(
                pattern="loan/lease payoff protection",
                normalized_category="GAP",
                normalized_label="GAP Insurance",
                default_sign="+",
                priority_rank=2
            ),
            KeywordPattern(
                pattern="debt protection",
                normalized_category="GAP",
                normalized_label="GAP Insurance",
                default_sign="+",
                priority_rank=2
            ),
            
            # PRIORITY 3: VSC / Warranty
            KeywordPattern(
                pattern="vsc",
                normalized_category="VSC",
                normalized_label="Vehicle Service Contract",
                default_sign="+",
                priority_rank=3
            ),
            KeywordPattern(
                pattern="service contract",
                normalized_category="VSC",
                normalized_label="Vehicle Service Contract",
                default_sign="+",
                priority_rank=3
            ),
            KeywordPattern(
                pattern="extended warranty",
                normalized_category="VSC",
                normalized_label="Extended Warranty",
                default_sign="+",
                priority_rank=3
            ),
            KeywordPattern(
                pattern="warranty",
                normalized_category="VSC",
                normalized_label="Warranty",
                default_sign="+",
                priority_rank=3
            ),
            KeywordPattern(
                pattern="protection plan",
                normalized_category="VSC",
                normalized_label="Protection Plan",
                default_sign="+",
                priority_rank=3
            ),
            
            # PRIORITY 4: Maintenance
            KeywordPattern(
                pattern="maintenance",
                normalized_category="MAINTENANCE",
                normalized_label="Maintenance Plan",
                default_sign="+",
                priority_rank=4
            ),
            KeywordPattern(
                pattern="prepaid maintenance",
                normalized_category="MAINTENANCE",
                normalized_label="Prepaid Maintenance",
                default_sign="+",
                priority_rank=4
            ),
            KeywordPattern(
                pattern="service plan",
                normalized_category="MAINTENANCE",
                normalized_label="Service Plan",
                default_sign="+",
                priority_rank=4
            ),
            
            # PRIORITY 5: Tire / Wheel / Protection
            KeywordPattern(
                pattern="tire",
                normalized_category="TIRE_WHEEL_PROTECTION",
                normalized_label="Tire Protection",
                default_sign="+",
                priority_rank=5
            ),
            KeywordPattern(
                pattern="wheel",
                normalized_category="TIRE_WHEEL_PROTECTION",
                normalized_label="Wheel Protection",
                default_sign="+",
                priority_rank=5
            ),
            KeywordPattern(
                pattern="tire and wheel",
                normalized_category="TIRE_WHEEL_PROTECTION",
                normalized_label="Tire & Wheel Protection",
                default_sign="+",
                priority_rank=5
            ),
            
            # PRIORITY 6: Add-On / Package
            KeywordPattern(
                pattern="propack",
                normalized_category="ADDON_PACKAGE",
                normalized_label="ProPack",
                default_sign="+",
                priority_rank=6
            ),
            KeywordPattern(
                pattern="pro pack",
                normalized_category="ADDON_PACKAGE",
                normalized_label="ProPack",
                default_sign="+",
                priority_rank=6
            ),
            KeywordPattern(
                pattern="package",
                normalized_category="ADDON_PACKAGE",
                normalized_label="Add-On Package",
                default_sign="+",
                priority_rank=6
            ),
            KeywordPattern(
                pattern="protection package",
                normalized_category="ADDON_PACKAGE",
                normalized_label="Protection Package",
                default_sign="+",
                priority_rank=6
            ),
            KeywordPattern(
                pattern="accessories",
                normalized_category="ADDON_PACKAGE",
                normalized_label="Accessories",
                default_sign="+",
                priority_rank=6
            ),
            
            # PRIORITY 7: Dealer Fee
            KeywordPattern(
                pattern="doc fee",
                normalized_category="DEALER_FEE",
                normalized_label="Documentation Fee",
                default_sign="+",
                priority_rank=7
            ),
            KeywordPattern(
                pattern="documentation fee",
                normalized_category="DEALER_FEE",
                normalized_label="Documentation Fee",
                default_sign="+",
                priority_rank=7
            ),
            KeywordPattern(
                pattern="dealer fee",
                normalized_category="DEALER_FEE",
                normalized_label="Dealer Fee",
                default_sign="+",
                priority_rank=7
            ),
            KeywordPattern(
                pattern="processing fee",
                normalized_category="DEALER_FEE",
                normalized_label="Processing Fee",
                default_sign="+",
                priority_rank=7
            ),
            KeywordPattern(
                pattern="addendum",
                normalized_category="DEALER_FEE",
                normalized_label="Addendum",
                default_sign="+",
                priority_rank=7
            ),
            
            # PRIORITY 8: Government Fee
            KeywordPattern(
                pattern="registration",
                normalized_category="GOV_FEE",
                normalized_label="Registration Fee",
                default_sign="+",
                priority_rank=8
            ),
            KeywordPattern(
                pattern="title",
                normalized_category="GOV_FEE",
                normalized_label="Title Fee",
                default_sign="+",
                priority_rank=8
            ),
            KeywordPattern(
                pattern="license",
                normalized_category="GOV_FEE",
                normalized_label="License Fee",
                default_sign="+",
                priority_rank=8
            ),
            KeywordPattern(
                pattern="dmv",
                normalized_category="GOV_FEE",
                normalized_label="DMV Fee",
                default_sign="+",
                priority_rank=8
            ),
            KeywordPattern(
                pattern="government fee",
                normalized_category="GOV_FEE",
                normalized_label="Government Fee",
                default_sign="+",
                priority_rank=8
            ),
            
            # PRIORITY 9: Tax
            KeywordPattern(
                pattern="sales tax",
                normalized_category="TAX",
                normalized_label="Sales Tax",
                default_sign="+",
                priority_rank=9
            ),
            KeywordPattern(
                pattern="tax",
                normalized_category="TAX",
                normalized_label="Tax",
                default_sign="+",
                priority_rank=9
            ),
            
            # PRIORITY 10: Market Adjustment
            KeywordPattern(
                pattern="market adjustment",
                normalized_category="MARKET_ADJUSTMENT",
                normalized_label="Market Adjustment",
                default_sign="+",
                priority_rank=10
            ),
            KeywordPattern(
                pattern="adm",
                normalized_category="MARKET_ADJUSTMENT",
                normalized_label="Additional Dealer Markup",
                default_sign="+",
                priority_rank=10
            ),
        ]
    
    @staticmethod
    def get_patterns_by_priority() -> Dict[int, List[KeywordPattern]]:
        """Group patterns by priority for efficient lookup"""
        patterns = OCRKeywordDictionary.get_patterns()
        grouped = {}
        for pattern in patterns:
            if pattern.priority_rank not in grouped:
                grouped[pattern.priority_rank] = []
            grouped[pattern.priority_rank].append(pattern)
        return grouped