from typing import List, Dict

class DiscountKeywords:
    """
    Keyword dictionary for discount/rebate/incentive detection.
    
    LOCKED - Do not modify without explicit approval.
    Keywords are organized by priority and type.
    """
    
    @staticmethod
    def get_discount_keywords() -> Dict[str, List[str]]:
        """
        Returns discount keywords grouped by type.
        Priority order: CONDITIONAL_FINANCE > CONDITIONAL_LEASE > UNCONDITIONAL > DEALER_DISCOUNT
        """
        return {
            # PRIORITY 1: Conditional Finance Incentives (check FIRST)
            "CONDITIONAL_FINANCE": [
                "finance certificate",
                "finance discount",
                "dealer financing discount",
                "with dealer financing",
                "with approved credit",
                "captive credit",
                "nmac",
                "in-lieu-of",
                "finance incentive",
                "financing incentive",
                "approval financing"
            ],
            
            # PRIORITY 2: Conditional Lease Incentives
            "CONDITIONAL_LEASE": [
                "lease cash",
                "lease certificate",
                "lease incentive",
                "lease bonus",
                "residual support",
                "lease discount"
            ],
            
            # PRIORITY 3: Unconditional Discounts/Rebates
            "UNCONDITIONAL": [
                "discount",
                "rebate",
                "incentive",
                "bonus cash",
                "customer cash",
                "consumer cash",
                "savings",
                "markdown",
                "price adjustment",
                "manufacturer rebate",
                "oem rebate",
                "loyalty",
                "military",
                "conquest"
            ],
            
            # PRIORITY 4: Dealer Discounts (check LAST among discount types)
            "DEALER_DISCOUNT": [
                "dealer discount",
                "internet price adjustment",
                "internet discount",
                "dealer contribution",
                "dealer rebate"
            ]
        }
    
    @staticmethod
    def get_explicit_sign_indicators() -> List[str]:
        """Keywords that indicate explicit negative/credit notation"""
        return [
            "credit",
            "cr",
            "rebate",
            "discount",
            "savings",
            "deduction"
        ]