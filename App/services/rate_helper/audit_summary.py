from typing import List, Dict, Optional
from .audit_flags import AuditFlag

class AuditSummary:
    """Generate clean audit summary with explicit labels"""
    
    @staticmethod
    def generate_summary(flags: List[AuditFlag]) -> str:
        """
        Generate audit summary with explicit labels.
        
        Format:
        ✅ Online Price Advantage
        ⚠️ Conditional Finance Incentive (Finance Certificate)
        ❌ Overpriced Add-On Package (Propack Plus)
        ⚠️ Long-Term Loan Risk
        ℹ️ GAP Recommended (Advisory)
        """
        if not flags:
            return "No significant issues detected."
        
        # Group flags by type
        green_flags = [f for f in flags if f.type == "green"]
        blue_flags = [f for f in flags if f.type == "blue"]
        red_flags = [f for f in flags if f.type == "red"]
        
        lines = []
        
        # Green flags (positive)
        for flag in green_flags:
            lines.append(f"✅ {flag.category}")
        
        # Blue flags (advisory/transparency)
        for flag in blue_flags:
            lines.append(f"⚠️ {flag.category}")
        
        # Red flags (issues)
        for flag in red_flags:
            lines.append(f"❌ {flag.category}")
        
        return "\n".join(lines)
    
    @staticmethod
    def validate_flag_clarity(flags: List[AuditFlag]) -> List[str]:
        """
        Validate that flags have clear, explicit labels.
        
        Returns list of validation issues.
        """
        issues = []
        
        vague_terms = ["item", "product", "fee", "charge", "unknown"]
        
        for flag in flags:
            # Check for vague category names
            if any(term in flag.category.lower() for term in vague_terms):
                if flag.category.lower() in vague_terms:
                    issues.append(f"Vague flag category: '{flag.category}'")
            
            # Check for empty messages
            if not flag.message or len(flag.message.strip()) < 10:
                issues.append(f"Flag '{flag.category}' has insufficient message")
        
        return issues