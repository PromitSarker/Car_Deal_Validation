import re
from typing import Optional
from .ocr_normalization_schema import NormalizedLineItem
from .ocr_keyword_dictionary import OCRKeywordDictionary

class OCRNormalizer:
    """
    Keyword-based normalization layer for dealer terms OCR.
    
    Rules:
    - One line item = one category (first match wins by priority)
    - Discounts ALWAYS negative
    - Products/fees ALWAYS positive (unless explicit credit)
    - No guessing - unknown items marked AMBIGUOUS
    """
    
    def __init__(self):
        self.patterns_by_priority = OCRKeywordDictionary.get_patterns_by_priority()
    
    def _clean_text(self, text: str) -> str:
        """Normalize text for matching"""
        # Lowercase
        text = text.lower()
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove common punctuation (keep spaces)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _extract_amount(self, amount_raw: str) -> float:
        """Extract numeric amount from raw string"""
        # Remove currency symbols, commas, spaces
        cleaned = re.sub(r'[^\d.-]', '', str(amount_raw))
        try:
            return float(cleaned)
        except ValueError:
            return 0.0
    
    def _apply_sign_rule(self, amount: float, sign_rule: str, category: str) -> float:
        """
        Apply global sign normalization rules.
        
        CRITICAL RULES:
        - Discounts/rebates/incentives → ALWAYS negative
        - Products/fees → ALWAYS positive
        """
        abs_amount = abs(amount)
        
        # Force negative for discounts (even if OCR shows positive)
        if category == "DISCOUNT_INCENTIVE":
            return -abs_amount
        
        # Force positive for products/fees
        if sign_rule == "+":
            return abs_amount
        
        # Explicit force rule from pattern
        if sign_rule == "-":
            return -abs_amount
        
        # Default: preserve sign
        return amount
    
    def normalize_line_item(
        self,
        raw_text: str,
        amount_raw: str
    ) -> NormalizedLineItem:
        """
        Classify and normalize a single OCR line item.
        
        Args:
            raw_text: Original OCR text
            amount_raw: Raw amount string from OCR
            
        Returns:
            NormalizedLineItem with classification and normalized amount
        """
        cleaned_text = self._clean_text(raw_text)
        amount = self._extract_amount(amount_raw)
        
        # Match in priority order (1-10)
        for priority in sorted(self.patterns_by_priority.keys()):
            patterns = self.patterns_by_priority[priority]
            
            for pattern in patterns:
                # Simple substring match (case-insensitive, cleaned)
                if pattern.pattern.lower() in cleaned_text:
                    # Match found - apply normalization
                    normalized_amount = self._apply_sign_rule(
                        amount,
                        pattern.default_sign,
                        pattern.normalized_category
                    )
                    
                    return NormalizedLineItem(
                        raw_text=raw_text,
                        amount_raw=amount_raw,
                        amount_normalized=normalized_amount,
                        normalized_category=pattern.normalized_category,
                        normalized_label=pattern.normalized_label,
                        matched_keyword=pattern.pattern,
                        confidence_score=1.0  # Direct keyword match = high confidence
                    )
        
        # No match found - mark as AMBIGUOUS
        return NormalizedLineItem(
            raw_text=raw_text,
            amount_raw=amount_raw,
            amount_normalized=amount,  # Preserve original sign
            normalized_category="AMBIGUOUS",
            normalized_label="Unknown Item",
            matched_keyword=None,
            confidence_score=0.0  # No match = low confidence
        )