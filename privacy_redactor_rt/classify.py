"""Classification engine for sensitive data detection with privacy protection."""

import logging
from typing import List, Optional, Dict, Any
from privacy_redactor_rt.types import Match, BBox
from privacy_redactor_rt.config import ClassificationConfig
from privacy_redactor_rt.patterns import PatternLibrary
from privacy_redactor_rt.validators import (
    luhn_check, 
    get_credit_card_brand,
    validate_phone_number,
    validate_email_rfc,
    is_high_entropy_string,
    extract_digits
)
from privacy_redactor_rt.address_rules import AddressDetector


logger = logging.getLogger(__name__)


class ClassificationEngine:
    """Engine for classifying text content into sensitive data categories."""
    
    def __init__(self, config: ClassificationConfig):
        """
        Initialize classification engine.
        
        Args:
            config: Classification configuration
        """
        self.config = config
        self.pattern_lib = PatternLibrary()
        self.address_detector = AddressDetector(
            use_spacy=config.use_spacy_ner,
            spacy_model=config.spacy_model
        )
        
        # Category-specific confidence thresholds
        self.category_thresholds = {
            'phone': 0.8,
            'credit_card': 0.9,
            'email': 0.7,
            'address': config.address_min_score,
            'api_key': 0.6
        }
        
        logger.info(f"Initialized ClassificationEngine with categories: {config.categories}")
    
    def classify_text(self, text: str, bbox: Optional[BBox] = None) -> List[Match]:
        """
        Classify text content for sensitive data.
        
        Args:
            text: Text content to classify
            bbox: Optional bounding box for the text
            
        Returns:
            List of classification matches
        """
        if not text or not text.strip():
            return []
        
        # Normalize text for processing
        normalized_text = self.pattern_lib.normalize_text(text)
        matches = []
        
        # Process each enabled category
        for category in self.config.categories:
            category_matches = self._classify_category(normalized_text, category, bbox)
            matches.extend(category_matches)
        
        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)
        
        return matches
    
    def _classify_category(self, text: str, category: str, bbox: Optional[BBox]) -> List[Match]:
        """
        Classify text for a specific category.
        
        Args:
            text: Normalized text content
            category: Category to classify for
            bbox: Optional bounding box
            
        Returns:
            List of matches for the category
        """
        if category == 'phone':
            return self._classify_phone(text, bbox)
        elif category == 'credit_card':
            return self._classify_credit_card(text, bbox)
        elif category == 'email':
            return self._classify_email(text, bbox)
        elif category == 'address':
            return self._classify_address(text, bbox)
        elif category == 'api_key':
            return self._classify_api_key(text, bbox)
        else:
            logger.warning(f"Unknown category: {category}")
            return []
    
    def _classify_phone(self, text: str, bbox: Optional[BBox]) -> List[Match]:
        """Classify phone numbers with phonenumbers library validation."""
        matches = []
        patterns = self.pattern_lib.get_patterns('phone')
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                phone_text = match.group()
                
                # Validate with phonenumbers library (US/CA defaults)
                is_valid, formatted = validate_phone_number(phone_text, ['US', 'CA'])
                
                if is_valid:
                    # Extract digits for additional validation
                    digits = extract_digits(phone_text)
                    
                    # Must have at least min_phone_digits
                    if len(digits) >= self.config.min_phone_digits:
                        confidence = self._calculate_phone_confidence(phone_text, formatted)
                        
                        if confidence >= self.category_thresholds['phone']:
                            masked_text = self.get_masked_preview(phone_text, 'phone')
                            
                            matches.append(Match(
                                category='phone',
                                confidence=confidence,
                                masked_text=masked_text,
                                bbox=bbox or BBox(0, 0, 100, 20, 1.0)
                            ))
        
        return matches
    
    def _classify_credit_card(self, text: str, bbox: Optional[BBox]) -> List[Match]:
        """Classify credit card numbers with Luhn validation and brand identification."""
        matches = []
        patterns = self.pattern_lib.get_patterns('credit_card')
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                card_text = match.group()
                digits = extract_digits(card_text)
                
                # Must have valid length for credit cards
                if self.config.min_credit_card_digits <= len(digits) <= 19:
                    # Validate with Luhn algorithm
                    if luhn_check(digits):
                        # Identify brand
                        brand = get_credit_card_brand(digits)
                        confidence = self._calculate_credit_card_confidence(digits, brand)
                        
                        if confidence >= self.category_thresholds['credit_card']:
                            masked_text = self.get_masked_preview(card_text, 'credit_card')
                            
                            matches.append(Match(
                                category='credit_card',
                                confidence=confidence,
                                masked_text=masked_text,
                                bbox=bbox or BBox(0, 0, 100, 20, 1.0)
                            ))
        
        return matches
    
    def _classify_email(self, text: str, bbox: Optional[BBox]) -> List[Match]:
        """Classify email addresses with RFC compliance validation."""
        matches = []
        patterns = self.pattern_lib.get_patterns('email')
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                email_text = match.group()
                
                # Validate with RFC compliance
                if validate_email_rfc(email_text):
                    confidence = self._calculate_email_confidence(email_text)
                    
                    if confidence >= self.category_thresholds['email']:
                        masked_text = self.get_masked_preview(email_text, 'email')
                        
                        matches.append(Match(
                            category='email',
                            confidence=confidence,
                            masked_text=masked_text,
                            bbox=bbox or BBox(0, 0, 100, 20, 1.0)
                        ))
        
        return matches
    
    def _classify_address(self, text: str, bbox: Optional[BBox]) -> List[Match]:
        """Classify addresses using rule-based scoring with optional spaCy NER."""
        matches = []
        
        # Use address detector for rule-based detection
        score = self.address_detector.score_address(text)
        
        if score >= self.config.address_min_score:
            confidence = min(score, 1.0)  # Cap at 1.0
            
            if confidence >= self.category_thresholds['address']:
                masked_text = self.get_masked_preview(text, 'address')
                
                matches.append(Match(
                    category='address',
                    confidence=confidence,
                    masked_text=masked_text,
                    bbox=bbox or BBox(0, 0, 100, 20, 1.0)
                ))
        
        return matches
    
    def _classify_api_key(self, text: str, bbox: Optional[BBox]) -> List[Match]:
        """Classify API keys for major vendors plus entropy-based fallback."""
        matches = []
        patterns = self.pattern_lib.get_patterns('api_key')
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                key_text = match.group()
                confidence = self._calculate_api_key_confidence(key_text)
                
                if confidence >= self.category_thresholds['api_key']:
                    masked_text = self.get_masked_preview(key_text, 'api_key')
                    
                    matches.append(Match(
                        category='api_key',
                        confidence=confidence,
                        masked_text=masked_text,
                        bbox=bbox or BBox(0, 0, 100, 20, 1.0)
                    ))
        
        # Entropy-based fallback for high-entropy strings
        if not matches and len(text) >= 16:
            if is_high_entropy_string(text, self.config.entropy_threshold_bits_per_char):
                confidence = 0.6  # Lower confidence for entropy-based detection
                
                if confidence >= self.category_thresholds['api_key']:
                    masked_text = self.get_masked_preview(text, 'api_key')
                    
                    matches.append(Match(
                        category='api_key',
                        confidence=confidence,
                        masked_text=masked_text,
                        bbox=bbox or BBox(0, 0, 100, 20, 1.0)
                    ))
        
        return matches
    
    def _calculate_phone_confidence(self, original: str, formatted: Optional[str]) -> float:
        """Calculate confidence score for phone number detection."""
        base_confidence = 0.8
        
        # Boost confidence if phonenumbers library successfully formatted
        if formatted:
            base_confidence = 0.9
        
        # Boost for common formatting patterns
        if any(char in original for char in ['(', ')', '-', '.']):
            base_confidence += 0.05
        
        # Boost for country code
        if original.strip().startswith(('+1', '1-', '1 ')):
            base_confidence += 0.05
        
        return min(base_confidence, 1.0)
    
    def _calculate_credit_card_confidence(self, digits: str, brand: Optional[str]) -> float:
        """Calculate confidence score for credit card detection."""
        base_confidence = 0.85  # High base since Luhn validation passed
        
        # Boost confidence if brand is identified
        if brand:
            base_confidence = 0.95
        
        # Boost for standard lengths
        if len(digits) in [13, 15, 16]:
            base_confidence += 0.02
        
        return min(base_confidence, 1.0)
    
    def _calculate_email_confidence(self, email: str) -> float:
        """Calculate confidence score for email detection."""
        base_confidence = 0.7
        
        # Boost for common TLDs
        common_tlds = ['.com', '.org', '.net', '.edu', '.gov', '.mil']
        if any(email.lower().endswith(tld) for tld in common_tlds):
            base_confidence += 0.1
        
        # Boost for proper structure
        if '.' in email.split('@')[1]:  # Domain has subdomain
            base_confidence += 0.05
        
        # Penalize for suspicious patterns
        if '..' in email or email.startswith('.') or email.endswith('.'):
            base_confidence -= 0.2
        
        return max(min(base_confidence, 1.0), 0.1)
    
    def _calculate_api_key_confidence(self, key: str) -> float:
        """Calculate confidence score for API key detection."""
        base_confidence = 0.6
        
        # Higher confidence for known vendor patterns
        vendor_patterns = {
            'AKIA': 0.95,  # AWS Access Key
            'ASIA': 0.95,  # AWS STS
            'AIza': 0.9,   # Google API Key
            'ghp_': 0.9,   # GitHub Token
            'sk_': 0.85,   # Stripe/OpenAI
            'xox': 0.85,   # Slack
            'hf_': 0.8,    # Hugging Face
        }
        
        for prefix, confidence in vendor_patterns.items():
            if key.startswith(prefix):
                return confidence
        
        # Entropy-based confidence for unknown patterns
        if is_high_entropy_string(key, self.config.entropy_threshold_bits_per_char):
            base_confidence = 0.7
        
        return base_confidence
    
    def get_masked_preview(self, text: str, category: str) -> str:
        """
        Create privacy-preserving masked version of text.
        
        Args:
            text: Original text to mask
            category: Category of sensitive data
            
        Returns:
            Masked text showing only first/last characters
        """
        if not text:
            return ""
        
        # Category-specific masking rules
        if category == 'credit_card':
            # Show only last 4 digits for credit cards
            digits = extract_digits(text)
            if len(digits) >= 4:
                return f"****-****-****-{digits[-4:]}"
            else:
                return "*" * len(text)
        
        elif category == 'phone':
            # Show area code and last 4 digits
            digits = extract_digits(text)
            if len(digits) >= 10:
                return f"({digits[:3]}) ***-{digits[-4:]}"
            else:
                return self.pattern_lib.mask_text(text, visible_chars=2)
        
        elif category == 'email':
            # Show first char of local part and domain
            if '@' in text:
                local, domain = text.rsplit('@', 1)
                if len(local) > 0:
                    masked_local = local[0] + '*' * (len(local) - 1)
                    return f"{masked_local}@{domain}"
            return self.pattern_lib.mask_text(text, visible_chars=2)
        
        elif category == 'api_key':
            # Show only prefix for API keys
            if len(text) > 8:
                return text[:4] + '*' * (len(text) - 4)
            else:
                return '*' * len(text)
        
        else:
            # Default masking for addresses and other categories
            return self.pattern_lib.mask_text(text, visible_chars=3)
    
    def get_category_stats(self) -> Dict[str, Any]:
        """Get statistics about classification categories."""
        return {
            'enabled_categories': self.config.categories,
            'category_thresholds': self.category_thresholds,
            'entropy_threshold': self.config.entropy_threshold_bits_per_char,
            'temporal_consensus_required': self.config.require_temporal_consensus,
            'spacy_ner_enabled': self.config.use_spacy_ner
        }