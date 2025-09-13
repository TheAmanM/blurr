"""Compiled regex patterns for sensitive data detection."""

import re
from typing import Dict, Pattern, List


class PatternLibrary:
    """Compiled regex patterns for all sensitive data categories."""
    
    def __init__(self):
        """Initialize compiled patterns."""
        self._patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[Pattern[str]]]:
        """Compile all regex patterns for sensitive data detection."""
        return {
            'phone': self._compile_phone_patterns(),
            'credit_card': self._compile_credit_card_patterns(),
            'email': self._compile_email_patterns(),
            'api_key': self._compile_api_key_patterns(),
        }
    
    def _compile_phone_patterns(self) -> List[Pattern[str]]:
        """Compile phone number patterns for US/Canadian numbers."""
        patterns = [
            # Standard formats with parentheses: (123) 456-7890
            r'\([2-9][0-8][0-9]\)\s?[2-9][0-9]{2}[-.\s]?[0-9]{4}',
            # Standard formats with dashes/dots: 123-456-7890, 123.456.7890  
            r'[2-9][0-8][0-9][-.\s][2-9][0-9]{2}[-.\s][0-9]{4}',
            # Compact formats: 1234567890
            r'\b[2-9][0-8][0-9][2-9][0-9]{2}[0-9]{4}\b',
            # More permissive exchange codes (allow 1xx exchange)
            r'\([2-9][0-8][0-9]\)\s?[1-9][0-9]{2}[-.\s]?[0-9]{4}',
            r'[2-9][0-8][0-9][-.\s][1-9][0-9]{2}[-.\s][0-9]{4}',
            # With country code
            r'(?:\+?1[-.\s]?)?[2-9][0-8][0-9][-.\s]?[1-9][0-9]{2}[-.\s]?[0-9]{4}',
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _compile_credit_card_patterns(self) -> List[Pattern[str]]:
        """Compile credit card number patterns."""
        patterns = [
            # 13-19 digits with optional spaces/dashes
            r'\b(?:[0-9]{4}[-\s]?){3}[0-9]{1,7}\b',
            # Compact format 13-19 digits
            r'\b[0-9]{13,19}\b',
            # Common formats with separators
            r'\b[0-9]{4}[-\s][0-9]{4}[-\s][0-9]{4}[-\s][0-9]{4}\b',
            r'\b[0-9]{4}[-\s][0-9]{6}[-\s][0-9]{5}\b',  # Amex format
        ]
        return [re.compile(pattern) for pattern in patterns]
    
    def _compile_email_patterns(self) -> List[Pattern[str]]:
        """Compile RFC-compliant email patterns."""
        patterns = [
            # Standard email pattern - must not have spaces in local part
            r'\b[a-zA-Z0-9][a-zA-Z0-9._%+-]*@[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}\b',
        ]
        return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    
    def _compile_api_key_patterns(self) -> List[Pattern[str]]:
        """Compile API key patterns for major vendors."""
        patterns = [
            # AWS Access Key ID
            r'\b(?:AKIA|ASIA|AROA|AIDA|AGPA|AIPA|ANPA|ANVA|APKA)[A-Z0-9]{16}\b',
            # AWS Secret Access Key (40 chars base64-like)
            r'\b[A-Za-z0-9/+=]{40}\b',
            # Google API Key
            r'\bAIza[0-9A-Za-z_-]{35}\b',
            # GitHub Token (classic) - fixed length
            r'\bghp_[A-Za-z0-9]{40}\b',
            # GitHub Token (fine-grained)
            r'\bgithub_pat_[A-Za-z0-9_]{82}\b',
            # Stripe API Key
            r'\b(?:sk|pk)_(?:live|test)_[A-Za-z0-9]{24,}\b',
            # Slack Token
            r'\bxox[bpoa]-[A-Za-z0-9-]{10,}\b',
            # Twilio API Key
            r'\bSK[a-f0-9]{32}\b',
            # OpenAI API Key
            r'\bsk-[A-Za-z0-9]{48}\b',
            # Hugging Face Token
            r'\bhf_[A-Za-z0-9]{38}\b',
            # Supabase API Key (JWT pattern)
            r'\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b',
            # Generic high-entropy base64 patterns (32+ chars)
            r'\b[A-Za-z0-9+/]{32,}={0,2}\b',
            # Generic hex patterns (32+ chars)
            r'\b[a-fA-F0-9]{32,}\b',
        ]
        return [re.compile(pattern) for pattern in patterns]
    
    def get_patterns(self, category: str) -> List[Pattern[str]]:
        """Get compiled patterns for a specific category."""
        return self._patterns.get(category, [])
    
    def get_all_patterns(self) -> Dict[str, List[Pattern[str]]]:
        """Get all compiled patterns."""
        return self._patterns.copy()
    
    def normalize_text(self, text: str) -> str:
        """Normalize text for pattern matching."""
        # Remove extra whitespace and normalize unicode
        import unicodedata
        text = unicodedata.normalize('NFKC', text)
        # Collapse multiple whitespace to single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def mask_text(self, text: str, visible_chars: int = 3) -> str:
        """Create privacy-preserving masked version of text."""
        if len(text) <= visible_chars * 2:
            return '*' * len(text)
        
        start = text[:visible_chars]
        end = text[-visible_chars:]
        middle_length = len(text) - visible_chars * 2
        middle = '*' * middle_length
        return f"{start}{middle}{end}"