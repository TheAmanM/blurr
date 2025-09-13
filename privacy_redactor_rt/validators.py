"""Validation utilities for sensitive data detection."""

import math
import re
from collections import Counter
from typing import Optional, Tuple


def luhn_check(number: str) -> bool:
    """
    Validate credit card number using Luhn algorithm.
    
    Args:
        number: Credit card number as string (digits only)
        
    Returns:
        True if number passes Luhn validation, False otherwise
    """
    # Remove non-digit characters
    digits = re.sub(r'\D', '', number)
    
    # Must be at least 13 digits
    if len(digits) < 13:
        return False
    
    # Reject all zeros or all same digit
    if len(set(digits)) <= 1:
        return False
    
    # Convert to list of integers
    digits = [int(d) for d in digits]
    
    # Apply Luhn algorithm
    checksum = 0
    is_even = False
    
    # Process digits from right to left
    for digit in reversed(digits):
        if is_even:
            digit *= 2
            if digit > 9:
                digit = digit // 10 + digit % 10
        checksum += digit
        is_even = not is_even
    
    return checksum % 10 == 0


def get_credit_card_brand(number: str) -> Optional[str]:
    """
    Identify credit card brand based on number patterns.
    
    Args:
        number: Credit card number as string
        
    Returns:
        Brand name or None if not recognized
    """
    # Remove non-digit characters
    digits = re.sub(r'\D', '', number)
    
    if len(digits) < 13:
        return None
    
    # Visa: starts with 4, 13-19 digits
    if digits.startswith('4') and len(digits) in [13, 16, 19]:
        return 'Visa'
    
    # Mastercard: starts with 5[1-5] or 2[2-7], 16 digits
    if len(digits) == 16:
        if digits.startswith(('51', '52', '53', '54', '55')):
            return 'Mastercard'
        if digits[:2] in [str(i) for i in range(22, 28)]:
            return 'Mastercard'
    
    # American Express: starts with 34 or 37, 15 digits
    if len(digits) == 15 and digits.startswith(('34', '37')):
        return 'American Express'
    
    # Discover: starts with 6011, 622126-622925, 644-649, 65, 16 digits
    if len(digits) == 16:
        if digits.startswith('6011') or digits.startswith('65'):
            return 'Discover'
        if digits.startswith('644') or digits.startswith('645') or \
           digits.startswith('646') or digits.startswith('647') or \
           digits.startswith('648') or digits.startswith('649'):
            return 'Discover'
        if digits.startswith('622') and 622126 <= int(digits[:6]) <= 622925:
            return 'Discover'
    
    # Diners Club: starts with 300-305, 36, 38, 14 digits
    if len(digits) == 14:
        if digits.startswith(('300', '301', '302', '303', '304', '305', '36', '38')):
            return 'Diners Club'
    
    # JCB: starts with 35, 2131, 1800, various lengths
    if digits.startswith('35') and len(digits) == 16:
        return 'JCB'
    if digits.startswith(('2131', '1800')):
        if len(digits) in [15, 16]:
            return 'JCB'
    
    return None


def calculate_entropy(text: str) -> float:
    """
    Calculate Shannon entropy of text in bits per character.
    
    Args:
        text: Input text string
        
    Returns:
        Shannon entropy in bits per character
    """
    if not text:
        return 0.0
    
    # Count character frequencies
    counter = Counter(text)
    length = len(text)
    
    # Calculate entropy
    entropy = 0.0
    for count in counter.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)
    
    return entropy


def validate_phone_number(text: str, country_codes: Optional[list] = None) -> Tuple[bool, Optional[str]]:
    """
    Validate phone number format using phonenumbers library.
    
    Args:
        text: Phone number text
        country_codes: List of country codes to try (default: ['US', 'CA'])
        
    Returns:
        Tuple of (is_valid, formatted_number)
    """
    try:
        import phonenumbers
        from phonenumbers import NumberParseException
    except ImportError:
        # Fallback to basic regex validation if phonenumbers not available
        return _validate_phone_regex(text), None
    
    if country_codes is None:
        country_codes = ['US', 'CA']
    
    # Clean the input text
    cleaned = re.sub(r'[^\d+()-.\s]', '', text)
    
    for country_code in country_codes:
        try:
            # Parse the number
            parsed = phonenumbers.parse(cleaned, country_code)
            
            # Validate the number
            if phonenumbers.is_valid_number(parsed):
                # Format in E164 format
                formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
                return True, formatted
                
        except NumberParseException:
            continue
    
    # Try parsing without country code (international format)
    try:
        parsed = phonenumbers.parse(cleaned, None)
        if phonenumbers.is_valid_number(parsed):
            formatted = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
            return True, formatted
    except NumberParseException:
        pass
    
    return False, None


def _validate_phone_regex(text: str) -> bool:
    """
    Fallback phone validation using regex patterns.
    
    Args:
        text: Phone number text
        
    Returns:
        True if text matches phone number patterns
    """
    # Test multiple formats
    patterns = [
        # (212) 123-4567
        r'^\([2-9][0-8][0-9]\)\s?[1-9][0-9]{2}[-.\s]?[0-9]{4}$',
        # 212-123-4567, 212.123.4567
        r'^[2-9][0-8][0-9][-.\s][1-9][0-9]{2}[-.\s][0-9]{4}$',
        # 2121234567
        r'^[2-9][0-8][0-9][1-9][0-9]{2}[0-9]{4}$',
        # +1 212 123 4567, 1-212-123-4567
        r'^(?:\+?1[-.\s]?)?[2-9][0-8][0-9][-.\s]?[1-9][0-9]{2}[-.\s]?[0-9]{4}$',
    ]
    
    for pattern in patterns:
        if re.match(pattern, text.strip()):
            return True
    
    return False


def validate_email_rfc(email: str) -> bool:
    """
    Validate email address with RFC-compliant checks.
    
    Args:
        email: Email address string
        
    Returns:
        True if email appears valid
    """
    # Basic length check
    if len(email) > 254:  # RFC 5321 limit
        return False
    
    # Must contain exactly one @
    if email.count('@') != 1:
        return False
    
    local, domain = email.rsplit('@', 1)
    
    # Local part validation
    if len(local) == 0 or len(local) > 64:  # RFC 5321 limit
        return False
    
    # Domain part validation
    if len(domain) == 0 or len(domain) > 253:
        return False
    
    # Domain must contain at least one dot
    if '.' not in domain:
        return False
    
    # Basic character validation
    local_pattern = r'^[a-zA-Z0-9._%+-]+$'
    domain_pattern = r'^[a-zA-Z0-9.-]+$'
    
    if not re.match(local_pattern, local):
        return False
    
    if not re.match(domain_pattern, domain):
        return False
    
    # Domain parts validation
    domain_parts = domain.split('.')
    if len(domain_parts) < 2:
        return False
    
    # TLD must be at least 2 characters
    if len(domain_parts[-1]) < 2:
        return False
    
    # Each domain part must not be empty and not start/end with hyphen
    for part in domain_parts:
        if not part or part.startswith('-') or part.endswith('-'):
            return False
    
    return True


def is_high_entropy_string(text: str, threshold: float = 3.5) -> bool:
    """
    Check if string has high entropy (likely random/encoded).
    
    Args:
        text: Input string
        threshold: Entropy threshold in bits per character
        
    Returns:
        True if entropy exceeds threshold
    """
    if len(text) < 8:  # Too short to be meaningful
        return False
    
    entropy = calculate_entropy(text)
    return entropy >= threshold


def extract_digits(text: str) -> str:
    """
    Extract only digits from text.
    
    Args:
        text: Input text
        
    Returns:
        String containing only digits
    """
    return re.sub(r'\D', '', text)