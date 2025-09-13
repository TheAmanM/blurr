"""Tests for validation utilities."""

import pytest
from privacy_redactor_rt.validators import (
    luhn_check,
    get_credit_card_brand,
    calculate_entropy,
    validate_phone_number,
    validate_email_rfc,
    is_high_entropy_string,
    extract_digits,
)


class TestLuhnValidation:
    """Test cases for Luhn algorithm validation."""
    
    def test_valid_credit_cards(self):
        """Test Luhn validation with valid credit card numbers."""
        valid_cards = [
            '4111111111111111',  # Visa test card
            '5555555555554444',  # Mastercard test card
            '378282246310005',   # Amex test card
            '6011111111111117',  # Discover test card
            '30569309025904',    # Diners Club test card
        ]
        
        for card in valid_cards:
            assert luhn_check(card), f"Card {card} should pass Luhn validation"
    
    def test_invalid_credit_cards(self):
        """Test Luhn validation with invalid credit card numbers."""
        invalid_cards = [
            '4111111111111112',  # Invalid Visa
            '5555555555554445',  # Invalid Mastercard
            '378282246310006',   # Invalid Amex
            '1234567890123456',  # Random invalid number
            '0000000000000000',  # All zeros
        ]
        
        for card in invalid_cards:
            assert not luhn_check(card), f"Card {card} should fail Luhn validation"
    
    def test_luhn_with_spaces_and_dashes(self):
        """Test Luhn validation with formatted numbers."""
        formatted_cards = [
            '4111 1111 1111 1111',
            '4111-1111-1111-1111',
            '4111.1111.1111.1111',
            '5555 5555 5555 4444',
            '3782 822463 10005',
        ]
        
        for card in formatted_cards:
            assert luhn_check(card), f"Formatted card {card} should pass Luhn validation"
    
    def test_luhn_short_numbers(self):
        """Test Luhn validation with numbers too short to be credit cards."""
        short_numbers = [
            '123456789012',  # 12 digits
            '12345678901',   # 11 digits
            '1234567890',    # 10 digits
        ]
        
        for number in short_numbers:
            assert not luhn_check(number), f"Short number {number} should fail validation"
    
    def test_luhn_non_numeric(self):
        """Test Luhn validation with non-numeric input."""
        non_numeric = [
            'abcd efgh ijkl mnop',
            '4111-abcd-1111-1111',
            '',
            'not a number',
        ]
        
        for text in non_numeric:
            assert not luhn_check(text), f"Non-numeric input '{text}' should fail validation"


class TestCreditCardBrand:
    """Test cases for credit card brand identification."""
    
    def test_visa_detection(self):
        """Test Visa card detection."""
        visa_cards = [
            '4111111111111111',  # 16 digits
            '4000000000000',     # 13 digits
            '4000000000000000000',  # 19 digits
        ]
        
        for card in visa_cards:
            brand = get_credit_card_brand(card)
            assert brand == 'Visa', f"Card {card} should be detected as Visa"
    
    def test_mastercard_detection(self):
        """Test Mastercard detection."""
        mastercard_cards = [
            '5555555555554444',  # Traditional Mastercard
            '2223000048400011',  # New Mastercard range
            '2720000000000000',  # New Mastercard range
        ]
        
        for card in mastercard_cards:
            brand = get_credit_card_brand(card)
            assert brand == 'Mastercard', f"Card {card} should be detected as Mastercard"
    
    def test_amex_detection(self):
        """Test American Express detection."""
        amex_cards = [
            '378282246310005',   # Amex starting with 37
            '341111111111111',   # Amex starting with 34
        ]
        
        for card in amex_cards:
            brand = get_credit_card_brand(card)
            assert brand == 'American Express', f"Card {card} should be detected as American Express"
    
    def test_discover_detection(self):
        """Test Discover card detection."""
        discover_cards = [
            '6011111111111117',  # Standard Discover
            '6221260000000000',  # Discover range
            '6441111111111111',  # Discover range
            '6500000000000000',  # Discover range
        ]
        
        for card in discover_cards:
            brand = get_credit_card_brand(card)
            assert brand == 'Discover', f"Card {card} should be detected as Discover"
    
    def test_diners_club_detection(self):
        """Test Diners Club detection."""
        diners_cards = [
            '30569309025904',    # Diners Club
            '36000000000000',    # Diners Club
            '38000000000000',    # Diners Club
        ]
        
        for card in diners_cards:
            brand = get_credit_card_brand(card)
            assert brand == 'Diners Club', f"Card {card} should be detected as Diners Club"
    
    def test_jcb_detection(self):
        """Test JCB detection."""
        jcb_cards = [
            '3530111333300000',  # JCB
            '2131000000000000',  # JCB
            '1800000000000000',  # JCB
        ]
        
        for card in jcb_cards:
            brand = get_credit_card_brand(card)
            assert brand == 'JCB', f"Card {card} should be detected as JCB"
    
    def test_unknown_brand(self):
        """Test unknown brand detection."""
        unknown_cards = [
            '1234567890123456',  # Invalid pattern
            '9999999999999999',  # Unknown pattern
        ]
        
        for card in unknown_cards:
            brand = get_credit_card_brand(card)
            assert brand is None, f"Card {card} should not be recognized"
    
    def test_brand_with_formatting(self):
        """Test brand detection with formatted numbers."""
        formatted_cards = [
            ('4111 1111 1111 1111', 'Visa'),
            ('5555-5555-5555-4444', 'Mastercard'),
            ('3782 822463 10005', 'American Express'),
        ]
        
        for card, expected_brand in formatted_cards:
            brand = get_credit_card_brand(card)
            assert brand == expected_brand, f"Formatted card {card} should be detected as {expected_brand}"


class TestEntropyCalculation:
    """Test cases for entropy calculation."""
    
    def test_entropy_uniform_distribution(self):
        """Test entropy calculation with uniform character distribution."""
        # String with all unique characters should have high entropy
        unique_chars = 'abcdefghijklmnop'
        entropy = calculate_entropy(unique_chars)
        assert entropy > 3.5, "Unique characters should have high entropy"
    
    def test_entropy_repeated_characters(self):
        """Test entropy calculation with repeated characters."""
        # String with repeated characters should have low entropy
        repeated = 'aaaaaaaaaaaaaaaa'
        entropy = calculate_entropy(repeated)
        assert entropy == 0.0, "Repeated characters should have zero entropy"
    
    def test_entropy_mixed_distribution(self):
        """Test entropy calculation with mixed character distribution."""
        # Mix of repeated and unique characters
        mixed = 'aaaabbbbccccdddd'
        entropy = calculate_entropy(mixed)
        assert 1.0 < entropy < 3.0, "Mixed distribution should have moderate entropy"
    
    def test_entropy_empty_string(self):
        """Test entropy calculation with empty string."""
        entropy = calculate_entropy('')
        assert entropy == 0.0, "Empty string should have zero entropy"
    
    def test_entropy_single_character(self):
        """Test entropy calculation with single character."""
        entropy = calculate_entropy('a')
        assert entropy == 0.0, "Single character should have zero entropy"
    
    def test_entropy_binary_distribution(self):
        """Test entropy calculation with binary distribution."""
        # Equal distribution of two characters should approach 1.0
        binary = 'ababababababab'
        entropy = calculate_entropy(binary)
        assert abs(entropy - 1.0) < 0.1, "Equal binary distribution should have entropy ~1.0"


class TestPhoneValidation:
    """Test cases for phone number validation."""
    
    def test_valid_us_phones(self):
        """Test validation of valid US phone numbers."""
        valid_phones = [
            '(212) 123-4567',
            '212-123-4567',
            '212.123.4567',
            '2121234567',
            '+1 212 123 4567',
            '1-212-123-4567',
        ]
        
        for phone in valid_phones:
            is_valid, formatted = validate_phone_number(phone)
            assert is_valid, f"Phone {phone} should be valid"
            if formatted:
                assert formatted.startswith('+1'), f"Formatted phone should start with +1"
    
    def test_valid_canadian_phones(self):
        """Test validation of valid Canadian phone numbers."""
        valid_phones = [
            '(416) 234-1234',
            '604-234-1234',
            '+1 416 234 1234',
        ]
        
        for phone in valid_phones:
            is_valid, formatted = validate_phone_number(phone, ['CA'])
            assert is_valid, f"Canadian phone {phone} should be valid"
    
    def test_invalid_phones(self):
        """Test validation of invalid phone numbers."""
        invalid_phones = [
            '123-456-7890',  # Invalid area code
            '555-023-4567',  # Invalid exchange
            '555-123-456',   # Too short
            '911',           # Emergency number
            'not-a-phone',   # Non-numeric
            '',              # Empty
        ]
        
        for phone in invalid_phones:
            is_valid, formatted = validate_phone_number(phone)
            assert not is_valid, f"Phone {phone} should be invalid"
    
    def test_international_format(self):
        """Test international format phone numbers."""
        international_phones = [
            '+1 212 123 4567',
            '+1-212-123-4567',
            '+12121234567',
        ]
        
        for phone in international_phones:
            is_valid, formatted = validate_phone_number(phone)
            assert is_valid, f"International phone {phone} should be valid"
            if formatted:
                assert formatted.startswith('+1'), f"Formatted phone should start with +1"


class TestEmailValidation:
    """Test cases for RFC-compliant email validation."""
    
    def test_valid_emails(self):
        """Test validation of valid email addresses."""
        valid_emails = [
            'user@example.com',
            'test.email@domain.org',
            'user+tag@example.co.uk',
            'firstname.lastname@company.com',
            'user123@test-domain.net',
            'a@b.co',
        ]
        
        for email in valid_emails:
            assert validate_email_rfc(email), f"Email {email} should be valid"
    
    def test_invalid_emails(self):
        """Test validation of invalid email addresses."""
        invalid_emails = [
            'plainaddress',
            '@missingdomain.com',
            'missing@.com',
            'missing@domain',
            'spaces in@email.com',
            'double@@domain.com',
            'user@',
            '@domain.com',
            'user@domain.',
            'user@domain.c',  # TLD too short
            'a' * 65 + '@domain.com',  # Local part too long
            'user@' + 'a' * 254 + '.com',  # Domain too long
        ]
        
        for email in invalid_emails:
            assert not validate_email_rfc(email), f"Email {email} should be invalid"
    
    def test_email_length_limits(self):
        """Test email length validation."""
        # Test maximum length (254 characters total)
        long_email = 'a' * 240 + '@example.com'
        assert not validate_email_rfc(long_email), "Email over 254 chars should be invalid"
        
        # Test local part limit (64 characters)
        long_local = 'a' * 65 + '@example.com'
        assert not validate_email_rfc(long_local), "Local part over 64 chars should be invalid"


class TestHighEntropyString:
    """Test cases for high entropy string detection."""
    
    def test_high_entropy_strings(self):
        """Test detection of high entropy strings."""
        high_entropy_strings = [
            'sk_test_1234567890abcdef1234567890abcdef',  # API key
            'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',  # AWS secret
            'AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI',  # Google API key
            'abcdefghijklmnopqrstuvwxyz1234567890',      # Mixed chars
        ]
        
        for string in high_entropy_strings:
            assert is_high_entropy_string(string), f"String {string} should be high entropy"
    
    def test_low_entropy_strings(self):
        """Test detection of low entropy strings."""
        low_entropy_strings = [
            'aaaaaaaaaaaaaaaa',  # Repeated chars
            'password123',       # Common pattern
            'hello world',       # Natural language
            '1234567890',        # Sequential numbers
            'abcdefghij',        # Sequential letters
        ]
        
        for string in low_entropy_strings:
            assert not is_high_entropy_string(string), f"String {string} should be low entropy"
    
    def test_entropy_threshold(self):
        """Test entropy threshold configuration."""
        # Test with different thresholds
        test_string = 'abcdef1234567890'
        
        # Should be high entropy with low threshold
        assert is_high_entropy_string(test_string, threshold=2.0)
        
        # Should be low entropy with high threshold
        assert not is_high_entropy_string(test_string, threshold=5.0)
    
    def test_short_strings(self):
        """Test that short strings are not considered high entropy."""
        short_strings = [
            'abc',
            '12345',
            'xy',
            '',
        ]
        
        for string in short_strings:
            assert not is_high_entropy_string(string), f"Short string {string} should not be high entropy"


class TestExtractDigits:
    """Test cases for digit extraction."""
    
    def test_extract_digits_mixed(self):
        """Test digit extraction from mixed content."""
        test_cases = [
            ('abc123def456', '123456'),
            ('(555) 123-4567', '5551234567'),
            ('4111-1111-1111-1111', '4111111111111111'),
            ('no digits here', ''),
            ('', ''),
            ('123', '123'),
        ]
        
        for input_text, expected in test_cases:
            result = extract_digits(input_text)
            assert result == expected, f"extract_digits('{input_text}') should return '{expected}'"
    
    def test_extract_digits_special_chars(self):
        """Test digit extraction with special characters."""
        text_with_special = '!@#123$%^456&*()789'
        result = extract_digits(text_with_special)
        assert result == '123456789'
    
    def test_extract_digits_unicode(self):
        """Test digit extraction with unicode characters."""
        unicode_text = 'café123naïve456'
        result = extract_digits(unicode_text)
        assert result == '123456'


class TestValidationIntegration:
    """Integration tests for validation utilities."""
    
    def test_credit_card_full_validation(self):
        """Test complete credit card validation pipeline."""
        test_card = '4111 1111 1111 1111'
        
        # Extract digits
        digits = extract_digits(test_card)
        assert digits == '4111111111111111'
        
        # Validate with Luhn
        assert luhn_check(digits)
        
        # Identify brand
        brand = get_credit_card_brand(digits)
        assert brand == 'Visa'
    
    def test_phone_number_full_validation(self):
        """Test complete phone number validation pipeline."""
        test_phone = '(212) 123-4567'
        
        # Validate phone
        is_valid, formatted = validate_phone_number(test_phone)
        assert is_valid
        
        if formatted:
            # Should be in E164 format
            assert formatted.startswith('+1')
            assert len(formatted) == 12  # +1 + 10 digits
    
    def test_api_key_entropy_validation(self):
        """Test API key validation using entropy."""
        api_keys = [
            'sk_test_1234567890abcdef1234567890abcdef',
            'AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI',
            'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
        ]
        
        for key in api_keys:
            # Should have high entropy
            assert is_high_entropy_string(key)
            
            # Should have reasonable entropy value
            entropy = calculate_entropy(key)
            assert entropy > 3.0, f"API key {key} should have entropy > 3.0"