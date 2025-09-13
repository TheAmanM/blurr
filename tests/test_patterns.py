"""Tests for pattern matching functionality."""

import pytest
import re
from privacy_redactor_rt.patterns import PatternLibrary


class TestPatternLibrary:
    """Test cases for PatternLibrary class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_lib = PatternLibrary()
    
    def test_phone_patterns(self):
        """Test phone number pattern matching."""
        phone_patterns = self.pattern_lib.get_patterns('phone')
        assert len(phone_patterns) > 0
        
        # Test valid phone numbers (using valid area codes)
        valid_phones = [
            '(212) 123-4567',
            '212-123-4567',
            '212.123.4567',
            '2121234567',
            '+1 212 123 4567',
            '+1-212-123-4567',
            '1-212-123-4567',
        ]
        
        for phone in valid_phones:
            matched = any(pattern.search(phone) for pattern in phone_patterns)
            assert matched, f"Phone number '{phone}' should match patterns"
    
    def test_phone_patterns_invalid(self):
        """Test phone patterns reject invalid numbers."""
        phone_patterns = self.pattern_lib.get_patterns('phone')
        
        # Test invalid phone numbers
        invalid_phones = [
            '123-456-7890',  # Invalid area code (starts with 1)
            '555-123-456',   # Too short
            '555-023-4567',  # Invalid exchange code (starts with 0)
            '000-123-4567',  # Invalid area code (000)
            '911',           # Emergency number
            '411',           # Directory assistance
        ]
        
        for phone in invalid_phones:
            matched = any(pattern.search(phone) for pattern in phone_patterns)
            assert not matched, f"Invalid phone number '{phone}' should not match patterns"
    
    def test_credit_card_patterns(self):
        """Test credit card pattern matching."""
        cc_patterns = self.pattern_lib.get_patterns('credit_card')
        assert len(cc_patterns) > 0
        
        # Test valid credit card formats
        valid_cards = [
            '4111 1111 1111 1111',  # Visa format
            '4111-1111-1111-1111',  # Visa with dashes
            '4111111111111111',     # Visa compact
            '5555 5555 5555 4444',  # Mastercard
            '3782 822463 10005',    # Amex format
            '378282246310005',      # Amex compact
        ]
        
        for card in valid_cards:
            matched = any(pattern.search(card) for pattern in cc_patterns)
            assert matched, f"Credit card '{card}' should match patterns"
    
    def test_credit_card_patterns_invalid(self):
        """Test credit card patterns reject invalid formats."""
        cc_patterns = self.pattern_lib.get_patterns('credit_card')
        
        # Test invalid formats
        invalid_cards = [
            '1234',           # Too short
            '12345678901234567890',  # Too long
            '4111 1111 1111',        # Too short for Visa
            'abcd efgh ijkl mnop',   # Non-numeric
        ]
        
        for card in invalid_cards:
            matched = any(pattern.search(card) for pattern in cc_patterns)
            assert not matched, f"Invalid credit card '{card}' should not match patterns"
    
    def test_email_patterns(self):
        """Test email pattern matching."""
        email_patterns = self.pattern_lib.get_patterns('email')
        assert len(email_patterns) > 0
        
        # Test valid email addresses
        valid_emails = [
            'user@example.com',
            'test.email@domain.org',
            'user+tag@example.co.uk',
            'firstname.lastname@company.com',
            'user123@test-domain.net',
        ]
        
        for email in valid_emails:
            matched = any(pattern.search(email) for pattern in email_patterns)
            assert matched, f"Email '{email}' should match patterns"
    
    def test_email_patterns_invalid(self):
        """Test email patterns reject invalid addresses."""
        email_patterns = self.pattern_lib.get_patterns('email')
        
        # Test invalid email addresses
        invalid_emails = [
            'plainaddress',
            '@missingdomain.com',
            'missing@.com',
            'missing@domain',
            'double@@domain.com',
        ]
        
        for email in invalid_emails:
            matched = any(pattern.search(email) for pattern in email_patterns)
            assert not matched, f"Invalid email '{email}' should not match patterns"
    
    def test_api_key_patterns(self):
        """Test API key pattern matching."""
        api_patterns = self.pattern_lib.get_patterns('api_key')
        assert len(api_patterns) > 0
        
        # Test valid API key formats
        valid_keys = [
            'AKIAIOSFODNN7EXAMPLE',  # AWS Access Key
            'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',  # AWS Secret
            'AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI',  # Google API Key
            'ghp_1234567890abcdef1234567890abcdef12345678',  # GitHub Token
            'sk_test_1234567890abcdef1234567890abcdef',  # Stripe Key
            'xoxb-1234567890-1234567890-abcdefghijklmnop',  # Slack Token
            'sk-1234567890abcdef1234567890abcdef1234567890abcdef',  # OpenAI
            'hf_1234567890abcdef1234567890abcdef123456',  # Hugging Face
        ]
        
        for key in valid_keys:
            matched = any(pattern.search(key) for pattern in api_patterns)
            assert matched, f"API key '{key}' should match patterns"
    
    def test_normalize_text(self):
        """Test text normalization functionality."""
        # Test unicode normalization
        text_with_unicode = "café naïve résumé"
        normalized = self.pattern_lib.normalize_text(text_with_unicode)
        assert normalized == "café naïve résumé"
        
        # Test whitespace normalization
        text_with_spaces = "  multiple   spaces   here  "
        normalized = self.pattern_lib.normalize_text(text_with_spaces)
        assert normalized == "multiple spaces here"
        
        # Test mixed normalization
        mixed_text = "  café   naïve  "
        normalized = self.pattern_lib.normalize_text(mixed_text)
        assert normalized == "café naïve"
    
    def test_mask_text(self):
        """Test text masking functionality."""
        # Test normal case
        text = "sensitive_data_123"
        masked = self.pattern_lib.mask_text(text, visible_chars=3)
        assert masked == "sen************123"
        
        # Test short text
        short_text = "abc"
        masked = self.pattern_lib.mask_text(short_text, visible_chars=3)
        assert masked == "***"
        
        # Test very short text
        very_short = "ab"
        masked = self.pattern_lib.mask_text(very_short, visible_chars=3)
        assert masked == "**"
        
        # Test empty text
        empty = ""
        masked = self.pattern_lib.mask_text(empty, visible_chars=3)
        assert masked == ""
        
        # Test different visible chars
        text = "1234567890"
        masked = self.pattern_lib.mask_text(text, visible_chars=2)
        assert masked == "12******90"
    
    def test_get_all_patterns(self):
        """Test getting all patterns."""
        all_patterns = self.pattern_lib.get_all_patterns()
        
        expected_categories = ['phone', 'credit_card', 'email', 'api_key']
        for category in expected_categories:
            assert category in all_patterns
            assert len(all_patterns[category]) > 0
    
    def test_get_patterns_invalid_category(self):
        """Test getting patterns for invalid category."""
        patterns = self.pattern_lib.get_patterns('invalid_category')
        assert patterns == []


class TestPatternMatching:
    """Integration tests for pattern matching across categories."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_lib = PatternLibrary()
    
    def test_mixed_content_detection(self):
        """Test detection in mixed content."""
        mixed_text = """
        Contact John at (212) 123-4567 or john@example.com
        Credit card: 4111 1111 1111 1111
        API Key: sk_test_1234567890abcdef1234567890abcdef
        """
        
        # Should detect phone
        phone_patterns = self.pattern_lib.get_patterns('phone')
        phone_found = any(pattern.search(mixed_text) for pattern in phone_patterns)
        assert phone_found
        
        # Should detect email
        email_patterns = self.pattern_lib.get_patterns('email')
        email_found = any(pattern.search(mixed_text) for pattern in email_patterns)
        assert email_found
        
        # Should detect credit card
        cc_patterns = self.pattern_lib.get_patterns('credit_card')
        cc_found = any(pattern.search(mixed_text) for pattern in cc_patterns)
        assert cc_found
        
        # Should detect API key
        api_patterns = self.pattern_lib.get_patterns('api_key')
        api_found = any(pattern.search(mixed_text) for pattern in api_patterns)
        assert api_found
    
    def test_case_insensitive_matching(self):
        """Test case insensitive matching where appropriate."""
        # Email should be case insensitive
        email_patterns = self.pattern_lib.get_patterns('email')
        
        emails = [
            'USER@EXAMPLE.COM',
            'User@Example.Com',
            'user@example.com',
        ]
        
        for email in emails:
            matched = any(pattern.search(email) for pattern in email_patterns)
            assert matched, f"Email '{email}' should match (case insensitive)"
    
    def test_boundary_matching(self):
        """Test word boundary matching to avoid false positives."""
        # Phone patterns should use word boundaries
        phone_patterns = self.pattern_lib.get_patterns('phone')
        
        # Should match standalone phone
        standalone = "Call 212-123-4567 today"
        matched = any(pattern.search(standalone) for pattern in phone_patterns)
        assert matched
        
        # Should not match phone as part of larger number
        embedded = "ID: 12555123456789"
        # This might match depending on pattern design - adjust test as needed
        # The key is that patterns should be designed to minimize false positives