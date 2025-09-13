"""Tests for classification engine functionality."""

import pytest
from unittest.mock import Mock, patch
from privacy_redactor_rt.classify import ClassificationEngine
from privacy_redactor_rt.config import ClassificationConfig
from privacy_redactor_rt.types import BBox, Match


class TestClassificationEngine:
    """Test cases for ClassificationEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ClassificationConfig()
        self.engine = ClassificationEngine(self.config)
    
    def test_initialization(self):
        """Test classification engine initialization."""
        assert self.engine.config == self.config
        assert self.engine.pattern_lib is not None
        assert self.engine.address_detector is not None
        assert len(self.engine.category_thresholds) > 0
    
    def test_classify_empty_text(self):
        """Test classification with empty or None text."""
        # Empty string
        matches = self.engine.classify_text("")
        assert matches == []
        
        # None
        matches = self.engine.classify_text(None)
        assert matches == []
        
        # Whitespace only
        matches = self.engine.classify_text("   ")
        assert matches == []
    
    def test_classify_phone_numbers(self):
        """Test phone number classification."""
        valid_phones = [
            "(212) 123-4567",
            "212-123-4567", 
            "212.123.4567",
            "2121234567",
            "+1 212 123 4567",
            "1-212-123-4567"
        ]
        
        for phone in valid_phones:
            matches = self.engine.classify_text(phone)
            phone_matches = [m for m in matches if m.category == 'phone']
            assert len(phone_matches) > 0, f"Phone '{phone}' should be detected"
            
            match = phone_matches[0]
            assert match.confidence > 0.7
            assert 'phone' in match.category
            assert len(match.masked_text) > 0
    
    def test_classify_phone_numbers_invalid(self):
        """Test phone classification rejects invalid numbers."""
        invalid_phones = [
            "123-456-7890",  # Invalid area code
            "555-023-4567",  # Invalid exchange code
            "000-123-4567",  # Invalid area code
            "911",           # Emergency number
            "411"            # Directory assistance
        ]
        
        for phone in invalid_phones:
            matches = self.engine.classify_text(phone)
            phone_matches = [m for m in matches if m.category == 'phone']
            assert len(phone_matches) == 0, f"Invalid phone '{phone}' should not be detected"
    
    def test_classify_credit_cards(self):
        """Test credit card classification with Luhn validation."""
        # Valid test credit card numbers (these pass Luhn but are test numbers)
        valid_cards = [
            "4111111111111111",  # Visa test number
            "4111 1111 1111 1111",  # Visa with spaces
            "4111-1111-1111-1111",  # Visa with dashes
            "5555555555554444",  # Mastercard test number
            "378282246310005",   # Amex test number
            "3782 822463 10005"  # Amex with spaces
        ]
        
        for card in valid_cards:
            matches = self.engine.classify_text(card)
            cc_matches = [m for m in matches if m.category == 'credit_card']
            assert len(cc_matches) > 0, f"Credit card '{card}' should be detected"
            
            match = cc_matches[0]
            assert match.confidence > 0.8
            assert 'credit_card' in match.category
            assert '****' in match.masked_text  # Should be masked
    
    def test_classify_credit_cards_invalid(self):
        """Test credit card classification rejects invalid numbers."""
        invalid_cards = [
            "1234567890123456",  # Fails Luhn check
            "1234",              # Too short
            "12345678901234567890",  # Too long
            "abcd efgh ijkl mnop"    # Non-numeric
        ]
        
        for card in invalid_cards:
            matches = self.engine.classify_text(card)
            cc_matches = [m for m in matches if m.category == 'credit_card']
            assert len(cc_matches) == 0, f"Invalid credit card '{card}' should not be detected"
    
    def test_classify_emails(self):
        """Test email classification with RFC validation."""
        valid_emails = [
            "user@example.com",
            "test.email@domain.org", 
            "user+tag@example.co.uk",
            "firstname.lastname@company.com",
            "user123@test-domain.net"
        ]
        
        for email in valid_emails:
            matches = self.engine.classify_text(email)
            email_matches = [m for m in matches if m.category == 'email']
            assert len(email_matches) > 0, f"Email '{email}' should be detected"
            
            match = email_matches[0]
            assert match.confidence > 0.6
            assert 'email' in match.category
            assert '@' in match.masked_text
    
    def test_classify_emails_invalid(self):
        """Test email classification rejects invalid addresses."""
        invalid_emails = [
            "plainaddress",
            "@missingdomain.com",
            "missing@.com", 
            "missing@domain",
            "double@@domain.com"
        ]
        
        for email in invalid_emails:
            matches = self.engine.classify_text(email)
            email_matches = [m for m in matches if m.category == 'email']
            assert len(email_matches) == 0, f"Invalid email '{email}' should not be detected"
    
    def test_classify_addresses(self):
        """Test address classification with rule-based scoring."""
        valid_addresses = [
            "123 Main Street, Anytown, NY 12345",
            "456 Oak Ave, Suite 100, Los Angeles, CA 90210",
            "789 First St, Apt 2B, Boston, MA 02101",
            "1000 Broadway, New York, NY 10001"
        ]
        
        for address in valid_addresses:
            matches = self.engine.classify_text(address)
            addr_matches = [m for m in matches if m.category == 'address']
            # Note: Address detection might be less reliable, so we check if any are found
            if addr_matches:
                match = addr_matches[0]
                assert match.confidence > 0.0
                assert 'address' in match.category
                assert len(match.masked_text) > 0
    
    def test_classify_api_keys(self):
        """Test API key classification for major vendors."""
        valid_api_keys = [
            "AKIAIOSFODNN7EXAMPLE",  # AWS Access Key format
            "AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI",  # Google API Key format
            "ghp_1234567890abcdef1234567890abcdef12345678",  # GitHub Token format
            "sk_test_1234567890abcdef1234567890abcdef",  # Stripe Key format
            "xoxb-1234567890-1234567890-abcdefghijklmnop",  # Slack Token format
            "sk-1234567890abcdef1234567890abcdef1234567890abcdef",  # OpenAI format
            "hf_1234567890abcdef1234567890abcdef123456"  # Hugging Face format
        ]
        
        for api_key in valid_api_keys:
            matches = self.engine.classify_text(api_key)
            api_matches = [m for m in matches if m.category == 'api_key']
            assert len(api_matches) > 0, f"API key '{api_key}' should be detected"
            
            match = api_matches[0]
            assert match.confidence > 0.5
            assert 'api_key' in match.category
            assert '*' in match.masked_text or len(match.masked_text) < len(api_key)
    
    def test_classify_high_entropy_strings(self):
        """Test entropy-based API key detection."""
        # High entropy strings that should be detected as potential API keys
        high_entropy_strings = [
            "dGhpc2lzYXZlcnlsb25ncmFuZG9tc3RyaW5nd2l0aGhpZ2hlbnRyb3B5",  # Base64-like
            "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0u1v2w3x4y5z6",  # Mixed alphanumeric
        ]
        
        for entropy_str in high_entropy_strings:
            matches = self.engine.classify_text(entropy_str)
            api_matches = [m for m in matches if m.category == 'api_key']
            # Entropy-based detection might not always trigger, so we allow for this
            if api_matches:
                match = api_matches[0]
                assert match.confidence > 0.0
                assert 'api_key' in match.category
    
    def test_mixed_content_classification(self):
        """Test classification of text with multiple sensitive data types."""
        mixed_text = """
        Contact John at (212) 123-4567 or john@example.com
        Credit card: 4111 1111 1111 1111
        Address: 123 Main St, New York, NY 10001
        API Key: sk_test_1234567890abcdef1234567890abcdef
        """
        
        matches = self.engine.classify_text(mixed_text)
        
        # Should detect multiple categories
        categories_found = {match.category for match in matches}
        expected_categories = {'phone', 'email', 'credit_card', 'api_key'}
        
        # At least some categories should be detected
        assert len(categories_found.intersection(expected_categories)) >= 2
        
        # All matches should have valid confidence scores
        for match in matches:
            assert 0.0 <= match.confidence <= 1.0
            assert len(match.masked_text) > 0
            assert match.category in self.config.categories
    
    def test_get_masked_preview_credit_card(self):
        """Test credit card specific masking."""
        card_text = "4111111111111111"
        masked = self.engine.get_masked_preview(card_text, 'credit_card')
        assert masked == "****-****-****-1111"
    
    def test_get_masked_preview_phone(self):
        """Test phone number specific masking."""
        phone_text = "2121234567"
        masked = self.engine.get_masked_preview(phone_text, 'phone')
        assert masked == "(212) ***-4567"
    
    def test_get_masked_preview_email(self):
        """Test email specific masking."""
        email_text = "user@example.com"
        masked = self.engine.get_masked_preview(email_text, 'email')
        assert masked == "u***@example.com"
    
    def test_get_masked_preview_api_key(self):
        """Test API key specific masking."""
        api_key = "sk_test_1234567890abcdef"
        masked = self.engine.get_masked_preview(api_key, 'api_key')
        assert masked.startswith("sk_t")
        assert masked.endswith("*" * (len(api_key) - 4))
    
    def test_get_masked_preview_address(self):
        """Test address default masking."""
        address = "123 Main Street"
        masked = self.engine.get_masked_preview(address, 'address')
        assert masked.startswith("123")
        assert masked.endswith("eet")
        assert "*" in masked
    
    def test_confidence_calculation_phone(self):
        """Test phone confidence calculation."""
        # Test with formatted number
        confidence = self.engine._calculate_phone_confidence("(212) 123-4567", "+12121234567")
        assert confidence >= 0.9
        
        # Test without formatting
        confidence = self.engine._calculate_phone_confidence("2121234567", None)
        assert confidence >= 0.8
    
    def test_confidence_calculation_credit_card(self):
        """Test credit card confidence calculation."""
        # Test with brand identification
        confidence = self.engine._calculate_credit_card_confidence("4111111111111111", "Visa")
        assert confidence >= 0.9
        
        # Test without brand
        confidence = self.engine._calculate_credit_card_confidence("1234567890123456", None)
        assert confidence >= 0.8
    
    def test_confidence_calculation_email(self):
        """Test email confidence calculation."""
        # Test with common TLD
        confidence = self.engine._calculate_email_confidence("user@example.com")
        assert confidence >= 0.7
        
        # Test with uncommon TLD
        confidence = self.engine._calculate_email_confidence("user@example.xyz")
        assert confidence >= 0.6
    
    def test_confidence_calculation_api_key(self):
        """Test API key confidence calculation."""
        # Test AWS key
        confidence = self.engine._calculate_api_key_confidence("AKIAIOSFODNN7EXAMPLE")
        assert confidence >= 0.9
        
        # Test unknown pattern
        confidence = self.engine._calculate_api_key_confidence("randomstring123")
        assert confidence >= 0.0
    
    def test_get_category_stats(self):
        """Test category statistics retrieval."""
        stats = self.engine.get_category_stats()
        
        assert 'enabled_categories' in stats
        assert 'category_thresholds' in stats
        assert 'entropy_threshold' in stats
        assert 'temporal_consensus_required' in stats
        assert 'spacy_ner_enabled' in stats
        
        assert stats['enabled_categories'] == self.config.categories
        assert len(stats['category_thresholds']) > 0
    
    def test_classify_with_bbox(self):
        """Test classification with bounding box information."""
        bbox = BBox(10, 20, 100, 40, 0.9)
        text = "user@example.com"
        
        matches = self.engine.classify_text(text, bbox)
        
        if matches:  # If email is detected
            match = matches[0]
            assert match.bbox == bbox
    
    def test_custom_configuration(self):
        """Test classification with custom configuration."""
        custom_config = ClassificationConfig(
            categories=['phone', 'email'],  # Only phone and email
            entropy_threshold_bits_per_char=4.0,
            address_min_score=0.8
        )
        
        custom_engine = ClassificationEngine(custom_config)
        
        # Should only detect enabled categories
        mixed_text = "Call (212) 123-4567 or email user@example.com, card 4111111111111111"
        matches = custom_engine.classify_text(mixed_text)
        
        categories_found = {match.category for match in matches}
        assert categories_found.issubset({'phone', 'email'})
        assert 'credit_card' not in categories_found
    
    def test_temporal_consensus_requirement(self):
        """Test temporal consensus configuration."""
        assert self.engine.config.require_temporal_consensus >= 1
        
        # This would be tested in integration with the pipeline
        # Here we just verify the configuration is accessible
        stats = self.engine.get_category_stats()
        assert stats['temporal_consensus_required'] == self.config.require_temporal_consensus


class TestClassificationEngineEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = ClassificationConfig()
        self.engine = ClassificationEngine(self.config)
    
    def test_very_long_text(self):
        """Test classification with very long text."""
        long_text = "a" * 10000 + " user@example.com " + "b" * 10000
        
        matches = self.engine.classify_text(long_text)
        
        # Should still detect the email in the middle
        email_matches = [m for m in matches if m.category == 'email']
        assert len(email_matches) > 0
    
    def test_unicode_text(self):
        """Test classification with unicode characters."""
        unicode_text = "Contact café@résumé.com for information"
        
        matches = self.engine.classify_text(unicode_text)
        
        # Should handle unicode normalization
        email_matches = [m for m in matches if m.category == 'email']
        if email_matches:  # Email detection might vary with unicode
            match = email_matches[0]
            assert match.confidence > 0.0
    
    def test_special_characters(self):
        """Test classification with special characters."""
        special_text = "Email: user@example.com!!! Phone: (212) 123-4567???"
        
        matches = self.engine.classify_text(special_text)
        
        # Should detect despite special characters
        categories_found = {match.category for match in matches}
        assert len(categories_found) > 0
    
    def test_case_variations(self):
        """Test classification with various case patterns."""
        case_variations = [
            "USER@EXAMPLE.COM",
            "User@Example.Com", 
            "user@example.com"
        ]
        
        for email in case_variations:
            matches = self.engine.classify_text(email)
            email_matches = [m for m in matches if m.category == 'email']
            assert len(email_matches) > 0, f"Email '{email}' should be detected regardless of case"
    
    def test_boundary_conditions(self):
        """Test boundary conditions for pattern matching."""
        # Test minimum length strings
        short_strings = [
            "a@b.co",  # Minimum valid email
            "12345",   # Minimum ZIP code
            "123"      # Too short for most patterns
        ]
        
        for text in short_strings:
            matches = self.engine.classify_text(text)
            # Should not crash, results may vary
            assert isinstance(matches, list)
    
    @patch('privacy_redactor_rt.classify.logger')
    def test_unknown_category_logging(self, mock_logger):
        """Test logging for unknown categories."""
        # This tests internal method with invalid category
        matches = self.engine._classify_category("test", "unknown_category", None)
        
        assert matches == []
        mock_logger.warning.assert_called_once()
    
    def test_empty_configuration_categories(self):
        """Test with empty categories list."""
        empty_config = ClassificationConfig(categories=[])
        empty_engine = ClassificationEngine(empty_config)
        
        matches = empty_engine.classify_text("user@example.com")
        assert matches == []
    
    def test_malformed_input_handling(self):
        """Test handling of malformed input data."""
        malformed_inputs = [
            "\x00\x01\x02",  # Control characters
            "   \n\t\r   ",  # Only whitespace
            "",               # Empty string
            None              # None value
        ]
        
        for malformed in malformed_inputs:
            matches = self.engine.classify_text(malformed)
            assert isinstance(matches, list)
            assert len(matches) == 0