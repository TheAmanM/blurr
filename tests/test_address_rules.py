"""Tests for address detection rules and scoring."""

import pytest
from privacy_redactor_rt.address_rules import AddressDetector


class TestAddressDetector:
    """Test cases for AddressDetector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector(use_spacy=False)
    
    def test_initialization(self):
        """Test AddressDetector initialization."""
        assert self.detector is not None
        assert not self.detector.use_spacy
        assert len(self.detector.street_suffixes) > 0
        assert len(self.detector.directionals) > 0
        assert len(self.detector.state_abbreviations) > 0
        assert len(self.detector.canadian_provinces) > 0
    
    def test_street_suffixes_loading(self):
        """Test street suffixes are properly loaded."""
        suffixes = self.detector.street_suffixes
        
        # Check common suffixes are present
        expected_suffixes = ['street', 'st', 'avenue', 'ave', 'road', 'rd', 'drive', 'dr']
        for suffix in expected_suffixes:
            assert suffix in suffixes, f"Suffix '{suffix}' should be in street_suffixes"
    
    def test_directionals_loading(self):
        """Test directional indicators are properly loaded."""
        directionals = self.detector.directionals
        
        expected_directionals = ['north', 'n', 'south', 's', 'east', 'e', 'west', 'w']
        for directional in expected_directionals:
            assert directional in directionals, f"Directional '{directional}' should be loaded"
    
    def test_state_abbreviations_loading(self):
        """Test US state abbreviations are properly loaded."""
        states = self.detector.state_abbreviations
        
        # Check some common states
        expected_states = ['ca', 'ny', 'tx', 'fl', 'il', 'pa', 'oh', 'ga', 'nc', 'mi']
        for state in expected_states:
            assert state in states, f"State '{state}' should be in state_abbreviations"
        
        # Should have all 50 states plus territories
        assert len(states) >= 50


class TestAddressScoring:
    """Test cases for address scoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector(use_spacy=False)
    
    def test_complete_address_high_score(self):
        """Test complete addresses receive high scores."""
        complete_addresses = [
            '123 Main Street, Anytown, CA 90210',
            '456 Oak Avenue, Springfield, IL 62701',
            '789 Pine Road, Austin, TX 73301',
            '321 Elm Drive, Miami, FL 33101',
        ]
        
        for address in complete_addresses:
            score = self.detector.score_address(address)
            assert score > 0.7, f"Complete address '{address}' should have high score, got {score}"
    
    def test_partial_address_medium_score(self):
        """Test partial addresses receive medium scores."""
        partial_addresses = [
            '123 Main Street',
            'Oak Avenue, CA 90210',
            '456 Pine Road, Springfield',
        ]
        
        for address in partial_addresses:
            score = self.detector.score_address(address)
            assert 0.2 < score < 0.8, f"Partial address '{address}' should have medium score, got {score}"
    
    def test_non_address_low_score(self):
        """Test non-addresses receive low scores."""
        non_addresses = [
            'Hello world',
            'This is not an address',
            'Random text here',
            '123456789',
            'email@example.com',
        ]
        
        for text in non_addresses:
            score = self.detector.score_address(text)
            assert score < 0.3, f"Non-address '{text}' should have low score, got {score}"
    
    def test_empty_or_short_text(self):
        """Test empty or very short text receives zero score."""
        short_texts = ['', 'a', 'ab', 'abc', '   ']
        
        for text in short_texts:
            score = self.detector.score_address(text)
            assert score == 0.0, f"Short text '{text}' should have zero score"
    
    def test_postal_code_scoring(self):
        """Test postal code presence increases score."""
        with_postal = '123 Main St, Anytown, CA 90210'
        without_postal = '123 Main St, Anytown, CA'
        
        score_with = self.detector.score_address(with_postal)
        score_without = self.detector.score_address(without_postal)
        
        assert score_with > score_without, "Address with postal code should score higher"
    
    def test_street_suffix_scoring(self):
        """Test street suffix presence increases score."""
        with_suffix = '123 Main Street'
        without_suffix = '123 Main'
        
        score_with = self.detector.score_address(with_suffix)
        score_without = self.detector.score_address(without_suffix)
        
        assert score_with > score_without, "Address with street suffix should score higher"
    
    def test_canadian_addresses(self):
        """Test Canadian address scoring."""
        canadian_addresses = [
            '123 Main Street, Toronto, ON M5V 3A8',
            '456 Rue Saint-Jacques, Montreal, QC H2Y 1L9',
            '789 Granville Street, Vancouver, BC V6Z 1K3',
        ]
        
        for address in canadian_addresses:
            score = self.detector.score_address(address)
            assert score > 0.5, f"Canadian address '{address}' should have good score, got {score}"


class TestPostalCodeDetection:
    """Test cases for postal code detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector()
    
    def test_us_zip_codes(self):
        """Test US ZIP code detection."""
        us_zips = [
            '90210',
            '12345-6789',
            '00501',
            '99950',
        ]
        
        for zip_code in us_zips:
            text = f'Address with ZIP {zip_code}'
            has_postal = self.detector._has_postal_code(text.lower())
            assert has_postal, f"Should detect US ZIP code {zip_code}"
    
    def test_canadian_postal_codes(self):
        """Test Canadian postal code detection."""
        ca_postal_codes = [
            'M5V 3A8',
            'H2Y1L9',
            'V6Z 1K3',
            'K1A 0A6',
        ]
        
        for postal_code in ca_postal_codes:
            text = f'Address with postal code {postal_code}'
            has_postal = self.detector._has_postal_code(text.lower())
            assert has_postal, f"Should detect Canadian postal code {postal_code}"
    
    def test_invalid_postal_codes(self):
        """Test invalid postal code patterns are not detected."""
        invalid_codes = [
            '1234',      # Too short
            '123456',    # Wrong format (6 digits, not ZIP)
            'ABCDEF',    # All letters
        ]
        
        for code in invalid_codes:
            text = f'Text with {code}'
            has_postal = self.detector._has_postal_code(text.lower())
            assert not has_postal, f"Should not detect invalid postal code {code}"


class TestAddressComponentExtraction:
    """Test cases for address component extraction."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector()
    
    def test_extract_us_address_components(self):
        """Test extraction of US address components."""
        address = '123 Main Street, Anytown, CA 90210'
        components = self.detector.extract_address_components(address)
        
        assert components['street_number'] == '123'
        assert components['street_suffix'] == 'street'
        assert components['state_province'] == 'CA'
        assert components['postal_code'] == '90210'
        assert components['country'] == 'US'
    
    def test_extract_canadian_address_components(self):
        """Test extraction of Canadian address components."""
        address = '456 Oak Avenue, Toronto, ON M5V 3A8'
        components = self.detector.extract_address_components(address)
        
        assert components['street_number'] == '456'
        assert components['street_suffix'] == 'avenue'
        assert components['state_province'] == 'ON'
        assert components['postal_code'] == 'M5V 3A8'
        assert components['country'] == 'CA'
    
    def test_extract_partial_components(self):
        """Test extraction from partial addresses."""
        partial_address = '789 Pine Road'
        components = self.detector.extract_address_components(partial_address)
        
        assert components['street_number'] == '789'
        assert components['street_suffix'] == 'road'
        assert components['state_province'] is None
        assert components['postal_code'] is None
    
    def test_extract_no_components(self):
        """Test extraction from non-address text."""
        non_address = 'This is not an address'
        components = self.detector.extract_address_components(non_address)
        
        # All components should be None
        for key, value in components.items():
            assert value is None, f"Component '{key}' should be None for non-address text"


class TestSpecialAddressTypes:
    """Test cases for special address types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector()
    
    def test_po_box_detection(self):
        """Test PO Box address detection."""
        po_box_addresses = [
            'PO Box 123',
            'P.O. Box 456',
            'Post Office Box 789',
            'po box 321',
        ]
        
        for address in po_box_addresses:
            assert self.detector.is_po_box(address), f"Should detect PO Box in '{address}'"
    
    def test_rural_route_detection(self):
        """Test rural route address detection."""
        rural_routes = [
            'RR 1',
            'Rural Route 2',
            'R.R. 3',
            'rr 4',
        ]
        
        for address in rural_routes:
            assert self.detector.is_rural_route(address), f"Should detect rural route in '{address}'"
    
    def test_non_special_addresses(self):
        """Test that regular addresses are not detected as special types."""
        regular_addresses = [
            '123 Main Street',
            '456 Oak Avenue',
            'Regular address text',
        ]
        
        for address in regular_addresses:
            assert not self.detector.is_po_box(address), f"Should not detect PO Box in '{address}'"
            assert not self.detector.is_rural_route(address), f"Should not detect rural route in '{address}'"


class TestPostalCodeValidation:
    """Test cases for postal code validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector()
    
    def test_validate_us_zip_codes(self):
        """Test US ZIP code validation."""
        valid_us_zips = [
            '90210',
            '12345-6789',
            '00501',
            '99950',
        ]
        
        for zip_code in valid_us_zips:
            assert self.detector.validate_postal_code(zip_code, 'US'), f"US ZIP {zip_code} should be valid"
    
    def test_validate_canadian_postal_codes(self):
        """Test Canadian postal code validation."""
        valid_ca_postal = [
            'M5V 3A8',
            'H2Y1L9',
            'V6Z 1K3',
            'K1A 0A6',
        ]
        
        for postal_code in valid_ca_postal:
            assert self.detector.validate_postal_code(postal_code, 'CA'), f"Canadian postal {postal_code} should be valid"
    
    def test_invalid_postal_codes(self):
        """Test invalid postal code validation."""
        invalid_codes = [
            ('1234', 'US'),      # Too short for US
            ('123456', 'US'),    # Wrong format for US
            ('ABCDEF', 'CA'),    # Wrong format for CA
            ('12345', 'CA'),     # US format for CA
        ]
        
        for code, country in invalid_codes:
            assert not self.detector.validate_postal_code(code, country), f"Invalid postal {code} for {country} should fail validation"


class TestAddressDetectorWithSpacy:
    """Test cases for AddressDetector with spaCy integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Try to initialize with spaCy, but don't fail if not available
        try:
            self.detector = AddressDetector(use_spacy=True)
        except (ImportError, OSError):
            pytest.skip("spaCy not available or model not installed")
    
    def test_spacy_initialization(self):
        """Test spaCy initialization."""
        if self.detector.use_spacy:
            assert self.detector._nlp is None  # Should be lazy loaded
    
    def test_spacy_scoring(self):
        """Test address scoring with spaCy NER."""
        if not self.detector.use_spacy:
            pytest.skip("spaCy not available")
        
        addresses_with_locations = [
            '123 Main Street, New York, NY 10001',
            '456 Oak Avenue, Los Angeles, CA 90210',
            '789 Pine Road, Chicago, IL 60601',
        ]
        
        for address in addresses_with_locations:
            score = self.detector.score_address(address)
            # spaCy should help identify location entities and increase score
            assert score > 0.5, f"Address with locations '{address}' should have good score with spaCy"


class TestAddressDetectorIntegration:
    """Integration tests for AddressDetector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = AddressDetector()
    
    def test_mixed_content_address_detection(self):
        """Test address detection in mixed content."""
        mixed_content = """
        Contact information:
        Name: John Doe
        Phone: (555) 123-4567
        Address: 123 Main Street, Anytown, CA 90210
        Email: john@example.com
        """
        
        # Should detect the address line
        lines = mixed_content.strip().split('\n')
        address_scores = [self.detector.score_address(line.strip()) for line in lines]
        
        # At least one line should have a high address score
        max_score = max(address_scores)
        assert max_score > 0.7, "Should detect address in mixed content"
    
    def test_multiple_address_formats(self):
        """Test detection of various address formats."""
        address_formats = [
            # Standard formats
            '123 Main Street, Anytown, CA 90210',
            '456 Oak Ave, Springfield IL 62701',
            '789 Pine Rd., Austin TX 73301',
            
            # With unit numbers
            '123 Main St Apt 4B, City, ST 12345',
            '456 Oak Avenue Suite 200, Town, ST 54321',
            
            # Canadian formats
            '123 Main Street, Toronto, ON M5V 3A8',
            '456 Rue Saint-Jacques, Montreal QC H2Y 1L9',
            
            # Special formats
            'PO Box 123, Anytown, ST 12345',
            'RR 1, Rural Town, ST 54321',
        ]
        
        for address in address_formats:
            score = self.detector.score_address(address)
            assert score > 0.5, f"Address format '{address}' should be detected, got score {score}"
    
    def test_address_vs_non_address_discrimination(self):
        """Test that detector can discriminate between addresses and non-addresses."""
        addresses = [
            '123 Main Street, Anytown, CA 90210',
            '456 Oak Avenue, Springfield, IL 62701',
        ]
        
        non_addresses = [
            'This is just regular text',
            'Phone: (555) 123-4567',
            'Email: user@example.com',
            'Random numbers: 123 456 789',
        ]
        
        # Addresses should score higher than non-addresses
        address_scores = [self.detector.score_address(addr) for addr in addresses]
        non_address_scores = [self.detector.score_address(text) for text in non_addresses]
        
        min_address_score = min(address_scores)
        max_non_address_score = max(non_address_scores)
        
        assert min_address_score > max_non_address_score, "Addresses should score higher than non-addresses"