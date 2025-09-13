"""Address detection rules and scoring."""

import re
from typing import List, Dict, Set, Optional, Tuple


class AddressDetector:
    """Rule-based address detection with scoring."""
    
    def __init__(self, use_spacy: bool = False, spacy_model: str = "en_core_web_sm"):
        """
        Initialize address detection rules.
        
        Args:
            use_spacy: Whether to use spaCy NER for enhanced detection
            spacy_model: spaCy model name to use
        """
        self.use_spacy = use_spacy
        self.spacy_model = spacy_model
        self._nlp = None
        
        # Initialize rule-based components
        self.street_suffixes = self._load_street_suffixes()
        self.directionals = self._load_directionals()
        self.unit_types = self._load_unit_types()
        self.state_abbreviations = self._load_state_abbreviations()
        self.canadian_provinces = self._load_canadian_provinces()
        
        # Compile regex patterns
        self.patterns = self._compile_patterns()
    
    def _load_street_suffixes(self) -> Set[str]:
        """Load common street suffixes."""
        return {
            # Primary suffixes
            'alley', 'ally', 'avenue', 'ave', 'boulevard', 'blvd', 'circle', 'cir',
            'court', 'ct', 'drive', 'dr', 'lane', 'ln', 'place', 'pl', 'road', 'rd',
            'street', 'st', 'way', 'trail', 'trl', 'parkway', 'pkwy', 'highway', 'hwy',
            'route', 'rte', 'square', 'sq', 'terrace', 'ter', 'crescent', 'cres',
            
            # Secondary suffixes
            'apartment', 'apt', 'building', 'bldg', 'floor', 'fl', 'room', 'rm',
            'suite', 'ste', 'unit', 'lot', 'pier', 'slip', 'space', 'spc', 'stop',
            'trailer', 'trlr', 'box', 'drawer', 'rural', 'rr', 'route', 'rt',
            
            # Directionals as suffixes
            'north', 'n', 'south', 's', 'east', 'e', 'west', 'w',
            'northeast', 'ne', 'northwest', 'nw', 'southeast', 'se', 'southwest', 'sw'
        }
    
    def _load_directionals(self) -> Set[str]:
        """Load directional indicators."""
        return {
            'north', 'n', 'south', 's', 'east', 'e', 'west', 'w',
            'northeast', 'ne', 'northwest', 'nw', 'southeast', 'se', 'southwest', 'sw'
        }
    
    def _load_unit_types(self) -> Set[str]:
        """Load unit type indicators."""
        return {
            'apartment', 'apt', 'building', 'bldg', 'floor', 'fl', 'room', 'rm',
            'suite', 'ste', 'unit', 'lot', 'space', 'spc', 'trailer', 'trlr',
            'box', 'drawer', 'slip', 'pier', '#', 'number', 'no'
        }
    
    def _load_state_abbreviations(self) -> Set[str]:
        """Load US state abbreviations."""
        return {
            'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga',
            'hi', 'id', 'il', 'in', 'ia', 'ks', 'ky', 'la', 'me', 'md',
            'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
            'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc',
            'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy',
            'dc', 'pr', 'vi', 'gu', 'as', 'mp'
        }
    
    def _load_canadian_provinces(self) -> Set[str]:
        """Load Canadian province abbreviations."""
        return {
            'ab', 'bc', 'mb', 'nb', 'nl', 'ns', 'nt', 'nu', 'on', 'pe', 'qc', 'sk', 'yt'
        }
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for address components."""
        return {
            # US ZIP codes: 12345 or 12345-6789 (exactly 4 digits after dash)
            'us_zip': re.compile(r'\b\d{5}(?:-\d{4})?\b'),
            
            # Canadian postal codes: A1A 1A1 or A1A1A1
            'ca_postal': re.compile(r'\b[A-Za-z]\d[A-Za-z][\s-]?\d[A-Za-z]\d\b'),
            
            # Street numbers: 123, 123A, 123-125
            'street_number': re.compile(r'\b\d+(?:[A-Za-z]|[-/]\d+)?\b'),
            
            # PO Box patterns
            'po_box': re.compile(r'\b(?:po|p\.o\.?|post\s+office)\s*box\s+\d+\b', re.IGNORECASE),
            
            # Rural route patterns
            'rural_route': re.compile(r'\b(?:rr|rural\s+route|r\.r\.)\s*\d+\b', re.IGNORECASE),
            
            # Highway patterns
            'highway': re.compile(r'\b(?:hwy|highway)\s+\d+\b', re.IGNORECASE),
        }
    
    def score_address(self, text: str) -> float:
        """
        Score text for address likelihood using rule-based approach.
        
        Args:
            text: Input text to score
            
        Returns:
            Score between 0.0 and 1.0 indicating address likelihood
        """
        if not text or len(text.strip()) < 5:
            return 0.0
        
        # Normalize text
        normalized = text.lower().strip()
        words = normalized.split()
        
        if len(words) < 2:
            return 0.0
        
        score = 0.0
        max_score = 0.0
        
        # Check for postal codes (high weight)
        if self._has_postal_code(normalized):
            score += 0.4
        max_score += 0.4
        
        # Check for street suffixes (high weight)
        suffix_score = self._score_street_suffixes(words)
        score += suffix_score * 0.3
        max_score += 0.3
        
        # Check for street numbers (medium weight)
        if self._has_street_number(normalized):
            score += 0.15
        max_score += 0.15
        
        # Check for directionals (low weight)
        directional_score = self._score_directionals(words)
        score += directional_score * 0.1
        max_score += 0.1
        
        # Check for unit indicators (low weight)
        unit_score = self._score_unit_indicators(words)
        score += unit_score * 0.05
        max_score += 0.05
        
        # Use spaCy NER if available
        if self.use_spacy:
            spacy_score = self._score_with_spacy(text)
            if spacy_score > 0:
                score += spacy_score * 0.2
                max_score += 0.2
        
        # Normalize score
        if max_score > 0:
            return min(1.0, score / max_score)
        
        return 0.0
    
    def _has_postal_code(self, text: str) -> bool:
        """Check if text contains US ZIP or Canadian postal code."""
        return (self.patterns['us_zip'].search(text) is not None or
                self.patterns['ca_postal'].search(text) is not None)
    
    def _has_street_number(self, text: str) -> bool:
        """Check if text contains street number pattern."""
        return self.patterns['street_number'].search(text) is not None
    
    def _score_street_suffixes(self, words: List[str]) -> float:
        """Score based on presence of street suffixes."""
        suffix_count = 0
        for word in words:
            # Remove punctuation for matching
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.street_suffixes:
                suffix_count += 1
        
        # Return normalized score (max 1.0 for multiple suffixes)
        return min(1.0, suffix_count / 2.0)
    
    def _score_directionals(self, words: List[str]) -> float:
        """Score based on presence of directional indicators."""
        directional_count = 0
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.directionals:
                directional_count += 1
        
        return min(1.0, directional_count / 2.0)
    
    def _score_unit_indicators(self, words: List[str]) -> float:
        """Score based on presence of unit type indicators."""
        unit_count = 0
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.unit_types:
                unit_count += 1
        
        return min(1.0, unit_count / 2.0)
    
    def _score_with_spacy(self, text: str) -> float:
        """Score using spaCy NER if available."""
        if not self.use_spacy:
            return 0.0
        
        try:
            if self._nlp is None:
                import spacy
                self._nlp = spacy.load(self.spacy_model)
        except (ImportError, OSError):
            # spaCy not available or model not found
            self.use_spacy = False
            return 0.0
        
        doc = self._nlp(text)
        
        # Look for location entities
        location_labels = {'GPE', 'LOC', 'FAC'}  # Geopolitical, Location, Facility
        location_count = 0
        total_tokens = len([token for token in doc if not token.is_space])
        
        for ent in doc.ents:
            if ent.label_ in location_labels:
                location_count += len(ent)
        
        if total_tokens > 0:
            return min(1.0, location_count / total_tokens)
        
        return 0.0
    
    def extract_address_components(self, text: str) -> Dict[str, Optional[str]]:
        """
        Extract individual address components from text.
        
        Args:
            text: Input address text
            
        Returns:
            Dictionary with extracted components
        """
        components = {
            'street_number': None,
            'street_name': None,
            'street_suffix': None,
            'unit_type': None,
            'unit_number': None,
            'city': None,
            'state_province': None,
            'postal_code': None,
            'country': None
        }
        
        normalized = text.lower().strip()
        words = normalized.split()
        
        # Extract postal code
        us_zip_match = self.patterns['us_zip'].search(text)
        ca_postal_match = self.patterns['ca_postal'].search(text)
        
        if us_zip_match:
            components['postal_code'] = us_zip_match.group()
            components['country'] = 'US'
        elif ca_postal_match:
            components['postal_code'] = ca_postal_match.group()
            components['country'] = 'CA'
        
        # Extract street number (usually first number)
        street_num_match = self.patterns['street_number'].search(text)
        if street_num_match:
            components['street_number'] = street_num_match.group()
        
        # Extract street suffix
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.street_suffixes:
                components['street_suffix'] = clean_word
                break
        
        # Extract state/province (look for 2-letter codes near postal code)
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.state_abbreviations:
                components['state_province'] = clean_word.upper()
                components['country'] = 'US'
                break
            elif clean_word in self.canadian_provinces:
                components['state_province'] = clean_word.upper()
                components['country'] = 'CA'
                break
        
        return components
    
    def is_po_box(self, text: str) -> bool:
        """Check if text represents a PO Box address."""
        return self.patterns['po_box'].search(text) is not None
    
    def is_rural_route(self, text: str) -> bool:
        """Check if text represents a rural route address."""
        return self.patterns['rural_route'].search(text) is not None
    
    def validate_postal_code(self, postal_code: str, country: str = 'US') -> bool:
        """
        Validate postal code format for given country.
        
        Args:
            postal_code: Postal code to validate
            country: Country code ('US' or 'CA')
            
        Returns:
            True if postal code format is valid
        """
        if country.upper() == 'US':
            return self.patterns['us_zip'].fullmatch(postal_code) is not None
        elif country.upper() == 'CA':
            return self.patterns['ca_postal'].fullmatch(postal_code) is not None
        
        return False