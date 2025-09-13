"""Unit tests for redaction engine."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock

from privacy_redactor_rt.redact import RedactionEngine, RedactionMethod
from privacy_redactor_rt.config import RedactionConfig
from privacy_redactor_rt.types import Track, BBox, Match


class TestRedactionEngine:
    """Test cases for RedactionEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = RedactionConfig()
        self.engine = RedactionEngine(self.config)
        
        # Create test frame (100x100 blue image)
        self.test_frame = np.full((100, 100, 3), [255, 0, 0], dtype=np.uint8)
        
        # Create test bounding box
        self.test_bbox = BBox(x1=20, y1=20, x2=60, y2=60, confidence=0.9)
        
        # Create test match
        self.test_match = Match(
            category="phone",
            confidence=0.8,
            masked_text="555***1234",
            bbox=self.test_bbox
        )
        
        # Create test track
        self.test_track = Track(
            id="track_1",
            bbox=self.test_bbox,
            matches=[self.test_match],
            age=5,
            hits=3,
            last_ocr_frame=0
        )
        
    def test_init_with_valid_config(self):
        """Test initialization with valid configuration."""
        config = RedactionConfig(
            default_method="gaussian",
            gaussian_kernel_size=15,
            pixelate_block_size=8
        )
        engine = RedactionEngine(config)
        assert engine.config == config
        
    def test_init_with_invalid_default_method(self):
        """Test initialization with invalid default method."""
        config = RedactionConfig(default_method="invalid")
        with pytest.raises(ValueError, match="Invalid default redaction method"):
            RedactionEngine(config)
            
    def test_init_with_invalid_category_method(self):
        """Test initialization with invalid category method."""
        config = RedactionConfig(category_methods={"phone": "invalid"})
        with pytest.raises(ValueError, match="Invalid redaction method"):
            RedactionEngine(config)
            
    def test_init_with_even_kernel_size(self):
        """Test initialization with even kernel size."""
        config = RedactionConfig(gaussian_kernel_size=14)
        with pytest.raises(ValueError, match="Gaussian kernel size must be odd"):
            RedactionEngine(config)
            
    def test_init_with_small_kernel_size(self):
        """Test initialization with too small kernel size."""
        config = RedactionConfig(gaussian_kernel_size=1)
        with pytest.raises(ValueError, match="Gaussian kernel size must be at least 3"):
            RedactionEngine(config)
            
    def test_init_with_small_pixelate_block(self):
        """Test initialization with too small pixelate block."""
        config = RedactionConfig(pixelate_block_size=1)
        with pytest.raises(ValueError, match="Pixelate block size must be at least 2"):
            RedactionEngine(config)
            
    def test_redact_regions_empty_tracks(self):
        """Test redaction with empty tracks list."""
        result = self.engine.redact_regions(self.test_frame, [])
        np.testing.assert_array_equal(result, self.test_frame)
        
    def test_redact_regions_track_without_matches(self):
        """Test redaction with track that has no matches."""
        track = Track(
            id="track_1",
            bbox=self.test_bbox,
            matches=[],
            age=1,
            hits=1,
            last_ocr_frame=0
        )
        result = self.engine.redact_regions(self.test_frame, [track])
        np.testing.assert_array_equal(result, self.test_frame)
        
    def test_redact_regions_gaussian_blur(self):
        """Test Gaussian blur redaction."""
        config = RedactionConfig(default_method="gaussian", gaussian_kernel_size=15)
        engine = RedactionEngine(config)
        
        result = engine.redact_regions(self.test_frame, [self.test_track])
        
        # Check that frame was modified
        assert not np.array_equal(result, self.test_frame)
        
        # Check that only the region was modified (approximately)
        # Areas outside the inflated bbox should be unchanged
        inflated = self.test_bbox.inflate(config.inflate_bbox_px)
        
        # Check corners are unchanged
        np.testing.assert_array_equal(
            result[0:10, 0:10], 
            self.test_frame[0:10, 0:10]
        )
        np.testing.assert_array_equal(
            result[90:100, 90:100], 
            self.test_frame[90:100, 90:100]
        )
        
    def test_redact_regions_pixelation(self):
        """Test pixelation redaction."""
        config = RedactionConfig(default_method="pixelate", pixelate_block_size=8)
        engine = RedactionEngine(config)
        
        result = engine.redact_regions(self.test_frame, [self.test_track])
        
        # Check that frame was modified
        assert not np.array_equal(result, self.test_frame)
        
        # Check that corners are unchanged
        np.testing.assert_array_equal(
            result[0:10, 0:10], 
            self.test_frame[0:10, 0:10]
        )
        
    def test_redact_regions_solid_color(self):
        """Test solid color redaction."""
        config = RedactionConfig(
            default_method="solid", 
            solid_color=(0, 255, 0)  # Green
        )
        engine = RedactionEngine(config)
        
        result = engine.redact_regions(self.test_frame, [self.test_track])
        
        # Check that frame was modified
        assert not np.array_equal(result, self.test_frame)
        
        # Check that redacted region is green
        inflated = self.test_bbox.inflate(config.inflate_bbox_px)
        x1, y1 = max(0, inflated.x1), max(0, inflated.y1)
        x2, y2 = min(100, inflated.x2), min(100, inflated.y2)
        
        redacted_region = result[y1:y2, x1:x2]
        expected_color = np.array([0, 255, 0], dtype=np.uint8)
        
        # Check that all pixels in region are green
        assert np.all(redacted_region == expected_color)
        
    def test_redact_regions_category_specific_method(self):
        """Test category-specific redaction method."""
        config = RedactionConfig(
            default_method="gaussian",
            category_methods={"phone": "solid"},
            solid_color=(255, 0, 255)  # Magenta
        )
        engine = RedactionEngine(config)
        
        result = engine.redact_regions(self.test_frame, [self.test_track])
        
        # Should use solid color for phone category
        inflated = self.test_bbox.inflate(config.inflate_bbox_px)
        x1, y1 = max(0, inflated.x1), max(0, inflated.y1)
        x2, y2 = min(100, inflated.x2), min(100, inflated.y2)
        
        redacted_region = result[y1:y2, x1:x2]
        expected_color = np.array([255, 0, 255], dtype=np.uint8)
        
        assert np.all(redacted_region == expected_color)
        
    def test_redact_regions_multiple_tracks(self):
        """Test redaction with multiple tracks."""
        # Create second track with different category
        bbox2 = BBox(x1=70, y1=70, x2=90, y2=90, confidence=0.8)
        match2 = Match(
            category="email",
            confidence=0.7,
            masked_text="tes***@example.com",
            bbox=bbox2
        )
        track2 = Track(
            id="track_2",
            bbox=bbox2,
            matches=[match2],
            age=3,
            hits=2,
            last_ocr_frame=0
        )
        
        config = RedactionConfig(
            default_method="gaussian",
            category_methods={"phone": "solid", "email": "pixelate"},
            solid_color=(255, 0, 0),  # Red
            pixelate_block_size=4
        )
        engine = RedactionEngine(config)
        
        result = engine.redact_regions(self.test_frame, [self.test_track, track2])
        
        # Both regions should be modified
        assert not np.array_equal(result, self.test_frame)
        
        # First region should be solid red
        inflated1 = self.test_bbox.inflate(config.inflate_bbox_px)
        x1, y1 = max(0, inflated1.x1), max(0, inflated1.y1)
        x2, y2 = min(100, inflated1.x2), min(100, inflated1.y2)
        
        redacted_region1 = result[y1:y2, x1:x2]
        expected_color = np.array([255, 0, 0], dtype=np.uint8)
        assert np.all(redacted_region1 == expected_color)
        
    def test_redact_regions_boundary_clamping(self):
        """Test that bounding boxes are properly clamped to frame boundaries."""
        # Create bbox that extends outside frame
        bbox = BBox(x1=-10, y1=-10, x2=110, y2=110, confidence=0.9)
        match = Match(
            category="phone",
            confidence=0.8,
            masked_text="555***1234",
            bbox=bbox
        )
        track = Track(
            id="track_1",
            bbox=bbox,
            matches=[match],
            age=1,
            hits=1,
            last_ocr_frame=0
        )
        
        # Should not raise exception
        result = self.engine.redact_regions(self.test_frame, [track])
        assert result.shape == self.test_frame.shape
        
    def test_redact_regions_small_region(self):
        """Test redaction of very small region."""
        # Create tiny bbox
        bbox = BBox(x1=50, y1=50, x2=51, y2=51, confidence=0.9)
        match = Match(
            category="phone",
            confidence=0.8,
            masked_text="5",
            bbox=bbox
        )
        track = Track(
            id="track_1",
            bbox=bbox,
            matches=[match],
            age=1,
            hits=1,
            last_ocr_frame=0
        )
        
        # Should handle gracefully (might skip redaction)
        result = self.engine.redact_regions(self.test_frame, [track])
        assert result.shape == self.test_frame.shape
        
    def test_apply_gaussian_blur(self):
        """Test Gaussian blur application."""
        roi = np.full((20, 20, 3), [255, 0, 0], dtype=np.uint8)
        result = self.engine._apply_gaussian_blur(roi)
        
        # Result should be different from original
        assert not np.array_equal(result, roi)
        # Result should have same shape
        assert result.shape == roi.shape
        # Result should be blurred (less sharp edges)
        assert result.dtype == roi.dtype
        
    def test_apply_pixelation(self):
        """Test pixelation application."""
        # Create ROI with gradient for visible pixelation
        roi = np.zeros((20, 20, 3), dtype=np.uint8)
        for i in range(20):
            for j in range(20):
                roi[i, j] = [i * 12, j * 12, 128]
                
        result = self.engine._apply_pixelation(roi)
        
        # Result should be different from original
        assert not np.array_equal(result, roi)
        # Result should have same shape
        assert result.shape == roi.shape
        assert result.dtype == roi.dtype
        
    def test_apply_pixelation_small_roi(self):
        """Test pixelation on ROI smaller than block size."""
        roi = np.full((4, 4, 3), [255, 0, 0], dtype=np.uint8)
        config = RedactionConfig(pixelate_block_size=8)
        engine = RedactionEngine(config)
        
        result = engine._apply_pixelation(roi)
        
        # Should return original ROI unchanged
        np.testing.assert_array_equal(result, roi)
        
    def test_apply_solid_color(self):
        """Test solid color application."""
        roi = np.full((20, 20, 3), [255, 0, 0], dtype=np.uint8)
        config = RedactionConfig(solid_color=(0, 255, 0))
        engine = RedactionEngine(config)
        
        result = engine._apply_solid_color(roi)
        
        # Result should be all green
        expected = np.full((20, 20, 3), [0, 255, 0], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)
        
    def test_get_method_for_category_default(self):
        """Test getting default method for category."""
        method = self.engine.get_method_for_category("phone")
        assert method == RedactionMethod.GAUSSIAN
        
    def test_get_method_for_category_specific(self):
        """Test getting category-specific method."""
        config = RedactionConfig(category_methods={"phone": "pixelate"})
        engine = RedactionEngine(config)
        
        method = engine.get_method_for_category("phone")
        assert method == RedactionMethod.PIXELATE
        
        # Other categories should use default
        method = engine.get_method_for_category("email")
        assert method == RedactionMethod.GAUSSIAN
        
    def test_set_method_for_category(self):
        """Test setting method for category."""
        self.engine.set_method_for_category("phone", RedactionMethod.SOLID)
        
        method = self.engine.get_method_for_category("phone")
        assert method == RedactionMethod.SOLID
        
        # Should be reflected in config
        assert self.engine.config.category_methods["phone"] == "solid"
        
    def test_get_supported_methods(self):
        """Test getting list of supported methods."""
        methods = self.engine.get_supported_methods()
        expected = ["gaussian", "pixelate", "solid"]
        assert set(methods) == set(expected)
        
    def test_redaction_preserves_frame_properties(self):
        """Test that redaction preserves frame properties."""
        original_shape = self.test_frame.shape
        original_dtype = self.test_frame.dtype
        
        result = self.engine.redact_regions(self.test_frame, [self.test_track])
        
        assert result.shape == original_shape
        assert result.dtype == original_dtype
        
    def test_redaction_does_not_modify_original(self):
        """Test that redaction does not modify the original frame."""
        original_copy = self.test_frame.copy()
        
        self.engine.redact_regions(self.test_frame, [self.test_track])
        
        # Original frame should be unchanged
        np.testing.assert_array_equal(self.test_frame, original_copy)


class TestRedactionVisualValidation:
    """Visual validation tests for redaction quality."""
    
    def setup_method(self):
        """Set up test fixtures for visual validation."""
        # Create more complex test image with text-like patterns
        self.test_frame = self._create_test_image()
        self.bbox = BBox(x1=50, y1=50, x2=150, y2=100, confidence=0.9)
        
    def _create_test_image(self) -> np.ndarray:
        """Create a test image with text-like patterns."""
        # Create 200x200 white image
        img = np.full((200, 200, 3), 255, dtype=np.uint8)
        
        # Add some text-like rectangular patterns
        cv2.rectangle(img, (60, 60), (140, 80), (0, 0, 0), -1)  # Black rectangle
        cv2.rectangle(img, (70, 65), (80, 75), (255, 255, 255), -1)  # White rectangle
        cv2.rectangle(img, (90, 65), (100, 75), (255, 255, 255), -1)  # White rectangle
        cv2.rectangle(img, (110, 65), (130, 75), (255, 255, 255), -1)  # White rectangle
        
        return img
        
    def test_gaussian_blur_effectiveness(self):
        """Test that Gaussian blur effectively obscures content."""
        config = RedactionConfig(
            default_method="gaussian",
            gaussian_kernel_size=21,
            gaussian_sigma=8.0
        )
        engine = RedactionEngine(config)
        
        # Extract original region
        x1, y1, x2, y2 = self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2
        original_roi = self.test_frame[y1:y2, x1:x2].copy()
        
        # Apply blur
        blurred_roi = engine._apply_gaussian_blur(original_roi)
        
        # Calculate difference metrics
        mse = np.mean((original_roi.astype(float) - blurred_roi.astype(float)) ** 2)
        
        # Blur should create significant difference
        assert mse > 100, "Gaussian blur should create significant visual difference"
        
        # Check that blur reduces high-frequency content
        original_gray = cv2.cvtColor(original_roi, cv2.COLOR_BGR2GRAY)
        blurred_gray = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate gradient magnitude (measure of sharpness)
        original_grad = cv2.Sobel(original_gray, cv2.CV_64F, 1, 1, ksize=3)
        blurred_grad = cv2.Sobel(blurred_gray, cv2.CV_64F, 1, 1, ksize=3)
        
        original_sharpness = np.mean(np.abs(original_grad))
        blurred_sharpness = np.mean(np.abs(blurred_grad))
        
        # Blurred image should be less sharp
        assert blurred_sharpness < original_sharpness * 0.5, "Blur should significantly reduce sharpness"
        
    def test_pixelation_effectiveness(self):
        """Test that pixelation effectively obscures content."""
        config = RedactionConfig(
            default_method="pixelate",
            pixelate_block_size=8
        )
        engine = RedactionEngine(config)
        
        # Extract original region
        x1, y1, x2, y2 = self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2
        original_roi = self.test_frame[y1:y2, x1:x2].copy()
        
        # Apply pixelation
        pixelated_roi = engine._apply_pixelation(original_roi)
        
        # Pixelated image should have block-like structure
        # Check that adjacent pixels in blocks have same values
        block_size = config.pixelate_block_size
        h, w = pixelated_roi.shape[:2]
        
        # Sample a few blocks and verify uniformity
        for by in range(0, h - block_size, block_size):
            for bx in range(0, w - block_size, block_size):
                block = pixelated_roi[by:by+block_size, bx:bx+block_size]
                
                # All pixels in block should be similar (allowing for some rounding)
                block_mean = np.mean(block, axis=(0, 1))
                block_std = np.std(block, axis=(0, 1))
                
                # Standard deviation should be very low within blocks
                assert np.all(block_std < 5), "Pixels within blocks should be very similar"
                
    def test_solid_color_effectiveness(self):
        """Test that solid color completely obscures content."""
        config = RedactionConfig(
            default_method="solid",
            solid_color=(128, 64, 192)  # Purple
        )
        engine = RedactionEngine(config)
        
        # Extract original region
        x1, y1, x2, y2 = self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2
        original_roi = self.test_frame[y1:y2, x1:x2].copy()
        
        # Apply solid color
        solid_roi = engine._apply_solid_color(original_roi)
        
        # All pixels should be exactly the specified color
        expected_color = np.array([128, 64, 192], dtype=np.uint8)
        assert np.all(solid_roi == expected_color), "All pixels should be exactly the solid color"
        
        # Should be completely different from original
        assert not np.array_equal(solid_roi, original_roi), "Solid color should completely replace original"
        
    def test_redaction_coverage_with_inflation(self):
        """Test that bounding box inflation provides adequate coverage."""
        config = RedactionConfig(
            default_method="solid",
            solid_color=(255, 0, 0),  # Red
            inflate_bbox_px=5
        )
        engine = RedactionEngine(config)
        
        # Create track
        match = Match(
            category="test",
            confidence=0.9,
            masked_text="test",
            bbox=self.bbox
        )
        track = Track(
            id="test_track",
            bbox=self.bbox,
            matches=[match],
            age=1,
            hits=1,
            last_ocr_frame=0
        )
        
        result = engine.redact_regions(self.test_frame, [track])
        
        # Check that inflated region is redacted
        inflated = self.bbox.inflate(config.inflate_bbox_px)
        x1 = max(0, inflated.x1)
        y1 = max(0, inflated.y1)
        x2 = min(self.test_frame.shape[1], inflated.x2)
        y2 = min(self.test_frame.shape[0], inflated.y2)
        
        redacted_region = result[y1:y2, x1:x2]
        expected_color = np.array([255, 0, 0], dtype=np.uint8)
        
        # All pixels in inflated region should be red
        assert np.all(redacted_region == expected_color), "Inflated region should be completely redacted"
        
        # Original bbox region should definitely be redacted
        orig_region = result[self.bbox.y1:self.bbox.y2, self.bbox.x1:self.bbox.x2]
        assert np.all(orig_region == expected_color), "Original bbox region should be redacted"


if __name__ == "__main__":
    pytest.main([__file__])