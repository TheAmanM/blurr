"""UI element detection for privacy settings interfaces."""

import logging
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from .types import BBox

logger = logging.getLogger(__name__)


@dataclass
class UIElement:
    """Represents a detected UI element."""
    bbox: BBox
    element_type: str  # 'checkbox', 'toggle', 'radio', 'button'
    state: str  # 'enabled', 'disabled', 'checked', 'unchecked'
    confidence: float
    text_label: Optional[str] = None


class PrivacyUIDetector:
    """Detector for privacy settings UI elements in images."""
    
    def __init__(self):
        """Initialize the UI detector."""
        self.checkbox_cascade = None
        self.toggle_cascade = None
        self._load_cascades()
    
    def _load_cascades(self):
        """Load Haar cascades for UI element detection."""
        try:
            # For now, we'll use basic computer vision techniques
            # In a production system, you'd train custom models or use pre-trained ones
            logger.info("UI detector initialized with basic CV methods")
        except Exception as e:
            logger.warning(f"Failed to load UI cascades: {e}")
    
    def detect_privacy_elements(self, image: np.ndarray) -> List[UIElement]:
        """Detect privacy-related UI elements in an image.
        
        Args:
            image: Input image as BGR numpy array
            
        Returns:
            List of detected UI elements
        """
        if image is None or image.size == 0:
            return []
        
        elements = []
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect different types of UI elements
        logger.debug("Detecting checkboxes...")
        checkboxes = self._detect_checkboxes(gray, image)
        logger.debug(f"Found {len(checkboxes)} checkboxes")
        elements.extend(checkboxes)
        
        logger.debug("Detecting toggles...")
        toggles = self._detect_toggles(gray, image)
        logger.debug(f"Found {len(toggles)} toggles")
        elements.extend(toggles)
        
        logger.debug("Detecting radio buttons...")
        radios = self._detect_radio_buttons(gray, image)
        logger.debug(f"Found {len(radios)} radio buttons")
        elements.extend(radios)
        
        logger.debug("Detecting buttons...")
        buttons = self._detect_privacy_buttons(gray, image)
        logger.debug(f"Found {len(buttons)} buttons")
        elements.extend(buttons)
        
        logger.info(f"Total elements detected: {len(elements)}")
        return elements
    
    def _detect_checkboxes(self, gray: np.ndarray, color_image: np.ndarray) -> List[UIElement]:
        """Detect checkbox elements and their states."""
        elements = []
        
        # Use edge detection to find rectangular shapes
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (checkboxes are typically small squares)
            if 15 <= w <= 35 and 15 <= h <= 35 and abs(w - h) <= 8:
                # Check if it's roughly rectangular
                area = cv2.contourArea(contour)
                rect_area = w * h
                
                if area > 0 and (area / rect_area) > 0.6:  # At least 60% filled
                    # Extract the region from color image for better analysis
                    roi_color = color_image[y:y+h, x:x+w]
                    roi_gray = gray[y:y+h, x:x+w]
                    
                    # Determine if checkbox is checked
                    state = self._analyze_checkbox_state_improved(roi_gray, roi_color)
                    
                    if state != 'unknown':
                        bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.8)
                        element = UIElement(
                            bbox=bbox,
                            element_type='checkbox',
                            state=state,
                            confidence=0.8
                        )
                        elements.append(element)
        
        return elements
    
    def _detect_toggles(self, gray: np.ndarray, color_image: np.ndarray) -> List[UIElement]:
        """Detect toggle switch elements and their states."""
        elements = []
        
        # Look for rounded rectangular shapes (toggle backgrounds)
        # Use morphological operations to find toggle-like shapes
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        morph = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Toggle switches are typically wider than they are tall
            if 30 <= w <= 80 and 15 <= h <= 35 and w > h * 1.5:
                # Extract the region
                roi_color = color_image[y:y+h, x:x+w]
                roi_gray = gray[y:y+h, x:x+w]
                
                # Look for circular elements (toggle knob) within this region
                circles = cv2.HoughCircles(
                    roi_gray,
                    cv2.HOUGH_GRADIENT,
                    dp=1,
                    minDist=20,
                    param1=50,
                    param2=15,
                    minRadius=5,
                    maxRadius=min(h//2, 15)
                )
                
                if circles is not None:
                    circles = np.round(circles[0, :]).astype("int")
                    
                    # Use the first (most prominent) circle
                    if len(circles) > 0:
                        circle_x, circle_y, r = circles[0]
                        
                        # Analyze toggle state based on circle position and colors
                        state = self._analyze_toggle_state_improved(roi_color, roi_gray, circle_x, w)
                        
                        if state != 'unknown':
                            bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.75)
                            element = UIElement(
                                bbox=bbox,
                                element_type='toggle',
                                state=state,
                                confidence=0.75
                            )
                            elements.append(element)
        
        return elements
    
    def _detect_radio_buttons(self, gray: np.ndarray, color_image: np.ndarray) -> List[UIElement]:
        """Detect radio button elements and their states."""
        elements = []
        
        # Use HoughCircles to detect circular radio buttons
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=20,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=15
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # Extract the circular region
                roi_size = r * 2 + 4
                roi_x1 = max(0, x - roi_size // 2)
                roi_y1 = max(0, y - roi_size // 2)
                roi_x2 = min(gray.shape[1], x + roi_size // 2)
                roi_y2 = min(gray.shape[0], y + roi_size // 2)
                
                roi = gray[roi_y1:roi_y2, roi_x1:roi_x2]
                
                # Analyze if radio button is selected
                state = self._analyze_radio_state(roi)
                
                if state != 'unknown':
                    bbox = BBox(x1=roi_x1, y1=roi_y1, x2=roi_x2, y2=roi_y2, confidence=0.75)
                    element = UIElement(
                        bbox=bbox,
                        element_type='radio',
                        state=state,
                        confidence=0.75
                    )
                    elements.append(element)
        
        return elements
    
    def _detect_privacy_buttons(self, gray: np.ndarray, color_image: np.ndarray) -> List[UIElement]:
        """Detect privacy-related buttons and their states."""
        elements = []
        
        # Look for rectangular button-like shapes
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size (buttons are typically larger than checkboxes)
            if 50 <= w <= 200 and 20 <= h <= 50:
                # Check if this looks like a button
                roi = gray[y:y+h, x:x+w]
                
                # Simple button detection based on edge density
                edges = cv2.Canny(roi, 50, 150)
                edge_density = np.sum(edges > 0) / (w * h)
                
                if 0.1 <= edge_density <= 0.4:  # Buttons have moderate edge density
                    # Determine button state (enabled/disabled) based on color analysis
                    color_roi = color_image[y:y+h, x:x+w]
                    state = self._analyze_button_state(color_roi)
                    
                    bbox = BBox(x1=x, y1=y, x2=x+w, y2=y+h, confidence=0.6)
                    element = UIElement(
                        bbox=bbox,
                        element_type='button',
                        state=state,
                        confidence=0.6
                    )
                    elements.append(element)
        
        return elements
    
    def _analyze_checkbox_state(self, roi: np.ndarray) -> str:
        """Analyze checkbox ROI to determine if it's checked or unchecked."""
        if roi.size == 0:
            return 'unknown'
        
        # Simple approach: check for dark pixels in the center (indicating a checkmark)
        center_region = roi[roi.shape[0]//4:3*roi.shape[0]//4, roi.shape[1]//4:3*roi.shape[1]//4]
        
        if center_region.size == 0:
            return 'unknown'
        
        # Calculate the percentage of dark pixels
        dark_pixels = np.sum(center_region < 128)
        total_pixels = center_region.size
        dark_ratio = dark_pixels / total_pixels
        
        # If more than 30% of center pixels are dark, consider it checked
        return 'checked' if dark_ratio > 0.3 else 'unchecked'
    
    def _analyze_toggle_state(self, roi: np.ndarray, circle_x: int, circle_y: int, radius: int) -> str:
        """Analyze toggle switch state based on circle position."""
        if roi.size == 0:
            return 'unknown'
        
        roi_width = roi.shape[1]
        
        # If circle is on the right side, toggle is likely enabled
        if circle_x > roi_width * 0.6:
            return 'enabled'
        elif circle_x < roi_width * 0.4:
            return 'disabled'
        else:
            return 'unknown'
    
    def _analyze_radio_state(self, roi: np.ndarray) -> str:
        """Analyze radio button state."""
        if roi.size == 0:
            return 'unknown'
        
        # Check for a filled center (indicating selection)
        center_region = roi[roi.shape[0]//3:2*roi.shape[0]//3, roi.shape[1]//3:2*roi.shape[1]//3]
        
        if center_region.size == 0:
            return 'unknown'
        
        dark_pixels = np.sum(center_region < 128)
        total_pixels = center_region.size
        dark_ratio = dark_pixels / total_pixels
        
        return 'selected' if dark_ratio > 0.4 else 'unselected'
    
    def _analyze_button_state(self, color_roi: np.ndarray) -> str:
        """Analyze button state based on color characteristics."""
        if color_roi.size == 0:
            return 'unknown'
        
        # Calculate average brightness
        gray_roi = cv2.cvtColor(color_roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray_roi)
        
        # Brighter buttons are typically enabled, darker ones disabled
        return 'enabled' if avg_brightness > 100 else 'disabled'
    
    def _analyze_checkbox_state_improved(self, roi_gray: np.ndarray, roi_color: np.ndarray) -> str:
        """Improved checkbox state analysis using color and pattern detection."""
        if roi_gray.size == 0:
            return 'unknown'
        
        # Look for checkmark patterns (lines forming a check)
        edges = cv2.Canny(roi_gray, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=10, minLineLength=5, maxLineGap=3)
        
        if lines is not None and len(lines) >= 2:
            # If we detect line patterns, it's likely checked
            return 'checked'
        
        # Fallback to color analysis
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        
        # Look for green colors (common for checked states)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        if green_ratio > 0.1:  # If more than 10% is green
            return 'checked'
        
        # Check for dark content in center (traditional checkmark)
        center_region = roi_gray[roi_gray.shape[0]//4:3*roi_gray.shape[0]//4, 
                                roi_gray.shape[1]//4:3*roi_gray.shape[1]//4]
        
        if center_region.size > 0:
            dark_pixels = np.sum(center_region < 128)
            total_pixels = center_region.size
            dark_ratio = dark_pixels / total_pixels
            
            return 'checked' if dark_ratio > 0.3 else 'unchecked'
        
        return 'unchecked'
    
    def _analyze_toggle_state_improved(self, roi_color: np.ndarray, roi_gray: np.ndarray, 
                                     circle_x: int, roi_width: int) -> str:
        """Improved toggle state analysis using position and color."""
        if roi_color.size == 0:
            return 'unknown'
        
        # Analyze position of the toggle knob
        position_ratio = circle_x / roi_width
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        
        # Look for green colors (enabled state)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_ratio = np.sum(green_mask > 0) / green_mask.size
        
        # Look for blue colors (also common for enabled state)
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
        
        # Combine position and color analysis
        if position_ratio > 0.6 or green_ratio > 0.2 or blue_ratio > 0.2:
            return 'enabled'
        elif position_ratio < 0.4:
            return 'disabled'
        else:
            return 'unknown'
    
    def filter_privacy_elements(self, elements: List[UIElement]) -> List[UIElement]:
        """Filter elements to only include those that are enabled/active privacy settings."""
        privacy_states = {'checked', 'enabled', 'selected'}
        return [elem for elem in elements if elem.state in privacy_states]
    
    def draw_bounding_boxes(self, image: np.ndarray, elements: List[UIElement]) -> np.ndarray:
        """Draw bounding boxes around detected UI elements.
        
        Args:
            image: Input image
            elements: List of UI elements to draw
            
        Returns:
            Image with bounding boxes drawn
        """
        result = image.copy()
        
        # Color mapping for different element types
        colors = {
            'checkbox': (0, 255, 0),    # Green
            'toggle': (255, 0, 0),      # Blue
            'radio': (0, 0, 255),       # Red
            'button': (255, 255, 0)     # Cyan
        }
        
        for element in elements:
            color = colors.get(element.element_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(
                result,
                (element.bbox.x1, element.bbox.y1),
                (element.bbox.x2, element.bbox.y2),
                color,
                2
            )
            
            # Add label
            label = f"{element.element_type}: {element.state}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background
            cv2.rectangle(
                result,
                (element.bbox.x1, element.bbox.y1 - label_size[1] - 5),
                (element.bbox.x1 + label_size[0], element.bbox.y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (element.bbox.x1, element.bbox.y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return result