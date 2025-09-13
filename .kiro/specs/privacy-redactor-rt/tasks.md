# Implementation Plan

- [x] 1. Set up project structure and core configuration

  - Create directory structure for privacy_redactor_rt package with all required modules
  - Implement pyproject.toml with pinned dependencies (Python 3.11, Streamlit, OpenCV, PaddleOCR, etc.)
  - Create basic **init**.py files and package structure
  - Set up development tooling configuration (ruff, black, isort, pytest)
  - _Requirements: 9.1, 9.2_

- [x] 2. Implement core data models and configuration system

  - Create types.py with BBox, Detection, Match, Track dataclasses
  - Implement config.py with Pydantic models for all configuration sections
  - Create default.yaml configuration file with all required parameters
  - Add configuration loading and validation logic
  - Write unit tests for data model serialization and config validation
  - _Requirements: 10.1, 10.2, 9.4_

- [ ] 3. Create pattern matching and validation engines

  - Implement patterns.py with compiled regex patterns for all sensitive data categories
  - Create validators.py with Luhn check, phone number validation, and entropy calculation
  - Implement address_rules.py with street suffix and postal code detection
  - Add text normalization and masking utilities
  - Write comprehensive unit tests for all pattern matching and validation logic
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.2, 9.4_

- [ ] 4. Implement classification engine with privacy protection

  - Create classify.py with ClassificationEngine class
  - Integrate phonenumbers library for phone validation with US/CA defaults
  - Implement credit card detection with Luhn validation and brand identification
  - Add email regex matching with RFC compliance
  - Implement address scoring with rule-based detection and optional spaCy NER
  - Add API key detection for major vendors plus entropy-based fallback
  - Implement privacy-preserving text masking for all categories
  - Write unit tests for each classification category with positive/negative test cases
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 8.2, 9.4_

- [ ] 5. Create text detection wrapper with lazy initialization

  - Implement text_detect.py with TextDetector class using PaddleOCR PP-OCRv4
  - Add lazy model initialization to avoid startup delays
  - Implement bounding box extraction and confidence filtering
  - Add quadrilateral to axis-aligned bounding box conversion
  - Create unit tests with mock detection results
  - _Requirements: 4.1, 9.4_

- [ ] 6. Implement asynchronous OCR worker with queue management

  - Create ocr.py with OCRWorker class for threaded text recognition
  - Implement bounded queue with configurable size and backpressure handling
  - Add text normalization (NFKC, whitespace handling, case preservation)
  - Implement result caching per track ID with thread-safe operations
  - Create unit tests for queue management and text processing
  - _Requirements: 4.3, 1.3, 9.4_

- [ ] 7. Create optical flow tracker with IoU association

  - Implement track.py with OpticalFlowTracker class
  - Add sparse optical flow calculation using cv2.calcOpticalFlowPyrLK
  - Implement IoU-based track association and lifecycle management
  - Add bounding box coordinate smoothing with moving averages
  - Create track aging and cleanup logic
  - Write unit tests with mock frame sequences and known transformations
  - _Requirements: 4.2, 4.5, 9.4_

- [x] 8. Implement redaction engine with multiple methods

  - Create redact.py with RedactionEngine class
  - Implement gaussian blur redaction with configurable kernel size
  - Add pixelation redaction with configurable block size
  - Implement solid color redaction with configurable color
  - Add per-category redaction method selection and bounding box inflation
  - Create unit tests for visual validation of redaction methods
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 9.4_

- [ ] 9. Create real-time pipeline orchestrator

  - Implement pipeline.py with RealtimePipeline class
  - Add frame-by-frame processing coordination with detection scheduling
  - Implement temporal consensus logic requiring multiple consecutive matches
  - Add track management and lifecycle coordination
  - Integrate all processing components (detection, OCR, classification, tracking, redaction)
  - Create integration tests for end-to-end pipeline processing
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 1.4, 9.4_

- [ ] 10. Implement video input normalization and frame processing

  - Create video_source.py with frame normalization to 1280×720 with letterboxing
  - Add FPS throttling to maintain consistent 30 FPS output
  - Implement support for multiple input formats (webcam, RTSP, file)
  - Add frame conversion utilities (BGR/RGB, aspect ratio preservation)
  - Write unit tests for frame normalization and FPS control
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 1.1, 1.2, 9.4_

- [ ] 11. Create WebRTC integration with performance monitoring

  - Implement webrtc_utils.py with VideoTransformer class for streamlit-webrtc
  - Add frame reception, processing, and output rendering
  - Implement performance statistics collection (FPS, latency with EMA)
  - Add backpressure management and frame dropping for real-time performance
  - Integrate RealtimePipeline for frame processing
  - Create integration tests for WebRTC frame handling
  - _Requirements: 1.3, 1.4, 6.5, 9.4_

- [ ] 12. Implement optional MP4 recording with audio preservation

  - Create recorder.py with FFmpeg pipe integration for MP4 encoding
  - Add libx264 encoding with configurable CRF and preset options
  - Implement audio track copying from WebRTC source when available
  - Add silent audio track generation when source lacks audio
  - Ensure constant frame rate (CFR) at 30 FPS and 1280×720 output
  - Write unit tests for recording functionality and audio handling
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.4_

- [ ] 13. Create audit logging system with privacy protection

  - Implement logging_utils.py with JSONL detection logging
  - Add timestamp, category, bounding box, and confidence logging
  - Implement privacy-preserving text masking (first/last 3 characters)
  - Add --no-log-text flag support for complete text preview disabling
  - Ensure all processing remains local-only with no network calls
  - Write unit tests for logging functionality and privacy protection
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 9.4_

- [ ] 14. Build Streamlit web interface with real-time controls

  - Create app.py with Streamlit interface and sidebar controls
  - Add input source selection (webcam, RTSP URL, file upload)
  - Implement category multi-select with all supported sensitive data types
  - Add redaction method configuration with per-category overrides
  - Implement detector/OCR stride and confidence threshold controls
  - Add recording toggle with MP4 output configuration
  - Display real-time FPS and latency statistics
  - Show live detection counters per category and masked event feed
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 9.4_

- [ ] 15. Integrate all components in main application

  - Wire together all pipeline components in the Streamlit app
  - Implement configuration loading from YAML and UI overrides
  - Add proper error handling and graceful degradation
  - Ensure thread-safe operations between UI and processing threads
  - Add startup initialization and cleanup procedures
  - Create integration tests for complete application workflow
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 9.4_

- [x] 16. Create CLI interface for offline processing

  - Implement cli.py with Typer-based command-line interface
  - Add redact-video command for offline file processing
  - Support input/output file specification and configuration loading
  - Implement non-real-time processing using the same pipeline components
  - Add progress reporting and batch processing capabilities
  - Write unit tests for CLI functionality and offline processing
  - _Requirements: 9.4_

- [ ] 17. Add comprehensive test suite and validation

  - Create test_patterns.py with positive/negative cases for all regex patterns
  - Implement test_luhn.py with masked credit card validation fixtures
  - Add test_address_rules.py with US/CA address detection test cases
  - Create test_cli_smoke.py with mock video stream processing
  - Implement visual validation tests for redaction quality
  - Add performance benchmark tests with standardized test videos
  - Create integration tests for WebRTC streaming and recording
  - _Requirements: 9.4_

- [ ] 18. Create deployment configuration and documentation

  - Implement Dockerfile with multi-stage build for production deployment
  - Create Makefile with setup, run, lint, format, test, and docker targets
  - Add comprehensive README.md with installation and usage instructions
  - Create .gitignore with appropriate exclusions for Python and media files
  - Add pre-commit configuration for code quality enforcement
  - Document performance tuning guidelines and hardware requirements
  - _Requirements: 9.2, 9.4_

- [ ] 19. Implement performance optimization and monitoring

  - Add automatic quality scaling based on real-time performance metrics
  - Implement memory pooling for frequently allocated arrays
  - Add vectorized operations for image processing where possible
  - Create performance profiling utilities for latency analysis
  - Implement resource usage monitoring with automatic alerts
  - Add configuration recommendations based on hardware detection
  - Write performance validation tests with various hardware configurations
  - _Requirements: 1.3, 1.4, 10.3, 10.4, 10.5_

- [ ] 20. Final integration testing and validation
  - Perform end-to-end testing with all supported input sources
  - Validate 720p @ 30 FPS performance on target hardware
  - Test all sensitive data detection categories with real-world examples
  - Verify temporal consensus and flicker prevention
  - Validate recording functionality with audio preservation
  - Test privacy protection in logging and audit trails
  - Perform stress testing with extended runtime scenarios
  - Validate cross-platform compatibility and deployment
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4, 3.5, 4.4, 7.1, 7.2, 8.1, 8.2_
