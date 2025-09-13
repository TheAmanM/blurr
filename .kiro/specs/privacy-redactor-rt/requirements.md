# Requirements Document

## Introduction

Privacy-Redactor RT is a real-time sensitive information detection and redaction system that processes live video streams to automatically blur or obscure sensitive data such as phone numbers, credit card numbers, emails, addresses, and API keys. The system targets 720p resolution at 30 FPS performance on CPU-only hardware while maintaining low latency and high accuracy. The application provides a Streamlit-based web interface with WebRTC integration for live video processing and optional recording capabilities.

## Requirements

### Requirement 1

**User Story:** As a privacy-conscious user, I want to process live video streams in real-time to automatically detect and redact sensitive information, so that I can protect personal data during video calls or recordings.

#### Acceptance Criteria

1. WHEN a live video stream is provided THEN the system SHALL process frames at 1280×720 resolution with letterboxing normalization
2. WHEN processing video frames THEN the system SHALL maintain a consistent 30 FPS output rate
3. WHEN the median processing latency exceeds 120ms THEN the system SHALL implement backpressure mechanisms to maintain real-time performance
4. IF the processing falls below 24 FPS THEN the system SHALL drop frames to recover performance while maintaining at least 24 FPS minimum

### Requirement 2

**User Story:** As a user, I want multiple video input sources available, so that I can process different types of video content based on my needs.

#### Acceptance Criteria

1. WHEN selecting input sources THEN the system SHALL support webcam capture via streamlit-webrtc as the default option
2. WHEN providing alternative sources THEN the system SHALL accept RTSP/HTTP URLs as live stream inputs
3. WHEN using file inputs THEN the system SHALL process local video files as simulated live sources at source FPS throttled to 30 FPS
4. WHEN normalizing input THEN the system SHALL convert all input formats to 1280×720 with letterboxing preservation

### Requirement 3

**User Story:** As a user, I want the system to detect multiple categories of sensitive information, so that I can comprehensively protect various types of personal data.

#### Acceptance Criteria

1. WHEN processing text THEN the system SHALL detect US/Canadian phone numbers using phonenumbers library with E.164 fallback
2. WHEN scanning for financial data THEN the system SHALL identify 13-19 digit credit card numbers with Luhn validation and brand detection
3. WHEN analyzing text content THEN the system SHALL recognize email addresses using RFC-compliant regex patterns
4. WHEN processing address information THEN the system SHALL detect mailing addresses using rule-based scoring with street suffixes, ZIP/postal codes, and optional spaCy NER
5. WHEN scanning for credentials THEN the system SHALL identify API keys from major vendors (AWS, Google, GitHub, Stripe, Slack, Twilio, OpenAI, Hugging Face, Supabase) plus high-entropy Base64 patterns

### Requirement 4

**User Story:** As a performance-conscious user, I want optimized real-time processing, so that the system can handle live video without significant delays or resource consumption.

#### Acceptance Criteria

1. WHEN processing frames THEN the system SHALL run text detection every N=3 frames (configurable)
2. WHEN between detection frames THEN the system SHALL propagate bounding boxes using sparse optical flow (cv2.calcOpticalFlowPyrLK) and IoU tracking
3. WHEN determining OCR necessity THEN the system SHALL only OCR new boxes or those with IoU < 0.5 compared to last OCR state, or every K=10 frames for refresh
4. WHEN applying detections THEN the system SHALL require ≥2 consecutive frame matches for temporal consensus to prevent flicker
5. WHEN managing overlapping detections THEN the system SHALL merge nearby boxes per category

### Requirement 5

**User Story:** As a user, I want configurable redaction methods, so that I can choose appropriate obscuration techniques for different types of sensitive data.

#### Acceptance Criteria

1. WHEN redacting content THEN the system SHALL support gaussian blur as the default method
2. WHEN alternative redaction is needed THEN the system SHALL provide pixelation and solid color options
3. WHEN configuring per-category THEN the system SHALL allow different redaction methods for each sensitive data type
4. WHEN applying redaction THEN the system SHALL inflate bounding boxes by configurable pixels to ensure complete coverage
5. WHEN rendering output THEN the system SHALL maintain 720p @ 30 FPS with preserved audio during recording

### Requirement 6

**User Story:** As a user, I want a web-based interface for controlling the redaction system, so that I can easily configure settings and monitor performance without technical complexity.

#### Acceptance Criteria

1. WHEN accessing the application THEN the system SHALL provide a Streamlit web interface with sidebar controls
2. WHEN configuring sources THEN the interface SHALL offer webcam, RTSP URL, and file input options
3. WHEN selecting detection categories THEN the interface SHALL provide multi-select controls for all supported sensitive data types
4. WHEN adjusting performance THEN the interface SHALL allow configuration of detector/OCR strides and confidence thresholds
5. WHEN monitoring performance THEN the interface SHALL display real-time FPS and latency statistics with EMA smoothing
6. WHEN viewing results THEN the interface SHALL show live counters per category and a masked event feed

### Requirement 7

**User Story:** As a user, I want optional recording capabilities, so that I can save redacted video streams for later use while preserving audio quality.

#### Acceptance Criteria

1. WHEN recording is enabled THEN the system SHALL save redacted streams as MP4 files at 1280×720 @ 30 FPS
2. WHEN source contains audio THEN the system SHALL preserve and copy audio tracks to the output
3. WHEN configuring recording THEN the system SHALL use libx264 encoding with configurable CRF and preset options
4. WHEN managing output THEN the system SHALL ensure constant frame rate (CFR) at exactly 30 FPS
5. IF source lacks audio THEN the system SHALL write a silent audio track to maintain compatibility

### Requirement 8

**User Story:** As a privacy-conscious user, I want comprehensive audit logging with privacy protection, so that I can track detections while ensuring sensitive data remains secure.

#### Acceptance Criteria

1. WHEN logging detections THEN the system SHALL write optional JSONL files with timestamps, categories, and bounding box coordinates
2. WHEN preserving privacy THEN the system SHALL mask all text previews showing only first/last 3 characters
3. WHEN maximum privacy is required THEN the system SHALL support --no-log-text flag to disable all text previews
4. WHEN processing data THEN the system SHALL ensure all operations remain local-only with no network calls
5. WHEN auditing THEN the system SHALL include redacted previews and detection confidence scores in logs

### Requirement 9

**User Story:** As a developer, I want a well-structured codebase with proper testing and tooling, so that the system is maintainable and reliable.

#### Acceptance Criteria

1. WHEN organizing code THEN the system SHALL use a modular architecture with separate components for detection, OCR, classification, tracking, and redaction
2. WHEN managing dependencies THEN the system SHALL use uv with pyproject.toml for package management with pinned versions
3. WHEN ensuring quality THEN the system SHALL include ruff, black, isort for code formatting and pytest for testing
4. WHEN testing functionality THEN the system SHALL provide comprehensive test coverage for pattern matching, Luhn validation, address rules, and CLI operations
5. WHEN deploying THEN the system SHALL include Docker support with multi-stage builds and proper dependency management

### Requirement 10

**User Story:** As a user, I want flexible configuration options, so that I can tune the system performance and behavior for my specific hardware and use case.

#### Acceptance Criteria

1. WHEN configuring detection THEN the system SHALL support YAML-based configuration for all detection parameters
2. WHEN tuning performance THEN the system SHALL allow adjustment of detector stride, OCR refresh intervals, and queue limits
3. WHEN setting thresholds THEN the system SHALL provide configurable confidence levels for text detection and OCR
4. WHEN managing tracking THEN the system SHALL support configurable IoU thresholds, track aging, and smoothing windows
5. WHEN customizing redaction THEN the system SHALL allow per-category redaction method overrides and parameter tuning