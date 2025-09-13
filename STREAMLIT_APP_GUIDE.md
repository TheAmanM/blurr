# Privacy Redactor RT - Streamlit Web Interface

This document describes the Streamlit web interface for Privacy Redactor RT, which provides real-time sensitive information detection and redaction capabilities through a user-friendly web application.

## Features

### 🎛️ Controls Panel (Sidebar)

#### 📹 Input Source Selection
- **Webcam**: Use your computer's camera for live video processing
- **RTSP Stream**: Connect to network cameras or streaming sources
- **Video File**: Upload and process video files

#### 🔍 Detection Categories
Multi-select control for enabling/disabling detection of:
- **Phone Numbers**: US/Canadian phone numbers with validation
- **Credit Cards**: Credit card numbers with Luhn validation and brand detection
- **Email Addresses**: RFC-compliant email address detection
- **Addresses**: Mailing addresses using rule-based scoring
- **API Keys**: API keys from major vendors plus entropy-based detection

#### 🎨 Redaction Methods
- **Default Method**: Choose from gaussian blur, pixelation, or solid color
- **Per-Category Overrides**: Set different redaction methods for each detection category
  - Gaussian blur with configurable kernel size and sigma
  - Pixelation with configurable block size
  - Solid color fill with configurable RGB color

#### ⚡ Performance Controls
- **Detector Stride**: Run text detection every N frames (1-10)
- **OCR Refresh Stride**: Force OCR refresh every N frames (5-30)
- **Text Detection Confidence**: Minimum confidence threshold (0.1-1.0)
- **OCR Confidence**: Minimum OCR confidence threshold (0.1-1.0)

#### 🎬 Recording Options
- **Enable Recording**: Toggle MP4 recording of redacted stream
- **Output Directory**: Specify where to save recordings
- **Video Quality (CRF)**: Control compression (0-51, lower = better quality)
- **Encoding Preset**: Speed vs compression tradeoff

### 📺 Live Video Stream

The main video area displays:
- Real-time video feed with applied redactions
- WebRTC-based streaming for low latency
- Automatic frame normalization to 1280×720 resolution
- Letterboxing to preserve aspect ratios

#### Control Buttons
- **🔄 Reset Statistics**: Clear performance metrics
- **⚙️ Reload Config**: Reload configuration from file
- **🟢/🔴 Stream Status**: Visual indicator of stream state

### 📊 Performance Monitor

Real-time performance metrics including:

#### Core Metrics
- **FPS**: Current frames per second vs target
- **Latency**: Current processing latency in milliseconds
- **Processing Time**: Time spent in detection/classification pipeline
- **Active Tracks**: Number of currently tracked objects

#### Frame Statistics
- **Processed**: Total frames processed successfully
- **Total**: Total frames received
- **Dropped**: Frames dropped for performance (with percentage)
- **OCR Queue**: Current OCR queue utilization
- **OCR Processed**: Total OCR operations completed

#### Health Indicator
- **🟢 Healthy**: Performance within acceptable thresholds
- **🟡 Degraded**: Performance below optimal levels

### 🔍 Detection Counters

Live counters showing total detections per category:
- Phone numbers found
- Credit cards detected
- Email addresses identified
- Addresses located
- API keys discovered

### 📝 Detection Events Feed

Scrollable feed of recent detection events showing:
- **Timestamp**: When the detection occurred
- **Category**: Type of sensitive data detected
- **Masked Text**: Privacy-preserving preview (first/last characters only)
- **Confidence**: Detection confidence score

Features:
- Shows last 10 events in reverse chronological order
- Auto-refresh every 5 seconds (toggleable)
- Privacy-safe text masking

## Usage Instructions

### 1. Installation

Ensure all dependencies are installed:
```bash
pip install -e .
```

### 2. Running the Application

#### Option A: Using CLI
```bash
python -m privacy_redactor_rt.cli run-app
```

#### Option B: Direct Streamlit
```bash
streamlit run -m privacy_redactor_rt.app
```

#### Option C: Custom Port/Host
```bash
python -m privacy_redactor_rt.cli run-app --port 8502 --host 0.0.0.0
```

### 3. Configuration

The app loads configuration from `default.yaml` by default. UI controls override these settings in real-time.

### 4. Basic Workflow

1. **Select Input Source**: Choose webcam, RTSP stream, or upload a video file
2. **Configure Detection**: Select which types of sensitive data to detect
3. **Set Redaction Methods**: Choose how to obscure detected information
4. **Tune Performance**: Adjust detection frequency and confidence thresholds
5. **Start Processing**: The video stream will show real-time redactions
6. **Monitor Performance**: Watch metrics to ensure smooth operation
7. **Optional Recording**: Enable recording to save redacted video

### 5. Performance Optimization

For best performance:
- Reduce detector stride (run detection less frequently)
- Increase confidence thresholds to reduce false positives
- Disable unused detection categories
- Use faster redaction methods (solid color > pixelation > gaussian blur)
- Monitor the performance health indicator

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface with sidebar controls
- **Video Processing**: WebRTC integration for low-latency streaming
- **Backend**: Real-time processing pipeline with temporal tracking
- **Configuration**: Pydantic models with YAML configuration files

### Performance Features
- **Backpressure Management**: Automatic frame dropping when processing falls behind
- **Temporal Consensus**: Requires multiple consecutive detections to prevent flicker
- **Optical Flow Tracking**: Efficient bounding box propagation between detection frames
- **Asynchronous OCR**: Non-blocking text recognition with bounded queues

### Privacy Protection
- **Local Processing**: All operations run locally, no network calls
- **Text Masking**: Sensitive text is masked in logs and UI
- **Configurable Logging**: Option to disable text previews entirely
- **Memory Management**: Automatic cleanup of cached sensitive data

## Troubleshooting

### Common Issues

1. **Low FPS/High Latency**
   - Increase detector stride
   - Reduce OCR refresh frequency
   - Disable unused detection categories
   - Lower video resolution if possible

2. **Missing Detections**
   - Lower confidence thresholds
   - Reduce detector stride (more frequent detection)
   - Check that relevant categories are enabled

3. **False Positives**
   - Increase confidence thresholds
   - Enable temporal consensus (require multiple frames)
   - Review pattern matching configuration

4. **WebRTC Connection Issues**
   - Check browser permissions for camera access
   - Ensure HTTPS for production deployments
   - Verify network connectivity for RTSP streams

### Performance Monitoring

Watch these indicators for optimal performance:
- FPS should stay close to target (30 FPS)
- Latency should remain under 120ms
- Drop rate should be under 10%
- OCR queue should not stay full consistently

## Configuration Reference

Key configuration sections that affect the UI:

```yaml
# Real-time processing
realtime:
  detector_stride: 3              # Detection frequency
  ocr_refresh_stride: 10         # OCR refresh frequency
  backpressure_threshold_ms: 120 # Latency threshold

# Detection settings
detection:
  min_text_confidence: 0.6       # Text detection threshold
  bbox_inflate_px: 6             # Bounding box padding

# Classification categories
classification:
  categories:                    # Enabled detection types
    - phone
    - credit_card
    - email
    - address
    - api_key
  require_temporal_consensus: 2  # Frames for confirmation

# Redaction methods
redaction:
  default_method: "gaussian"     # Default redaction type
  category_methods: {}           # Per-category overrides
```

For complete configuration options, see `default.yaml`.