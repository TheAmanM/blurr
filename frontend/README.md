# Privacy Redactor RT

Real-time sensitive information detection and redaction system for video streams.

## Features

- Real-time video processing at 720p @ 30 FPS
- Multi-category sensitive data detection (phone numbers, credit cards, emails, addresses, API keys)
- Configurable redaction methods (gaussian blur, pixelation, solid color)
- WebRTC integration for live streaming
- Optional MP4 recording with audio preservation
- Privacy-preserving audit logging
- Streamlit web interface
- CLI for offline processing

## Installation

### Requirements

- Python 3.11+
- CPU-only operation (no GPU required)

### Setup

```bash
# Clone repository
git clone <repository-url>
cd privacy-redactor-rt

# Set up development environment
make setup

# Or install in production mode
make install
```

## Usage

### Web Interface

```bash
# Run Streamlit app
make run-app
# or
streamlit run privacy_redactor_rt/app.py
```

### CLI Interface

```bash
# Process video file
privacy-redactor-rt redact-video input.mp4 output.mp4

# With custom configuration
privacy-redactor-rt redact-video input.mp4 output.mp4 --config-file custom.yaml
```

## Configuration

Configuration is managed through YAML files. See `default.yaml` for all available options.

Key configuration sections:
- `io`: Input/output settings (resolution, FPS)
- `realtime`: Performance tuning (detection stride, queue limits)
- `detection`: Text detection parameters
- `classification`: Sensitive data categories and thresholds
- `redaction`: Redaction methods and parameters

## Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Run tests
make test
```

### Docker

```bash
# Build image
make docker-build

# Run container
make docker-run
```

## Architecture

The system uses a multi-threaded pipeline architecture:

1. **Video Input**: Webcam, RTSP, or file input with frame normalization
2. **Text Detection**: PaddleOCR-based text detection with configurable stride
3. **Optical Flow Tracking**: Sparse optical flow for bounding box propagation
4. **OCR Processing**: Asynchronous text recognition with queue management
5. **Classification**: Multi-category pattern matching and validation
6. **Temporal Consensus**: Flicker prevention through consecutive frame matching
7. **Redaction**: Configurable blur/pixelate/solid methods
8. **Output**: WebRTC streaming and optional MP4 recording

## Performance

Target performance: 720p @ 30 FPS on CPU-only hardware

Optimization features:
- Intelligent frame skipping (detection every N frames)
- Optical flow propagation between detections
- Asynchronous OCR processing
- Temporal consensus for stability
- Automatic quality scaling under load

## Privacy

- All processing is local-only (no network calls)
- Audit logs use privacy-preserving text masking
- Optional `--no-log-text` flag for maximum privacy
- Configurable text preview masking (first/last 3 characters)

## License

MIT License - see LICENSE file for details.