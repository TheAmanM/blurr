# Privacy Redactor RT

Real-time sensitive information detection and redaction system for video streams and uploaded videos.

## Features

- **Real-time Video Processing**: Live webcam privacy redaction with WebRTC integration
- **Video Upload Processing**: Upload and process entire video files for privacy redaction
- **Multi-format Support**: Handles MP4, AVI, MOV, MKV and other common video formats
- **Sensitive Data Detection**: Phone numbers, credit cards, emails, addresses, API keys
- **Flexible Redaction**: Configurable methods (gaussian blur, pixelate, solid color)
- **Audio Preservation**: Maintains audio tracks in processed videos (with FFmpeg)
- **Privacy-First**: All processing happens locally, no cloud dependencies
- **Web Interface**: User-friendly Gradio interface with live preview and upload tabs

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

### Web Interface (Recommended)

```bash
# Launch Gradio interface with video upload support
python demo_gradio_app.py
```

Access the interface at `http://localhost:7860` with two main features:
- **📹 Live Video**: Real-time webcam privacy redaction
- **📁 Video Upload**: Process entire video files for privacy redaction

### Alternative Interfaces

```bash
# Run the Streamlit web interface
streamlit run privacy_redactor_rt/app.py

# CLI usage for batch processing
privacy-redactor-rt redact-video input.mp4 output.mp4
privacy-redactor-rt batch-process input_folder output_folder
```

### Quick Video Processing Test

```bash
# Test video upload functionality
python test_gradio_video.py
```

## Configuration

Configuration is managed through YAML files. See `default.yaml` for all available options.

### Key Configuration Sections

- **Detection**: Text detection confidence and parameters
- **Classification**: Sensitive information categories and thresholds  
- **Redaction**: Blur/pixelate methods and intensity
- **Recording**: Video output codec and quality settings
- **Performance**: Processing optimization settings

## Video Upload Feature

The new video upload feature allows you to:

1. **Upload Videos**: Support for MP4, AVI, MOV, MKV formats
2. **Configure Detection**: Select specific categories (phone, email, etc.)
3. **Choose Redaction**: Gaussian blur, pixelate, or solid color methods
4. **Download Results**: Get privacy-redacted video with preserved audio

See [VIDEO_UPLOAD_GUIDE.md](VIDEO_UPLOAD_GUIDE.md) for detailed instructions.

## Development

Run tests:
```bash
pytest
```

Code formatting:
```bash
black privacy_redactor_rt tests
isort privacy_redactor_rt tests
ruff privacy_redactor_rt tests
```