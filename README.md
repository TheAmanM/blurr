# Privacy Redactor RT

Real-time sensitive information detection and redaction system for video streams.

## Features

- Real-time detection and redaction of sensitive information
- Support for multiple data types: phone numbers, credit cards, emails, addresses, API keys
- WebRTC integration for live video processing
- Configurable redaction methods (blur, pixelate, solid color)
- Optional recording with audio preservation
- Privacy-preserving audit logging

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run the Streamlit web interface
streamlit run privacy_redactor_rt/app.py

# CLI usage
privacy-redactor-rt --help
```

## Configuration

Configuration is managed through YAML files. See `default.yaml` for all available options.

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