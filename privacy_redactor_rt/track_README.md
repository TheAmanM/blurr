# Optical Flow Tracker Implementation

## Overview

The `OpticalFlowTracker` class implements real-time object tracking using sparse optical flow and IoU-based association. It's designed to maintain temporal consistency of detected text regions across video frames while minimizing computational overhead.

## Key Features

### 1. IoU-Based Association
- Associates new detections with existing tracks using Intersection over Union (IoU)
- Configurable IoU threshold (default: 0.5)
- Greedy matching algorithm for real-time performance

### 2. Sparse Optical Flow Propagation
- Uses Lucas-Kanade optical flow (`cv2.calcOpticalFlowPyrLK`) when available
- Propagates track positions between detection frames
- Handles tracking failures gracefully with fallback mechanisms

### 3. Track Lifecycle Management
- Automatic track creation for unmatched detections
- Track aging and cleanup based on configurable thresholds
- Hit rate tracking for quality assessment

### 4. Bounding Box Smoothing
- Exponential moving average for coordinate smoothing
- Configurable smoothing factor (default: 0.3)
- Reduces jitter in track positions

### 5. OCR Scheduling Integration
- Tracks when OCR was last performed on each track
- Intelligent scheduling based on movement and refresh intervals
- Minimizes redundant OCR operations

## Usage Example

```python
from privacy_redactor_rt.track import OpticalFlowTracker
from privacy_redactor_rt.config import TrackingConfig
from privacy_redactor_rt.types import Detection, BBox

# Initialize tracker
config = TrackingConfig(
    iou_threshold=0.5,
    max_age=30,
    min_hits=3,
    smoothing_factor=0.3
)
tracker = OpticalFlowTracker(config)

# Process detections
detections = [
    Detection(
        bbox=BBox(x1=100, y1=100, x2=200, y2=200, confidence=0.8),
        text="detected text",
        timestamp=1.0
    )
]

# Associate detections with tracks
tracker.associate_detections(detections)

# Propagate tracks using optical flow (between detection frames)
tracker.propagate_tracks(current_frame, previous_frame)

# Get active tracks for processing
active_tracks = tracker.get_active_tracks()

# Cleanup expired tracks
tracker.cleanup_tracks()
```

## Configuration Parameters

- `iou_threshold`: Minimum IoU for track association (0.1-0.9)
- `max_age`: Maximum frames a track can exist without hits (5-300)
- `min_hits`: Minimum hits required for track to be considered active (1-10)
- `smoothing_factor`: Coordinate smoothing strength (0.0-1.0)
- `max_flow_error`: Maximum optical flow displacement before failure (1.0-200.0)
- `flow_quality_level`: Feature detection quality threshold (0.001-0.1)
- `flow_min_distance`: Minimum distance between flow features (1-50)
- `flow_block_size`: Feature detection block size (3-15)

## Performance Considerations

### Computational Complexity
- O(N×M) for IoU calculation (N=tracks, M=detections)
- O(K) for optical flow (K=feature points per track)
- Greedy matching avoids expensive Hungarian algorithm

### Memory Usage
- Tracks store minimal state (bbox, metadata, flow points)
- Automatic cleanup prevents memory leaks
- Flow points are recomputed as needed

### Real-time Optimization
- Lazy optical flow initialization
- Fallback mechanisms for tracking failures
- Configurable quality vs. performance trade-offs

## Error Handling

### Optical Flow Failures
- Large displacement detection (> `max_flow_error`)
- Insufficient feature points (< 3 good points)
- Graceful degradation to previous positions

### Track Management
- Automatic cleanup of expired tracks
- Hit rate monitoring for quality assessment
- Bounds checking for frame boundaries

### OpenCV Compatibility
- Graceful fallback when OpenCV is unavailable
- Mock implementations for testing environments
- Runtime detection of cv2 availability

## Testing

The implementation includes comprehensive tests covering:

- Track creation and association logic
- IoU calculation accuracy
- Lifecycle management (aging, cleanup)
- OCR scheduling integration
- Error handling and edge cases
- Performance characteristics

Run tests with:
```bash
python -m pytest tests/test_track.py -v
```

## Integration Points

### With Detection Pipeline
- Receives `Detection` objects from text detection
- Provides track IDs for OCR scheduling
- Maintains temporal consistency

### With OCR Worker
- Tracks last OCR frame per track
- Provides scheduling recommendations
- Minimizes redundant processing

### With Classification Engine
- Associates matches with tracks
- Maintains classification history
- Supports temporal consensus

## Future Enhancements

1. **Kalman Filter Integration**: More sophisticated motion prediction
2. **Multi-Scale Tracking**: Handle scale changes in detected regions  
3. **Deep Learning Features**: Use learned features instead of corner detection
4. **GPU Acceleration**: CUDA-based optical flow for higher throughput
5. **Adaptive Parameters**: Dynamic adjustment based on scene complexity