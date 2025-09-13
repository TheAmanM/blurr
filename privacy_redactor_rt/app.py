"""Streamlit web interface with real-time controls."""

import streamlit as st
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import tempfile
import os

# Handle optional dependencies
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False
    st.error("streamlit-webrtc not available. Please install with: pip install streamlit-webrtc")

from privacy_redactor_rt.config import Config, load_config
from privacy_redactor_rt.pipeline import RealtimePipeline
from privacy_redactor_rt.webrtc_utils import VideoTransformer

logger = logging.getLogger(__name__)


class StreamlitApp:
    """Main Streamlit application with real-time controls."""
    
    def __init__(self):
        """Initialize the Streamlit application with proper startup procedures."""
        self.config: Optional[Config] = None
        self.pipeline: Optional[RealtimePipeline] = None
        self.video_transformer: Optional[VideoTransformer] = None
        self.detection_counters: Dict[str, int] = {}
        self.event_feed: List[Dict[str, Any]] = []
        self.max_events = 50
        self._initialized = False
        self._cleanup_registered = False
        
        # Initialize session state
        if 'config_loaded' not in st.session_state:
            st.session_state.config_loaded = False
        if 'pipeline_active' not in st.session_state:
            st.session_state.pipeline_active = False
        if 'recording_active' not in st.session_state:
            st.session_state.recording_active = False
        if 'app_instance' not in st.session_state:
            st.session_state.app_instance = self
        
        # Register cleanup on session end
        self._register_cleanup()
    
    def run(self):
        """Run the main Streamlit application."""
        st.set_page_config(
            page_title="Privacy Redactor RT",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🔒 Privacy Redactor RT")
        st.markdown("Real-time sensitive information detection and redaction system")
        
        # Load configuration
        self._load_configuration()
        
        if not self.config:
            st.error("Failed to load configuration. Please check your setup.")
            return
        
        # Create sidebar controls
        self._create_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._create_video_interface()
        
        with col2:
            self._create_monitoring_panel()
    
    def _load_configuration(self):
        """Load and initialize configuration with proper error handling."""
        try:
            # Load base configuration
            config_path = Path("default.yaml")
            
            if config_path.exists():
                self.config = load_config(config_path)
                logger.info(f"Configuration loaded from {config_path}")
            else:
                # Use default configuration if file doesn't exist
                self.config = Config()
                logger.warning(f"Configuration file {config_path} not found, using defaults")
            
            # Apply UI overrides if available
            ui_overrides = self._get_ui_config_overrides()
            if ui_overrides:
                try:
                    self.config = self.config.merge_overrides(ui_overrides)
                    logger.debug("UI configuration overrides applied")
                except Exception as e:
                    logger.warning(f"Failed to apply UI overrides: {e}")
            
            st.session_state.config_loaded = True
            self._initialized = True
            
        except FileNotFoundError as e:
            error_msg = f"Configuration file not found: {e}"
            st.error(error_msg)
            logger.error(error_msg)
            # Try to use default config
            try:
                self.config = Config()
                st.warning("Using default configuration")
                st.session_state.config_loaded = True
            except Exception as fallback_error:
                logger.error(f"Failed to create default config: {fallback_error}")
                
        except Exception as e:
            error_msg = f"Failed to load configuration: {e}"
            st.error(error_msg)
            logger.error(f"Configuration loading error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _create_sidebar(self):
        """Create sidebar with all configuration controls."""
        st.sidebar.header("🎛️ Controls")
        
        # Input source selection
        st.sidebar.subheader("📹 Input Source")
        input_source = st.sidebar.selectbox(
            "Select input source:",
            ["Webcam", "RTSP Stream", "Video File"],
            key="input_source"
        )
        
        if input_source == "RTSP Stream":
            rtsp_url = st.sidebar.text_input(
                "RTSP URL:",
                placeholder="rtsp://example.com/stream",
                key="rtsp_url"
            )
        elif input_source == "Video File":
            uploaded_file = st.sidebar.file_uploader(
                "Upload video file:",
                type=['mp4', 'avi', 'mov', 'mkv'],
                key="uploaded_file"
            )
        
        st.sidebar.divider()
        
        # Detection categories
        st.sidebar.subheader("🔍 Detection Categories")
        available_categories = ["phone", "credit_card", "email", "address", "api_key"]
        selected_categories = st.sidebar.multiselect(
            "Select categories to detect:",
            available_categories,
            default=self.config.classification.categories if self.config else available_categories,
            key="selected_categories"
        )
        
        st.sidebar.divider()
        
        # Redaction configuration
        st.sidebar.subheader("🎨 Redaction Methods")
        default_method = st.sidebar.selectbox(
            "Default redaction method:",
            ["gaussian", "pixelate", "solid"],
            index=0 if not self.config else ["gaussian", "pixelate", "solid"].index(self.config.redaction.default_method),
            key="default_redaction_method"
        )
        
        # Per-category redaction overrides
        st.sidebar.write("**Per-category overrides:**")
        category_methods = {}
        for category in selected_categories:
            method = st.sidebar.selectbox(
                f"{category.replace('_', ' ').title()}:",
                ["default", "gaussian", "pixelate", "solid"],
                key=f"redaction_{category}"
            )
            if method != "default":
                category_methods[category] = method
        
        st.sidebar.divider()
        
        # Performance controls
        st.sidebar.subheader("⚡ Performance")
        detector_stride = st.sidebar.slider(
            "Detector stride (frames):",
            min_value=1, max_value=10,
            value=self.config.realtime.detector_stride if self.config else 3,
            help="Run text detection every N frames",
            key="detector_stride"
        )
        
        ocr_refresh_stride = st.sidebar.slider(
            "OCR refresh stride (frames):",
            min_value=5, max_value=30,
            value=self.config.realtime.ocr_refresh_stride if self.config else 10,
            help="Force OCR refresh every N frames",
            key="ocr_refresh_stride"
        )
        
        text_confidence = st.sidebar.slider(
            "Text detection confidence:",
            min_value=0.1, max_value=1.0, step=0.05,
            value=self.config.detection.min_text_confidence if self.config else 0.6,
            help="Minimum confidence for text detection",
            key="text_confidence"
        )
        
        ocr_confidence = st.sidebar.slider(
            "OCR confidence threshold:",
            min_value=0.1, max_value=1.0, step=0.05,
            value=self.config.ocr.min_ocr_confidence if self.config else 0.7,
            help="Minimum confidence for OCR results",
            key="ocr_confidence"
        )
        
        st.sidebar.divider()
        
        # Recording controls
        st.sidebar.subheader("🎬 Recording")
        recording_enabled = st.sidebar.checkbox(
            "Enable recording",
            value=False,
            key="recording_enabled"
        )
        
        if recording_enabled:
            output_dir = st.sidebar.text_input(
                "Output directory:",
                value="recordings",
                key="output_dir"
            )
            
            crf = st.sidebar.slider(
                "Video quality (CRF):",
                min_value=0, max_value=51,
                value=self.config.recording.crf if self.config else 23,
                help="Lower values = better quality, larger files",
                key="recording_crf"
            )
            
            preset = st.sidebar.selectbox(
                "Encoding preset:",
                ["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                index=5,  # medium
                help="Speed vs compression tradeoff",
                key="recording_preset"
            )
    
    def _get_ui_config_overrides(self) -> Dict[str, Any]:
        """Get configuration overrides from UI controls."""
        if not hasattr(st.session_state, 'selected_categories'):
            return {}
        
        overrides = {
            'classification': {
                'categories': st.session_state.get('selected_categories', [])
            },
            'redaction': {
                'default_method': st.session_state.get('default_redaction_method', 'gaussian'),
                'category_methods': {}
            },
            'realtime': {
                'detector_stride': st.session_state.get('detector_stride', 3),
                'ocr_refresh_stride': st.session_state.get('ocr_refresh_stride', 10)
            },
            'detection': {
                'min_text_confidence': st.session_state.get('text_confidence', 0.6)
            },
            'ocr': {
                'min_ocr_confidence': st.session_state.get('ocr_confidence', 0.7)
            },
            'recording': {
                'enabled': st.session_state.get('recording_enabled', False),
                'output_dir': st.session_state.get('output_dir', 'recordings'),
                'crf': st.session_state.get('recording_crf', 23),
                'preset': st.session_state.get('recording_preset', 'medium')
            }
        }
        
        # Add per-category redaction methods
        for category in st.session_state.get('selected_categories', []):
            method_key = f'redaction_{category}'
            if method_key in st.session_state and st.session_state[method_key] != 'default':
                overrides['redaction']['category_methods'][category] = st.session_state[method_key]
        
        return overrides
    
    def _create_video_interface(self):
        """Create the main video interface with proper error handling."""
        st.subheader("📺 Live Video Stream")
        
        if not HAS_WEBRTC:
            st.error("WebRTC dependencies not available. Please install streamlit-webrtc.")
            st.code("pip install streamlit-webrtc")
            return
        
        if not self.config:
            st.error("Configuration not loaded. Please check your setup.")
            return
        
        # Initialize pipeline if needed
        if not st.session_state.pipeline_active and self.config:
            with st.spinner("Initializing processing pipeline..."):
                self._initialize_pipeline()
        
        if not self.pipeline:
            st.error("Pipeline initialization failed. Check logs for details.")
            return
        
        # WebRTC configuration with error handling
        try:
            rtc_configuration = RTCConfiguration({
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            })
            
            # Create video transformer with thread safety
            if self.pipeline and not self.video_transformer:
                try:
                    self.video_transformer = VideoTransformer(
                        self.config, 
                        self.pipeline, 
                        event_callback=self.add_detection_event
                    )
                    logger.info("Video transformer created successfully")
                except Exception as e:
                    st.error(f"Failed to create video transformer: {e}")
                    logger.error(f"Video transformer creation failed: {e}")
                    return
            
            # WebRTC streamer with error handling
            try:
                webrtc_ctx = webrtc_streamer(
                    key="privacy-redactor",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=rtc_configuration,
                    video_transformer_factory=lambda: self.video_transformer,
                    media_stream_constraints={
                        "video": {
                            "width": {"min": 640, "ideal": 1280, "max": 1920},
                            "height": {"min": 480, "ideal": 720, "max": 1080},
                            "frameRate": {"min": 15, "ideal": 30, "max": 30}
                        },
                        "audio": True
                    },
                    async_processing=True,
                )
            except Exception as e:
                st.error(f"Failed to initialize WebRTC streamer: {e}")
                logger.error(f"WebRTC streamer initialization failed: {e}")
                return
            
            # Control buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🔄 Reset Statistics", key="reset_stats"):
                    try:
                        if self.video_transformer:
                            self.video_transformer.reset_stats()
                            st.success("Statistics reset!")
                        else:
                            st.warning("No video transformer available")
                    except Exception as e:
                        st.error(f"Failed to reset statistics: {e}")
            
            with col2:
                if st.button("⚙️ Reload Config", key="reload_config"):
                    try:
                        with st.spinner("Reloading configuration..."):
                            self._load_configuration()
                            if self.pipeline:
                                self.pipeline.stop()
                                self._initialize_pipeline()
                        st.success("Configuration reloaded!")
                    except Exception as e:
                        st.error(f"Failed to reload configuration: {e}")
            
            with col3:
                try:
                    if webrtc_ctx.state.playing:
                        st.success("🟢 Stream Active")
                    else:
                        st.info("🔴 Stream Inactive")
                except Exception as e:
                    st.warning(f"Unable to determine stream state: {e}")
                    
        except Exception as e:
            st.error(f"Error setting up video interface: {e}")
            logger.error(f"Video interface setup failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _initialize_pipeline(self):
        """Initialize the processing pipeline with proper error handling."""
        try:
            # Stop existing pipeline if running
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception as e:
                    logger.warning(f"Error stopping existing pipeline: {e}")
            
            # Create new pipeline instance
            self.pipeline = RealtimePipeline(self.config)
            
            # Start pipeline components
            self.pipeline.start()
            st.session_state.pipeline_active = True
            
            logger.info("Pipeline initialized successfully")
            
        except ImportError as e:
            error_msg = f"Missing required dependencies: {e}"
            st.error(error_msg)
            logger.error(f"Pipeline initialization failed - missing dependencies: {e}")
            self.pipeline = None
            
        except Exception as e:
            error_msg = f"Failed to initialize pipeline: {e}"
            st.error(error_msg)
            logger.error(f"Pipeline initialization error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            self.pipeline = None
    
    def _create_monitoring_panel(self):
        """Create the monitoring and statistics panel with error handling."""
        st.subheader("📊 Performance Monitor")
        
        # Performance statistics
        try:
            if self.video_transformer:
                stats = self.video_transformer.get_stats()
                
                # FPS and latency metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "FPS",
                        f"{stats.get('fps_current', 0):.1f}",
                        delta=f"Target: {stats.get('fps_target', 30)}"
                    )
                    
                    st.metric(
                        "Latency",
                        f"{stats.get('latency_current_ms', 0):.1f}ms",
                        delta=f"Avg: {stats.get('latency_avg_ms', 0):.1f}ms"
                    )
                
                with col2:
                    st.metric(
                        "Processing Time",
                        f"{stats.get('processing_current_ms', 0):.1f}ms",
                        delta=f"Avg: {stats.get('processing_avg_ms', 0):.1f}ms"
                    )
                    
                    st.metric(
                        "Active Tracks",
                        stats.get('tracks_active', 0),
                        delta=f"Total: {stats.get('total_tracks', 0)}"
                    )
                
                # Frame statistics
                st.write("**Frame Statistics:**")
                frames_processed = stats.get('frames_processed', 0)
                frames_total = stats.get('frames_total', 0)
                frames_dropped = stats.get('frames_dropped', 0)
                drop_rate = stats.get('drop_rate_percent', 0)
                
                st.write(f"- Processed: {frames_processed:,}")
                st.write(f"- Total: {frames_total:,}")
                st.write(f"- Dropped: {frames_dropped:,} ({drop_rate:.1f}%)")
                
                # OCR queue status
                ocr_queue_size = stats.get('ocr_queue_size', 0)
                ocr_processed = stats.get('ocr_processed', 0)
                max_queue = self.config.realtime.max_queue if self.config else 2
                st.write(f"- OCR Queue: {ocr_queue_size}/{max_queue}")
                st.write(f"- OCR Processed: {ocr_processed:,}")
                
                # Performance health indicator
                try:
                    is_healthy = self.video_transformer.is_performance_healthy()
                    if is_healthy:
                        st.success("🟢 Performance: Healthy")
                    else:
                        st.warning("🟡 Performance: Degraded")
                except Exception as e:
                    st.error(f"Error checking performance health: {e}")
            else:
                st.info("Video transformer not initialized")
                
        except Exception as e:
            st.error(f"Error displaying performance statistics: {e}")
            logger.error(f"Performance monitoring error: {e}")
        
        st.divider()
        
        # Detection counters
        st.subheader("🔍 Detection Counters")
        
        try:
            # Get pipeline stats for detection counts
            if self.pipeline:
                try:
                    pipeline_stats = self.pipeline.get_stats()
                except Exception as e:
                    logger.warning(f"Failed to get pipeline stats: {e}")
                    pipeline_stats = {}
                
                # Display counters per category
                categories = self.config.classification.categories if self.config else []
                
                if categories:
                    for category in categories:
                        count = self.detection_counters.get(category, 0)
                        st.metric(
                            category.replace('_', ' ').title(),
                            count,
                            delta=None
                        )
                else:
                    st.info("No detection categories selected")
            else:
                st.info("Pipeline not initialized")
                
        except Exception as e:
            st.error(f"Error displaying detection counters: {e}")
            logger.error(f"Detection counter error: {e}")
        
        st.divider()
        
        # Event feed
        st.subheader("📝 Detection Events")
        
        try:
            # Create scrollable event feed
            event_container = st.container()
            
            with event_container:
                if self.event_feed:
                    # Show recent events (most recent first)
                    for event in reversed(self.event_feed[-10:]):
                        timestamp = event.get('timestamp', '')
                        category = event.get('category', 'unknown')
                        masked_text = event.get('masked_text', '')
                        confidence = event.get('confidence', 0)
                        
                        # Format event display
                        st.write(f"**{timestamp}** - {category.title()}")
                        st.write(f"Text: `{masked_text}` (conf: {confidence:.2f})")
                        st.write("---")
                else:
                    st.info("No detection events yet")
                    
        except Exception as e:
            st.error(f"Error displaying event feed: {e}")
            logger.error(f"Event feed error: {e}")
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True, key="auto_refresh")
        
        if auto_refresh:
            time.sleep(5)
            st.rerun()
    
    def _register_cleanup(self):
        """Register cleanup procedures for application shutdown."""
        if not self._cleanup_registered:
            import atexit
            atexit.register(self._cleanup_on_exit)
            self._cleanup_registered = True
    
    def _cleanup_on_exit(self):
        """Cleanup resources on application exit."""
        try:
            logger.info("Performing application cleanup...")
            
            # Stop pipeline
            if self.pipeline:
                try:
                    self.pipeline.stop()
                    logger.info("Pipeline stopped successfully")
                except Exception as e:
                    logger.error(f"Error stopping pipeline: {e}")
            
            # Reset video transformer
            if self.video_transformer:
                try:
                    self.video_transformer.reset_stats()
                    logger.info("Video transformer reset")
                except Exception as e:
                    logger.error(f"Error resetting video transformer: {e}")
            
            # Clear session state
            try:
                st.session_state.pipeline_active = False
                st.session_state.recording_active = False
                logger.info("Session state cleared")
            except Exception as e:
                logger.warning(f"Error clearing session state: {e}")
                
        except Exception as e:
            logger.error(f"Error during application cleanup: {e}")
    
    def shutdown(self):
        """Gracefully shutdown the application."""
        self._cleanup_on_exit()
    
    def add_detection_event(self, category: str, masked_text: str, confidence: float):
        """Add a detection event to the feed with thread safety.
        
        Args:
            category: Detection category
            masked_text: Privacy-masked text
            confidence: Detection confidence
        """
        try:
            # Thread-safe counter update
            import threading
            if not hasattr(self, '_counter_lock'):
                self._counter_lock = threading.Lock()
            
            with self._counter_lock:
                # Update counter
                self.detection_counters[category] = self.detection_counters.get(category, 0) + 1
                
                # Add to event feed
                event = {
                    'timestamp': datetime.now().strftime('%H:%M:%S'),
                    'category': category,
                    'masked_text': masked_text,
                    'confidence': confidence
                }
                
                self.event_feed.append(event)
                
                # Keep only recent events
                if len(self.event_feed) > self.max_events:
                    self.event_feed = self.event_feed[-self.max_events:]
                    
        except Exception as e:
            logger.error(f"Error adding detection event: {e}")


def main():
    """Main entry point for the Streamlit application with comprehensive error handling."""
    try:
        # Configure logging with proper formatting
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('privacy_redactor_rt.log', mode='a')
            ]
        )
        
        logger.info("Starting Privacy Redactor RT application")
        
        # Check for required dependencies
        missing_deps = []
        
        try:
            import streamlit as st
        except ImportError:
            missing_deps.append("streamlit")
        
        if not HAS_WEBRTC:
            missing_deps.append("streamlit-webrtc")
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
            logger.error(error_msg)
            if 'streamlit' not in missing_deps:
                st.error(error_msg)
                st.code(f"pip install {' '.join(missing_deps)}")
            else:
                print(error_msg)
                print(f"Install with: pip install {' '.join(missing_deps)}")
            return
        
        # Create and run the app
        try:
            app = StreamlitApp()
            app.run()
        except Exception as e:
            logger.error(f"Application runtime error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            
            # Try to show error in Streamlit if available
            try:
                st.error(f"Application error: {e}")
                st.code(traceback.format_exc())
            except:
                print(f"Application error: {e}")
                traceback.print_exc()
                
    except Exception as e:
        # Fallback error handling
        print(f"Critical application error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()