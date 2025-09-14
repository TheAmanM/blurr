"""Gradio web interface for Privacy Redactor RT."""

import gradio as gr
import cv2
import numpy as np
import logging
import tempfile
import time
from pathlib import Path
from typing import Optional, Tuple

from privacy_redactor_rt.config import Config, load_config
from privacy_redactor_rt.pipeline import RealtimePipeline
from privacy_redactor_rt.video_source import VideoSource
from privacy_redactor_rt.recorder import MP4Recorder
from privacy_redactor_rt.ui_detect import PrivacyUIDetector
from privacy_redactor_rt.realtime_detector import RealtimePrivacyDetector, PrivacyViolationCounter

logger = logging.getLogger(__name__)


class GradioApp:
    """Gradio-based web interface for privacy redaction."""
    
    def __init__(self):
        """Initialize the Gradio application."""
        self.config: Optional[Config] = None
        self.pipeline: Optional[RealtimePipeline] = None
        self.ui_detector = PrivacyUIDetector()
        self.realtime_detector: Optional[RealtimePrivacyDetector] = None
        self.violation_counter = PrivacyViolationCounter()
        self._load_config()
        self._initialize_pipeline()
        self._initialize_realtime_detector()
    
    def _load_config(self):
        """Load configuration."""
        try:
            config_path = Path("default.yaml")
            if config_path.exists():
                self.config = load_config(config_path)
            else:
                self.config = Config()
            
            # Ensure recording is enabled for video processing
            self.config.recording.enabled = True
            
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = Config()
            # Ensure recording is enabled even with default config
            self.config.recording.enabled = True
    
    def _initialize_pipeline(self):
        """Initialize the processing pipeline."""
        try:
            if self.config:
                self.pipeline = RealtimePipeline(self.config)
                self.pipeline.start()
                logger.info("Pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            self.pipeline = None
    
    def _initialize_realtime_detector(self):
        """Initialize the real-time privacy detector."""
        try:
            self.realtime_detector = RealtimePrivacyDetector()
            self.realtime_detector.start()
            logger.info("Real-time privacy detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize real-time detector: {e}")
            self.realtime_detector = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame through the privacy redaction pipeline.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Processed frame with redactions applied
        """
        if frame is None:
            return frame
            
        if self.pipeline is None:
            return frame
        
        # Validate frame
        if frame.size == 0:
            return frame
        
        try:
            # Ensure frame is contiguous and has proper shape
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Convert RGB to BGR (OpenCV format)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Ensure BGR frame is also contiguous
            if not frame_bgr.flags['C_CONTIGUOUS']:
                frame_bgr = np.ascontiguousarray(frame_bgr)
            
            # Process through pipeline with incremental frame counter
            if not hasattr(self, '_frame_counter'):
                self._frame_counter = 0
            self._frame_counter += 1
            
            processed_frame = self.pipeline.process_frame(frame_bgr, self._frame_counter)
            
            # Convert back to RGB for display
            if len(processed_frame.shape) == 3 and processed_frame.shape[2] == 3:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            return processed_frame
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame
    
    def process_video_stream(self, frame):
        """Process video stream frame by frame.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with privacy redactions
        """
        return self.process_frame(frame)
    
    def process_uploaded_video(self, video_file) -> Tuple[Optional[str], str]:
        """Process uploaded video file and return redacted video.
        
        Args:
            video_file: Uploaded video file path
            
        Returns:
            Tuple of (output_video_path, status_message)
        """
        if video_file is None:
            return None, "No video file uploaded"
        
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_output:
                output_path = tmp_output.name
            
            # Initialize video processing components
            video_source = VideoSource(self.config.io)
            
            # Open input video
            if not video_source.open_file(video_file):
                return None, f"Failed to open video file: {video_file}"
            
            # Get video info
            source_info = video_source.get_source_info()
            total_frames = source_info.get('total_frames', 0)
            
            # Initialize recorder with video dimensions
            recorder = MP4Recorder(
                self.config.recording,
                width=self.config.io.target_width,
                height=self.config.io.target_height,
                fps=self.config.io.target_fps
            )
            
            # Start recording
            if not recorder.start_recording(output_path):
                return None, f"Failed to start recording to output file"
            
            # Start pipeline
            if self.pipeline:
                self.pipeline.start()
            
            # Process video frame by frame
            frame_count = 0
            start_time = time.time()
            
            for frame, scale, offset in video_source.get_frame_iterator(throttled=False):
                # Process frame through pipeline
                if self.pipeline:
                    processed_frame = self.pipeline.process_frame(frame, frame_count)
                else:
                    processed_frame = frame
                
                # Write to output
                if not recorder.write_frame(processed_frame):
                    logger.warning(f"Failed to write frame {frame_count}")
                
                frame_count += 1
                
                # Progress logging every 100 frames
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
            
            # Finalize recording
            recording_stats = recorder.stop_recording()
            
            # Cleanup
            video_source.close()
            
            processing_time = time.time() - start_time
            status_msg = (
                f"✅ Video processed successfully!\n"
                f"📊 Processed {frame_count} frames in {processing_time:.2f} seconds\n"
                f"🎯 Average FPS: {frame_count / processing_time:.2f}\n"
                f"📁 Output size: {recording_stats.get('file_size_bytes', 0) / (1024*1024):.2f} MB"
            )
            
            return output_path, status_msg
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return None, f"❌ Error processing video: {str(e)}"
    
    def update_categories(self, categories):
        """Update detection categories.
        
        Args:
            categories: List of selected categories
        """
        if self.config and categories:
            self.config.classification.categories = categories
            logger.info(f"Updated categories: {categories}")
    
    def update_redaction_method(self, method):
        """Update redaction method.
        
        Args:
            method: Selected redaction method
        """
        if self.config:
            self.config.redaction.default_method = method
            logger.info(f"Updated redaction method: {method}")
    
    def process_privacy_settings_image(self, image) -> Tuple[Optional[np.ndarray], str]:
        """Process uploaded image to detect and highlight privacy settings.
        
        Args:
            image: Uploaded image as numpy array
            
        Returns:
            Tuple of (processed_image, status_message)
        """
        if image is None:
            return None, "No image uploaded"
        
        try:
            # Convert RGB to BGR for OpenCV processing
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Detect privacy UI elements
            elements = self.ui_detector.detect_privacy_elements(image_bgr)
            
            # Filter to only enabled/active privacy settings
            enabled_elements = self.ui_detector.filter_privacy_elements(elements)
            
            # Draw bounding boxes around enabled privacy settings
            result_image = self.ui_detector.draw_bounding_boxes(image_bgr, enabled_elements)
            
            # Convert back to RGB for display
            if len(result_image.shape) == 3 and result_image.shape[2] == 3:
                result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
            
            # Create status message
            total_elements = len(elements)
            enabled_count = len(enabled_elements)
            
            status_msg = (
                f"✅ Privacy settings analysis complete!\n"
                f"🔍 Total UI elements detected: {total_elements}\n"
                f"🟢 Enabled privacy settings: {enabled_count}\n"
                f"📊 Element breakdown:\n"
            )
            
            # Count by type and state
            element_stats = {}
            for elem in elements:
                key = f"{elem.element_type}_{elem.state}"
                element_stats[key] = element_stats.get(key, 0) + 1
            
            for elem_type, count in element_stats.items():
                status_msg += f"   • {elem_type.replace('_', ' ').title()}: {count}\n"
            
            if enabled_count == 0:
                status_msg += "\n⚠️ No enabled privacy settings detected in this image."
            else:
                status_msg += f"\n🎯 {enabled_count} privacy settings are currently enabled (highlighted in green boxes)."
            
            return result_image, status_msg
            
        except Exception as e:
            logger.error(f"Error processing privacy settings image: {e}")
            return None, f"❌ Error processing image: {str(e)}"
    
    def process_realtime_privacy_detection(self, frame):
        """Process video frame for real-time privacy violation detection.
        
        Args:
            frame: Input frame from webcam
            
        Returns:
            Processed frame with privacy violations highlighted
        """
        if frame is None:
            return frame
        
        if self.realtime_detector is None:
            return frame
        
        try:
            # Process frame through real-time detector
            result_frame = self.realtime_detector.process_frame(frame)
            
            # Update violation counter
            current_detections = self.realtime_detector.get_current_detections()
            self.violation_counter.update(current_detections)
            
            return result_frame
            
        except Exception as e:
            logger.error(f"Error in real-time privacy detection: {e}")
            return frame
    
    def get_privacy_stats(self):
        """Get current privacy violation statistics."""
        if self.realtime_detector is None:
            return "Real-time detector not available"
        
        try:
            detector_stats = self.realtime_detector.get_stats()
            violation_stats = self.violation_counter.get_stats()
            
            stats_text = (
                f"🔍 Real-time Privacy Detection Stats\n"
                f"📊 Current FPS: {detector_stats.get('current_fps', 0):.1f}\n"
                f"🎯 Active Detections: {detector_stats.get('active_detections', 0)}\n"
                f"⚡ Provider: {detector_stats.get('provider', 'Unknown')}\n"
                f"⏱️ Avg Inference: {detector_stats.get('avg_inference_time', 0)*1000:.1f}ms\n\n"
                f"📈 Violation Statistics (last 30 frames):\n"
                f"🚨 Total Violations: {violation_stats.get('total_violations', 0)}\n"
                f"📊 Avg per Frame: {violation_stats.get('avg_violations_per_frame', 0):.1f}\n"
                f"📈 Trend: {violation_stats.get('recent_trend', 'stable').title()}\n\n"
                f"🏷️ By Category:\n"
            )
            
            category_counts = violation_stats.get('category_counts', {})
            if category_counts:
                for category, count in category_counts.items():
                    stats_text += f"   • {category.replace('_', ' ').title()}: {count}\n"
            else:
                stats_text += "   • No violations detected\n"
            
            return stats_text
            
        except Exception as e:
            logger.error(f"Error getting privacy stats: {e}")
            return f"Error getting stats: {str(e)}"
    
    def update_detection_settings(self, confidence_threshold, draw_boxes, show_labels):
        """Update real-time detection settings."""
        if self.realtime_detector:
            self.realtime_detector.set_confidence_threshold(confidence_threshold)
            self.realtime_detector.set_draw_boxes(draw_boxes)
            self.realtime_detector.set_show_labels(show_labels)
            logger.info(f"Updated detection settings: confidence={confidence_threshold}, boxes={draw_boxes}, labels={show_labels}")
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Available options
        categories = ["phone", "credit_card", "email", "address", "api_key"]
        redaction_methods = ["gaussian", "pixelate", "solid"]
        
        with gr.Blocks(title="Privacy Redactor RT", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🔒 Privacy Redactor RT")
            gr.Markdown("Real-time sensitive information detection and redaction system")
            
            with gr.Tabs():
                # Real-time Privacy Detection Tab
                with gr.TabItem("🚨 Real-time Privacy Detection"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Video interface for privacy detection
                            privacy_video_input = gr.Image(
                                sources=["webcam"],
                                streaming=True,
                                label="Live Video Feed"
                            )
                            
                            privacy_video_output = gr.Image(
                                label="Privacy Violations Detected (Real-time)",
                                streaming=True
                            )
                        
                        with gr.Column(scale=1):
                            # Real-time controls
                            gr.Markdown("## 🎛️ Detection Settings")
                            
                            confidence_slider = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.5,
                                step=0.05,
                                label="Confidence Threshold",
                                info="Minimum confidence for detections"
                            )
                            
                            draw_boxes_checkbox = gr.Checkbox(
                                value=True,
                                label="Draw Bounding Boxes",
                                info="Show detection boxes on video"
                            )
                            
                            show_labels_checkbox = gr.Checkbox(
                                value=True,
                                label="Show Labels & Stats",
                                info="Display detection labels and performance stats"
                            )
                            
                            gr.Markdown("## 📊 Live Statistics")
                            
                            stats_display = gr.Textbox(
                                label="Privacy Detection Stats",
                                value="Starting real-time detection...",
                                interactive=False,
                                lines=15,
                                max_lines=20
                            )
                            
                            # Auto-refresh stats every 2 seconds
                            stats_refresh_btn = gr.Button(
                                "🔄 Refresh Stats",
                                variant="secondary",
                                size="sm"
                            )
                            
                            gr.Markdown("## 🎯 Detected Categories")
                            gr.Markdown(
                                "**Privacy Violations Detected:**\n"
                                "- 👤 **Faces**: Personal identity\n"
                                "- 🚗 **License Plates**: Vehicle identification\n"
                                "- 📄 **Documents**: Sensitive papers\n"
                                "- 💳 **ID Cards**: Identity documents\n"
                                "- 💳 **Credit Cards**: Financial information\n"
                                "- 📱 **Phone Numbers**: Contact information\n"
                                "- 📧 **Email Addresses**: Contact information\n"
                                "- 🏠 **Addresses**: Location information"
                            )
                
                # Live Video Tab (Original)
                with gr.TabItem("📹 Live Video (Text Detection)"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Video interface
                            video_input = gr.Image(
                                sources=["webcam"],
                                streaming=True,
                                label="Live Video Feed"
                            )
                            
                            video_output = gr.Image(
                                label="Processed Video (Privacy Redacted)",
                                streaming=True
                            )
                        
                        with gr.Column(scale=1):
                            # Controls
                            gr.Markdown("## 🎛️ Controls")
                            
                            selected_categories = gr.CheckboxGroup(
                                choices=categories,
                                value=categories,
                                label="Detection Categories",
                                info="Select types of sensitive information to detect"
                            )
                            
                            redaction_method = gr.Dropdown(
                                choices=redaction_methods,
                                value="gaussian",
                                label="Redaction Method",
                                info="How to redact detected information"
                            )
                            
                            gr.Markdown("## 📊 Performance")
                            
                            # Performance metrics (placeholder)
                            fps_display = gr.Textbox(
                                label="FPS",
                                value="30",
                                interactive=False
                            )
                            
                            latency_display = gr.Textbox(
                                label="Latency (ms)",
                                value="50",
                                interactive=False
                            )
                            
                            gr.Markdown("## 🔍 Detection Stats")
                            
                            # Detection counters (placeholder)
                            phone_count = gr.Number(label="Phone Numbers", value=0, interactive=False)
                            email_count = gr.Number(label="Email Addresses", value=0, interactive=False)
                            card_count = gr.Number(label="Credit Cards", value=0, interactive=False)
                
                # Privacy Settings Image Tab
                with gr.TabItem("🔒 Privacy Settings Image"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("## Privacy Settings Detection")
                            gr.Markdown("Upload an image of privacy settings (like a settings screen, privacy dashboard, etc.) to detect and highlight enabled privacy options.")
                            
                            # Image upload
                            upload_image = gr.Image(
                                label="Upload Privacy Settings Image",
                                sources=["upload"],
                                type="numpy"
                            )
                            
                            # Process button
                            process_image_btn = gr.Button(
                                "🔍 Analyze Privacy Settings", 
                                variant="primary",
                                size="lg"
                            )
                            
                            # Status
                            image_status_text = gr.Textbox(
                                label="Analysis Results",
                                value="Ready to analyze privacy settings...",
                                interactive=False,
                                lines=8
                            )
                            
                            # Output image
                            output_image = gr.Image(
                                label="Privacy Settings Analysis (Enabled Settings Highlighted)",
                                visible=False
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("## 🎯 Detection Info")
                            gr.Markdown(
                                "This tool detects common privacy UI elements:\n\n"
                                "**Checkboxes** - ✅ Checked boxes indicate enabled settings\n"
                                "**Toggle Switches** - 🔘 Right position means enabled\n"
                                "**Radio Buttons** - ⚫ Filled circles show selection\n"
                                "**Buttons** - 🔲 Bright buttons are typically active\n\n"
                                "**Color Legend:**\n"
                                "- 🟢 Green: Checkboxes\n"
                                "- 🔵 Blue: Toggle switches\n"
                                "- 🔴 Red: Radio buttons\n"
                                "- 🟡 Yellow: Buttons"
                            )
                            
                            gr.Markdown("## 📱 Supported Interfaces")
                            gr.Markdown(
                                "- Mobile app settings screens\n"
                                "- Web browser privacy pages\n"
                                "- Desktop application preferences\n"
                                "- Social media privacy dashboards\n"
                                "- Operating system privacy controls"
                            )
                            
                            gr.Markdown("## 💡 Tips")
                            gr.Markdown(
                                "- Use clear, high-resolution screenshots\n"
                                "- Ensure good contrast between UI elements\n"
                                "- Crop to focus on privacy settings area\n"
                                "- Works best with standard UI patterns"
                            )
                
                # Video Upload Tab
                with gr.TabItem("📁 Video Upload"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("## Upload Video for Privacy Redaction")
                            gr.Markdown("Upload a video file to detect and redact sensitive information. Supported formats: MP4, AVI, MOV, MKV")
                            
                            # Video upload
                            upload_video = gr.Video(
                                label="Upload Video File",
                                sources=["upload"],
                                format="mp4"
                            )
                            
                            # Process button
                            process_btn = gr.Button(
                                "🔒 Process Video", 
                                variant="primary",
                                size="lg"
                            )
                            
                            # Status and progress
                            status_text = gr.Textbox(
                                label="Processing Status",
                                value="Ready to process video...",
                                interactive=False,
                                lines=4
                            )
                            
                            # Output video
                            output_video = gr.Video(
                                label="Redacted Video Output",
                                visible=False
                            )
                            
                            # Download button
                            download_btn = gr.DownloadButton(
                                "📥 Download Redacted Video",
                                visible=False,
                                variant="secondary"
                            )
                        
                        with gr.Column(scale=1):
                            # Upload-specific controls
                            gr.Markdown("## 🎛️ Processing Settings")
                            
                            upload_categories = gr.CheckboxGroup(
                                choices=categories,
                                value=categories,
                                label="Detection Categories",
                                info="Select types of sensitive information to detect"
                            )
                            
                            upload_redaction_method = gr.Dropdown(
                                choices=redaction_methods,
                                value="gaussian",
                                label="Redaction Method",
                                info="How to redact detected information"
                            )
                            
                            gr.Markdown("## ℹ️ Processing Info")
                            gr.Markdown(
                                "- Processing time depends on video length and resolution\n"
                                "- Larger videos may take several minutes to process\n"
                                "- The output video will have the same duration as input\n"
                                "- Audio track is preserved in the output"
                            )
                            
                            gr.Markdown("## 🎯 Supported Formats")
                            gr.Markdown(
                                "**Input:** MP4, AVI, MOV, MKV, WMV, FLV\n"
                                "**Output:** MP4 (H.264 + AAC)"
                            )
            
            # Event handlers for real-time privacy detection tab
            privacy_video_input.stream(
                fn=self.process_realtime_privacy_detection,
                inputs=[privacy_video_input],
                outputs=[privacy_video_output],
                stream_every=0.05,  # Process every 50ms for real-time performance
                show_progress=False
            )
            
            # Settings update handlers
            confidence_slider.change(
                fn=lambda conf, boxes, labels: self.update_detection_settings(conf, boxes, labels),
                inputs=[confidence_slider, draw_boxes_checkbox, show_labels_checkbox]
            )
            
            draw_boxes_checkbox.change(
                fn=lambda conf, boxes, labels: self.update_detection_settings(conf, boxes, labels),
                inputs=[confidence_slider, draw_boxes_checkbox, show_labels_checkbox]
            )
            
            show_labels_checkbox.change(
                fn=lambda conf, boxes, labels: self.update_detection_settings(conf, boxes, labels),
                inputs=[confidence_slider, draw_boxes_checkbox, show_labels_checkbox]
            )
            
            # Stats refresh handler
            stats_refresh_btn.click(
                fn=self.get_privacy_stats,
                outputs=[stats_display]
            )
            
            # Auto-refresh stats periodically
            def auto_refresh_stats():
                return self.get_privacy_stats()
            
            # Note: Auto-refresh will be handled by manual refresh button
            # Gradio's auto-refresh capabilities vary by version
            
            # Event handlers for live video tab (original text detection)
            video_input.stream(
                fn=self.process_video_stream,
                inputs=[video_input],
                outputs=[video_output],
                stream_every=0.1,  # Process every 100ms
                show_progress=False
            )
            
            # Update handlers for live video
            selected_categories.change(
                fn=self.update_categories,
                inputs=[selected_categories]
            )
            
            redaction_method.change(
                fn=self.update_redaction_method,
                inputs=[redaction_method]
            )
            
            # Event handlers for video upload tab
            def process_video_wrapper(video_file, categories, method):
                """Wrapper to update config and process video."""
                if categories:
                    self.config.classification.categories = categories
                if method:
                    self.config.redaction.default_method = method
                
                return self.process_uploaded_video(video_file)
            
            def handle_video_processing(video_file, categories, method):
                """Handle video processing and return UI updates."""
                if video_file is None:
                    return (
                        "❌ Please upload a video file first",
                        gr.Video(visible=False),
                        gr.DownloadButton(visible=False)
                    )
                
                # Update config
                if categories:
                    self.config.classification.categories = categories
                if method:
                    self.config.redaction.default_method = method
                
                # Process video
                output_path, status = self.process_uploaded_video(video_file)
                
                if output_path:
                    return (
                        status,
                        gr.Video(value=output_path, visible=True),
                        gr.DownloadButton(value=output_path, visible=True)
                    )
                else:
                    return (
                        status,
                        gr.Video(visible=False),
                        gr.DownloadButton(visible=False)
                    )
            
            # Process button click handler
            process_btn.click(
                fn=handle_video_processing,
                inputs=[upload_video, upload_categories, upload_redaction_method],
                outputs=[status_text, output_video, download_btn],
                show_progress=True
            )
            
            # Update handlers for upload settings
            upload_categories.change(
                fn=self.update_categories,
                inputs=[upload_categories]
            )
            
            upload_redaction_method.change(
                fn=self.update_redaction_method,
                inputs=[upload_redaction_method]
            )
            
            # Event handlers for privacy settings image tab
            def handle_image_processing(image):
                """Handle privacy settings image processing and return UI updates."""
                if image is None:
                    return (
                        "❌ Please upload an image first",
                        gr.Image(visible=False)
                    )
                
                # Process image
                result_image, status = self.process_privacy_settings_image(image)
                
                if result_image is not None:
                    return (
                        status,
                        gr.Image(value=result_image, visible=True)
                    )
                else:
                    return (
                        status,
                        gr.Image(visible=False)
                    )
            
            # Process button click handler for image
            process_image_btn.click(
                fn=handle_image_processing,
                inputs=[upload_image],
                outputs=[image_status_text, output_image],
                show_progress=True
            )
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        return interface.launch(**kwargs)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.pipeline:
            self.pipeline.stop()
        if self.realtime_detector:
            self.realtime_detector.stop()


def main():
    """Main entry point for Gradio app."""
    logging.basicConfig(level=logging.INFO)
    
    app = GradioApp()
    
    try:
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True
        )
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()