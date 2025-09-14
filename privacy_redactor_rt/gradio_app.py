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

logger = logging.getLogger(__name__)


class GradioApp:
    """Gradio-based web interface for privacy redaction."""
    
    def __init__(self):
        """Initialize the Gradio application."""
        self.config: Optional[Config] = None
        self.pipeline: Optional[RealtimePipeline] = None
        self._load_config()
        self._initialize_pipeline()
    
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
    
    def create_interface(self):
        """Create the Gradio interface."""
        
        # Available options
        categories = ["phone", "credit_card", "email", "address", "api_key"]
        redaction_methods = ["gaussian", "pixelate", "solid"]
        
        with gr.Blocks(title="Privacy Redactor RT", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🔒 Privacy Redactor RT")
            gr.Markdown("Real-time sensitive information detection and redaction system")
            
            with gr.Tabs():
                # Live Video Tab
                with gr.TabItem("📹 Live Video"):
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
            
            # Event handlers for live video tab
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
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        return interface.launch(**kwargs)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.pipeline:
            self.pipeline.stop()


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