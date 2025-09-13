"""Gradio web interface for Privacy Redactor RT."""

import gradio as gr
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from privacy_redactor_rt.config import Config, load_config
from privacy_redactor_rt.pipeline import RealtimePipeline

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
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            self.config = Config()
    
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
        
        try:
            # Convert RGB to BGR (OpenCV format)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Process through pipeline
            processed_frame = self.pipeline.process_frame(frame_bgr, 0)
            
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
            
            # Set up streaming
            video_input.stream(
                fn=self.process_video_stream,
                inputs=[video_input],
                outputs=[video_output],
                stream_every=0.1,  # Process every 100ms
                show_progress=False
            )
            
            # Update handlers
            selected_categories.change(
                fn=self.update_categories,
                inputs=[selected_categories]
            )
            
            redaction_method.change(
                fn=self.update_redaction_method,
                inputs=[redaction_method]
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