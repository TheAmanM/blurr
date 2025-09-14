"""Optimized Gradio app with smooth performance and primary/secondary face detection."""

import gradio as gr
import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple, List
import threading
from queue import Queue, Empty

from .optimized_detector import OptimizedPrivacyDetector
from .optimized_realtime import OptimizedRealtimeDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedGradioApp:
    """Optimized Gradio app for privacy detection."""
    
    def __init__(self):
        """Initialize the optimized Gradio app."""
        self.detector = OptimizedPrivacyDetector()
        self.realtime_detector: Optional[OptimizedRealtimeDetector] = None
        self.is_realtime_active = False
        
        # Performance settings
        self.target_fps = 60
        self.confidence_threshold = 0.6
        self.adaptive_quality = True
        self.frame_skipping = True
        
        logger.info("Optimized Gradio app initialized")
    
    def process_image(self, image: np.ndarray, confidence_threshold: float = 0.6) -> Tuple[np.ndarray, str]:
        """Process a single image with optimized detection."""
        if image is None:
            return None, "No image provided"
        
        # Update confidence thresholds
        self.detector.face_confidence_threshold = confidence_threshold
        self.detector.plate_confidence_threshold = confidence_threshold + 0.1
        self.detector.document_confidence_threshold = confidence_threshold - 0.1
        self.detector.screen_confidence_threshold = confidence_threshold
        
        start_time = time.time()
        
        # Detect privacy violations
        detections = self.detector.detect_privacy_violations(image)
        
        processing_time = time.time() - start_time
        
        # Draw results
        result_image = self.detector.draw_detections(image, detections)
        
        # Generate report
        report = self._generate_detection_report(detections, processing_time)
        
        return result_image, report
    
    def process_video_frame(self, frame: np.ndarray, target_fps: int = 60, 
                          confidence_threshold: float = 0.6, 
                          adaptive_quality: bool = True,
                          frame_skipping: bool = True) -> Tuple[np.ndarray, str]:
        """Process video frame with real-time optimizations."""
        if frame is None:
            return None, "No frame provided"
        
        # Initialize or update real-time detector
        if not self.is_realtime_active or self.realtime_detector is None:
            if self.realtime_detector:
                self.realtime_detector.stop()
            
            self.realtime_detector = OptimizedRealtimeDetector()
            self.realtime_detector.start()
            self.is_realtime_active = True
        
        # Update settings
        self.realtime_detector.set_target_fps(target_fps)
        self.realtime_detector.set_adaptive_quality(adaptive_quality)
        self.realtime_detector.set_frame_skipping(frame_skipping)
        
        # Update detector confidence thresholds
        self.realtime_detector.detector.face_confidence_threshold = confidence_threshold
        self.realtime_detector.detector.plate_confidence_threshold = confidence_threshold + 0.1
        self.realtime_detector.detector.document_confidence_threshold = confidence_threshold - 0.1
        self.realtime_detector.detector.screen_confidence_threshold = confidence_threshold
        
        # Process frame
        result_frame = self.realtime_detector.process_frame(frame)
        
        # Get current detections and stats
        detections = self.realtime_detector.get_current_detections()
        stats = self.realtime_detector.get_comprehensive_stats()
        
        # Generate real-time report
        report = self._generate_realtime_report(detections, stats)
        
        return result_frame, report
    
    def _generate_detection_report(self, detections: List, processing_time: float) -> str:
        """Generate detection report for single image."""
        report_lines = [
            f"🚀 Optimized Privacy Detection Results",
            f"⏱️ Processing Time: {processing_time*1000:.1f}ms",
            f"🎯 Total Detections: {len(detections)}",
            ""
        ]
        
        # Categorize detections
        primary_faces = [d for d in detections if d.category == 'face' and d.priority == 'primary']
        secondary_faces = [d for d in detections if d.category == 'face' and d.priority == 'secondary']
        other_faces = [d for d in detections if d.category == 'face' and d.priority == 'normal']
        other_detections = [d for d in detections if d.category != 'face']
        
        # Face detection summary
        if primary_faces or secondary_faces or other_faces:
            report_lines.append("👤 Face Detection:")
            
            if primary_faces:
                report_lines.append(f"  🎯 Primary Faces: {len(primary_faces)}")
                for i, face in enumerate(primary_faces):
                    quality = face.features.get('quality', 0) if face.features else 0
                    report_lines.append(f"    {i+1}. {face.sub_category} (conf: {face.confidence:.2f}, quality: {quality:.2f})")
            
            if secondary_faces:
                report_lines.append(f"  🔸 Secondary Faces: {len(secondary_faces)}")
                for i, face in enumerate(secondary_faces):
                    quality = face.features.get('quality', 0) if face.features else 0
                    report_lines.append(f"    {i+1}. {face.sub_category} (conf: {face.confidence:.2f}, quality: {quality:.2f})")
            
            if other_faces:
                report_lines.append(f"  👥 Other Faces: {len(other_faces)}")
            
            report_lines.append("")
        
        # Other privacy elements
        if other_detections:
            report_lines.append("🔒 Other Privacy Elements:")
            
            categories = {}
            for detection in other_detections:
                if detection.category not in categories:
                    categories[detection.category] = []
                categories[detection.category].append(detection)
            
            for category, items in categories.items():
                icon = {"license_plate": "🚗", "document": "📄", "screen": "💻"}.get(category, "🔍")
                report_lines.append(f"  {icon} {category.replace('_', ' ').title()}: {len(items)}")
                
                for i, item in enumerate(items):
                    report_lines.append(f"    {i+1}. {item.sub_category} (conf: {item.confidence:.2f})")
        
        # Performance info
        stats = self.detector.get_stats()
        report_lines.extend([
            "",
            "📊 Performance:",
            f"  • Estimated FPS: {stats['fps']:.1f}",
            f"  • Cache Hit Rate: {stats['cache_hit_rate']*100:.1f}%",
            f"  • Total Processed: {stats['frames_processed']} frames"
        ])
        
        return "\n".join(report_lines)
    
    def _generate_realtime_report(self, detections: List, stats: dict) -> str:
        """Generate real-time detection report."""
        report_lines = [
            f"⚡ Real-time Optimized Detection",
            f"🎯 Current FPS: {stats.get('current_fps', 0):.1f} / {stats.get('target_fps', 60)}",
            f"📊 Quality Scale: {stats.get('quality_scale', 1.0):.2f}",
            f"🔍 Active Detections: {len(detections)}",
            ""
        ]
        
        # Performance status
        performance_ratio = stats.get('performance_ratio', 0)
        if performance_ratio >= 1.5:
            status = "🟢 EXCELLENT"
        elif performance_ratio >= 1.0:
            status = "🟢 SMOOTH"
        elif performance_ratio >= 0.8:
            status = "🟡 GOOD"
        else:
            status = "🔴 CHOPPY"
        
        report_lines.append(f"Status: {status}")
        report_lines.append("")
        
        # Current detections
        if detections:
            primary_faces = sum(1 for d in detections if d.category == 'face' and d.priority == 'primary')
            secondary_faces = sum(1 for d in detections if d.category == 'face' and d.priority == 'secondary')
            other_detections = len([d for d in detections if d.category != 'face'])
            
            if primary_faces > 0:
                report_lines.append(f"🎯 Primary Faces: {primary_faces}")
            if secondary_faces > 0:
                report_lines.append(f"🔸 Secondary Faces: {secondary_faces}")
            if other_detections > 0:
                report_lines.append(f"🔒 Other Elements: {other_detections}")
            
            report_lines.append("")
        
        # Performance metrics
        report_lines.extend([
            "📈 Performance Metrics:",
            f"  • Frames Processed: {stats.get('frames_processed', 0)}",
            f"  • Frames Skipped: {stats.get('frames_skipped', 0)}",
            f"  • Quality Adjustments: {stats.get('quality_adjustments', 0)}",
            f"  • Primary Faces Total: {stats.get('primary_faces_detected', 0)}",
            f"  • Secondary Faces Total: {stats.get('secondary_faces_detected', 0)}"
        ])
        
        return "\n".join(report_lines)
    
    def create_interface(self):
        """Create the Gradio interface."""
        with gr.Blocks(title="🚀 Optimized Privacy Detection", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🚀 Optimized Privacy Detection System
            
            **High-performance privacy detection with primary/secondary face recognition**
            
            ✨ **Features:**
            - 🎯 Primary & Secondary Face Detection
            - ⚡ 60+ FPS Real-time Processing  
            - 🔍 Accurate Privacy Element Classification
            - 📊 Adaptive Quality & Performance
            """)
            
            with gr.Tabs():
                # Image Processing Tab
                with gr.Tab("📸 Image Detection"):
                    gr.Markdown("### Upload an image for optimized privacy detection")
                    
                    with gr.Row():
                        with gr.Column():
                            image_input = gr.Image(label="Input Image", type="numpy")
                            confidence_slider = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                                label="Confidence Threshold"
                            )
                            process_btn = gr.Button("🚀 Process Image", variant="primary")
                        
                        with gr.Column():
                            image_output = gr.Image(label="Detection Results", type="numpy")
                            report_output = gr.Textbox(
                                label="Detection Report", 
                                lines=15, 
                                max_lines=20
                            )
                    
                    process_btn.click(
                        fn=self.process_image,
                        inputs=[image_input, confidence_slider],
                        outputs=[image_output, report_output]
                    )
                
                # Real-time Processing Tab
                with gr.Tab("⚡ Real-time Detection"):
                    gr.Markdown("### Real-time video processing with optimized performance")
                    
                    with gr.Row():
                        with gr.Column():
                            video_input = gr.Image(label="Video Frame", type="numpy", source="webcam")
                            
                            with gr.Row():
                                fps_slider = gr.Slider(
                                    minimum=15, maximum=120, value=60, step=5,
                                    label="Target FPS"
                                )
                                confidence_slider_rt = gr.Slider(
                                    minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                                    label="Confidence Threshold"
                                )
                            
                            with gr.Row():
                                adaptive_quality_cb = gr.Checkbox(
                                    label="Adaptive Quality", value=True
                                )
                                frame_skipping_cb = gr.Checkbox(
                                    label="Frame Skipping", value=True
                                )
                        
                        with gr.Column():
                            video_output = gr.Image(label="Real-time Results", type="numpy")
                            realtime_report = gr.Textbox(
                                label="Real-time Report", 
                                lines=15, 
                                max_lines=20
                            )
                    
                    video_input.change(
                        fn=self.process_video_frame,
                        inputs=[
                            video_input, fps_slider, confidence_slider_rt,
                            adaptive_quality_cb, frame_skipping_cb
                        ],
                        outputs=[video_output, realtime_report]
                    )
                
                # Performance Info Tab
                with gr.Tab("📊 Performance Info"):
                    gr.Markdown("""
                    ### Performance Optimizations
                    
                    **🚀 Speed Improvements:**
                    - Optimized face detection algorithms
                    - Streamlined classification pipeline
                    - Intelligent frame caching
                    - Adaptive quality scaling
                    
                    **🎯 Accuracy Improvements:**
                    - Primary/Secondary face prioritization
                    - Specialized privacy element classifiers
                    - Enhanced confidence scoring
                    - Reduced false positives
                    
                    **⚡ Real-time Features:**
                    - Target FPS: 60+ FPS
                    - Adaptive quality scaling
                    - Smart frame skipping
                    - Performance monitoring
                    
                    **🔍 Detection Categories:**
                    - **Faces**: Primary, Secondary, Profile detection
                    - **License Plates**: US, EU, Custom formats
                    - **Documents**: A4, Letter, Business cards
                    - **Screens**: 16:9, 4:3, Phone, Ultrawide
                    """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        # Default launch settings
        launch_settings = {
            'server_name': '0.0.0.0',
            'server_port': 7860,
            'share': False,
            'debug': False
        }
        launch_settings.update(kwargs)
        
        logger.info(f"Launching optimized Gradio app on {launch_settings['server_name']}:{launch_settings['server_port']}")
        
        try:
            interface.launch(**launch_settings)
        except KeyboardInterrupt:
            logger.info("Shutting down optimized Gradio app...")
            if self.realtime_detector:
                self.realtime_detector.stop()
            self.detector.cleanup()
        except Exception as e:
            logger.error(f"Error launching Gradio app: {e}")
            if self.realtime_detector:
                self.realtime_detector.stop()
            self.detector.cleanup()
            raise


def main():
    """Main function to launch the optimized Gradio app."""
    app = OptimizedGradioApp()
    app.launch()


if __name__ == "__main__":
    main()