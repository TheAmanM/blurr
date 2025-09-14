"""CLI interface for offline processing."""

import typer
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from .config import Config, load_config
from .pipeline import RealtimePipeline
from .video_source import VideoSource
from .recorder import MP4Recorder
from .logging_utils import setup_logging
from .ui_detect import PrivacyUIDetector


app = typer.Typer(help="Privacy Redactor RT - Real-time sensitive information detection and redaction")
console = Console()


class OfflineProcessor:
    """Handles offline video processing with progress reporting."""
    
    def __init__(self, config: Config):
        """Initialize offline processor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.pipeline: Optional[RealtimePipeline] = None
        self.video_source: Optional[VideoSource] = None
        self.recorder: Optional[MP4Recorder] = None
        self.stats = {
            'frames_processed': 0,
            'detections_found': 0,
            'processing_time': 0.0,
            'avg_fps': 0.0
        }
    
    def process_video(self, input_path: Path, output_path: Path, 
                     progress_callback=None) -> Dict[str, Any]:
        """Process video file with redaction.
        
        Args:
            input_path: Input video file path
            output_path: Output video file path
            progress_callback: Optional progress callback function
            
        Returns:
            Processing statistics dictionary
        """
        start_time = time.time()
        
        try:
            # Initialize components
            self.pipeline = RealtimePipeline(self.config)
            self.video_source = VideoSource(self.config.io)
            
            # Open input video
            if not self.video_source.open_file(input_path):
                raise ValueError(f"Failed to open input video: {input_path}")
            
            # Get video info
            source_info = self.video_source.get_source_info()
            total_frames = source_info.get('total_frames', 0)
            
            # Initialize recorder
            self.recorder = MP4Recorder(
                self.config.recording,
                width=self.config.io.target_width,
                height=self.config.io.target_height,
                fps=self.config.io.target_fps
            )
            
            # Start recording
            if not self.recorder.start_recording(str(output_path)):
                raise ValueError(f"Failed to start recording to: {output_path}")
            
            # Start pipeline
            self.pipeline.start()
            
            # Process frames
            frame_idx = 0
            for frame, scale, offset in self.video_source.get_frame_iterator(throttled=False):
                # Process frame through pipeline
                processed_frame = self.pipeline.process_frame(frame, frame_idx)
                
                # Write to output
                if not self.recorder.write_frame(processed_frame):
                    logging.warning(f"Failed to write frame {frame_idx}")
                
                frame_idx += 1
                self.stats['frames_processed'] = frame_idx
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = frame_idx / total_frames
                    progress_callback(progress, frame_idx, total_frames)
                
                # Update detection stats
                pipeline_stats = self.pipeline.get_stats()
                self.stats['detections_found'] = pipeline_stats.get('tracks_active', 0)
            
            # Finalize recording
            recording_stats = self.recorder.stop_recording()
            
            # Calculate final statistics
            self.stats['processing_time'] = time.time() - start_time
            if self.stats['processing_time'] > 0:
                self.stats['avg_fps'] = self.stats['frames_processed'] / self.stats['processing_time']
            
            # Merge all statistics
            final_stats = {
                **self.stats,
                **recording_stats,
                'source_info': source_info,
                'pipeline_stats': self.pipeline.get_stats()
            }
            
            return final_stats
            
        finally:
            # Cleanup
            if self.pipeline:
                self.pipeline.stop()
            if self.video_source:
                self.video_source.close()
            if self.recorder and self.recorder.is_recording():
                self.recorder.stop_recording()


def validate_input_file(input_path: Path) -> None:
    """Validate input video file."""
    if not input_path.exists():
        raise typer.BadParameter(f"Input file does not exist: {input_path}")
    
    if not input_path.is_file():
        raise typer.BadParameter(f"Input path is not a file: {input_path}")
    
    # Check if it's a video file by trying to open with OpenCV
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        cap.release()
        raise typer.BadParameter(f"Cannot open video file: {input_path}")
    cap.release()


def validate_output_file(output_path: Path) -> None:
    """Validate output video file path."""
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if we can write to the location
    try:
        # Try to create a temporary file
        temp_file = output_path.with_suffix('.tmp')
        temp_file.touch()
        temp_file.unlink()
    except Exception as e:
        raise typer.BadParameter(f"Cannot write to output location: {e}")


def load_and_validate_config(config_path: Path) -> Config:
    """Load and validate configuration file."""
    try:
        if config_path.exists():
            config = Config.from_yaml(config_path)
        else:
            console.print(f"[yellow]Config file not found: {config_path}[/yellow]")
            console.print("[yellow]Using default configuration[/yellow]")
            config = Config()
        
        # Ensure recording is enabled for CLI processing
        config.recording.enabled = True
        
        return config
        
    except Exception as e:
        raise typer.BadParameter(f"Invalid configuration file: {e}")


@app.command()
def redact_video(
    input_file: Path = typer.Argument(..., help="Input video file path"),
    output_file: Path = typer.Argument(..., help="Output video file path"),
    config_file: Path = typer.Option("default.yaml", "--config", "-c", help="Configuration file path"),
    categories: Optional[List[str]] = typer.Option(None, "--category", help="Specific categories to detect (phone, credit_card, email, address, api_key)"),
    confidence: Optional[float] = typer.Option(None, "--confidence", help="Detection confidence threshold (0.0-1.0)"),
    method: Optional[str] = typer.Option(None, "--method", help="Redaction method (gaussian, pixelate, solid)"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Disable progress bar"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
):
    """Redact sensitive information from video file.
    
    This command processes a video file offline, detecting and redacting
    sensitive information such as phone numbers, credit cards, emails,
    addresses, and API keys.
    
    Examples:
    
        # Basic usage
        privacy-redactor redact-video input.mp4 output.mp4
        
        # Specify categories and confidence
        privacy-redactor redact-video input.mp4 output.mp4 --category phone --category email --confidence 0.8
        
        # Use pixelation instead of blur
        privacy-redactor redact-video input.mp4 output.mp4 --method pixelate
        
        # Use custom config
        privacy-redactor redact-video input.mp4 output.mp4 --config my_config.yaml
    """
    # Set up logging
    log_level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    setup_logging(level=log_level)
    
    if not quiet:
        console.print(Panel.fit(
            "[bold blue]Privacy Redactor RT[/bold blue]\n"
            "[dim]Offline Video Processing[/dim]",
            border_style="blue"
        ))
    
    try:
        # Validate inputs
        validate_input_file(input_file)
        validate_output_file(output_file)
        
        # Load configuration
        config = load_and_validate_config(config_file)
        
        # Apply CLI overrides
        if categories:
            valid_categories = {"phone", "credit_card", "email", "address", "api_key"}
            invalid = set(categories) - valid_categories
            if invalid:
                raise typer.BadParameter(f"Invalid categories: {invalid}")
            config.classification.categories = categories
        
        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise typer.BadParameter("Confidence must be between 0.0 and 1.0")
            config.detection.min_text_confidence = confidence
            config.ocr.min_ocr_confidence = confidence
        
        if method:
            if method not in ["gaussian", "pixelate", "solid"]:
                raise typer.BadParameter("Method must be one of: gaussian, pixelate, solid")
            config.redaction.default_method = method
        
        if not quiet:
            # Display configuration summary
            table = Table(title="Processing Configuration")
            table.add_column("Setting", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Input File", str(input_file))
            table.add_row("Output File", str(output_file))
            table.add_row("Categories", ", ".join(config.classification.categories))
            table.add_row("Detection Confidence", f"{config.detection.min_text_confidence:.2f}")
            table.add_row("OCR Confidence", f"{config.ocr.min_ocr_confidence:.2f}")
            table.add_row("Redaction Method", config.redaction.default_method)
            table.add_row("Target Resolution", f"{config.io.target_width}x{config.io.target_height}")
            table.add_row("Target FPS", str(config.io.target_fps))
            
            console.print(table)
            console.print()
        
        # Initialize processor
        processor = OfflineProcessor(config)
        
        # Process with progress bar
        if no_progress or quiet:
            # Process without progress bar
            stats = processor.process_video(input_file, output_file)
        else:
            # Process with progress bar
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TextColumn("({task.completed}/{task.total} frames)"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                
                task_id = progress.add_task("Processing video...", total=100)
                
                def update_progress(progress_pct: float, frame_idx: int, total_frames: int):
                    progress.update(
                        task_id,
                        completed=int(progress_pct * 100),
                        description=f"Processing frame {frame_idx}/{total_frames}"
                    )
                
                stats = processor.process_video(input_file, output_file, update_progress)
        
        if not quiet:
            # Display results
            console.print("\n[bold green]✓ Processing completed successfully![/bold green]")
            
            results_table = Table(title="Processing Results")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")
            
            results_table.add_row("Frames Processed", str(stats['frames_processed']))
            results_table.add_row("Processing Time", f"{stats['processing_time']:.2f} seconds")
            results_table.add_row("Average FPS", f"{stats['avg_fps']:.2f}")
            results_table.add_row("Output File Size", f"{stats.get('file_size_bytes', 0) / (1024*1024):.2f} MB")
            results_table.add_row("Output Duration", f"{stats.get('duration_seconds', 0):.2f} seconds")
            
            console.print(results_table)
            
            # Display pipeline statistics
            pipeline_stats = stats.get('pipeline_stats', {})
            if pipeline_stats:
                console.print(f"\n[dim]Detections: {pipeline_stats.get('tracks_active', 0)} active tracks[/dim]")
                console.print(f"[dim]OCR Requests: {pipeline_stats.get('ocr_processed', 0)}[/dim]")
        
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Processing interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def batch_process(
    input_dir: Path = typer.Argument(..., help="Directory containing input video files"),
    output_dir: Path = typer.Argument(..., help="Directory for output video files"),
    config_file: Path = typer.Option("default.yaml", "--config", "-c", help="Configuration file path"),
    pattern: str = typer.Option("*.mp4", "--pattern", help="File pattern to match (e.g., '*.mp4', '*.avi')"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process subdirectories recursively"),
    parallel: int = typer.Option(1, "--parallel", "-j", help="Number of parallel processes"),
    continue_on_error: bool = typer.Option(False, "--continue-on-error", help="Continue processing other files if one fails"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Batch process multiple video files.
    
    This command processes all video files in a directory, applying
    the same redaction settings to each file.
    
    Examples:
    
        # Process all MP4 files in a directory
        privacy-redactor batch-process ./input ./output
        
        # Process all video files recursively
        privacy-redactor batch-process ./input ./output --pattern "*.{mp4,avi,mov}" --recursive
        
        # Process with 4 parallel workers
        privacy-redactor batch-process ./input ./output --parallel 4
    """
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(level=log_level)
    
    console.print(Panel.fit(
        "[bold blue]Privacy Redactor RT[/bold blue]\n"
        "[dim]Batch Processing[/dim]",
        border_style="blue"
    ))
    
    try:
        # Validate directories
        if not input_dir.exists() or not input_dir.is_dir():
            raise typer.BadParameter(f"Input directory does not exist: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find input files
        if recursive:
            input_files = list(input_dir.rglob(pattern))
        else:
            input_files = list(input_dir.glob(pattern))
        
        if not input_files:
            console.print(f"[yellow]No files found matching pattern: {pattern}[/yellow]")
            return
        
        console.print(f"Found {len(input_files)} files to process")
        
        # Load configuration
        config = load_and_validate_config(config_file)
        
        # Process files
        successful = 0
        failed = 0
        
        with Progress(console=console) as progress:
            overall_task = progress.add_task("Overall Progress", total=len(input_files))
            
            for input_file in input_files:
                # Generate output path
                relative_path = input_file.relative_to(input_dir)
                output_file = output_dir / relative_path.with_suffix('.redacted.mp4')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    console.print(f"\nProcessing: {input_file.name}")
                    
                    # Create file-specific progress task
                    file_task = progress.add_task(f"Processing {input_file.name}", total=100)
                    
                    def update_file_progress(progress_pct: float, frame_idx: int, total_frames: int):
                        progress.update(file_task, completed=int(progress_pct * 100))
                    
                    # Process file
                    processor = OfflineProcessor(config)
                    stats = processor.process_video(input_file, output_file, update_file_progress)
                    
                    successful += 1
                    console.print(f"[green]✓ Completed: {input_file.name}[/green]")
                    
                except Exception as e:
                    failed += 1
                    console.print(f"[red]✗ Failed: {input_file.name} - {e}[/red]")
                    
                    if not continue_on_error:
                        raise
                
                finally:
                    progress.remove_task(file_task)
                    progress.advance(overall_task)
        
        # Summary
        console.print(f"\n[bold]Batch processing completed![/bold]")
        console.print(f"[green]Successful: {successful}[/green]")
        console.print(f"[red]Failed: {failed}[/red]")
        
        if failed > 0 and not continue_on_error:
            raise typer.Exit(1)
            
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Batch processing interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def analyze_privacy_image(
    input_file: Path = typer.Argument(..., help="Input image file path"),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output image file path (optional)"),
    show_all: bool = typer.Option(False, "--show-all", help="Show all detected elements, not just enabled ones"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
):
    """Analyze privacy settings in an image and highlight enabled options.
    
    This command processes an image of privacy settings (like a settings screen,
    privacy dashboard, etc.) to detect and highlight enabled privacy options.
    
    Examples:
    
        # Basic usage - analyze and display results
        privacy-redactor analyze-privacy-image settings_screenshot.png
        
        # Save result to file
        privacy-redactor analyze-privacy-image settings.png --output result.png
        
        # Show all detected elements, not just enabled ones
        privacy-redactor analyze-privacy-image settings.png --show-all
    """
    # Set up logging
    log_level = logging.ERROR if quiet else (logging.DEBUG if verbose else logging.INFO)
    setup_logging(level=log_level)
    
    if not quiet:
        console.print(Panel.fit(
            "[bold blue]Privacy Redactor RT[/bold blue]\n"
            "[dim]Privacy Settings Image Analysis[/dim]",
            border_style="blue"
        ))
    
    try:
        # Validate input file
        if not input_file.exists():
            raise typer.BadParameter(f"Input file does not exist: {input_file}")
        
        if not input_file.is_file():
            raise typer.BadParameter(f"Input path is not a file: {input_file}")
        
        # Try to load the image
        image = cv2.imread(str(input_file))
        if image is None:
            raise typer.BadParameter(f"Cannot open image file: {input_file}")
        
        if not quiet:
            console.print(f"[cyan]Analyzing image: {input_file}[/cyan]")
            console.print(f"[dim]Image dimensions: {image.shape[1]}x{image.shape[0]}[/dim]")
        
        # Initialize detector
        detector = PrivacyUIDetector()
        
        # Detect UI elements
        if not quiet:
            console.print("[yellow]Detecting UI elements...[/yellow]")
        
        elements = detector.detect_privacy_elements(image)
        
        if not quiet:
            console.print(f"[green]Detected {len(elements)} UI elements[/green]")
        
        # Filter elements based on options
        if show_all:
            display_elements = elements
            filter_desc = "all detected"
        else:
            display_elements = detector.filter_privacy_elements(elements)
            filter_desc = "enabled privacy settings"
        
        # Display results
        if not quiet:
            if display_elements:
                table = Table(title=f"Detected Elements ({filter_desc})")
                table.add_column("Type", style="cyan")
                table.add_column("State", style="green")
                table.add_column("Position", style="yellow")
                table.add_column("Confidence", style="magenta")
                
                for i, element in enumerate(display_elements):
                    table.add_row(
                        element.element_type.title(),
                        element.state.title(),
                        f"({element.bbox.x1}, {element.bbox.y1}) - ({element.bbox.x2}, {element.bbox.y2})",
                        f"{element.confidence:.2f}"
                    )
                
                console.print(table)
                
                # Summary statistics
                console.print(f"\n[bold]Summary:[/bold]")
                console.print(f"Total elements detected: {len(elements)}")
                console.print(f"Enabled privacy settings: {len(detector.filter_privacy_elements(elements))}")
                
                # Count by type
                type_counts = {}
                state_counts = {}
                for elem in elements:
                    type_counts[elem.element_type] = type_counts.get(elem.element_type, 0) + 1
                    state_counts[elem.state] = state_counts.get(elem.state, 0) + 1
                
                console.print(f"\nBy type: {dict(type_counts)}")
                console.print(f"By state: {dict(state_counts)}")
                
            else:
                console.print("[yellow]No elements detected matching the criteria[/yellow]")
        
        # Generate output image if requested
        if output_file or not quiet:
            result_image = detector.draw_bounding_boxes(image, display_elements)
            
            if output_file:
                # Validate output path
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                success = cv2.imwrite(str(output_file), result_image)
                if not success:
                    raise typer.BadParameter(f"Failed to save output image: {output_file}")
                
                if not quiet:
                    console.print(f"[green]✓ Result saved to: {output_file}[/green]")
            
            # If no output file specified but not quiet, save to temp file for display info
            elif not quiet and display_elements:
                temp_output = input_file.with_suffix('.analyzed.png')
                cv2.imwrite(str(temp_output), result_image)
                console.print(f"[dim]Preview saved to: {temp_output}[/dim]")
        
        # Exit with appropriate code
        enabled_count = len(detector.filter_privacy_elements(elements))
        if enabled_count > 0:
            if not quiet:
                console.print(f"\n[bold green]✓ Found {enabled_count} enabled privacy settings[/bold green]")
        else:
            if not quiet:
                console.print(f"\n[yellow]⚠ No enabled privacy settings detected[/yellow]")
        
    except typer.BadParameter as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def run_app(
    port: int = typer.Option(8501, help="Port to run Streamlit app on"),
    host: str = typer.Option("localhost", help="Host to bind to"),
    config_file: Path = typer.Option("default.yaml", help="Configuration file"),
):
    """Run the Streamlit web interface."""
    console.print(f"Starting Privacy Redactor RT web interface on {host}:{port}")
    console.print(f"Using configuration: {config_file}")
    
    # Construct streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        "--server.port", str(port),
        "--server.address", host,
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
        "privacy_redactor_rt/app.py"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error running Streamlit app: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/yellow]")
        raise typer.Exit(0)


if __name__ == "__main__":
    app()