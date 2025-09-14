"""MP4 recording functionality with FFmpeg integration and audio preservation."""

import subprocess
import threading
import time
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
import queue
import cv2

from .config import RecordingConfig


logger = logging.getLogger(__name__)


class MP4Recorder:
    """MP4 video recorder using FFmpeg with audio preservation, with OpenCV fallback."""
    
    def __init__(self, config: RecordingConfig, width: int = 1280, height: int = 720, fps: int = 30):
        """Initialize MP4 recorder.
        
        Args:
            config: Recording configuration
            width: Output video width
            height: Output video height  
            fps: Output frame rate
        """
        self.config = config
        self.width = width
        self.height = height
        self.fps = fps
        
        # Check if FFmpeg is available
        self._has_ffmpeg = shutil.which('ffmpeg') is not None
        
        # FFmpeg-specific attributes
        self._process: Optional[subprocess.Popen] = None
        
        # OpenCV VideoWriter fallback
        self._cv_writer: Optional[cv2.VideoWriter] = None
        
        # Common attributes
        self._output_path: Optional[Path] = None
        self._frame_count = 0
        self._start_time: Optional[float] = None
        self._is_recording = False
        self._lock = threading.Lock()
        
        # Audio handling (FFmpeg only)
        self._has_audio = False
        self._audio_sample_rate = 44100
        self._audio_channels = 2
        
        # Frame queue for constant frame rate (FFmpeg only)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=fps * 2)  # 2 second buffer
        self._writer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        if not self._has_ffmpeg:
            logger.warning("FFmpeg not found, using OpenCV VideoWriter (no audio support)")
        
    def start_recording(self, output_path: Optional[str] = None, has_audio: bool = False,
                       audio_sample_rate: int = 44100, audio_channels: int = 2) -> bool:
        """Start MP4 recording.
        
        Args:
            output_path: Custom output file path (optional)
            has_audio: Whether source has audio track
            audio_sample_rate: Audio sample rate in Hz
            audio_channels: Number of audio channels
            
        Returns:
            True if recording started successfully
        """
        with self._lock:
            if self._is_recording:
                logger.warning("Recording already in progress")
                return False
                
            if not self.config.enabled:
                logger.info("Recording disabled in configuration")
                return False
                
            try:
                # Generate output path
                self._output_path = self._generate_output_path(output_path)
                self._output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Store audio parameters
                self._has_audio = has_audio and self._has_ffmpeg  # Audio only with FFmpeg
                self._audio_sample_rate = audio_sample_rate
                self._audio_channels = audio_channels
                
                if self._has_ffmpeg:
                    return self._start_ffmpeg_recording()
                else:
                    return self._start_opencv_recording()
                
            except Exception as e:
                logger.error(f"Failed to start recording: {e}")
                self._cleanup()
                return False
    
    def _start_ffmpeg_recording(self) -> bool:
        """Start recording using FFmpeg."""
        try:
            # Build FFmpeg command
            cmd = self._build_ffmpeg_command()
            
            logger.info(f"Starting FFmpeg recording to: {self._output_path}")
            logger.debug(f"FFmpeg command: {' '.join(cmd)}")
            
            # Start FFmpeg process
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0
            )
            
            # Reset counters
            self._frame_count = 0
            self._start_time = time.time()
            self._is_recording = True
            self._stop_event.clear()
            
            # Start frame writer thread for CFR
            if self.config.constant_framerate:
                self._writer_thread = threading.Thread(
                    target=self._cfr_writer_loop,
                    daemon=True
                )
                self._writer_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start FFmpeg recording: {e}")
            return False
    
    def _start_opencv_recording(self) -> bool:
        """Start recording using OpenCV VideoWriter."""
        try:
            # Use MP4V codec for better compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            logger.info(f"Starting OpenCV recording to: {self._output_path}")
            
            # Initialize OpenCV VideoWriter
            self._cv_writer = cv2.VideoWriter(
                str(self._output_path),
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self._cv_writer.isOpened():
                logger.error("Failed to open OpenCV VideoWriter")
                return False
            
            # Reset counters
            self._frame_count = 0
            self._start_time = time.time()
            self._is_recording = True
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start OpenCV recording: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """Write a video frame.
        
        Args:
            frame: BGR frame as numpy array (height, width, 3)
            
        Returns:
            True if frame written successfully
        """
        if not self._is_recording:
            return False
            
        try:
            # Ensure frame is correct size and format
            if frame.shape != (self.height, self.width, 3):
                logger.warning(f"Frame size mismatch: expected ({self.height}, {self.width}, 3), got {frame.shape}")
                return False
            
            if self._has_ffmpeg and self._process is not None:
                return self._write_frame_ffmpeg(frame)
            elif self._cv_writer is not None:
                return self._write_frame_opencv(frame)
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False
    
    def _write_frame_ffmpeg(self, frame: np.ndarray) -> bool:
        """Write frame using FFmpeg."""
        # Convert BGR to RGB for FFmpeg
        rgb_frame = frame[:, :, ::-1]
        
        if self.config.constant_framerate and self._writer_thread is not None:
            # Use CFR queue
            try:
                self._frame_queue.put(rgb_frame, timeout=0.1)
                return True
            except queue.Full:
                logger.warning("Frame queue full, dropping frame")
                return False
        else:
            # Direct write
            return self._write_frame_direct(rgb_frame)
    
    def _write_frame_opencv(self, frame: np.ndarray) -> bool:
        """Write frame using OpenCV VideoWriter."""
        try:
            # OpenCV expects BGR format
            self._cv_writer.write(frame)
            self._frame_count += 1
            return True
        except Exception as e:
            logger.error(f"Error writing frame with OpenCV: {e}")
            return False
    
    def write_audio_samples(self, samples: np.ndarray) -> bool:
        """Write audio samples (if audio input is available).
        
        Args:
            samples: Audio samples as numpy array
            
        Returns:
            True if samples written successfully
        """
        if not self._is_recording or self._process is None or not self._has_audio:
            return False
            
        try:
            # Convert to bytes and write to stdin
            audio_bytes = samples.astype(np.int16).tobytes()
            self._process.stdin.write(audio_bytes)
            return True
            
        except Exception as e:
            logger.error(f"Error writing audio samples: {e}")
            return False
    
    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and finalize MP4 file.
        
        Returns:
            Recording statistics dictionary
        """
        with self._lock:
            if not self._is_recording:
                return {}
                
            try:
                if self._has_ffmpeg and self._process is not None:
                    return self._stop_ffmpeg_recording()
                elif self._cv_writer is not None:
                    return self._stop_opencv_recording()
                else:
                    return {}
                
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
                return {}
            finally:
                self._cleanup()
    
    def _stop_ffmpeg_recording(self) -> Dict[str, Any]:
        """Stop FFmpeg recording."""
        # Signal stop to CFR thread
        self._stop_event.set()
        
        # Wait for CFR thread to finish
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        
        # Close FFmpeg stdin to signal end
        if self._process and self._process.stdin:
            self._process.stdin.close()
        
        # Wait for FFmpeg to finish
        if self._process:
            try:
                stdout, stderr = self._process.communicate(timeout=30.0)
                return_code = self._process.returncode
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg process timeout, terminating")
                self._process.terminate()
                stdout, stderr = self._process.communicate(timeout=5.0)
                return_code = self._process.returncode
            
            if return_code != 0:
                logger.error(f"FFmpeg failed with code {return_code}")
                logger.error(f"FFmpeg stderr: {stderr.decode()}")
            else:
                logger.info("FFmpeg recording completed successfully")
        
        return self._get_recording_stats()
    
    def _stop_opencv_recording(self) -> Dict[str, Any]:
        """Stop OpenCV recording."""
        if self._cv_writer:
            self._cv_writer.release()
            self._cv_writer = None
            logger.info("OpenCV recording completed successfully")
        
        return self._get_recording_stats()
    
    def _get_recording_stats(self) -> Dict[str, Any]:
        """Get recording statistics."""
        duration = time.time() - self._start_time if self._start_time else 0
        stats = {
            'output_path': str(self._output_path) if self._output_path else None,
            'duration_seconds': duration,
            'frame_count': self._frame_count,
            'average_fps': self._frame_count / duration if duration > 0 else 0,
            'file_size_bytes': self._output_path.stat().st_size if self._output_path and self._output_path.exists() else 0
        }
        return stats
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._is_recording
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current recording statistics."""
        if not self._is_recording or self._start_time is None:
            return {}
            
        duration = time.time() - self._start_time
        return {
            'duration_seconds': duration,
            'frame_count': self._frame_count,
            'current_fps': self._frame_count / duration if duration > 0 else 0,
            'output_path': str(self._output_path) if self._output_path else None,
            'has_audio': self._has_audio
        }
    
    def _generate_output_path(self, custom_path: Optional[str] = None) -> Path:
        """Generate output file path."""
        if custom_path:
            return Path(custom_path)
            
        # Use template from config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.config.filename_template.format(timestamp=timestamp)
        
        return Path(self.config.output_dir) / filename
    
    def _build_ffmpeg_command(self) -> list[str]:
        """Build FFmpeg command for MP4 encoding."""
        cmd = ['ffmpeg', '-y']  # -y to overwrite output file
        
        # Video input from stdin
        cmd.extend([
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'rgb24',
            '-r', str(self.fps),
            '-i', '-'  # stdin
        ])
        
        # Audio input handling
        if self._has_audio:
            # Audio from stdin (interleaved with video)
            cmd.extend([
                '-f', 's16le',
                '-ar', str(self._audio_sample_rate),
                '-ac', str(self._audio_channels),
                '-i', '-'  # stdin for audio
            ])
        else:
            # Generate silent audio track
            cmd.extend([
                '-f', 'lavfi',
                '-i', f'anullsrc=channel_layout=stereo:sample_rate={self._audio_sample_rate}'
            ])
        
        # Video encoding options
        cmd.extend([
            '-c:v', self.config.codec,
            '-crf', str(self.config.crf),
            '-preset', self.config.preset
        ])
        
        # Audio encoding options
        cmd.extend([
            '-c:a', self.config.audio_codec,
            '-b:a', self.config.audio_bitrate
        ])
        
        # Constant frame rate
        if self.config.constant_framerate:
            cmd.extend(['-r', str(self.fps)])
        
        # Output format and file
        cmd.extend([
            '-f', 'mp4',
            '-movflags', '+faststart',  # Enable streaming
            str(self._output_path)
        ])
        
        return cmd
    
    def _write_frame_direct(self, rgb_frame: np.ndarray) -> bool:
        """Write frame directly to FFmpeg stdin."""
        try:
            if self._process and self._process.stdin:
                frame_bytes = rgb_frame.tobytes()
                self._process.stdin.write(frame_bytes)
                self._frame_count += 1
                return True
            return False
        except Exception as e:
            logger.error(f"Error writing frame directly: {e}")
            return False
    
    def _cfr_writer_loop(self) -> None:
        """Constant frame rate writer thread loop."""
        target_interval = 1.0 / self.fps
        last_write_time = time.time()
        
        while not self._stop_event.is_set():
            try:
                # Get frame from queue with timeout
                try:
                    frame = self._frame_queue.get(timeout=target_interval)
                except queue.Empty:
                    # No new frame, repeat last frame if we have one
                    if hasattr(self, '_last_frame'):
                        frame = self._last_frame
                    else:
                        continue
                
                # Maintain constant frame rate timing
                current_time = time.time()
                elapsed = current_time - last_write_time
                
                if elapsed < target_interval:
                    time.sleep(target_interval - elapsed)
                
                # Write frame
                if self._write_frame_direct(frame):
                    self._last_frame = frame
                    last_write_time = time.time()
                
                # Mark task done
                try:
                    self._frame_queue.task_done()
                except ValueError:
                    pass  # Queue might be empty
                    
            except Exception as e:
                logger.error(f"Error in CFR writer loop: {e}")
                break
    
    def _cleanup(self) -> None:
        """Clean up resources."""
        self._is_recording = False
        
        # Cleanup FFmpeg process
        if self._process:
            try:
                if self._process.poll() is None:  # Process still running
                    self._process.terminate()
                    self._process.wait(timeout=5.0)
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process: {e}")
            finally:
                self._process = None
        
        # Cleanup OpenCV VideoWriter
        if self._cv_writer:
            try:
                self._cv_writer.release()
            except Exception as e:
                logger.error(f"Error releasing OpenCV VideoWriter: {e}")
            finally:
                self._cv_writer = None
        
        # Clear frame queue
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        
        self._writer_thread = None
        self._output_path = None
        self._frame_count = 0
        self._start_time = None


class RecorderManager:
    """Manager for multiple recording sessions."""
    
    def __init__(self, config: RecordingConfig):
        """Initialize recorder manager.
        
        Args:
            config: Recording configuration
        """
        self.config = config
        self._active_recorders: Dict[str, MP4Recorder] = {}
        self._lock = threading.Lock()
    
    def create_recorder(self, session_id: str, width: int = 1280, height: int = 720, 
                       fps: int = 30) -> MP4Recorder:
        """Create a new recorder instance.
        
        Args:
            session_id: Unique session identifier
            width: Video width
            height: Video height
            fps: Frame rate
            
        Returns:
            MP4Recorder instance
        """
        with self._lock:
            if session_id in self._active_recorders:
                # Stop existing recorder
                self._active_recorders[session_id].stop_recording()
            
            recorder = MP4Recorder(self.config, width, height, fps)
            self._active_recorders[session_id] = recorder
            return recorder
    
    def get_recorder(self, session_id: str) -> Optional[MP4Recorder]:
        """Get existing recorder by session ID."""
        return self._active_recorders.get(session_id)
    
    def stop_all_recordings(self) -> Dict[str, Dict[str, Any]]:
        """Stop all active recordings.
        
        Returns:
            Dictionary of session_id -> recording stats
        """
        stats = {}
        with self._lock:
            for session_id, recorder in self._active_recorders.items():
                if recorder.is_recording():
                    stats[session_id] = recorder.stop_recording()
            
            self._active_recorders.clear()
        
        return stats
    
    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active recorders."""
        stats = {}
        with self._lock:
            for session_id, recorder in self._active_recorders.items():
                stats[session_id] = recorder.get_stats()
        
        return stats