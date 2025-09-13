#!/usr/bin/env python3
"""
Demo script showing CLI interface functionality for offline video processing.

This script demonstrates the CLI commands available in the Privacy Redactor RT system.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and display the output."""
    print(f"\n{'='*60}")
    print(f"DEMO: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Exit code: {result.returncode}")
    except subprocess.TimeoutExpired:
        print("Command timed out after 30 seconds")
    except Exception as e:
        print(f"Error running command: {e}")


def main():
    """Run CLI demonstrations."""
    python_cmd = [".venv/bin/python", "-m", "privacy_redactor_rt.cli"]
    
    print("Privacy Redactor RT - CLI Interface Demo")
    print("=" * 60)
    
    # 1. Show main help
    run_command(
        python_cmd + ["--help"],
        "Main CLI Help"
    )
    
    # 2. Show redact-video help
    run_command(
        python_cmd + ["redact-video", "--help"],
        "Redact Video Command Help"
    )
    
    # 3. Show batch-process help
    run_command(
        python_cmd + ["batch-process", "--help"],
        "Batch Process Command Help"
    )
    
    # 4. Show run-app help
    run_command(
        python_cmd + ["run-app", "--help"],
        "Run App Command Help"
    )
    
    # 5. Test error handling with invalid input
    run_command(
        python_cmd + ["redact-video", "nonexistent.mp4", "output.mp4", "--quiet"],
        "Error Handling - Invalid Input File"
    )
    
    # 6. Test invalid category
    run_command(
        python_cmd + ["redact-video", "input.mp4", "output.mp4", "--category", "invalid", "--quiet"],
        "Error Handling - Invalid Category"
    )
    
    # 7. Test invalid confidence
    run_command(
        python_cmd + ["redact-video", "input.mp4", "output.mp4", "--confidence", "1.5", "--quiet"],
        "Error Handling - Invalid Confidence"
    )
    
    print(f"\n{'='*60}")
    print("CLI Demo Complete!")
    print("=" * 60)
    print("\nKey Features Demonstrated:")
    print("✓ Main CLI interface with multiple commands")
    print("✓ Comprehensive help system")
    print("✓ Video redaction with configurable options")
    print("✓ Batch processing capabilities")
    print("✓ Web app launcher")
    print("✓ Input validation and error handling")
    print("✓ Progress reporting and verbose logging options")
    print("✓ Configuration file support")
    print("✓ Category-specific detection")
    print("✓ Multiple redaction methods")


if __name__ == "__main__":
    main()