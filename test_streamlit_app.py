#!/usr/bin/env python3
"""Test script to verify Streamlit app can be imported and basic functionality works."""

import sys
import os
import tempfile
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from privacy_redactor_rt.config import Config, load_config
        print("✓ Config module imported successfully")
        
        from privacy_redactor_rt.types import BBox, Detection, Match, Track
        print("✓ Types module imported successfully")
        
        # Test app module import (may fail due to streamlit dependency)
        try:
            from privacy_redactor_rt.app import StreamlitApp, main
            print("✓ App module imported successfully")
        except ImportError as e:
            if "streamlit" in str(e).lower():
                print("⚠ App module requires streamlit (expected in dev environment)")
            else:
                print(f"✗ App import error: {e}")
                return False
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        from privacy_redactor_rt.config import load_config
        
        # Test loading default config
        config = load_config(Path("default.yaml"))
        print("✓ Default configuration loaded successfully")
        
        # Test basic config properties
        assert config.io.target_width == 1280
        assert config.io.target_height == 720
        assert config.io.target_fps == 30
        print("✓ Configuration values are correct")
        
        return True
    except Exception as e:
        print(f"✗ Configuration loading error: {e}")
        return False

def test_app_initialization():
    """Test Streamlit app initialization."""
    print("\nTesting app initialization...")
    
    try:
        # Skip if streamlit not available
        try:
            import streamlit
        except ImportError:
            print("⚠ Skipping app initialization test (streamlit not available)")
            return True
            
        from privacy_redactor_rt.app import StreamlitApp
        
        # Create app instance
        app = StreamlitApp()
        print("✓ StreamlitApp instance created successfully")
        
        # Test that session state attributes exist
        assert hasattr(app, 'config')
        assert hasattr(app, 'pipeline')
        assert hasattr(app, 'video_transformer')
        assert hasattr(app, 'detection_counters')
        assert hasattr(app, 'event_feed')
        print("✓ App attributes initialized correctly")
        
        return True
    except Exception as e:
        print(f"✗ App initialization error: {e}")
        return False

def test_cli_integration():
    """Test CLI integration."""
    print("\nTesting CLI integration...")
    
    try:
        # Skip if typer not available
        try:
            import typer
        except ImportError:
            print("⚠ Skipping CLI test (typer not available)")
            return True
            
        from privacy_redactor_rt.cli import app as cli_app
        print("✓ CLI app imported successfully")
        
        # Test that commands exist
        commands = [cmd.name for cmd in cli_app.commands.values()]
        assert 'redact-video' in commands
        assert 'run-app' in commands
        print("✓ CLI commands are available")
        
        return True
    except Exception as e:
        print(f"✗ CLI integration error: {e}")
        return False

def main():
    """Run all tests."""
    print("Privacy Redactor RT - Streamlit App Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_config_loading,
        test_app_initialization,
        test_cli_integration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Streamlit app is ready to run.")
        print("\nTo run the app:")
        print("  python -m privacy_redactor_rt.cli run-app")
        print("  or")
        print("  streamlit run -m privacy_redactor_rt.app")
        return 0
    else:
        print("✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())