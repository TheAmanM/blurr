"""Streamlit web interface with real-time controls."""

import streamlit as st
from .config import Config


def main():
    """Main Streamlit application."""
    st.title("Privacy Redactor RT")
    st.sidebar.header("Configuration")
    
    # Placeholder for Streamlit interface
    st.write("Real-time privacy redaction system")


if __name__ == "__main__":
    main()