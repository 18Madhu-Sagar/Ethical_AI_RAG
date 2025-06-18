#!/usr/bin/env python3
"""
Simple startup script for Ethical AI RAG System
Avoids atexit registration issues during deployment
"""

import os
import sys
from pathlib import Path

# Set up environment variables
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('TRANSFORMERS_CACHE', './cache')

# Create cache directory if it doesn't exist
cache_dir = Path('./cache')
cache_dir.mkdir(exist_ok=True)

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Start the Streamlit app with minimal initialization."""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up Streamlit arguments
        sys.argv = [
            "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ]
        
        # Run Streamlit
        sys.exit(stcli.main())
        
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 