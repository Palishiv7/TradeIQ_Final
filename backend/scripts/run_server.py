#!/usr/bin/env python3
"""
Backend server runner script.

This script starts the FastAPI server with the appropriate configuration.
"""

import os
import sys
import logging
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent.parent
sys.path.append(str(backend_dir))

import uvicorn
from backend.main import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def main():
    """Run the backend server."""
    try:
        # Get configuration from environment or use defaults
        host = os.getenv("HOST", "0.0.0.0")
        port = int(os.getenv("PORT", 8000))
        reload_enabled = os.getenv("RELOAD", "true").lower() == "true"
        
        logger.info(f"Starting server on {host}:{port} (reload: {reload_enabled})")
        
        # Run the application
        uvicorn.run(
            "backend.main:app",
            host=host,
            port=port,
            reload=reload_enabled,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 