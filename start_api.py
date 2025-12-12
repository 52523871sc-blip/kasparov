"""
Startup script for Chess AI Play API
"""

import subprocess
import sys
import time
import requests
from threading import Thread

def start_api_server():
    """Start the API server"""
    print("Starting Chess AI Play API Server...")
    subprocess.run([sys.executable, "chess_api.py"])

def test_api_when_ready():
    """Test API once it's ready"""
    print("Waiting for API to start...")
    
    # Wait for server to be ready
    for i in range(30):  # Wait up to 30 seconds
        try:
            response = requests.get("http://localhost:5000/api/health", timeout=1)
            if response.status_code == 200:
                print("✓ API server is ready!")
                break
        except:
            time.sleep(1)
    else:
        print("✗ API server failed to start")
        return
    
    # Run API client demo
    print("\\nRunning API demo...")
    subprocess.run([sys.executable, "api_client.py"])

if __name__ == "__main__":
    print("Chess AI Play API Startup")
    print("=" * 40)
    
    # Start server in background thread
    server_thread = Thread(target=start_api_server, daemon=True)
    server_thread.start()
    
    # Test API
    test_api_when_ready()