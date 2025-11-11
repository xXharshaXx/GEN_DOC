"""
Test script for the GEN_DOC API
Tests the chat endpoint and verifies token counting and logging
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_chat_endpoint():
    """Test the /v1/chat endpoint"""
    print("=" * 60)
    print("Testing /v1/chat endpoint")
    print("=" * 60)
    
    # Test 1: Simple chat message
    print("\n1. Testing simple chat message...")
    response = requests.post(
        f"{BASE_URL}/v1/chat",
        params={
            "message": "Hello, how are you?",
            "session_id": "test_session_1"
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    # Test 2: Code analysis (triggers analysis path)
    print("\n2. Testing code analysis (should trigger token counting)...")
    response = requests.post(
        f"{BASE_URL}/v1/chat",
        params={
            "message": "Analyze this code: def hello(): print('world')",
            "session_id": "test_session_2"
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    # Test 3: File analysis with Python keywords
    print("\n3. Testing file analysis with Python keywords...")
    response = requests.post(
        f"{BASE_URL}/v1/chat",
        params={
            "message": "import torch\nmodel = torch.nn.Linear(10, 1)\noptimizer = torch.optim.Adam(model.parameters(), lr=0.001)",
            "session_id": "test_session_3"
        }
    )
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"Response: {json.dumps(data, indent=2)}")
    else:
        print(f"Error: {response.text}")
    
    print("\n" + "=" * 60)
    print("Check app.log file for token usage and timing logs!")
    print("=" * 60)

def check_logs():
    """Check if logs are being generated"""
    import os
    log_path = "app.log"
    if os.path.exists(log_path):
        print(f"\n✓ Log file found: {log_path}")
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            print(f"  Total log entries: {len(lines)}")
            # Show last 5 lines
            print("\n  Last 5 log entries:")
            for line in lines[-5:]:
                print(f"  {line.strip()}")
    else:
        print(f"\n⚠ Log file not found: {log_path}")

if __name__ == "__main__":
    print("🧪 GEN_DOC API Test Script")
    print("=" * 60)
    print("Make sure the API server is running:")
    print("  uvicorn api.main:app --host 0.0.0.0 --port 8000")
    print("=" * 60)
    
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/docs", timeout=2)
        print("✓ Server is running!\n")
        
        test_chat_endpoint()
        time.sleep(1)  # Wait a bit for logs to be written
        check_logs()
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to API server.")
        print("   Make sure the server is running:")
        print("   uvicorn api.main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"\n❌ Error: {e}")

