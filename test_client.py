# test_client.py

import requests
import json
import time

# Point to the local FastAPI server
API_URL = "http://127.0.0.1:8000/api/v1/video/analyze"

# Ensure you have a small, multi-scene video here
TEST_VIDEO_PATH = "data/raw/test_scene.mp4"

def test_pipeline():
    print(f"Sending {TEST_VIDEO_PATH} to the FrameTune API...")
    
    start_time = time.time()
    
    try:
        with open(TEST_VIDEO_PATH, "rb") as video_file:
            # FastAPI expects the key 'file' to match the parameter in video_routes.py
            files = {"file": video_file}
            
            # Send the POST request
            response = requests.post(API_URL, files=files)
            
        # Check if the server crashed
        response.raise_for_status()
        
        # Parse and print the JSON payload
        data = response.json()
        print("\n✅ Success! Received JSON Payload:\n")
        print(json.dumps(data, indent=4))
        
        end_time = time.time()
        print(f"\nTotal Processing Time: {round(end_time - start_time, 2)} seconds")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ API Request Failed: {e}")
        if response is not None:
            print("Server Response:", response.text)

if __name__ == "__main__":
    test_pipeline()