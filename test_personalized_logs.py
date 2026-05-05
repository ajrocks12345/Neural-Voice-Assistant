import requests
import os
import time

BASE_URL = "http://127.0.0.1:5000"

def test_user(name, age, symptoms):
    print(f"\n--- Testing User: {name} (Age: {age}) ---")
    
    # 1. Send Predict Request
    payload = {
        "message": "diagnose now", # force a log entry
        "history": symptoms,
        "force": True,
        "user_name": name,
        "user_age": str(age)
    }
    res = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Predict Status: {res.status_code}")
    
    time.sleep(1) # wait for file write
    
    # 2. Check if file was created
    expected_file = f"logs/{name}_{age}_history.csv"
    if os.path.exists(expected_file):
        print(f"SUCCESS: Log file created -> {expected_file}")
    else:
        print(f"FAILED: Log file NOT found -> {expected_file}")
        
    # 3. Test Download Endpoint
    download_res = requests.get(f"{BASE_URL}/download-history?name={name}&age={age}")
    if download_res.status_code == 200:
        print(f"SUCCESS: Download endpoint returned file for {name}")
    else:
        print(f"FAILED: Download endpoint returned {download_res.status_code}")

print("Starting personalized health report tests...")

test_user("raj", 25, ["muscle_pain", "mild_fever"])
test_user("ram", 30, ["headache", "vomiting"])

print("\n--- Testing /users Endpoint ---")
user_res = requests.get(f"{BASE_URL}/users")
if user_res.status_code == 200:
    print("SUCCESS: /users endpoint returned:", user_res.json())
else:
    print("FAILED: /users endpoint returned", user_res.status_code)
