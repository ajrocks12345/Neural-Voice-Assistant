import requests
import os
import pandas as pd
import csv

url = "http://localhost:5000/predict"

def test_csv_update():
    print("\n--- Testing CSV Header Update ---")
    
    user_id = "csv_test_user"
    age = "25"
    profile = {
        "gender": "Female",
        "conditions": "Asthma",
        "allergies": "Nuts",
        "lifestyle": "Sedentary",
        "medications": "Inhaler"
    }
    
    log_path = f"logs/{user_id}_{age}_history.csv"
    if os.path.exists(log_path):
        os.remove(log_path) # Start fresh

    # 1. First Prediction (Creates CSV)
    payload = {
        "message": "i am coughing",
        "history": [],
        "user_name": user_id,
        "user_age": age,
        "user_profile": profile
    }
    
    requests.post(url, json=payload)
    
    if os.path.exists(log_path):
        print(f"Success: CSV created at {log_path}")
        with open(log_path, 'r') as f:
            lines = f.readlines()
            # Check headers
            print(f"Row 1: {lines[0].strip()}")
            print(f"Row 3: {lines[2].strip()}")
            print(f"Row 5: {lines[4].strip()}")
            
            if "Gender: Female" in lines[2] and "Allergies: Nuts" in lines[4]:
                print("Success: Profile found in CSV header.")
            else:
                print("Failure: Profile missing or incorrect in CSV header.")
    else:
        print("Failure: CSV not created.")
        return

    # 2. Test Recurring Symptom Detection with new header (skiprows=8)
    # Add TWO manual entries to history to trigger detection (past_count >= 2)
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["2026-03-11 10:00:00", "coughing a lot", "cough", "Asthma", "0.90"])
        writer.writerow(["2026-03-12 09:00:00", "still coughing", "cough", "Cold", "0.80"])

    # Now predict again with cough
    payload["message"] = "i still have a cough"
    payload["history"] = []
    
    r = requests.post(url, json=payload)
    data = r.json()
    resp = data.get('response', '')
    print(f"Bot Response: {resp}")
    if "notice you've experienced cough" in resp:
        print("Success: Recurring symptom detected with new CSV format!")
    else:
        print("Failure: Recurring symptom NOT detected. Check skiprows logic.")

if __name__ == "__main__":
    test_csv_update()
