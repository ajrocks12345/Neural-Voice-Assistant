import requests
import json
import os

url = "http://localhost:5000/predict"

def test_personalization():
    print("\n--- Testing Personalization & Adaptive Questioning ---")
    
    # 1. New User Onboarding (Simulated via payload)
    user_id = "test_perso_user"
    age = "30"
    profile = {
        "gender": "Male",
        "conditions": "Heart problems",
        "allergies": "Aspirin",
        "lifestyle": "Smoking",
        "medications": "None"
    }

    # 2. Test Adaptive Questioning: User has "Heart problems" and mentions "chest pain"
    print("\nStep 1: Mentioning Chest Pain (Adaptive Trigger)")
    payload = {
        "message": "I have chest pain",
        "history": [],
        "user_name": user_id,
        "user_age": age,
        "user_profile": profile
    }
    
    r = requests.post(url, json=payload)
    data = r.json()
    print(f"Bot Response: {data.get('response')}")
    print(f"Suggestions (should include adaptive ones like 'shortness of breath'): {data.get('suggestions')}")
    
    # 3. Test Diagnosis with Personalized Advice
    print("\nStep 2: Adding more symptoms for diagnosis")
    payload["message"] = "I also have coughing and shortness of breath"
    payload["history"] = ["chest_pain"]
    payload["force"] = True # Force diagnosis to see advice
    
    r = requests.post(url, json=payload)
    data = r.json()
    print(f"Bot Response Summary: {data.get('response')[:200]}...")
    
    # Check if advice is present
    resp = data.get('response', '')
    if "Allergy Warning" in resp:
        print("Success: Allergy Warning (Aspirin) found in response.")
    if "Lifestyle Note" in resp:
        print("Success: Lifestyle Note (Smoking) found in response.")

    # 4. Test Recurring Symptom Detection
    # To trigger recurring detection, we need to create a log with 2+ previous mentions of a symptom
    print("\nStep 3: Simulating previous sessions for Recurring Symptom Detection")
    log_path = f"logs/{user_id}_{age}_history.csv"
    with open(log_path, 'w') as f:
        f.write(f"Name: {user_id.capitalize()}\n")
        f.write(f"Age: {age}\n")
        f.write("\n")
        f.write("Date,User_Input,Extracted_Symptoms,Predicted_Disease,Confidence\n")
        f.write("2026-01-01 10:00:00,i have headache,headache,Migraine,0.90\n")
        f.write("2026-02-01 10:00:00,my head hurts again,headache,Migraine,0.85\n")

    # Now test with headache again
    payload["message"] = "my head is aching"
    payload["history"] = []
    payload["force"] = False

    r = requests.post(url, json=payload)
    data = r.json()
    resp = data.get('response', '')
    print(f"Bot Response with History: {resp}")
    if "notice you've experienced headache" in resp:
        print("Success: Recurring symptom (headache) detected and mentioned!")
    else:
        print("Failure: Recurring symptom not mentioned.")

    # 5. Profile Persistence Check
    profile_path = f"logs/{user_id}_{age}_profile.json"
    if os.path.exists(profile_path):
        print(f"Success: Profile JSON correctly saved at {profile_path}")

if __name__ == "__main__":
    test_personalization()
