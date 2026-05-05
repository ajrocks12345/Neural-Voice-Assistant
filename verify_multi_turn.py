import requests
import json

def test_multi_turn():
    url = "http://localhost:5000/predict"
    
    print("--- Turn 1: Single Symptom ---")
    payload1 = {"message": "i have a fever", "history": []}
    r1 = requests.post(url, json=payload1)
    res1 = r1.json()
    print(f"Status: {res1.get('status')}")
    print(f"Response: {res1.get('response')}")
    print(f"Symptoms so far: {res1.get('extracted_symptoms')}")
    
    history1 = res1.get('extracted_symptoms', [])
    
    print("\n--- Turn 2: Second Symptom ---")
    payload2 = {"message": "i have a cough", "history": history1}
    r2 = requests.post(url, json=payload2)
    res2 = r2.json()
    print(f"Status: {res2.get('status')}")
    print(f"Response: {res2.get('response')}")
    print(f"Symptoms so far: {res2.get('extracted_symptoms')}")
    
    history2 = res2.get('extracted_symptoms', [])
    
    print("\n--- Turn 3: Third Symptom (Should trigger diagnosis) ---")
    payload3 = {"message": "and my body aches", "history": history2}
    r3 = requests.post(url, json=payload3)
    res3 = r3.json()
    print(f"Status: {res3.get('status')}")
    print(f"Response: {res3.get('response')}")
    if res3.get('disease'):
        print(f"Diagnosis: {res3.get('disease')} ({res3.get('confidence')*100:.1f}%)")

if __name__ == "__main__":
    test_multi_turn()
