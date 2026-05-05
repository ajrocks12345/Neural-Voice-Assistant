import requests
import json

url = "http://localhost:5000/predict"

def test_interaction(msg, history, name):
    print(f"\n--- Testing: {name} ---")
    payload = {
        "message": msg,
        "history": history,
        "user_name": "test_user",
        "user_age": "25"
    }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        print(f"Input: {msg}")
        print(f"Status: {data.get('status')}")
        print(f"Response: {data.get('response')}")
        if 'suggestions' in data:
            print(f"Suggestions: {data.get('suggestions')}")
        return data
    except Exception as e:
        print(f"Error: {e}")

# Test Non-Ambiguous Single Symptom
test_interaction("i have a cough", [], "Cough Flow")
