import requests
import json

url = "http://localhost:5000/predict"

def test_interaction(msg, history, name):
    print(f"\n--- Testing: {name} ---")
    payload = {
        "message": msg,
        "history": history,
        "user_name": "test_loop_user",
        "user_age": "25"
    }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        print(f"Input: {msg}")
        print(f"Status: {data.get('status')}")
        print(f"Response: {data.get('response')}")
        return data
    except Exception as e:
        print(f"Error: {e}")

# 1. Start with ambiguous "fever"
d1 = test_interaction("i have fever", [], "Initial Ambiguity")

# 2. Respond with "mild fever" (history should be empty or contain fever if we wanted, but app.py uses history)
# In app.py: total_symptoms = list(set(history + new_symptoms))
# already_clarified = any(opt in total_symptoms for opt in specific_options)
# If user selects "mild fever", extraction should catch it.
test_interaction("mild fever", [], "Clarification Response")
