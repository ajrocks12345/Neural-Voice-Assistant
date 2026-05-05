import requests
import json

url = "http://localhost:5000/predict"

def test_interaction(msg, history, name):
    print(f"\n--- Testing: {name} ---")
    payload = {
        "message": msg,
        "history": history,
        "user_name": "test_ambig_user",
        "user_age": "25"
    }
    try:
        r = requests.post(url, json=payload)
        data = r.json()
        print(f"Input: {msg}")
        print(f"Status: {data.get('status')}")
        print(f"Response: {data.get('response')[:100]}...")
        if 'top_predictions' in data and data['top_predictions']:
            p1 = data['top_predictions'][0]
            print(f"Top 1: {p1['disease']} ({p1['confidence']:.4f})")
            if len(data['top_predictions']) > 1:
                p2 = data['top_predictions'][1]
                print(f"Top 2: {p2['disease']} ({p2['confidence']:.4f})")
                print(f"Gap: {p1['confidence'] - p2['confidence']:.4f}")
        return data
    except Exception as e:
        print(f"Error: {e}")

# Simulate the user's flow
# 1. Start with throat irritation and mild fever
hist1 = ["throat_irritation", "mild_fever"]
test_interaction("I have throat irritation", ["mild_fever"], "Initial Ambiguity Check")

# 2. Add headache
test_interaction("I have headache", ["mild_fever", "throat_irritation"], "Adding Headache")
