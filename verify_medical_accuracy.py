import requests
import json

def test_prediction(message, history=[]):
    url = "http://localhost:5000/predict"
    payload = {
        "message": message,
        "history": history,
        "user_name": "test_user",
        "user_age": "25"
    }
    response = requests.post(url, json=payload)
    return response.json()

def main():
    print("--- TEST 1: Ambiguous Input (Toe itching + Systemic symptoms) ---")
    res1 = test_prediction("I have toe itching, itching, mild fever, headache, and loss of appetite")
    print(f"Status: {res1.get('status')}")
    print(f"Top 3 Predictions:")
    for p in res1.get('top_predictions', []):
        print(f"  - {p['disease']}: {p['confidence']:.2f}")
    print(f"Response: {res1.get('response')[:100]}...")

    print("\n--- TEST 2: Strictly Localized (Toe itching + Scaling) ---")
    res2 = test_prediction("itching between toes and my skin is peeling there")
    print(f"Status: {res2.get('status')}")
    print(f"Top 3 Predictions:")
    for p in res2.get('top_predictions', []):
        print(f"  - {p['disease']}: {p['confidence']:.2f}")

    print("\n--- TEST 3: Strictly Systemic (High fever + Rash + Headache) ---")
    res3 = test_prediction("I have a high fever, severe headache and some pink spots on my chest")
    print(f"Status: {res3.get('status')}")
    print(f"Top 3 Predictions:")
    for p in res3.get('top_predictions', []):
        print(f"  - {p['disease']}: {p['confidence']:.2f}")

if __name__ == "__main__":
    main()
