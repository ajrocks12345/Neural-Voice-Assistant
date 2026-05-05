import requests

def test_refined_suggestions():
    url = "http://localhost:5000/predict"
    
    print("--- Testing: Relevant Suggestions (Eye Pain) ---")
    r = requests.post(url, json={"message": "i have eye pain", "history": []})
    data = r.json()
    print(f"Response: {data.get('response')}")
    print(f"Extracted: {data.get('extracted_symptoms')}")
    print(f"Suggestions: {data.get('suggestions')}")
    
    print("\n--- Testing: None of These ---")
    r = requests.post(url, json={"none_of_these": True, "history": ["eye_pain"]})
    data = r.json()
    print(f"Response: {data.get('response')}")
    print(f"Status: {data.get('status')}")

if __name__ == "__main__":
    test_refined_suggestions()
