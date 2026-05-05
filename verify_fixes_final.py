import requests
import json

url = "http://localhost:5000/predict"

tests = [
    {
        "name": "Extraction: vomitting",
        "payload": {
            "message": "i have vomitting",
            "history": [],
            "user_name": "test",
            "user_age": "25"
        }
    },
    {
        "name": "Suggestions: Low Confidence",
        "payload": {
            "message": "i have fever", # Low confidence/ambiguous
            "history": [],
            "user_name": "test",
            "user_age": "25"
        }
    }
]

for t in tests:
    print(f"\n--- {t['name']} ---")
    try:
        response = requests.post(url, json=t['payload'])
        data = response.json()
        print("Status:", data.get('status'))
        print("Extracted:", data.get('extracted_symptoms'))
        print("Suggestions:", data.get('suggestions'))
        print("Predictions Count:", len(data.get('top_predictions', [])))
    except Exception as e:
        print(f"Error: {e}")
