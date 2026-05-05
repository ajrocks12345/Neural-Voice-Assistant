import requests
import json

url = "http://localhost:5000/predict"
payload = {
    "message": "I have high fever, headache, and pain behind the eyes",
    "history": [],
    "user_name": "test",
    "user_age": "25"
}

try:
    response = requests.post(url, json=payload)
    data = response.json()
    print("Status:", data.get('status'))
    print("Disease:", data.get('disease'))
    print("Description (Root):", data.get('description'))
    print("Symptoms (Root):", data.get('symptoms'))
    
    # Check top_predictions too
    if 'top_predictions' in data and len(data['top_predictions']) > 0:
        print("First Top Prediction Description:", data['top_predictions'][0].get('description'))

except Exception as e:
    print(f"Error connecting to backend: {e}")
