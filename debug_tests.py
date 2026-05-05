import requests

def test_debug():
    url = "http://localhost:5000/predict"
    
    test_cases = [
        {
            "name": "Typo Fever",
            "message": "i thin i have fever",
            "expected_contains": "mild_fever"
        },
        {
            "name": "No symptoms identified",
            "message": "i dont think i have any symptoms",
            "check_suggestions": True
        }
    ]
    
    for test in test_cases:
        print(f"--- Testing: {test['name']} ---")
        try:
            r = requests.post(url, json={"message": test['message'], "history": []})
            data = r.json()
            print(f"Response: {data.get('response')}")
            print(f"Extracted: {data.get('extracted_symptoms')}")
            if "check_suggestions" in test:
                print(f"Suggestions: {data.get('suggestions')}")
        except Exception as e:
            print(f"Error: {e}")
        print()

if __name__ == "__main__":
    test_debug()
