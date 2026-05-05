import requests

def verify_expansion():
    url = "http://localhost:5000/predict"
    
    test_cases = [
        {
            "name": "Appendicitis",
            "message": "i have pain in my lower right stomach and feel like vomiting",
            "expected_symptoms": ["abdominal_pain_lower_right", "vomiting"]
        },
        {
            "name": "UTI",
            "message": "it burns when i pee and i have to go very often",
            "expected_symptoms": ["burning_urination", "frequent_urination"]
        },
        {
            "name": "Anemia",
            "message": "i feel very tired and my skin looks pale",
            "expected_symptoms": ["fatigue", "pale_skin"]
        },
        {
            "name": "Typhoid",
            "message": "i have a high fever and my stomach hurts bad",
            "expected_symptoms": ["high_fever", "stomach_pain"]
        },
        {
            "name": "Gout",
            "message": "my big toe is red and hurts a lot",
            "expected_symptoms": ["toe_redness"]
        }
    ]
    
    for test in test_cases:
        print(f"--- Testing: {test['name']} ---")
        try:
            r = requests.post(url, json={"message": test['message'], "history": []})
            res = r.json()
            print(f"Response: {res.get('response')}")
            print(f"Extracted: {res.get('extracted_symptoms')}")
            print(f"Status: {res.get('status')}")
            if res.get('suggestions'):
                print(f"Suggestions: {res.get('suggestions')}")
        except Exception as e:
            print(f"Error testing {test['name']}: {e}")
        print()

if __name__ == "__main__":
    verify_expansion()
