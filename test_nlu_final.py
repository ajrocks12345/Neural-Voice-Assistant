import requests

BASE_URL = "http://127.0.0.1:5000/predict"

tests = [
    ("i have leg pain",        "Should detect muscle_pain"),
    ("i have fever",           "Should detect mild_fever"),
    ("my head hurts so much",  "Should detect headache"),
    ("i cant breathe properly","Should detect breathlessness"),
    ("stomach hurts",          "Should detect stomach_pain"),
    ("i am really scared my chest hurts", "Empathy + chest_pain"),
    ("i have a sore throat and cough",    "Should detect throat + cough"),
    ("i feel very tired and dizzy",       "Should detect fatigue + visual_disturbances"),
    ("i feel bad",             "Vague phrase → ask for more info"),
]

print("=" * 60)
for text, note in tests:
    res = requests.post(BASE_URL, json={"message": text, "history": []})
    data = res.json()
    print(f"\n[INPUT] {text!r}")
    print(f"[NOTE]  {note}")
    print(f"[SYMPTOMS] {data.get('extracted_symptoms', [])}")
    print(f"[STATUS]   {data.get('status')}")
    print(f"[RESPONSE] {data.get('response', '')[:120]}")
print("\n" + "=" * 60)
