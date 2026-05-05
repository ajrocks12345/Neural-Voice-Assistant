import sys
import os
import pickle
import numpy as np

# Add backend to path
sys.path.append(os.path.abspath('backend'))
from utils import extract_symptoms

def test_new_aiments():
    try:
        with open('backend/model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('backend/model_data.pkl', 'rb') as f:
            feature_names = pickle.load(f)
    except Exception as e:
        print(f"Error: {e}")
        return

    test_cases = [
        "i have ear pain and difficulty hearing",
        "i am constipated and my abdomen is bloating",
        "i have blurred vision and dry eyes from looking at screens",
        "i have a sore throat and it hurts when I swallow",
        "i feel very hot and dizzy after being in the sun",
        "i have hard stool and its difficult to pass"
    ]

    for phrase in test_cases:
        print(f"\nUser: {phrase}")
        input_vector, extracted = extract_symptoms(phrase, feature_names)
        print(f"Extracted: {extracted}")
        
        probs = model.predict_proba(input_vector)[0]
        diseases = model.classes_
        results = sorted(zip(diseases, probs), key=lambda x: x[1], reverse=True)
        
        print("Top 3 Predictions:")
        for d, p in results[:3]:
            print(f" - {d}: {p:.2%}")

if __name__ == "__main__":
    test_new_aiments()
