import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from utils import extract_symptoms
import pickle

# Load features
with open('backend/model_data.pkl', 'rb') as f:
    feature_names = pickle.load(f)

tests = [
    "I have toe itching, itching, mild fever, headache, and loss of appetite",
    "itching between toes and my skin is peeling there",
    "I have a high fever, severe headache and some pink spots on my chest"
]

for t in tests:
    print(f"\nText: {t}")
    _, extracted = extract_symptoms(t, feature_names)
    print(f"Extracted: {extracted}")
