import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from utils import DISEASE_PROFILES, DISEASE_DESCRIPTIONS, get_precautions

print(f"Total Diseases in profiles: {len(DISEASE_PROFILES)}")
print(f"Total Diseases in descriptions: {len(DISEASE_DESCRIPTIONS)}")

missing_desc = []
missing_prec = []

for disease in DISEASE_PROFILES.keys():
    if disease not in DISEASE_DESCRIPTIONS:
        missing_desc.append(disease)
    
    # Test get_precautions
    # We check if the result is the default message
    p = get_precautions(disease)
    if "Please consult a medical professional" in p[0]:
        missing_prec.append(disease)

print(f"\nMissing Descriptions ({len(missing_desc)}): {missing_desc}")
print(f"Missing Precautions ({len(missing_prec)}): {missing_prec}")

# Check for mismatches (extra entries in desc/prec that aren't in profiles)
extra_desc = [d for d in DISEASE_DESCRIPTIONS.keys() if d not in DISEASE_PROFILES]
print(f"\nExtra Descriptions ({len(extra_desc)}): {extra_desc}")
