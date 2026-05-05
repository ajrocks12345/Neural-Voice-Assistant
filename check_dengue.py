import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from utils import DISEASE_DESCRIPTIONS, DISEASE_PROFILES

d = "Dengue Fever"
print(f"Checking '{d}':")
print(f"  In Profiles: {d in DISEASE_PROFILES}")
print(f"  In Descriptions: {d in DISEASE_DESCRIPTIONS}")
print(f"  Description: {DISEASE_DESCRIPTIONS.get(d)}")

# Print all keys to look for tiny mismatches
print("\nFirst 5 Profile Keys:", list(DISEASE_PROFILES.keys())[:5])
print("First 5 Description Keys:", list(DISEASE_DESCRIPTIONS.keys())[:5])

# Find index of Dengue Fever
try:
    idx = list(DISEASE_PROFILES.keys()).index(d)
    print(f"\nIndex of '{d}' in Profiles: {idx}")
except:
    print(f"\n'{d}' NOT FOUND in Profiles")

try:
    idx_d = list(DISEASE_DESCRIPTIONS.keys()).index(d)
    print(f"Index of '{d}' in Descriptions: {idx_d}")
except:
    print(f"'{d}' NOT FOUND in Descriptions")

# Check for hidden characters
if d in DISEASE_DESCRIPTIONS:
    print(f"\nExact match for '{d}' found.")
else:
    print("\nNO EXACT MATCH. Looking for fuzzy match...")
    for k in DISEASE_DESCRIPTIONS.keys():
        if d.lower() in k.lower():
            print(f"  Found close match: '{k}' (hex: {k.encode().hex()})")
    print(f"  Search string '{d}' hex: {d.encode().hex()}")
