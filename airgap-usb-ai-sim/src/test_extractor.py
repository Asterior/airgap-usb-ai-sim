# test_extractor.py - test utils/feature_map.py
import sys, os
sys.path.append(os.path.abspath(".."))

import json

from utils.feature_map import extract_features, label_to_int, FEATURES

# Load a sample JSON
with open("../data/ransomware_sample.json") as f:
    log = json.load(f)

# Extract features
x = extract_features(log)
y = label_to_int(log["label"])

print("Features extracted:")
for name, val in zip(FEATURES, x):
    print(f"  {name}: {val}")

print("Label:", y)
