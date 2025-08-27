# usb_sim_manual.py — Manual Approval USB Simulator

import os, json
from pathlib import Path
from utils.feature_map import extract_features
import joblib

# Paths
USB_FOLDER = Path("usb")  # your USB simulation folder
MODEL_PATH = Path("model/model.pkl")

# Load model
model = joblib.load(MODEL_PATH)
print("✅ Model loaded:", MODEL_PATH)

def predict_log(log_dict):
    import numpy as np
    x = extract_features(log_dict)
    if isinstance(x, dict):
        x = np.array(list(x.values())).reshape(1, -1)
    else:
        x = x.reshape(1, -1)
    
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(x)
        if proba_all.shape[1] == 1:
            proba = float(proba_all[0][0])
            pred = 1 if proba >= 0.5 else 0
        else:
            proba = float(proba_all[0][1])
            pred = 1 if proba >= 0.5 else 0
    else:
        pred = model.predict(x)[0]
        proba = 1.0 if pred == 1 else 0.0
    
    return pred, proba

# Scan all files with manual approval
usb_files = list(USB_FOLDER.glob("*.json"))
if not usb_files:
    print("No files found in USB folder.")
else:
    print("🛡️ Air-Gapped USB Simulator Running...")
    for f in usb_files:
        print(f"\nFile detected: {f.name}")
        decision = input("Approve transfer to sandbox? (y/n): ").strip().lower()
        if decision != "y":
            print(f"❌ Transfer denied for {f.name}")
            continue  # skip scanning this file

        # Transfer approved, scan
        log_dict = json.loads(f.read_text())
        pred, proba = predict_log(log_dict)
        verdict = "Malicious" if pred == 1 else "Safe"
        print(f"✅ Scan result: {f.name} → {verdict} ({proba*100:.2f}%)")
        
        # Simulate real-time isolation
        if pred == 1:
            print(f"⚠️ ALERT: {f.name} flagged as malicious! Simulating USB disconnect.")
            break  # stop further file transfers
