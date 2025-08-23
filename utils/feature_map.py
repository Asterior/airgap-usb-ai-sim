# utils/feature_map.py
import numpy as np

FEATURES = [
    "size_kb", "entropy",
    "api_CreateFile", "api_ReadFile", "api_WriteFile", "api_CryptEncrypt",
    "fs_files_created", "fs_files_renamed"
]

def extract_features(log_dict):
    api = log_dict.get("api_calls", {})
    fs = log_dict.get("fs_activity", {})

    vec = [
        float(log_dict.get("size_kb", 0)),
        float(log_dict.get("entropy", 0)),
        float(api.get("CreateFile", 0)),
        float(api.get("ReadFile", 0)),
        float(api.get("WriteFile", 0)),
        float(api.get("CryptEncrypt", 0)),
        float(fs.get("files_created", 0)),
        float(fs.get("files_renamed", 0)),
    ]
    return np.array(vec, dtype=float)

def label_to_int(lbl):
    return 1 if str(lbl).lower() == "malicious" else 0
