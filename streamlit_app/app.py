# streamlit_app/app.py - Air-Gapped USB AI Simulator

import streamlit as st
import json, os, hashlib, joblib
import numpy as np
from pathlib import Path
from datetime import datetime
from utils.feature_map import extract_features, FEATURES

st.set_page_config(page_title="Air-Gapped USB AI Simulator", page_icon="🛡️")

# Paths
MODEL_PATH = Path("model/model.pkl")
DATA_DIR = Path(__file__).parent.parent / "data"
samples = sorted([p.name for p in DATA_DIR.glob("*.json")])
LOG_DIR = Path("../logs"); LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "events.jsonl"
CHAIN_FILE = LOG_DIR / "chain.hash"

# Load model once
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# SHA-256 hashing helper
def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def load_prev_hash():
    if CHAIN_FILE.exists():
        return CHAIN_FILE.read_text().strip()
    return "0"*64  # genesis

def append_log_and_chain(record: dict):
    # Append record to JSONL
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Update hash chain
    prev = load_prev_hash()
    curr_hash = sha256_hex((prev + json.dumps(record, sort_keys=True)).encode("utf-8"))
    CHAIN_FILE.write_text(curr_hash)

# AI prediction
def predict_one(log_dict):
    x = extract_features(log_dict).reshape(1, -1)
    
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(x)
        # If only one class in model, assign 0 or 1 probability
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


st.title("🛡️ Air-Gapped USB Malware Detection — Online Simulation")

# USB simulation sidebar
samples = sorted([p.name for p in DATA_DIR.glob("*.json")])
st.sidebar.header("USB Controller (Simulated)")
choice = st.sidebar.selectbox("Insert USB & pick sample", samples)
multi = st.sidebar.checkbox("Multi-USB (select multiple)")
if multi:
    choice_multi = st.sidebar.multiselect("Pick multiple samples", samples, default=samples[:2])

st.subheader("USB Firewall ➜ Sandbox ➜ AI Verdict ➜ Secure Log")

def run_flow(sample_name):
    p = DATA_DIR / sample_name
    log_dict = json.loads(p.read_text())
    # Hash original log
    raw_hash = sha256_hex(json.dumps(log_dict, sort_keys=True).encode("utf-8"))

    # AI inference
    pred, proba = predict_one(log_dict)
    verdict = "Malicious" if pred == 1 else "Safe"
    action = "USB power cut (simulated)" if pred == 1 else "Transfer allowed"

    # Tamper-evident record
    record = {
        "ts": datetime.utcnow().isoformat()+"Z",
        "sample": sample_name,
        "verdict": verdict,
        "confidence": round(proba, 3),
        "raw_log_sha256": raw_hash
    }
    append_log_and_chain(record)
    return log_dict, verdict, proba, action

col1, col2 = st.columns(2)

with col1:
    if not multi:
        if st.button("▶️ Scan Selected USB"):
            d, verdict, proba, action = run_flow(choice)
            st.success(f"Verdict: {verdict} (confidence {proba:.2f})")
            st.info(f"Action: {action}")
            with st.expander("Behavior summary (from sandbox log)"):
                st.json(d)
    else:
        if st.button("▶️ Scan All Selected"):
            rows = []
            for s in choice_multi:
                d, verdict, proba, action = run_flow(s)
                rows.append((s, verdict, f"{proba:.2f}", action))
            st.table(rows)

with col2:
    st.markdown("**Tamper-evident log**")
    if LOG_FILE.exists():
        st.download_button("⬇️ Download events.jsonl", data=LOG_FILE.read_bytes(), file_name="events.jsonl")
    if CHAIN_FILE.exists():
        st.download_button("⬇️ Download chain.hash", data=CHAIN_FILE.read_bytes(), file_name="chain.hash")
    st.caption("Each record is chained via SHA-256 (prev_hash + record). Any edit breaks the chain.")
