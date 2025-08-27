

# streamlit_app/app.py - Air-Gapped USB AI Simulator with Visualizations
import sys, os
sys.path.insert(0, os.path.abspath(os.getcwd()))

import streamlit as st
import json, os, hashlib, joblib
import numpy as np
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from utils.feature_map import extract_features, FEATURES

st.set_page_config(
    page_title="Air-Gapped USB Malware Simulator",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
MODEL_PATH = Path("model/model.pkl")
DATA_DIR = Path(__file__).parent.parent / "data" / "usb"
LOG_DIR = Path(os.path.abspath(os.path.join(os.getcwd(), "logs")))
LOG_DIR.mkdir(exist_ok=True)
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
    import uuid
    # Get previous hash
    prev = load_prev_hash()
    record['id'] = str(uuid.uuid4())
    record['prev_hash'] = prev or ''
    # For compatibility, use sample or file_name
    file_name = record.get('sample') or record.get('file_name', '')
    # Compute hash
    data = f"{record['id']}{record['ts']}{file_name}{record['verdict']}{record.get('confidence', '')}{prev or ''}"
    record['hash'] = sha256_hex(data.encode('utf-8'))
    # Append record to JSONL
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Update hash chain
    CHAIN_FILE.write_text(record['hash'])

# AI prediction
def predict_one(log_dict):
    x = extract_features(log_dict).reshape(1, -1)

    
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(x)
        # Handle single-class model
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

# Visualization: Feature Importance
def show_feature_importance():
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({"Feature": FEATURES, "Importance": importances})
        df = df.sort_values("Importance", ascending=False)
        st.subheader("📊 Feature Importance (RandomForest)")
        fig, ax = plt.subplots()
        ax.barh(df["Feature"], df["Importance"])
        ax.invert_yaxis()
        st.pyplot(fig)

# Visualization: Event timeline
def show_timeline(log_file=LOG_FILE):
    if log_file.exists():
        df = pd.read_json(log_file, lines=True)
        df['ts'] = pd.to_datetime(df['ts'])
        st.subheader("⏱️ Event Timeline")
        st.line_chart(df.set_index('ts')['confidence'])

# --- Streamlit Layout ---
st.title("🛡️ Air-Gapped USB Malware Detection — Online Simulation")

# Sidebar
st.sidebar.header("📂 USB Controller (Simulated)")
samples = sorted([p.name for p in DATA_DIR.glob("*.json")])
choice = st.sidebar.selectbox("Select a USB sample", samples)
multi = st.sidebar.checkbox("Enable Multi-USB Scan")
if multi:
    choice_multi = st.sidebar.multiselect("Select multiple samples", samples, default=samples[:2])

st.subheader("USB Firewall ➜ Sandbox ➜ AI Verdict ➜ Secure Log")

def run_flow(sample_name):
    p = DATA_DIR / sample_name
    log_dict = json.loads(p.read_text())
    raw_hash = sha256_hex(json.dumps(log_dict, sort_keys=True).encode("utf-8"))
    pred, proba = predict_one(log_dict)
    verdict = "Malicious" if pred == 1 else "Safe"
    action = "USB power cut (simulated)" if pred == 1 else "Transfer allowed"

    record = {
        "ts": datetime.utcnow().isoformat()+"Z",
        "sample": sample_name,
        "verdict": verdict,
        "confidence": round(proba, 3),
        "raw_log_sha256": raw_hash
    }
    append_log_and_chain(record)
    return log_dict, verdict, proba, action

def load_event_logs():
    if LOG_FILE.exists():
        records = []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))
        return pd.DataFrame(records)
    return pd.DataFrame()


col1, col2 = st.columns(2)

with col1:
    if not multi:
        if st.button("▶️ Scan Selected USB"):
            d, verdict, proba, action = run_flow(choice)
            st.success(f"Verdict: {verdict} (confidence {proba:.2f})")
            st.info(f"Action: {action}")
            with st.expander("Behavior summary (from sandbox log)"):
                st.json(d)
            show_feature_importance()
            show_timeline()
    else:
        if st.button("▶️ Scan All Selected"):
            rows = []
            for s in choice_multi:
                d, verdict, proba, action = run_flow(s)
                rows.append((s, verdict, f"{proba:.2f}", action))
            st.table(rows)
            show_feature_importance()
            show_timeline()

with col2:
    st.markdown("**Tamper-evident log**")
    if LOG_FILE.exists():
        st.download_button("⬇️ Download events.jsonl", data=LOG_FILE.read_bytes(), file_name="events.jsonl")
    if CHAIN_FILE.exists():
        st.download_button("⬇️ Download chain.hash", data=CHAIN_FILE.read_bytes(), file_name="chain.hash")
    st.caption("Each record is chained via SHA-256 (prev_hash + record). Any edit breaks the chain.")

    st.subheader("✅ Log Integrity Checker")
    if st.button("Run Integrity Check"):
        import subprocess
        result = subprocess.run(["python", "utils/log_integrity_checker.py"], capture_output=True, text=True)
        st.code(result.stdout or result.stderr)

    st.subheader("⏱️ USB Event Timeline")
    df_logs = load_event_logs()
if not df_logs.empty:
    # Convert timestamp to datetime
    df_logs["ts"] = pd.to_datetime(df_logs["ts"])
    
    timeline = alt.Chart(df_logs).mark_circle(size=100).encode(
        x="ts:T",
        y=alt.Y("sample:N", title="USB Sample"),
        color=alt.Color("verdict:N", scale=alt.Scale(domain=["Malicious","Safe"], range=["red","green"])),
        tooltip=["ts:T", "sample:N", "verdict:N", "confidence:Q"]
    ).properties(
        width=400,
        height=300,
        title="Timeline of USB Scan Events"
    )
    
    st.altair_chart(timeline)
else:
    st.info("No scan events yet to display in the timeline.")

