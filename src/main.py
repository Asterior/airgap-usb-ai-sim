# main.py - Airgap USB AI Simulator (JSON sandbox logs)

import os, json
import uuid
from datetime import datetime
import hashlib

DATA_DIR = os.path.join("data", "usb")

LOG_FILE = os.path.join("logs", "events.jsonl")
CHAIN_FILE = os.path.join("logs", "chain.hash")

def compute_entry_hash(entry, prev_hash):
    data = f"{entry['id']}{entry['ts']}{entry['file_name']}{entry['verdict']}{entry.get('confidence', '')}{prev_hash or ''}"
    return hashlib.sha256(data.encode()).hexdigest()

def append_log_entry(entry, prev_hash):
    entry['prev_hash'] = prev_hash or ''
    entry['hash'] = compute_entry_hash(entry, prev_hash)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    with open(CHAIN_FILE, "w") as f:
        f.write(entry['hash'])

def load_sandbox_logs():
    logs = []
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".json"):
            path = os.path.join(DATA_DIR, fname)
            with open(path, "r") as f:
                d = json.load(f)
                logs.append(d)
    return logs

def simulate_sandbox():
    print("🧪 Sandbox Behavior Analysis")
    logs = load_sandbox_logs()

    if not logs:
        print("⚠️ No sandbox logs found in data/")
        return

    # Get previous hash
    prev_hash = None
    if os.path.exists(CHAIN_FILE):
        with open(CHAIN_FILE) as f:
            prev_hash = f.read().strip() or None

    for log in logs:
        verdict = "Safe" if log.get("label") == "benign" else "Malicious"
        file_name = log.get('file_name') or log.get('sample') or 'unknown'
        print(f"  → {file_name} : {'✅' if verdict=='Safe' else '⚠️'} {verdict}")
        entry = {
            "id": str(uuid.uuid4()),
            "ts": datetime.utcnow().isoformat() + "Z",
            "file_name": file_name,
            "verdict": verdict,
            "confidence": log.get("confidence", 1.0)
        }
        append_log_entry(entry, prev_hash)
        prev_hash = entry['hash']

def main():
    print("=== Airgap USB AI Simulator ===")
    simulate_sandbox()

if __name__ == "__main__":
    main()
