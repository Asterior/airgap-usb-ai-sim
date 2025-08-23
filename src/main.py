# main.py - Airgap USB AI Simulator (JSON sandbox logs)

import os, json

DATA_DIR = "../data"

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

    for log in logs:
        verdict = "✅ Safe" if log.get("label") == "benign" else "⚠️ Malicious"
        print(f"  → {log['file_name']} : {verdict}")

def main():
    print("=== Airgap USB AI Simulator ===")
    simulate_sandbox()

if __name__ == "__main__":
    main()
