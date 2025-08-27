# log_integrity_checker.py
"""
Log Integrity Checker
---------------------
Verifies the integrity of the hash-chained log file (events.jsonl + chain.hash).
Checks that each log entry's hash matches the expected chain.
"""
import json
import hashlib
import os
from datetime import datetime

LOG_FILE = os.path.join("logs", "events.jsonl")
CHAIN_FILE = os.path.join("logs", "chain.hash")


def compute_entry_hash(entry, prev_hash):
    # Use id, timestamp, sample/file_name, verdict, confidence, prev_hash
    file_name = entry.get('sample') or entry.get('file_name', '')
    data = f"{entry['id']}{entry['ts']}{file_name}{entry['verdict']}{entry.get('confidence', '')}{prev_hash or ''}"
    return hashlib.sha256(data.encode()).hexdigest()


def verify_log_chain():
    with open(LOG_FILE) as f:
        logs = [json.loads(line) for line in f if line.strip()]
    with open(CHAIN_FILE) as f:
        chain_hash = f.read().strip()
    prev_hash = None
    for entry in logs:
        computed_hash = compute_entry_hash(entry, prev_hash)
        # Check prev_hash field
        if prev_hash and entry.get('prev_hash', '') != prev_hash:
            print(f"Chain broken at entry {entry.get('id', 'unknown')}")
            return False
        # Check hash field
        if entry.get('hash', '') != computed_hash:
            print(f"Hash mismatch at entry {entry.get('id', 'unknown')}")
            return False
        prev_hash = computed_hash
    if prev_hash != chain_hash:
        print("Final chain hash mismatch!")
        return False
    print("Log integrity verified.")
    return True

if __name__ == "__main__":
    verify_log_chain()
