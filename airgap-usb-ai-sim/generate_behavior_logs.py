import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


RANDOM_SEED = 42


@dataclass
class BehaviorLog:
    timestamp: int
    device_id: str
    file_ops: int
    process_creations: int
    registry_edits: int
    network_calls: int
    avg_entropy: float
    autorun_present: int
    macro_exec: int
    signed_binary: int
    sandbox_score: float
    label: int  # 0 = benign, 1 = malicious


def _sample_benign(n: int, rng: np.random.Generator) -> pd.DataFrame:
    file_ops = rng.poisson(lam=10, size=n)
    process_creations = rng.poisson(lam=1.2, size=n)
    registry_edits = rng.poisson(lam=0.6, size=n)
    network_calls = rng.poisson(lam=3, size=n)
    avg_entropy = np.clip(rng.normal(loc=5.2, scale=0.6, size=n), 0.0, 8.0)
    autorun_present = rng.binomial(1, 0.02, size=n)
    macro_exec = rng.binomial(1, 0.03, size=n)
    signed_binary = rng.binomial(1, 0.85, size=n)
    sandbox_score = np.clip(
        0.15
        + 0.02 * file_ops
        + 0.05 * process_creations
        + 0.06 * registry_edits
        + 0.03 * network_calls
        - 0.04 * signed_binary
        + rng.normal(0, 0.15, size=n),
        0.0,
        1.0,
    )
    label = np.zeros(n, dtype=int)
    return pd.DataFrame(
        {
            "file_ops": file_ops,
            "process_creations": process_creations,
            "registry_edits": registry_edits,
            "network_calls": network_calls,
            "avg_entropy": avg_entropy,
            "autorun_present": autorun_present,
            "macro_exec": macro_exec,
            "signed_binary": signed_binary,
            "sandbox_score": sandbox_score,
            "label": label,
        }
    )


def _sample_malicious(n: int, rng: np.random.Generator) -> pd.DataFrame:
    file_ops = rng.poisson(lam=80, size=n)
    process_creations = rng.poisson(lam=6.5, size=n)
    registry_edits = rng.poisson(lam=5.0, size=n)
    network_calls = rng.poisson(lam=15, size=n)
    avg_entropy = np.clip(rng.normal(loc=6.7, scale=0.7, size=n), 0.0, 8.0)
    autorun_present = rng.binomial(1, 0.55, size=n)
    macro_exec = rng.binomial(1, 0.35, size=n)
    signed_binary = rng.binomial(1, 0.15, size=n)
    sandbox_score = np.clip(
        0.55
        + 0.004 * file_ops
        + 0.05 * process_creations
        + 0.06 * registry_edits
        + 0.03 * network_calls
        + 0.06 * autorun_present
        + 0.08 * macro_exec
        - 0.08 * signed_binary
        + rng.normal(0, 0.12, size=n),
        0.0,
        1.0,
    )
    label = np.ones(n, dtype=int)
    return pd.DataFrame(
        {
            "file_ops": file_ops,
            "process_creations": process_creations,
            "registry_edits": registry_edits,
            "network_calls": network_calls,
            "avg_entropy": avg_entropy,
            "autorun_present": autorun_present,
            "macro_exec": macro_exec,
            "signed_binary": signed_binary,
            "sandbox_score": sandbox_score,
            "label": label,
        }
    )


def generate_dataset(
    num_rows: int,
    benign_ratio: float,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    num_benign = int(num_rows * benign_ratio)
    num_malicious = num_rows - num_benign
    benign_df = _sample_benign(num_benign, rng)
    malicious_df = _sample_malicious(num_malicious, rng)
    df = pd.concat([benign_df, malicious_df], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    # Add synthetic ids and timestamps
    df["timestamp"] = np.arange(len(df)) + rng.integers(1_700_000_000, 1_800_000_000)
    df["device_id"] = [f"USB-{rng.integers(1000, 9999)}" for _ in range(len(df))]
    cols = [
        "timestamp",
        "device_id",
        "file_ops",
        "process_creations",
        "registry_edits",
        "network_calls",
        "avg_entropy",
        "autorun_present",
        "macro_exec",
        "signed_binary",
        "sandbox_score",
        "label",
    ]
    return df[cols]


def save_jsonl(df: pd.DataFrame, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")


def maybe_make_dirs(*paths: str) -> None:
    for p in paths:
        os.makedirs(p, exist_ok=True)


def split_train_test(df: pd.DataFrame, test_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_idx = int(len(df_shuffled) * (1 - test_ratio))
    return df_shuffled.iloc[:split_idx].copy(), df_shuffled.iloc[split_idx:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic simulated USB behavior logs")
    parser.add_argument("--num_rows", type=int, default=5000, help="Total number of rows to generate")
    parser.add_argument("--benign_ratio", type=float, default=0.6, help="Proportion of benign samples")
    parser.add_argument("--out_dir", type=str, default="data", help="Output directory root")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test split ratio")
    args = parser.parse_args()

    maybe_make_dirs(args.out_dir)
    df = generate_dataset(args.num_rows, args.benign_ratio, seed=args.seed)

    csv_path = os.path.join(args.out_dir, "behavior_logs.csv")
    jsonl_path = os.path.join(args.out_dir, "behavior_logs.jsonl")
    df.to_csv(csv_path, index=False)
    save_jsonl(df, jsonl_path)

    train_df, test_df = split_train_test(df, test_ratio=args.test_ratio, seed=args.seed)
    train_csv = os.path.join(args.out_dir, "behavior_train.csv")
    test_csv = os.path.join(args.out_dir, "behavior_test.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Wrote {len(df)} rows to {csv_path} and {jsonl_path}")
    print(f"Train/Test split: {len(train_df)}/{len(test_df)} -> {train_csv}, {test_csv}")


if __name__ == "__main__":
    main()


