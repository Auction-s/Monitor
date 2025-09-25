#!/usr/bin/env python3
# src/make_demo_data.py
"""
Create a demo dataset for scoring / demo purposes.

- Reads:  data/flows_features.csv
- Writes:  data/demo_flows.csv
- Behavior:
    * Samples `sample_size` rows (random)
    * Perturbs `perturb_frac` fraction of those rows using small/medium perturbations
      (packet counts multiplicative jitter, duration noise, swap ratios, protocol flips)
    * Adds a column `__perturbed` (0/1) so you can see which rows were changed
    * Keeps original attack_label (so evaluation still meaningful)
"""

import os
import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
ROOT = os.path.expanduser("~")
PROJECT_DATA_DIR = os.path.join(ROOT, "Documents", "Projects", "Bridge", "AI", "Anomaly", "data")

INPUT_CSV = os.path.join(PROJECT_DATA_DIR, "flows_features.csv")   # source (engineered) dataset
OUTPUT_CSV = os.path.join(PROJECT_DATA_DIR, "demo_flows.csv")     # demo output

SAMPLE_SIZE = 5000          # number of demo rows to produce
PERTURB_FRAC = 0.20         # fraction of sampled rows to perturb (e.g. 0.2 -> 20% perturbed)
RANDOM_SEED = 42
# ----------------------------------------

np.random.seed(RANDOM_SEED)

def safe_load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    return pd.read_csv(path)

def choose_numeric_cols(df):
    # heuristic list of columns we expect to perturb; fall back to numeric detection
    candidates = [
        "flow_duration",
        "total_fwd_packets",
        "total_backward_packets",
        "total_length_fwd_packets",
        "total_length_bwd_packets",
        "bytes_per_sec",
        "pkts_per_sec",
        "fwd_to_bwd_ratio"
    ]
    present = [c for c in candidates if c in df.columns]
    if present:
        return present
    # fallback to numeric-only columns (exclude label)
    return [c for c in df.columns if c not in ("attack_label", "is_attack") and np.issubdtype(df[c].dtype, np.number)]

def perturb_rows(df, idx_to_perturb):
    df = df.copy()
    numeric_cols = choose_numeric_cols(df)
    n = len(idx_to_perturb)

    # 1) Multiplicative jitter on packet counts and lengths (moderate)
    pkt_cols = [c for c in ["total_fwd_packets", "total_backward_packets", "total_length_fwd_packets", "total_length_bwd_packets"] if c in df.columns]
    if pkt_cols:
        # factors between 0.5x and 3.0x (some big anomalies)
        factors = np.random.uniform(0.5, 3.0, size=(n, len(pkt_cols)))
        for i, col in enumerate(pkt_cols):
            df.loc[idx_to_perturb, col] = (df.loc[idx_to_perturb, col].astype(float) * factors[:, i]).round().astype(int)

    # 2) Additive noise to flow_duration (can create tiny or huge durations)
    if "flow_duration" in df.columns:
        noise = np.random.normal(loc=0.0, scale=0.2, size=n)  # relative noise
        orig = df.loc[idx_to_perturb, "flow_duration"].astype(float).values
        # inject occasional large jumps
        large_mask = np.random.rand(n) < 0.05
        large_noise = np.where(large_mask, orig * np.random.uniform(5.0, 20.0, size=n), orig * noise)
        new_dur = np.clip(orig + large_noise, a_min=1.0, a_max=None)
        df.loc[idx_to_perturb, "flow_duration"] = new_dur.astype(int)

    # 3) Recompute derived features if present (bytes_per_sec, pkts_per_sec, fwd_to_bwd_ratio)
    if "bytes_per_sec" in df.columns:
        df.loc[idx_to_perturb, "bytes_per_sec"] = (
            (df.loc[idx_to_perturb, "total_length_fwd_packets"].astype(float).fillna(0)
             + df.loc[idx_to_perturb, "total_length_bwd_packets"].astype(float).fillna(0))
            / df.loc[idx_to_perturb, "flow_duration"].clip(lower=1)
        )

    if "pkts_per_sec" in df.columns:
        df.loc[idx_to_perturb, "pkts_per_sec"] = (
            (df.loc[idx_to_perturb, "total_fwd_packets"].astype(float).fillna(0)
             + df.loc[idx_to_perturb, "total_backward_packets"].astype(float).fillna(0))
            / df.loc[idx_to_perturb, "flow_duration"].clip(lower=1)
        )

    if "fwd_to_bwd_ratio" in df.columns:
        df.loc[idx_to_perturb, "fwd_to_bwd_ratio"] = (
            df.loc[idx_to_perturb, "total_fwd_packets"].astype(float).fillna(0)
            / (df.loc[idx_to_perturb, "total_backward_packets"].astype(float).fillna(0) + 1.0)
        )

    # 4) Protocol flips (if protocol one-hot columns exist, flip them)
    # If original dataset has protocol columns (protocol_tcp, protocol_udp), try to flip randomly.
    proto_cols = [c for c in df.columns if c.startswith("protocol")]
    if proto_cols:
        # randomly assign a new protocol among present types for perturbed rows
        possible = proto_cols
        for ridx in idx_to_perturb:
            new = np.random.choice(possible)
            # zero all protocol cols for that row then set chosen one to 1/True
            for c in possible:
                # handle boolean or numeric columns
                if df[c].dtype == "bool":
                    df.at[ridx, c] = False
                else:
                    df.at[ridx, c] = 0
            # set chosen to 1 / True
            if df[new].dtype == "bool":
                df.at[ridx, new] = True
            else:
                df.at[ridx, new] = 1

    # 5) Optionally permute flow_id or source/destination to make synthetic unseen ips (if present)
    for col in ("flow_id", "source_ip", "source_port", "destination_ip", "destination_port"):
        if col in df.columns:
            # small chance to shuffle among selected rows
            if np.random.rand() < 0.5:
                vals = df.loc[idx_to_perturb, col].values
                np.random.shuffle(vals)
                df.loc[idx_to_perturb, col] = vals

    return df

def make_demo():
    print("Loading input CSV:", INPUT_CSV)
    df = safe_load(INPUT_CSV)
    n_total = len(df)
    print("Loaded rows:", n_total)

    # sample
    if SAMPLE_SIZE >= n_total:
        sample_df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        sample_df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)
    print("Sampled rows:", len(sample_df))

    # pick indices to perturb
    n_perturb = int(len(sample_df) * PERTURB_FRAC)
    perturb_idx = sample_df.index.to_numpy()
    if n_perturb > 0:
        perturb_idx = np.random.choice(sample_df.index.to_numpy(), size=n_perturb, replace=False)

    # create output copy and mark perturbation column
    sample_out = sample_df.copy()
    sample_out["__perturbed"] = 0

    if n_perturb > 0:
        print("Perturbing", n_perturb, "rows")
        perturbed = perturb_rows(sample_out, perturb_idx)
        perturbed.loc[perturb_idx, "__perturbed"] = 1
        sample_out = perturbed

    # final save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    sample_out.to_csv(OUTPUT_CSV, index=False)
    print("Saved demo CSV to:", OUTPUT_CSV)
    print("Perturbed rows:", int(sample_out["__perturbed"].sum()))

if __name__ == "__main__":
    make_demo()
