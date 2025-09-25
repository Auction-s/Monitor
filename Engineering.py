# ============= Feature Engineering =============
import pandas as pd
import os

ROOT = os.path.expanduser("~")
input_path = os.path.join(
    ROOT, "Documents", "Projects", "Bridge", "AI", "Anomaly", "data", "flows_clean.csv"
)
output_path = os.path.join(
    ROOT, "Documents", "Projects", "Bridge", "AI", "Anomaly", "data", "flows_features.csv"
)

try:
    # Load cleaned dataset
    df = pd.read_csv(input_path)
    print("✅ Clean dataset loaded:", df.shape)

    # Derived features
    df["bytes_per_sec"] = (df["total_length_fwd_packets"] + df["total_length_bwd_packets"]) / df["flow_duration"].clip(lower=1)
    df["pkts_per_sec"] = (df["total_fwd_packets"] + df["total_backward_packets"]) / df["flow_duration"].clip(lower=1)
    df["fwd_to_bwd_ratio"] = df["total_fwd_packets"] / (df["total_backward_packets"] + 1)

    # One-hot encode protocol
    df = pd.get_dummies(df, columns=["protocol"], prefix="protocol")

    # Save engineered dataset
    df.to_csv(output_path, index=False)
    print(f"✅ Features saved at: {output_path}")
    print("New Shape:", df.shape)
    print("Preview:\n", df.head())

except Exception as e:
    print(f"❌ Error in feature engineering: {e}")
