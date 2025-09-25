# ============= Preprocess Dataset ==============
import pandas as pd
import os

print("pandas imported successfully")

# Input + Output paths
ROOT = os.path.expanduser("~")
data_path = os.path.join(
    ROOT, "Documents", "Projects", "Bridge", "AI", "Anomaly", "data", "CICIDS_Flow.parquet"
)
output_path = os.path.join(
    ROOT, "Documents", "Projects", "Bridge", "AI", "Anomaly", "data", "flows_clean.csv"
)

try:
    # Load dataset
    df = pd.read_parquet(data_path)
    print("Dataset loaded successfully")

    # Select only relevant columns (make sure names match exactly!)
    selected_cols = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "protocol",
        "attack_label",
    ]
    df_clean = df[selected_cols].copy()

    # Rename columns to snake_case for easier ML processing
    df_clean.rename(
        columns={
            "Flow Duration": "flow_duration",
            "Total Fwd Packets": "total_fwd_packets",
            "Total Backward Packets": "total_backward_packets",
            "Total Length of Fwd Packets": "total_length_fwd_packets",
            "Total Length of Bwd Packets": "total_length_bwd_packets",
            "protocol": "protocol",
            "attack_label": "attack_label",
        },
        inplace=True,
    )

    # Save to CSV
    df_clean.to_csv(output_path, index=False)
    print(f"✅ Clean dataset saved at: {output_path}")
    print("Shape:", df_clean.shape)
    print("Preview:\n", df_clean.head())

except Exception as e:
    print(f"❌ Error preprocessing dataset: {e}")
