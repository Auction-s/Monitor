# train_models.py
"""
Train an unsupervised Isolation Forest for NIDS.

- Loads features from data/flows_features.csv
- Trains IsolationForest on BENIGN samples only
- Tests on a held-out set (benign + attacks)
- Saves model + scaler in models/
- Saves metrics + eval results in models/ (instead of reports/)

All paths are hard-set for your project:
- Input:  C:/Users/LENOVO/Documents/Projects/Bridge/AI/Anomaly/data/flows_features.csv
- Output: C:/Users/LENOVO/Documents/Projects/Bridge/AI/Anomaly/models
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# --- Hardcoded paths ---
INPUT_PATH = r"C:\Users\LENOVO\Documents\Projects\Bridge\AI\Anomaly\data\flows_features.csv"
OUTPUT_DIR = r"C:\Users\LENOVO\Documents\Projects\Bridge\AI\Anomaly\models"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_numeric_feature_columns(df):
    exclude = {'attack_label', 'is_attack'}
    return [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

def main():
    contamination = 0.4
    sample_size = 200000
    rs = 42

    print(f"Loading data from: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print("Total rows in CSV:", len(df))

    # Optional subsample
    if sample_size and sample_size > 0 and sample_size < len(df):
        print(f"Sampling {sample_size} rows (stratified by label)...")
        frac = sample_size / len(df)
        df = df.groupby('attack_label', group_keys=False).apply(
            lambda x: x.sample(frac=frac, random_state=rs)
        )
        df = df.sample(frac=1, random_state=rs).reset_index(drop=True)
        print("Rows after sampling:", len(df))

    # Map labels
    df['is_attack'] = (df['attack_label'].str.upper() != 'BENIGN').astype(int)

    # Features
    feature_cols = get_numeric_feature_columns(df)
    print("Numeric feature columns used:", feature_cols)

    X = df[feature_cols].fillna(0).astype(float)
    y = df['is_attack'].values

    # Train/test split
    benign_mask = (df['is_attack'] == 0)
    X_benign = X[benign_mask]
    print("Benign rows available for training:", len(X_benign))

    rng = np.random.RandomState(rs)
    n_benign = len(X_benign)
    train_frac = 0.6
    train_idx = rng.choice(n_benign, size=int(train_frac * n_benign), replace=False)
    X_benign_train = X_benign.iloc[train_idx]

    # Test set = remaining benign + all attacks
    train_indices = X_benign_train.index
    test_df = df.drop(index=train_indices)
    X_test = X.loc[test_df.index]
    y_test = test_df['is_attack'].values

    print("Train (benign-only) shape:", X_benign_train.shape)
    print("Test shape (benign+attacks):", X_test.shape)

    # --------- Scale features ------------- 
    scaler = StandardScaler()
    scaler.fit(X_benign_train)
    X_train_scaled = scaler.transform(X_benign_train)
    X_test_scaled = scaler.transform(X_test)

    # --------- Train IsolationForest -------------
    print(f"Training IsolationForest (contamination={contamination}) ...")
    iso = IsolationForest(contamination=contamination, random_state=rs, n_jobs=-1)
    iso.fit(X_train_scaled)
    print("Model trained.")

    # --------- Predict ------------
    preds = iso.predict(X_test_scaled)
    y_pred = (preds == -1).astype(int)

    # --------- Evaluate -----------
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("Evaluation on test set:")
    print("Confusion matrix (rows=true, cols=pred):\n", cm)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # --------- Save eval results ----------
    test_scores = iso.decision_function(X_test_scaled)
    out_eval_df = test_df.copy()
    out_eval_df = out_eval_df.assign(
        pred_anomaly=y_pred,
        score=test_scores
    )
    eval_csv = os.path.join(OUTPUT_DIR, 'eval_results.csv')
    out_eval_df.to_csv(eval_csv, index=False)
    print("Saved per-flow evaluation CSV to:", eval_csv)

    # ------------ Save model + scaler -------------
    joblib.dump(iso, os.path.join(OUTPUT_DIR, 'isoforest.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'scaler.pkl'))
    print("Saved model and scaler in:", OUTPUT_DIR)

    # Save summary
    summary = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'n_train_benign': int(X_benign_train.shape[0]),
        'n_test': int(X_test.shape[0])
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'metrics_summary.csv'), index=False)
    print("Saved metrics summary.")

    print("\nClassification report (anomaly=1):")
    print(classification_report(y_test, y_pred, zero_division=0))

if __name__ == '__main__':
    main()
