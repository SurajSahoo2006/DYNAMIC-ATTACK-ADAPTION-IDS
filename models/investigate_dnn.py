"""
models/investigate_dnn.py
=========================
Investigates why the PyTorch DNN model achieves only ~76.8% on 8 classes.
Hypotheses tested:
  1. Class Imbalance: CrossEntropyLoss without weights heavily favors majority classes (DDoS/DoS)
  2. Dropout Rate: 0.30 dropout on only 19 tabular features drops critical feature signals
  3. Feature Distribution / Outlier z-scores: Extreme values in standardized flow features
  4. Learning rate & training duration
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATA_PROCESSED_DIR,
    ENCODER_8_PATH,
    SELECTED_FEATURES_PATH,
    RANDOM_SEED,
)

def run_investigation():
    print("=" * 70)
    print("  DNN INVESTIGATION: DIAGNOSING 76.8% PERFORMANCE ON 8 CLASSES")
    print("=" * 70)

    with open(SELECTED_FEATURES_PATH, "r") as f:
        features = json.load(f)
    le_8 = joblib.load(ENCODER_8_PATH)
    classes = list(le_8.classes_)

    # Load un-SMOTEd train, val, test
    df_tr = pd.read_parquet(DATA_PROCESSED_DIR / "train_unsmoted_19.parquet")
    X_tr = df_tr[features].to_numpy(dtype=np.float32)
    y_tr = df_tr["label_8"].to_numpy(dtype=np.int64)

    df_val = pd.read_parquet(DATA_PROCESSED_DIR / "val_phase2.parquet")
    X_val = df_val[features].to_numpy(dtype=np.float32)
    y_val = df_val["label_8"].to_numpy(dtype=np.int64)

    df_te = pd.read_parquet(DATA_PROCESSED_DIR / "test_phase2.parquet")
    X_te = df_te[features].to_numpy(dtype=np.float32)
    y_te = df_te["label_8"].to_numpy(dtype=np.int64)

    # 1. Feature Distribution Inspection
    print("\n[DIAGNOSIS 1] Checking Feature Outliers & Min/Max Z-scores:")
    mins = np.min(X_tr, axis=0)
    maxs = np.max(X_tr, axis=0)
    print(f"  Feature min values range: {mins.min():.2f} to {mins.max():.2f}")
    print(f"  Feature max values range: {maxs.min():.2f} to {maxs.max():.2f}")
    outliers = (np.abs(X_tr) > 10.0).sum()
    print(f"  Total cells with |z-score| > 10: {outliers:,} (out of {X_tr.size:,})")

    # 2. Baseline Architecture Evaluation
    print("\n[TEST A] Baseline DNN (Dropout 0.30, Unweighted Loss, lr=0.001):")
    # Train 20 epochs
    class BaselineDNN(nn.Module):
        def __init__(self, drop=0.3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(19, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(drop),
                nn.Linear(64, 8),
            )
        def forward(self, x):
            return self.net(x)

    def fit_and_eval(name, model, criterion, lr=0.001, epochs=25, clip_grad=False):
        torch.manual_seed(RANDOM_SEED)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        loader = DataLoader(train_ds, batch_size=512, shuffle=True)

        for ep in range(epochs):
            model.train()
            for bx, by in loader:
                opt.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()

        model.eval()
        with torch.no_grad():
            val_out = model(torch.from_numpy(X_val)).argmax(dim=1).numpy()
            te_out = model(torch.from_numpy(X_te)).argmax(dim=1).numpy()

        val_acc = accuracy_score(y_val, val_out) * 100.0
        val_f1 = f1_score(y_val, val_out, average="macro") * 100.0
        te_acc = accuracy_score(y_te, te_out) * 100.0
        te_f1 = f1_score(y_te, te_out, average="macro") * 100.0
        print(f"  -> {name:<45}: Val Acc={val_acc:.2f}%, Val F1={val_f1:.2f}% | Test Acc={te_acc:.2f}%, Test F1={te_f1:.2f}%")
        return {"val_acc": val_acc, "val_f1": val_f1, "test_acc": te_acc, "test_f1": te_f1}

    # Test 1: Baseline
    res_base = fit_and_eval("1. Baseline DNN (drop=0.30, unweighted)", BaselineDNN(0.3), nn.CrossEntropyLoss())

    # Test 2: Fix A: Lower Dropout to 0.10 (preserve tabular signal)
    res_drop = fit_and_eval("2. Fix A: Lower Dropout (drop=0.10)", BaselineDNN(0.1), nn.CrossEntropyLoss())

    # Test 3: Fix B: Class-Weighted CrossEntropyLoss
    weights = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    res_cw = fit_and_eval("3. Fix B: Balanced CrossEntropyLoss", BaselineDNN(0.1), nn.CrossEntropyLoss(weight=weights_tensor))

    # Test 4: Fix C: Learning Rate Tuning + Gradient Clipping
    res_lr = fit_and_eval("4. Fix C: LR=0.002 + Grad Clip 1.0", BaselineDNN(0.1), nn.CrossEntropyLoss(weight=weights_tensor), lr=0.002, clip_grad=True)

    print("\n--- Summary of DNN Diagnosis ---")
    print("Baseline Test Accuracy:      ", f"{res_base['test_acc']:.2f}% (Macro F1: {res_base['test_f1']:.2f}%)")
    print("Fix A (Dropout 0.10):        ", f"{res_drop['test_acc']:.2f}% (Macro F1: {res_drop['test_f1']:.2f}%)")
    print("Fix B (Class-Weighted Loss): ", f"{res_cw['test_acc']:.2f}% (Macro F1: {res_cw['test_f1']:.2f}%)")
    print("Fix C (LR 0.002 + Grad Clip):", f"{res_lr['test_acc']:.2f}% (Macro F1: {res_lr['test_f1']:.2f}%)")

if __name__ == "__main__":
    run_investigation()
