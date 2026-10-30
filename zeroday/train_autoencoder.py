#!/usr/bin/env python3
"""
zeroday/train_autoencoder.py - Train PyTorch Autoencoder on Benign flows only.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import (
    TRAIN_PARQUET,
    SELECTED_FEATURES_PATH,
    ENCODER_8_PATH,
    SCALER_PATH,
)
from models.load_validation import load_validation_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_autoencoder")

MODELS_DIR = PROJECT_ROOT / "zeroday" / "models"
RESULTS_DIR = PROJECT_ROOT / "zeroday" / "results"


class BenignAutoencoder(nn.Module):
    """
    Autoencoder architecture: 19 -> 12 -> 6 -> 12 -> 19 with ReLU activations.
    """
    def __init__(self, in_features: int = 19):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features, 12),
            nn.ReLU(),
            nn.Linear(12, 6),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU(),
            nn.Linear(12, in_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    seed = 42
    set_seed(seed)

    # 1. Load artifacts
    with open(SELECTED_FEATURES_PATH, "r", encoding="utf-8") as f:
        selected_features = json.load(f)
    if isinstance(selected_features, dict) and "selected_features" in selected_features:
        selected_features = selected_features["selected_features"]

    le_8 = joblib.load(ENCODER_8_PATH)
    benign_idx = int(np.where(le_8.classes_ == "Benign")[0][0])
    log.info(f"Benign class index in 8-class encoder: {benign_idx}")

    scaler = None
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)

    # 1a. Load training data (Phase 1-2 train split, Benign only, real rows only, cap 100k)
    train_file = TRAIN_PARQUET
    if not train_file.exists():
        train_file = PROJECT_ROOT / "data" / "processed" / "train.parquet"
    if not train_file.exists():
        raise FileNotFoundError(f"Training parquet not found at {train_file}")

    log.info(f"Loading training data from {train_file.name} ...")
    df_train = pd.read_parquet(train_file)
    label_col = "label_8" if "label_8" in df_train.columns else "label"

    # Filter Benign rows only (real rows, cap at 100,000)
    df_benign = df_train[df_train[label_col] == benign_idx].copy()
    raw_benign_count = len(df_benign)
    if raw_benign_count > 100_000:
        df_benign = df_benign.sample(n=100_000, random_state=seed)

    X_train_benign = df_benign[selected_features].to_numpy(dtype=np.float32)
    n_train_benign = len(X_train_benign)
    log.info(f"Extracted {n_train_benign:,} Benign training flows (cap: 100,000).")

    # Apply scaler if dimensions match
    if scaler is not None and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(selected_features):
        X_train_benign = scaler.transform(X_train_benign).astype(np.float32)

    # 1b. Load validation split using run_phase3_ensemble loading function
    log.info("Loading validation data using load_validation_data() ...")
    X_val_all, y_val_8, _ = load_validation_data()
    val_benign_mask = (y_val_8 == benign_idx)
    X_val_benign = X_val_all[val_benign_mask][selected_features].to_numpy(dtype=np.float32)
    n_val_benign = len(X_val_benign)
    log.info(f"Extracted {n_val_benign:,} Benign validation flows.")

    if scaler is not None and hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(selected_features):
        X_val_benign = scaler.transform(X_val_benign).astype(np.float32)

    # 1c. Build PyTorch Autoencoder (19 -> 12 -> 6 -> 12 -> 19)
    device = torch.device("cpu")
    model = BenignAutoencoder(in_features=len(selected_features)).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_dataset = TensorDataset(torch.from_numpy(X_train_benign.copy()))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

    val_tensor = torch.from_numpy(X_val_benign.copy()).to(device)

    # Allow up to 150 epochs with early stopping (patience 10)
    max_epochs = 150
    patience = 10
    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0
    epochs_run = 0

    log.info(f"Starting Autoencoder training on Benign flows (CPU, max {max_epochs} epochs, patience {patience}) ...")
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        train_loss = 0.0
        n_batches = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            loss = criterion(reconstruction, batch_x)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        avg_train_loss = train_loss / max(n_batches, 1)

        # Validation loss on Benign validation set
        model.eval()
        with torch.no_grad():
            val_recon = model(val_tensor)
            val_loss = criterion(val_recon, val_tensor).item()

        if epoch % 5 == 0 or epoch == 1 or val_loss < best_val_loss - 1e-6:
            log.info(
                f"Epoch {epoch:03d}/{max_epochs} - Train MSE: {avg_train_loss:.6f} | Val MSE: {val_loss:.6f}"
            )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"Early stopping triggered at epoch {epoch} (patience={patience}).")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    # 1d & 1e. Compute Anomaly Score and Thresholds (95th, 97.5th, 99th percentiles on val Benign)
    model.eval()
    with torch.no_grad():
        val_recon = model(val_tensor)
        val_scores = ((val_tensor - val_recon) ** 2).mean(dim=1).cpu().numpy()

    thresh_95 = float(np.percentile(val_scores, 95.0))
    thresh_97_5 = float(np.percentile(val_scores, 97.5))
    thresh_99 = float(np.percentile(val_scores, 99.0))

    threshold = thresh_95
    log.info(f"Validation Benign thresholds: 95th={thresh_95:.6f}, 97.5th={thresh_97_5:.6f}, 99th={thresh_99:.6f}")

    # 1f. Save model and results
    model_path = MODELS_DIR / "autoencoder.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "in_features": len(selected_features),
        "threshold": threshold,
        "thresholds_by_percentile": {
            "95.0": thresh_95,
            "97.5": thresh_97_5,
            "99.0": thresh_99,
        },
    }, model_path)
    log.info(f"Model saved to {model_path}")

    results_data = {
        "epochs_run": int(epochs_run),
        "best_val_loss": round(float(best_val_loss), 6),
        "threshold": round(float(threshold), 6),
        "thresholds_by_percentile": {
            "95.0": round(float(thresh_95), 6),
            "97.5": round(float(thresh_97_5), 6),
            "99.0": round(float(thresh_99), 6),
        },
        "n_train_benign": int(n_train_benign),
        "train_cap": 100000,
        "train_below_cap": bool(n_train_benign < 100000),
        "train_cap_note": f"Real Benign training rows ({n_train_benign:,}) are fewer than the 100,000 cap.",
        "n_val_benign": int(n_val_benign),
        "seed": int(seed),
    }

    results_path = RESULTS_DIR / "ae_training.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)
    log.info(f"Training results saved to {results_path}")
    print("[AUTOENCODER] Training complete.")


if __name__ == "__main__":
    main()
