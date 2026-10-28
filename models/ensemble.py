"""
models/ensemble.py
==================
Phase 3 - Ensemble Classifier Architecture
Combines:
  1. Random Forest (scikit-learn)
  2. XGBoost (hist gradient boosting)
  3. Deep Neural Network (PyTorch CPU build: 19 -> 128 -> 64 -> C)

Provides:
  - Uniform soft voting (probability averaging)
  - Validation-weighted soft voting (weights optimized strictly on validation data)
  - Serialization to models/ directory
  - Per-class and macro metric evaluation with low-support caution flags
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    MODELS_DIR,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_N_JOBS,
    XGB_N_ESTIMATORS,
    XGB_MAX_DEPTH,
    XGB_LEARNING_RATE,
    XGB_EARLY_STOPPING,
    XGB_N_JOBS,
    DNN_HIDDEN_1,
    DNN_HIDDEN_2,
    DNN_DROPOUT,
    DNN_BATCH_SIZE,
    DNN_LEARNING_RATE,
    DNN_EPOCHS,
    DNN_PATIENCE,
)

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 1. PyTorch Deep Neural Network Module
# ----------------------------------------------------------------------
class PyTorchDNN(nn.Module):
    """
    Feed-forward Neural Network:
      Input (19) -> Linear(128) -> BatchNorm1d(128) -> ReLU() -> Dropout(0.3)
                 -> Linear(64)  -> BatchNorm1d(64)  -> ReLU() -> Dropout(0.3)
                 -> Linear(num_classes)
    """

    def __init__(
        self,
        input_dim: int = 19,
        hidden_1: int = DNN_HIDDEN_1,
        hidden_2: int = DNN_HIDDEN_2,
        num_classes: int = 8,
        dropout: float = DNN_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.BatchNorm1d(hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.BatchNorm1d(hidden_2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, X_np: np.ndarray, batch_size: int = 1024) -> np.ndarray:
        """Computes softmax probabilities in batches."""
        self.eval()
        probs_list: List[np.ndarray] = []
        X_tensor = torch.from_numpy(X_np.astype(np.float32))

        with torch.no_grad():
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i : i + batch_size]
                logits = self(batch)
                batch_probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_list.append(batch_probs)

        return np.vstack(probs_list)


def train_pytorch_dnn(
    model: PyTorchDNN,
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    X_val_np: np.ndarray,
    y_val_np: np.ndarray,
    epochs: int = DNN_EPOCHS,
    batch_size: int = DNN_BATCH_SIZE,
    lr: float = DNN_LEARNING_RATE,
    patience: int = DNN_PATIENCE,
    seed: int = RANDOM_SEED,
) -> Tuple[PyTorchDNN, Dict[str, Any]]:
    """
    Trains PyTorchDNN on CPU with validation early stopping.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = TensorDataset(
        torch.from_numpy(X_train_np.astype(np.float32)),
        torch.from_numpy(y_train_np.astype(np.int64)),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val_np.astype(np.float32)),
        torch.from_numpy(y_val_np.astype(np.int64)),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = float("inf")
    best_weights = None
    no_improve_epochs = 0
    history: List[Dict[str, float]] = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0

        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            correct_train += (preds == by).sum().item()
            total_train += len(by)

        train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for bx, by in val_loader:
                logits = model(bx)
                loss = criterion(logits, by)
                total_val_loss += loss.item() * len(by)
                preds = logits.argmax(dim=1)
                correct_val += (preds == by).sum().item()
                total_val += len(by)

        val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if epoch % 5 == 0 or epoch == 1 or no_improve_epochs >= patience:
            print(
                f"  [DNN Epoch {epoch:>2}/{epochs}] "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

        if no_improve_epochs >= patience:
            print(f"  [DNN] Early stopping triggered at epoch {epoch} (best val loss: {best_loss:.4f})")
            break

    train_time = time.time() - t0
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, {
        "epochs_trained": len(history),
        "best_val_loss": best_loss,
        "train_time_sec": train_time,
        "history": history,
    }


# ----------------------------------------------------------------------
# 2. Ensemble Classifier Manager
# ----------------------------------------------------------------------
class EnsembleClassifier:
    """
    Orchestrates RF + XGBoost + PyTorch DNN.
    Computes both uniform soft voting and validation-weighted soft voting.
    """

    def __init__(
        self,
        target_name: str = "8class",
        num_classes: int = 8,
        class_names: Optional[List[str]] = None,
        random_state: int = RANDOM_SEED,
    ):
        self.target_name = target_name
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        self.random_state = random_state

        # Models
        self.rf_model = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            n_jobs=RF_N_JOBS,
            random_state=random_state,
        )
        self.xgb_model = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            tree_method="hist",
            early_stopping_rounds=XGB_EARLY_STOPPING,
            eval_metric="mlogloss",
            n_jobs=XGB_N_JOBS,
            random_state=random_state,
        )
        self.dnn_model = PyTorchDNN(
            input_dim=19,
            hidden_1=DNN_HIDDEN_1,
            hidden_2=DNN_HIDDEN_2,
            num_classes=num_classes,
            dropout=DNN_DROPOUT,
        )

        # Optimal weights found on validation set
        self.weights = {"rf": 1.0 / 3.0, "xgb": 1.0 / 3.0, "dnn": 1.0 / 3.0}
        self.training_telemetry: Dict[str, Any] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> "EnsembleClassifier":
        """
        Trains all 3 models and tunes ensemble weights strictly on X_val, y_val.
        """
        X_tr_np = X_train.to_numpy(dtype=np.float32)
        X_val_np = X_val.to_numpy(dtype=np.float32)
        y_tr_np = np.asarray(y_train, dtype=np.int64)
        y_val_np = np.asarray(y_val, dtype=np.int64)

        print(f"\n============================================================")
        print(f"Training Ensemble on target '{self.target_name}' ({self.num_classes} classes)")
        print(f"Train samples: {len(X_tr_np):,} | Validation samples: {len(X_val_np):,}")
        print(f"============================================================")

        # 1. Random Forest
        print(f"\n[1/3] Training Random Forest (trees={RF_N_ESTIMATORS}, depth={RF_MAX_DEPTH}, min_leaf={RF_MIN_SAMPLES_LEAF}) ...")
        t0_rf = time.time()
        self.rf_model.fit(X_tr_np, y_tr_np)
        rf_time = time.time() - t0_rf
        print(f"      [RF Finished] Training time: {rf_time:.2f} s")

        # 2. XGBoost
        print(f"\n[2/3] Training XGBoost (hist, max_depth={XGB_MAX_DEPTH}, lr={XGB_LEARNING_RATE}, early_stop={XGB_EARLY_STOPPING}) ...")
        t0_xgb = time.time()
        self.xgb_model.fit(
            X_tr_np,
            y_tr_np,
            eval_set=[(X_val_np, y_val_np)],
            verbose=False,
        )
        xgb_time = time.time() - t0_xgb
        best_iter = getattr(self.xgb_model, "best_iteration", XGB_N_ESTIMATORS)
        print(f"      [XGB Finished] Training time: {xgb_time:.2f} s (best iteration: {best_iter})")

        # 3. PyTorch DNN
        print(f"\n[3/3] Training PyTorch DNN (19 -> 128 -> 64 -> {self.num_classes}, Adam, CPU) ...")
        self.dnn_model, dnn_info = train_pytorch_dnn(
            self.dnn_model,
            X_tr_np,
            y_tr_np,
            X_val_np,
            y_val_np,
            epochs=DNN_EPOCHS,
            batch_size=DNN_BATCH_SIZE,
            lr=DNN_LEARNING_RATE,
            patience=DNN_PATIENCE,
            seed=self.random_state,
        )
        dnn_time = dnn_info["train_time_sec"]
        print(f"      [DNN Finished] Training time: {dnn_time:.2f} s (epochs: {dnn_info['epochs_trained']})")

        # 4. Weight Optimization on Validation Probabilities
        print(f"\n[Ensemble] Tuning ensemble voting weights strictly on validation set ...")
        P_val_rf = self._predict_proba_rf(X_val_np)
        P_val_xgb = self.xgb_model.predict_proba(X_val_np)
        P_val_dnn = self.dnn_model.predict_proba(X_val_np)

        best_acc = 0.0
        best_weights = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

        # 0.05 step grid search over simplex w1 + w2 + w3 = 1
        steps = np.arange(0.0, 1.01, 0.05)
        for w1 in steps:
            for w2 in steps:
                if w1 + w2 > 1.001:
                    continue
                w3 = 1.0 - w1 - w2
                if w3 < -1e-6:
                    continue
                w3 = max(0.0, w3)

                P_comb = w1 * P_val_rf + w2 * P_val_xgb + w3 * P_val_dnn
                preds = np.argmax(P_comb, axis=1)
                acc = np.mean(preds == y_val_np)

                if acc > best_acc:
                    best_acc = acc
                    best_weights = (float(w1), float(w2), float(w3))

        self.weights = {
            "rf": round(best_weights[0], 3),
            "xgb": round(best_weights[1], 3),
            "dnn": round(best_weights[2], 3),
        }
        print(
            f"  Validation Accuracies: "
            f"RF={accuracy_score(y_val_np, np.argmax(P_val_rf, axis=1)):.4f}, "
            f"XGB={accuracy_score(y_val_np, np.argmax(P_val_xgb, axis=1)):.4f}, "
            f"DNN={accuracy_score(y_val_np, np.argmax(P_val_dnn, axis=1)):.4f}"
        )
        print(
            f"  Uniform Soft Voting Val Acc: "
            f"{accuracy_score(y_val_np, np.argmax((P_val_rf + P_val_xgb + P_val_dnn) / 3.0, axis=1)):.4f}"
        )
        print(
            f"  Optimal Weights on Validation: "
            f"RF={self.weights['rf']}, XGB={self.weights['xgb']}, DNN={self.weights['dnn']} "
            f"--> Best Val Acc: {best_acc:.4f}"
        )

        self.training_telemetry = {
            "rf_time_sec": round(rf_time, 2),
            "xgb_time_sec": round(xgb_time, 2),
            "dnn_time_sec": round(dnn_time, 2),
            "total_train_sec": round(rf_time + xgb_time + dnn_time, 2),
            "weights": self.weights,
            "val_acc_weighted": round(float(best_acc), 4),
        }

        return self

    def _predict_proba_rf(self, X_np: np.ndarray) -> np.ndarray:
        """RF predict_proba ensuring full (N, num_classes) shape."""
        probs = self.rf_model.predict_proba(X_np)
        if probs.shape[1] == self.num_classes:
            return probs
        # If any class missing in RF train split, pad with zeros
        full_probs = np.zeros((len(X_np), self.num_classes), dtype=np.float32)
        for idx, cls_idx in enumerate(self.rf_model.classes_):
            if cls_idx < self.num_classes:
                full_probs[:, cls_idx] = probs[:, idx]
        return full_probs

    def predict_probabilities(
        self, X: pd.DataFrame | np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Computes probabilities for RF, XGB, DNN, Uniform Soft Voting, and Weighted Soft Voting.
        """
        X_np = X.to_numpy(dtype=np.float32) if isinstance(X, pd.DataFrame) else X.astype(np.float32)

        P_rf = self._predict_proba_rf(X_np)
        P_xgb = self.xgb_model.predict_proba(X_np)
        P_dnn = self.dnn_model.predict_proba(X_np)

        P_soft = (P_rf + P_xgb + P_dnn) / 3.0
        w_rf = self.weights["rf"]
        w_xgb = self.weights["xgb"]
        w_dnn = self.weights["dnn"]
        P_weighted = w_rf * P_rf + w_xgb * P_xgb + w_dnn * P_dnn

        return {
            "rf": P_rf,
            "xgb": P_xgb,
            "dnn": P_dnn,
            "soft": P_soft,
            "weighted": P_weighted,
        }

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        mode: str = "weighted",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (predicted_class_indices, confidence_scores).
        """
        probs_dict = self.predict_probabilities(X)
        P = probs_dict.get(mode, probs_dict["weighted"])
        preds = np.argmax(P, axis=1)
        conf = np.max(P, axis=1)
        return preds, conf

    def evaluate(
        self,
        X_test: pd.DataFrame | np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluates individual models and ensemble variants on untouched test data.
        Returns accuracy, macro metrics, per-class metrics, support, and inference latencies.
        """
        X_np = X_test.to_numpy(dtype=np.float32) if isinstance(X_test, pd.DataFrame) else X_test.astype(np.float32)
        y_true = np.asarray(y_test, dtype=np.int64)
        n_samples = len(y_true)

        # Inference timing
        t0 = time.time()
        probs_dict = self.predict_probabilities(X_np)
        total_inf_time = time.time() - t0
        ms_per_sample = (total_inf_time / max(1, n_samples)) * 1000.0

        model_keys = ["rf", "xgb", "dnn", "soft", "weighted"]
        model_names = {
            "rf": "Random Forest",
            "xgb": "XGBoost",
            "dnn": "PyTorch DNN",
            "soft": "Ensemble (Soft Voting)",
            "weighted": "Ensemble (Weighted Voting)",
        }

        results: Dict[str, Any] = {
            "target": self.target_name,
            "n_test_samples": n_samples,
            "inference_time_sec": round(total_inf_time, 3),
            "ms_per_sample": round(ms_per_sample, 4),
            "models": {},
        }

        for k in model_keys:
            preds = np.argmax(probs_dict[k], axis=1)
            acc = float(accuracy_score(y_true, preds))
            prec = float(precision_score(y_true, preds, average="macro", zero_division=0))
            rec = float(recall_score(y_true, preds, average="macro", zero_division=0))
            f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

            results["models"][k] = {
                "name": model_names[k],
                "accuracy": round(acc, 4),
                "macro_precision": round(prec, 4),
                "macro_recall": round(rec, 4),
                "macro_f1": round(f1, 4),
                "predictions": preds,
            }

        # Detailed per-class metrics for the primary Weighted Ensemble
        weighted_preds = results["models"]["weighted"]["predictions"]
        report_dict = classification_report(
            y_true,
            weighted_preds,
            target_names=[str(c) for c in self.class_names],
            output_dict=True,
            zero_division=0,
        )

        per_class_table = []
        for i, cname in enumerate(self.class_names):
            c_str = str(cname)
            metrics = report_dict.get(c_str, {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
            sup = int(metrics["support"])
            flag = "⚠️ low support, interpret with caution" if sup < 100 else "Normal"
            per_class_table.append({
                "class_idx": i,
                "class_name": c_str,
                "precision": round(metrics["precision"], 4),
                "recall": round(metrics["recall"], 4),
                "f1": round(metrics["f1-score"], 4),
                "support": sup,
                "flag": flag,
            })

        results["per_class_table"] = per_class_table
        results["confusion_matrix"] = confusion_matrix(y_true, weighted_preds).tolist()

        return results

    def save(
        self,
        base_dir: Path = MODELS_DIR,
    ) -> None:
        """Saves RF, XGBoost, PyTorch DNN, and weights configuration."""
        base_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{self.target_name}"

        # 1. RF
        rf_path = base_dir / f"rf{suffix}.joblib"
        joblib.dump(self.rf_model, rf_path)

        # 2. XGB
        xgb_path = base_dir / f"xgb{suffix}.joblib"
        joblib.dump(self.xgb_model, xgb_path)

        # 3. DNN
        dnn_path = base_dir / f"dnn{suffix}.pt"
        torch.save({
            "model_state_dict": self.dnn_model.state_dict(),
            "input_dim": 19,
            "hidden_1": DNN_HIDDEN_1,
            "hidden_2": DNN_HIDDEN_2,
            "num_classes": self.num_classes,
        }, dnn_path)

        # 4. Weights & Metadata
        weights_path = base_dir / f"ensemble_weights{suffix}.json"
        metadata = {
            "target_name": self.target_name,
            "num_classes": self.num_classes,
            "class_names": [str(c) for c in self.class_names],
            "weights": self.weights,
            "telemetry": self.training_telemetry,
        }
        with open(weights_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        print(f"[SAVE] Models and weights for '{self.target_name}' saved to {base_dir}")

    @classmethod
    def load(
        cls,
        base_dir: Path = MODELS_DIR,
        target_name: str = "8class",
    ) -> "EnsembleClassifier":
        """Loads models and weights from disk."""
        suffix = f"_{target_name}"
        weights_path = base_dir / f"ensemble_weights{suffix}.json"
        with open(weights_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        instance = cls(
            target_name=meta["target_name"],
            num_classes=meta["num_classes"],
            class_names=meta["class_names"],
        )
        instance.weights = meta["weights"]
        instance.training_telemetry = meta.get("telemetry", {})

        # Load RF
        rf_path = base_dir / f"rf{suffix}.joblib"
        instance.rf_model = joblib.load(rf_path)

        # Load XGB
        xgb_path = base_dir / f"xgb{suffix}.joblib"
        instance.xgb_model = joblib.load(xgb_path)

        # Load DNN
        dnn_path = base_dir / f"dnn{suffix}.pt"
        checkpoint = torch.load(dnn_path, map_location="cpu")
        instance.dnn_model = PyTorchDNN(
            input_dim=checkpoint.get("input_dim", 19),
            hidden_1=checkpoint.get("hidden_1", DNN_HIDDEN_1),
            hidden_2=checkpoint.get("hidden_2", DNN_HIDDEN_2),
            num_classes=checkpoint.get("num_classes", meta["num_classes"]),
        )
        instance.dnn_model.load_state_dict(checkpoint["model_state_dict"])
        instance.dnn_model.eval()

        return instance
