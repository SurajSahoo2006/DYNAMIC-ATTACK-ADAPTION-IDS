#!/usr/bin/env python3
"""
federated/partition.py - Non-IID client data partitioning for federated learning simulation.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEDERATED_DATA_DIR = PROJECT_ROOT / "federated" / "data"


def load_data():
    candidate_train_paths = [
        PROJECT_ROOT / "data" / "processed" / "train.parquet",
        PROJECT_ROOT / "data" / "processed" / "train_phase2.parquet",
    ]
    df_train = None
    for p in candidate_train_paths:
        if p.exists():
            print(f"[PARTITION] Loading training data from {p}")
            df_train = pd.read_parquet(p)
            break
    if df_train is None:
        raise FileNotFoundError("Could not find train.parquet or train_phase2.parquet")

    feat_path = PROJECT_ROOT / "artifacts" / "selected_features.json"
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if isinstance(meta, dict) and "selected_features" in meta:
        features = meta["selected_features"]
    elif isinstance(meta, list):
        features = meta
    else:
        raise ValueError(f"Unexpected format in {feat_path}")

    le_path = PROJECT_ROOT / "artifacts" / "label_encoder_8.pkl"
    le_8 = joblib.load(le_path)

    return df_train, features, le_8


def partition_clients(df_train, features, le_8, random_state=42):
    rng = np.random.RandomState(random_state)
    name_to_idx = {str(name): i for i, name in enumerate(le_8.classes_)}
    idx_to_name = {i: str(name) for i, name in enumerate(le_8.classes_)}

    label_col = "label_8" if "label_8" in df_train.columns else "label"

    available_by_class = {
        cls_idx: list(df_train[df_train[label_col] == cls_idx].index)
        for cls_idx in range(len(le_8.classes_))
    }
    for c in available_by_class:
        rng.shuffle(available_by_class[c])

    def take_from_class(cls_idx, n):
        if len(available_by_class[cls_idx]) < n:
            raise ValueError(
                f"Not enough samples in class {idx_to_name[cls_idx]}: {len(available_by_class[cls_idx])} < {n}"
            )
        taken = available_by_class[cls_idx][:n]
        available_by_class[cls_idx] = available_by_class[cls_idx][n:]
        return taken

    def take_other(exclude_class, n):
        pool = []
        for c in available_by_class:
            if c != exclude_class:
                pool.extend(available_by_class[c])
        if len(pool) < n:
            raise ValueError(f"Not enough other samples: {len(pool)} < {n}")
        chosen_indices = rng.choice(len(pool), size=n, replace=False)
        chosen_items = [pool[i] for i in chosen_indices]
        chosen_set = set(chosen_items)
        for c in available_by_class:
            if c != exclude_class:
                available_by_class[c] = [x for x in available_by_class[c] if x not in chosen_set]
        return chosen_items

    # Client 0: DDoS-heavy — 60% DDoS rows + 40% random other rows (8,000 total)
    c0_idx = take_from_class(name_to_idx["DDoS"], 4800) + take_other(name_to_idx["DDoS"], 3200)

    # Client 1: DoS-heavy — 60% DoS rows + 40% random other rows (8,000 total)
    c1_idx = take_from_class(name_to_idx["DoS"], 4800) + take_other(name_to_idx["DoS"], 3200)

    # Client 2: Recon-heavy — 60% Recon rows + 40% random other rows (8,000 total)
    c2_idx = take_from_class(name_to_idx["Recon"], 4800) + take_other(name_to_idx["Recon"], 3200)

    # Client 3: Balanced — uniform random sample across all classes (1,000 each = 8,000 total)
    c3_idx = []
    for c in range(len(le_8.classes_)):
        c3_idx.extend(take_from_class(c, 1000))

    # Client 4: Benign-heavy — 60% Benign rows + 40% random other rows (2,400 Benign + 1,600 other = 4,000 total)
    c4_idx = take_from_class(name_to_idx["Benign"], 2400) + take_other(name_to_idx["Benign"], 1600)

    client_specs = [
        (0, "DDoS", c0_idx),
        (1, "DoS", c1_idx),
        (2, "Recon", c2_idx),
        (3, "Balanced", c3_idx),
        (4, "Benign", c4_idx),
    ]

    # Verify no overlaps across all clients
    all_indices = []
    for _, _, idxs in client_specs:
        all_indices.extend(idxs)
    if len(all_indices) != len(set(all_indices)):
        raise AssertionError("Overlap detected between clients in non-IID partition!")

    clients_meta = []
    for cid, dom_name, idxs in client_specs:
        sub_df = df_train.loc[idxs].copy().reset_index(drop=True)
        counts = sub_df[label_col].value_counts().to_dict()
        class_dist = {str(idx_to_name[int(k)]): int(v) for k, v in counts.items()}

        out_path = FEDERATED_DATA_DIR / f"client_{cid}_train.parquet"
        sub_df.to_parquet(out_path, index=False)
        print(f"[PARTITION] Client {cid} ({dom_name}) saved to {out_path.name} (n={len(sub_df)})")

        clients_meta.append({
            "client_id": cid,
            "dominant_class": dom_name,
            "n_rows": int(len(sub_df)),
            "class_distribution": class_dist,
        })

    metadata = {
        "n_clients": 5,
        "max_rows_per_client": 8000,
        "partition_strategy": "non-iid class dominance",
        "clients": clients_meta,
    }

    meta_path = FEDERATED_DATA_DIR / "partition_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[PARTITION] Metadata saved to {meta_path.name}")


def main():
    FEDERATED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_train, features, le_8 = load_data()
    partition_clients(df_train, features, le_8)
    print("[PARTITION] Done")


if __name__ == "__main__":
    main()
