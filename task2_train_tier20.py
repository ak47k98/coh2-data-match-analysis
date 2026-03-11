import os
import glob
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from task2_features import (
    match_mode, safe_json_loads, iter_matches_from_jsonl,
    build_features_from_counters, MODE_1V1, MODE_2V2, MODE_TEAM, RACE_IDS
)
from task2_tiers import TierConfig, build_tier_lookup

# -------------------------
# Config (hyperparameters)
# -------------------------
@dataclass
class TrainConfig:
    data_glob: str = "data/players/*.jsonl"
    out_dir: str = "models/task2_tier20"

    # tiers
    tier_cfg: TierConfig = TierConfig()

    # filtering
    min_gt_min: float = 2.0

    # model
    hidden_sizes: Tuple[int, ...] = (256, 128)
    dropout: float = 0.15

    # training
    epochs: int = 5000
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # validation
    test_size: float = 0.2
    seed: int = 42

    # handle imbalance
    use_class_weights: bool = True

    # store
    save_best: bool = True


# -------------------------
# Dataset
# -------------------------
class TierDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float, out_dim: int):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# Build match-level dataframe with tier labels
# -------------------------
def load_all_matches(cfg: TrainConfig) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(cfg.data_glob):
        for match in iter_matches_from_jsonl(fp):
            maxplayers = int(match.get("maxplayers", 0) or 0)
            mode = match_mode(maxplayers)
            start = int(match.get("startgametime", 0) or 0)
            match_id = match.get("id")

            # matchhistorymember gives oldrating per player
            mhm = {int(m.get("profile_id", -1)): m for m in match.get("matchhistorymember", [])}

            for r in match.get("matchhistoryreportresults", []):
                pid = int(r.get("profile_id", -1) or -1)
                
                race_id = r.get("race_id", None)


                if race_id not in RACE_IDS:
                    continue
                if pid not in mhm:
                    continue

                oldrating = mhm[pid].get("oldrating", None)
                if oldrating is None:
                    continue

                counters = r.get("counters", "{}")
                c = safe_json_loads(counters) if isinstance(counters, str) else (counters or {})
                feats = build_features_from_counters(c)
                if feats["gt_min"] < cfg.min_gt_min:
                    continue

                rows.append({
                    "match_id": match_id,
                    "profile_id": pid,
                    "race_id": race_id,
                    "mode": mode,
                    "startgametime": start,
                    "oldrating": float(oldrating),
                    **{f"f_{k}": float(v) for k, v in feats.items()},
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def build_race_dataset(df: pd.DataFrame, tier_lookup: Dict[Tuple[str, int], int], race_id: int) -> pd.DataFrame:
    sub = df[df["race_id"] == race_id].copy()
    if sub.empty:
        return sub

    # label: tier from (mode, profile_id) leaderboard computed on our dataset
    # tier in [1..20] => convert to 0..19
    tiers = []
    for _, row in sub.iterrows():
        key = (row["mode"], int(row["profile_id"]))
        if key not in tier_lookup:
            tiers.append(np.nan)
        else:
            tiers.append(float(tier_lookup[key]))
    sub["tier"] = tiers
    sub = sub[sub["tier"].notna()].copy()
    sub["tier"] = sub["tier"].astype(int)
    sub["y"] = sub["tier"] - 1  # 0..19
    return sub


def split_train_test(df: pd.DataFrame, test_size: float, seed: int):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_test = int(round(test_size * len(df)))
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return df.iloc[train_idx].copy(), df.iloc[test_idx].copy()


def standardize(train_X: np.ndarray, test_X: np.ndarray):
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True) + 1e-8
    return (train_X - mu) / sd, (test_X - mu) / sd, mu, sd


def train_one_race(cfg: TrainConfig, df_race: pd.DataFrame, race_id: int) -> Dict[str, Any]:
    if df_race.empty or df_race["y"].nunique() < 2:
        return {"skipped": True, "reason": "not enough data/classes"}

    feat_cols = [c for c in df_race.columns if c.startswith("f_")]
    X = df_race[feat_cols].to_numpy(dtype=np.float32)
    y = df_race["y"].to_numpy(dtype=np.int64)

    train_df, test_df = split_train_test(df_race, cfg.test_size, cfg.seed)
    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["y"].to_numpy(dtype=np.int64)
    X_test = test_df[feat_cols].to_numpy(dtype=np.float32)
    y_test = test_df["y"].to_numpy(dtype=np.int64)

    X_train, X_test, mu, sd = standardize(X_train, X_test)

    num_classes = 20  # fixed output head (0..19)
    model = MLP(in_dim=X_train.shape[1], hidden=cfg.hidden_sizes, dropout=cfg.dropout, out_dim=num_classes).to(cfg.device)

    # class weights (handle missing tiers by giving them 0 weight)
    if cfg.use_class_weights:
        counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
        weights = np.zeros(num_classes, dtype=np.float32)
        nonzero = counts > 0
        # inverse frequency for present classes
        weights[nonzero] = (counts[nonzero].sum() / (counts[nonzero] + 1e-6))
        weights = weights / (weights[nonzero].mean() + 1e-6) if nonzero.any() else weights
        class_w = torch.tensor(weights, dtype=torch.float32, device=cfg.device)
    else:
        class_w = None

    criterion = nn.CrossEntropyLoss(weight=class_w)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    train_loader = DataLoader(TierDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(TierDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    best_mae = 1e9
    best_state = None

    def eval_loader(loader):
        model.eval()
        all_pred = []
        all_true = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(cfg.device)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                all_pred.append(pred)
                all_true.append(yb.numpy())
        p = np.concatenate(all_pred)
        t = np.concatenate(all_true)
        # ordered metrics (tier is ordinal)
        mae = float(np.mean(np.abs(p - t)))
        within2 = float(np.mean(np.abs(p - t) <= 2))
        acc = float(np.mean(p == t))
        return {"mae": mae, "within2": within2, "acc": acc}

    history = []
    for ep in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)

            optim.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optim.step()

        metrics = eval_loader(test_loader)
        history.append({"epoch": ep, **metrics})

        if metrics["mae"] < best_mae:
            best_mae = metrics["mae"]
            if cfg.save_best:
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # save artifacts
    os.makedirs(cfg.out_dir, exist_ok=True)
    save_path = os.path.join(cfg.out_dir, f"tier20_race{race_id}.pt")
    meta_path = os.path.join(cfg.out_dir, f"tier20_race{race_id}.json")

    if cfg.save_best and best_state is not None:
        torch.save(best_state, save_path)
    else:
        torch.save(model.state_dict(), save_path)

    meta = {
        "race_id": race_id,
        "feature_cols": feat_cols,
        "scaler_mean": mu.squeeze().tolist(),
        "scaler_std": sd.squeeze().tolist(),
        "config": {
            "hidden_sizes": cfg.hidden_sizes,
            "dropout": cfg.dropout,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "min_gt_min": cfg.min_gt_min,
            "test_size": cfg.test_size,
            "seed": cfg.seed,
        },
        "class_counts_train": np.bincount(y_train, minlength=20).tolist(),
        "history": history,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"skipped": False, "race_id": race_id, "model_path": save_path, "meta_path": meta_path, "best_mae": best_mae, "history": history}


def main():
    cfg = TrainConfig()
    print(f"[Task2] Loading matches from {cfg.data_glob}")
    df = load_all_matches(cfg)
    if df.empty:
        print("[Task2] No data.")
        return

    print(f"[Task2] Match-level samples: {len(df)}")
    print(df[["mode", "race_id"]].groupby(["mode", "race_id"]).size())

    # Build per-mode leaderboard tier lookup using latest rating per (mode, player)
    tier_lookup = build_tier_lookup(df[["profile_id", "mode", "startgametime", "oldrating"]].copy(), cfg.tier_cfg)
    print(f"[Task2] Tier lookup size: {len(tier_lookup)}")

    results = []
    for rid in RACE_IDS:
        df_r = build_race_dataset(df, tier_lookup, rid)
        print(f"\n[Task2] Train race={rid}: samples={len(df_r)}, classes_present={df_r['y'].nunique() if not df_r.empty else 0}")
        res = train_one_race(cfg, df_r, rid)
        if res.get("skipped"):
            print("  -> skipped:", res.get("reason"))
        else:
            print(f"  -> saved: {res['model_path']}, best_mae={res['best_mae']:.3f}, within2(last)={res['history'][-1]['within2']:.3f}")
        results.append(res)

    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Task2] Done. Saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()
