import os
import glob
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

RACE_IDS = [0, 1, 2, 3, 4]
MODE_1V1 = "1v1"
MODE_2V2 = "2v2"
MODE_TEAM = "team"

@dataclass
class TrainConfig:
    data_glob: str = "data/players/*.jsonl"
    out_dir: str = "models/task2_calib"

    min_gt_min: float = 3.0

    use_vkill_features: bool = True
    use_vet_features: bool = True
    use_resource_float_features: bool = True

    hidden_sizes: Tuple[int, ...] = (256, 128)
    dropout: float = 0.2
    epochs: int = 20
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4

    seed: int = 42
    test_size: float = 0.2

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def match_mode(maxplayers: Any) -> str:
    mp = int(maxplayers) if maxplayers is not None else 0
    if mp == 2:
        return MODE_1V1
    if mp == 4:
        return MODE_2V2
    return MODE_TEAM


def iter_matches_from_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_features_from_counters(c: Dict[str, Any], cfg: TrainConfig) -> Dict[str, float]:
    gt = float(c.get("gt", 0.0) or 0.0)
    minutes = max(gt / 60.0, 1e-6)

    def per_min(key: str) -> float:
        return float(c.get(key, 0.0) or 0.0) / minutes

    def ratio(num_key: str, den_key: str) -> float:
        return float(c.get(num_key, 0.0) or 0.0) / (float(c.get(den_key, 0.0) or 0.0) + 1.0)

    pcap = float(c.get("pcap", 0.0) or 0.0)
    plost = float(c.get("plost", 0.0) or 0.0)

    feats = {
        "gt_min": minutes,

        "dmg_per_min": per_min("dmgdone"),
        "ekills_per_min": per_min("ekills"),
        "edeaths_per_min": per_min("edeaths"),

        "pcap_per_min": per_min("pcap"),
        "plost_per_min": per_min("plost"),
        "cap_pressure_per_min": (pcap + plost) / minutes,
        "cap_eff": pcap / (pcap + plost + 1.0),

        "fuelearn_per_min": per_min("fuelearn"),
        "manearn_per_min": per_min("manearn"),
        "munearn_per_min": per_min("munearn"),

        "sqprod_per_min": per_min("sqprod"),
        "sqlost_per_min": per_min("sqlost"),
        "sqkilled_per_min": per_min("sqkilled"),

        "man_spend_ratio": ratio("manspnt", "manearn"),
        "mun_spend_ratio": ratio("munspnt", "munearn"),
        "fuel_spend_ratio": ratio("fuelspnt", "fuelearn"),

        "trade_eff_kd": ratio("ekills", "edeaths"),
        "dmg_per_kill": ratio("dmgdone", "ekills"),
        "dmg_per_death": ratio("dmgdone", "edeaths"),

        "dmg_per_man": ratio("dmgdone", "manearn"),
        "kills_per_man": ratio("ekills", "manearn"),
        "dmg_per_fuel": ratio("dmgdone", "fuelspnt"),

        "squad_loss_ratio": ratio("sqlost", "sqprod"),
        "squad_net_ratio": (float(c.get("sqprod", 0.0) or 0.0) - float(c.get("sqlost", 0.0) or 0.0)) / (float(c.get("sqprod", 0.0) or 0.0) + 1.0),

        "utypes": float(c.get("utypes", 0.0) or 0.0),
        "upg": float(c.get("upg", 0.0) or 0.0),
        "abil": float(c.get("abil", 0.0) or 0.0),
        "cabil": float(c.get("cabil", 0.0) or 0.0),
        "bprod": float(c.get("bprod", 0.0) or 0.0),
        "blost": float(c.get("blost", 0.0) or 0.0),
    }

    if cfg.use_vkill_features:
        feats.update({
            "vkill_per_min": per_min("vkill"),
            "vlost_per_min": per_min("vlost"),
            "vkill_per_fuel": ratio("vkill", "fuelspnt"),
            "vloss_per_fuel": ratio("vlost", "fuelspnt"),
        })

    if cfg.use_resource_float_features:
        feats.update({
            "man_float_ratio": ratio("manmax", "manearn"),
            "mun_float_ratio": ratio("munmax", "munearn"),
            "fuel_float_ratio": ratio("fuelmax", "fuelearn"),
        })

    if cfg.use_vet_features:
        feats.update({
            "svetxp_per_min": per_min("svetxp"),
            "vvetxp_per_min": per_min("vvetxp"),
            "svetrank": float(c.get("svetrank", 0.0) or 0.0),
            "vvetrank": float(c.get("vvetrank", 0.0) or 0.0),
        })

    return feats


def load_match_level_dataset(cfg: TrainConfig) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(cfg.data_glob):
        for match in iter_matches_from_jsonl(fp):
            maxplayers = match.get("maxplayers", None)
            mode = match_mode(maxplayers)
            start = match.get("startgametime", None)
            start = int(start) if start is not None else 0

            mhm = {int(m.get("profile_id")): m for m in (match.get("matchhistorymember") or []) if m.get("profile_id") is not None}

            for r in (match.get("matchhistoryreportresults") or []):
                pid_raw = r.get("profile_id", None)
                if pid_raw is None:
                    continue
                pid = int(pid_raw)

                race_raw = r.get("race_id", None)
                if race_raw is None:
                    continue
                race_id = int(race_raw)
                if race_id not in RACE_IDS:
                    continue

                if pid not in mhm:
                    continue
                oldrating = mhm[pid].get("oldrating", None)
                if oldrating is None:
                    continue

                y_win = 1 if int(r.get("resulttype", 0) or 0) == 1 else 0

                counters = r.get("counters", "{}")
                c = safe_json_loads(counters) if isinstance(counters, str) else (counters or {})
                feats = build_features_from_counters(c, cfg)
                if feats["gt_min"] < cfg.min_gt_min:
                    continue

                rows.append({
                    "profile_id": pid,
                    "race_id": race_id,
                    "mode": mode,
                    "startgametime": start,
                    "oldrating": float(oldrating),
                    "y_win": y_win,
                    **feats,
                })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


def save_global_feature_means(df: pd.DataFrame, cfg: TrainConfig):
    """
    Save global means per race_id for the feature keys you care about.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    # Your requested feature list (must exist in df columns)
    feature_keys = [
        "gt_min","dmg_per_min","ekills_per_min","edeaths_per_min","pcap_per_min","plost_per_min",
        "fuelearn_per_min","manearn_per_min","munearn_per_min",
        "vkill_per_min","vlost_per_min",
        "sqprod_per_min","sqlost_per_min","sqkilled_per_min",
        "cap_pressure_per_min","cap_eff",
        "man_spend_ratio","mun_spend_ratio","fuel_spend_ratio",
        "trade_eff_kd","dmg_per_kill","dmg_per_death",
        "dmg_per_man","kills_per_man","dmg_per_fuel","vkill_per_fuel","vloss_per_fuel",
        "squad_loss_ratio","cabil","squad_net_ratio","bprod","man_float_ratio",
    ]

    # Some keys may not exist if toggles are off; filter
    feature_keys = [k for k in feature_keys if k in df.columns]

    global_all = df[feature_keys].mean(numeric_only=True).to_dict()
    global_by_race = (
        df.groupby("race_id")[feature_keys]
        .mean(numeric_only=True)
        .reset_index()
        .to_dict(orient="records")
    )

    out = {
        "feature_keys": feature_keys,
        "global_all": {k: float(v) for k, v in global_all.items()},
        "global_by_race": [{**r, **{k: float(r[k]) for k in feature_keys}} for r in global_by_race],
        "n_samples": int(len(df)),
        "n_by_race": df["race_id"].value_counts().to_dict(),
        "min_gt_min": cfg.min_gt_min,
    }

    with open(os.path.join(cfg.out_dir, "global_feature_means.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ---------- Torch model (same as你现有版本) ----------
class MatchDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CalibMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def standardize(train_X: np.ndarray, test_X: np.ndarray):
    mu = train_X.mean(axis=0, keepdims=True)
    sd = train_X.std(axis=0, keepdims=True) + 1e-8
    return (train_X - mu) / sd, (test_X - mu) / sd, mu.squeeze(), sd.squeeze()


def train_one_race(cfg: TrainConfig, df: pd.DataFrame, race_id: int) -> Dict[str, Any]:
    sub = df[df["race_id"] == race_id].copy()
    if sub.empty:
        return {"skipped": True, "reason": "no samples"}

    # features for model training: use all numeric engineered features except meta/labels
    drop_cols = {"profile_id","race_id","mode","startgametime","oldrating","y_win"}
    feat_cols = [c for c in sub.columns if c not in drop_cols]

    X = sub[feat_cols].to_numpy(dtype=np.float32)
    y = sub["y_win"].to_numpy(dtype=np.float32)

    rng = np.random.RandomState(cfg.seed + race_id * 17)
    idx = np.arange(len(sub))
    rng.shuffle(idx)
    n_test = int(round(cfg.test_size * len(sub)))
    te = idx[:n_test]
    tr = idx[n_test:]

    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]
    X_train, X_test, mu, sd = standardize(X_train, X_test)

    model = CalibMLP(in_dim=X_train.shape[1], hidden=cfg.hidden_sizes, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(MatchDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(MatchDataset(X_test, y_test), batch_size=cfg.batch_size, shuffle=False)

    best_auc = -1
    best_state = None
    history = []

    def eval_loader():
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(cfg.device)
                logits = model(xb).cpu().numpy()
                p = 1 / (1 + np.exp(-logits))
                all_p.append(p)
                all_t.append(yb.numpy())
        p = np.concatenate(all_p)
        t = np.concatenate(all_t)
        auc = float(roc_auc_score(t, p)) if len(np.unique(t)) > 1 else 0.5
        pred = (p >= 0.5).astype(int)
        acc = float(accuracy_score(t, pred))
        f1 = float(f1_score(t, pred))
        return {"auc": auc, "acc": acc, "f1": f1}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(cfg.device)
            yb = yb.to(cfg.device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()

        m = eval_loader()
        history.append({"epoch": ep, **m})
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    os.makedirs(cfg.out_dir, exist_ok=True)
    pt_path = os.path.join(cfg.out_dir, f"calib_race{race_id}.pt")
    meta_path = os.path.join(cfg.out_dir, f"calib_race{race_id}.json")

    torch.save(best_state if best_state is not None else model.state_dict(), pt_path)
    meta = {
        "race_id": race_id,
        "feature_cols": feat_cols,
        "scaler_mean": mu.tolist(),
        "scaler_std": sd.tolist(),
        "history": history,
        "best_auc": best_auc,
        "config": cfg.__dict__,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return {"skipped": False, "race_id": race_id, "best_auc": best_auc, "pt_path": pt_path, "meta_path": meta_path}


def main():
    cfg = TrainConfig()
    print(f"[Task2] Loading match-level dataset from {cfg.data_glob}")
    df = load_match_level_dataset(cfg)
    if df.empty:
        print("[Task2] No data.")
        return

    print(f"[Task2] Samples: {len(df)}")
    print(df[["mode", "race_id"]].groupby(["mode", "race_id"]).size())

    # NEW: save global means for visualization
    save_global_feature_means(df, cfg)
    print(f"[Task2] Saved global_feature_means.json to {cfg.out_dir}")

    results = []
    for rid in RACE_IDS:
        print(f"\n[Task2] Train calibrator for race={rid}")
        res = train_one_race(cfg, df, rid)
        if res.get("skipped"):
            print("  -> skipped:", res.get("reason"))
        else:
            print(f"  -> best AUC={res['best_auc']:.3f}, saved={res['pt_path']}")
        results.append(res)

    with open(os.path.join(cfg.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Task2] Done. Saved to {cfg.out_dir}")


if __name__ == "__main__":
    main()