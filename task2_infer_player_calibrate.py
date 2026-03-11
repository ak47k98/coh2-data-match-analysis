import os
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import numpy as np
import torch

from task2_features import (
    AggConfig, parse_player_file, MODE_1V1, MODE_2V2, MODE_TEAM, RACE_IDS
)
from task2_train_calibrator import CalibMLP  # 你训练脚本里定义的网络结构


@dataclass
class InferConfig:
    model_dir: str = "models/task2_calib"
    min_games_per_race: int = 10
    w_1v1: float = 0.7
    w_2v2: float = 0.3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    min_gt_min: float = 3.0


def load_race_model(model_dir: str, race_id: int, device: str):
    meta_path = os.path.join(model_dir, f"calib_race{race_id}.json")
    pt_path = os.path.join(model_dir, f"calib_race{race_id}.pt")
    if not (os.path.exists(meta_path) and os.path.exists(pt_path)):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_cols = meta["feature_cols"]  # 裸名
    mu = np.array(meta["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["scaler_std"], dtype=np.float32)

    hidden = tuple(meta["config"]["hidden_sizes"])
    dropout = float(meta["config"]["dropout"])

    model = CalibMLP(in_dim=len(feat_cols), hidden=hidden, dropout=dropout).to(device)
    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "feat_cols": feat_cols, "mu": mu, "sd": sd}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def build_row_dict(row: Dict[str, float], feat_cols: List[str]) -> Dict[str, float]:
    """
    统一从 row 中提取特征：
    - 优先裸名 col
    - 不存在则尝试 f_{col}
    """
    out = {}
    for col in feat_cols:
        v = row.get(col, None)
        if v is None:
            v = row.get(f"f_{col}", 0.0)
        out[col] = float(v)
    return out


def predict_win_prob(bundle, row_dict: Dict[str, float], device: str) -> float:
    feat_cols = bundle["feat_cols"]
    mu = bundle["mu"]
    sd = bundle["sd"]

    x = np.array([float(row_dict.get(col, 0.0)) for col in feat_cols], dtype=np.float32)
    x = (x - mu) / (sd + 1e-8)

    xb = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logit = bundle["model"](xb).detach().cpu().numpy()[0]
    return float(sigmoid(np.array([logit]))[0])


def mean_or_zero(xs: List[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def infer_player(player_jsonl: str, cfg: InferConfig, target_profile_id: int) -> Dict[str, Any]:
    df = parse_player_file(
        player_jsonl,
        AggConfig(min_gt_min=cfg.min_gt_min, min_games_per_race=cfg.min_games_per_race),
        target_profile_id=target_profile_id,
    )
    if df.empty:
        return {"ok": False, "reason": "no_valid_matches_after_filter", "target_profile_id": target_profile_id}

    # 将 f_* 列重命名为裸名，避免取值失败
    rename = {c: c[2:] for c in df.columns if c.startswith("f_")}
    df = df.rename(columns=rename)

    mode_pref = df["mode"].value_counts().to_dict()
    race_pref = df["race_id"].value_counts().to_dict()

    per_race = {}
    for rid in RACE_IDS:
        sub = df[df["race_id"] == rid].copy()
        if len(sub) < cfg.min_games_per_race:
            per_race[str(rid)] = {"status": "insufficient_games", "games": int(len(sub))}
            continue

        bundle = load_race_model(cfg.model_dir, rid, cfg.device)
        if bundle is None:
            per_race[str(rid)] = {"status": "model_not_found"}
            continue

        probs_1v1, probs_2v2, probs_team = [], [], []
        for _, row in sub.iterrows():
            row_dict = build_row_dict(row.to_dict(), bundle["feat_cols"])
            p = predict_win_prob(bundle, row_dict, cfg.device)
            if row["mode"] == MODE_1V1:
                probs_1v1.append(p)
            elif row["mode"] == MODE_2V2:
                probs_2v2.append(p)
            else:
                probs_team.append(p)

        # priority fusion: 1v1/2v2 (7:3) else team fallback
        if (len(probs_1v1) + len(probs_2v2)) >= cfg.min_games_per_race and (len(probs_1v1) + len(probs_2v2)) > 0:
            parts = []
            weights = []
            if probs_1v1:
                parts.append(mean_or_zero(probs_1v1)); weights.append(cfg.w_1v1)
            if probs_2v2:
                parts.append(mean_or_zero(probs_2v2)); weights.append(cfg.w_2v2)
            weights = [w / sum(weights) for w in weights]
            skill = float(sum(p * w for p, w in zip(parts, weights)))
            used = "1v1+2v2"
        else:
            skill = mean_or_zero(probs_team)
            used = "team_fallback"

        per_race[str(rid)] = {
            "status": "ok",
            "games": int(len(sub)),
            "games_1v1": int((sub["mode"] == MODE_1V1).sum()),
            "games_2v2": int((sub["mode"] == MODE_2V2).sum()),
            "games_team": int((sub["mode"] == MODE_TEAM).sum()),
            "used": used,
            "skill_score": float(skill),
            "pred_tier": None,
            "current_tier": None,
        }

    return {
        "ok": True,
        "player_jsonl": player_jsonl,
        "target_profile_id": target_profile_id,
        "mode_preference": {k: int(v) for k, v in mode_pref.items()},
        "race_preference": {str(k): int(v) for k, v in race_pref.items()},
        "per_race": per_race,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--player_jsonl", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--model_dir", default="models/task2_calib")
    ap.add_argument("--target_profile_id", type=int, default=None)
    ap.add_argument("--min_games_per_race", type=int, default=10)
    ap.add_argument("--w1", type=float, default=0.7)
    ap.add_argument("--w2", type=float, default=0.3)
    ap.add_argument("--min_gt_min", type=float, default=3.0)
    args = ap.parse_args()

    target_pid = args.target_profile_id
    if target_pid is None:
        base = os.path.splitext(os.path.basename(args.player_jsonl))[0]
        if base.isdigit():
            target_pid = int(base)
    if target_pid is None:
        raise ValueError("Cannot infer target_profile_id from filename; please pass --target_profile_id")

    cfg = InferConfig(
        model_dir=args.model_dir,
        min_games_per_race=args.min_games_per_race,
        w_1v1=args.w1,
        w_2v2=args.w2,
        min_gt_min=args.min_gt_min,
    )

    out = infer_player(args.player_jsonl, cfg, target_pid)
    s = json.dumps(out, ensure_ascii=False, indent=2)
    print(s)
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            f.write(s)


if __name__ == "__main__":
    main()