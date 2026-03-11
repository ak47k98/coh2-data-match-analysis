import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch
import torch.nn as nn

from task2_features import (
    AggConfig, parse_player_file, aggregate_matches,
    MODE_1V1, MODE_2V2, MODE_TEAM, RACE_IDS
)

# -------------------------
# Font for Chinese
# -------------------------
def setup_chinese_font_ubuntu():
    candidates = ["Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC", "AR PL UMing CN", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: Tuple[int, ...], dropout: float, out_dim: int = 20):
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

@dataclass
class InferConfig:
    model_dir: str = "models/task2_tier20"
    out_dir: str = "models/task2_tier20/player_reports"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # inference rules
    min_games_per_race: int = 10
    agg_cfg: AggConfig = AggConfig()

    # mode priority in inference: (1v1,2v2)=7:3 ; fallback to team if insufficient
    w_1v1: float = 0.7
    w_2v2: float = 0.3


def load_model_for_race(cfg: InferConfig, race_id: int):
    meta_path = os.path.join(cfg.model_dir, f"tier20_race{race_id}.json")
    pt_path = os.path.join(cfg.model_dir, f"tier20_race{race_id}.pt")
    if not (os.path.exists(meta_path) and os.path.exists(pt_path)):
        return None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feat_cols = meta["feature_cols"]
    mu = np.array(meta["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["scaler_std"], dtype=np.float32)

    hidden = tuple(meta["config"]["hidden_sizes"])
    dropout = float(meta["config"]["dropout"])

    model = MLP(in_dim=len(feat_cols), hidden=hidden, dropout=dropout, out_dim=20).to(cfg.device)
    state = torch.load(pt_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    return {"model": model, "meta": meta, "feat_cols": feat_cols, "mu": mu, "sd": sd}


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / (e.sum() + 1e-12)


def predict_from_agg(model_bundle, agg_features: Dict[str, float]) -> Dict[str, Any]:
    feat_cols = model_bundle["feat_cols"]
    x = np.array([agg_features.get(col.replace("f_", ""), 0.0) if col.startswith("f_") else agg_features.get(col, 0.0) for col in feat_cols], dtype=np.float32)

    # NOTE: our training feature cols are f_* (match-level).
    # For aggregation we used base__wmean etc, so here we need consistent columns.
    # Simplify: use only "__wmean" stats for the same base names -> build in caller.
    # In this function, agg_features already contains values aligned to feat_cols.

    x = (x - model_bundle["mu"]) / (model_bundle["sd"] + 1e-8)
    xb = torch.tensor(x, dtype=torch.float32, device=next(model_bundle["model"].parameters()).device).unsqueeze(0)
    with torch.no_grad():
        logits = model_bundle["model"](xb).cpu().numpy()[0]
    proba = softmax_np(logits)
    pred = int(np.argmax(proba)) + 1  # 1..20
    return {"pred_level": pred, "proba": float(proba[pred - 1]), "proba_vec": proba.tolist()}


def mode_pref_stats(df: pd.DataFrame) -> Dict[str, Any]:
    total = len(df)
    if total == 0:
        return {}
    return {
        "total": total,
        "1v1": int((df["mode"] == MODE_1V1).sum()),
        "2v2": int((df["mode"] == MODE_2V2).sum()),
        "team": int((df["mode"] == MODE_TEAM).sum()),
        "ratio_1v1": float((df["mode"] == MODE_1V1).mean()),
        "ratio_2v2": float((df["mode"] == MODE_2V2).mean()),
        "ratio_team": float((df["mode"] == MODE_TEAM).mean()),
    }


def infer_player(player_jsonl: str, cfg: InferConfig) -> Dict[str, Any]:
    df = parse_player_file(player_jsonl, cfg.agg_cfg, only_creator=True)
    if df.empty:
        return {"ok": False, "reason": "no_valid_matches"}

    # preference stats
    pref_mode = mode_pref_stats(df)
    pref_race = df["race_id"].value_counts().to_dict()

    per_race = {}
    overall_feature_pool = []

    for rid in RACE_IDS:
        sub = df[df["race_id"] == rid].copy()
        if len(sub) < cfg.min_games_per_race:
            per_race[str(rid)] = {"status": "insufficient_games", "games": int(len(sub))}
            continue

        # split by mode groups for priority fusion
        sub_1v1 = sub[sub["mode"] == MODE_1V1]
        sub_2v2 = sub[sub["mode"] == MODE_2V2]
        sub_team = sub[sub["mode"] == MODE_TEAM]

        def agg_to_model_input(g: pd.DataFrame) -> Dict[str, float]:
            # aggregate to base__wmean etc.
            agg = aggregate_matches(g, cfg.agg_cfg)
            # use only __wmean to align with base feature names
            out = {}
            for k, v in agg.items():
                if k.endswith("__wmean"):
                    base = k.replace("__wmean", "")
                    out[f"f_{base}"] = float(v)
            return out

        # choose data: 1v1+2v2 weighted, else fallback team
        use_main = (len(sub_1v1) + len(sub_2v2)) >= cfg.min_games_per_race
        if use_main:
            # aggregate separately then fuse by 7:3 (if one missing, normalize)
            parts = []
            weights = []
            if len(sub_1v1) > 0:
                parts.append(agg_to_model_input(sub_1v1))
                weights.append(cfg.w_1v1)
            if len(sub_2v2) > 0:
                parts.append(agg_to_model_input(sub_2v2))
                weights.append(cfg.w_2v2)
            wsum = sum(weights)
            weights = [w / wsum for w in weights]

            fused = {}
            for p, w in zip(parts, weights):
                for kk, vv in p.items():
                    fused[kk] = fused.get(kk, 0.0) + w * vv
        else:
            # fallback to team
            if len(sub_team) < cfg.min_games_per_race:
                per_race[str(rid)] = {"status": "insufficient_main_and_team", "games": int(len(sub))}
                continue
            fused = agg_to_model_input(sub_team)

        model_bundle = load_model_for_race(cfg, rid)
        if model_bundle is None:
            per_race[str(rid)] = {"status": "model_not_found"}
            continue

        pred = predict_from_agg(model_bundle, fused)

        per_race[str(rid)] = {
            "status": "ok",
            "games_total": int(len(sub)),
            "games_1v1": int(len(sub_1v1)),
            "games_2v2": int(len(sub_2v2)),
            "games_team": int(len(sub_team)),
            "used_main_1v1_2v2": bool(use_main),
            "pred_level": pred["pred_level"],
            "pred_conf": pred["proba"],
            "proba_vec": pred["proba_vec"],
        }

        overall_feature_pool.append(fused)

    # overall technical stats: mean of available races (weighted by game count)
    # (only uses the fused feature vectors we built)
    tech_summary = {}
    if overall_feature_pool:
        keys = list(overall_feature_pool[0].keys())
        mat = np.array([[d.get(k, 0.0) for k in keys] for d in overall_feature_pool], dtype=float)
        mu = mat.mean(axis=0)
        tech_summary = {k: float(v) for k, v in zip(keys, mu)}

    return {
        "ok": True,
        "player_jsonl": player_jsonl,
        "mode_preference": pref_mode,
        "race_preference_counts": {str(k): int(v) for k, v in pref_race.items()},
        "per_race": per_race,
        "tech_summary": tech_summary,
    }


def plot_outputs(report: Dict[str, Any], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    setup_chinese_font_ubuntu()

    # 1) Race predicted levels bar
    races = []
    pred = []
    conf = []
    for rid in RACE_IDS:
        info = report["per_race"].get(str(rid), {})
        if info.get("status") != "ok":
            continue
        races.append(f"Race {rid}")
        pred.append(info["pred_level"])
        conf.append(info["pred_conf"])

    if races:
        plt.figure(figsize=(9, 4.5))
        plt.bar(races, pred)
        for i, c in enumerate(conf):
            plt.text(i, pred[i] + 0.2, f"{c:.2f}", ha="center", fontsize=10)
        plt.ylim(0, 20.5)
        plt.title("五阵营预测段位（柱高=段位，数字=置信度）")
        plt.ylabel("预测段位(1-20)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pred_levels_by_race.png"), dpi=220)
        plt.close()

    # 2) Mode preference pie
    mp = report.get("mode_preference", {})
    if mp:
        labels = ["1v1", "2v2", "team(3v3+4v4)"]
        values = [mp.get("1v1", 0), mp.get("2v2", 0), mp.get("team", 0)]
        plt.figure(figsize=(6, 5))
        plt.pie(values, labels=labels, autopct="%1.1f%%")
        plt.title("模式偏好分布（按局数）")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "mode_preference.png"), dpi=220)
        plt.close()

    # 3) Race preference bar
    rp = report.get("race_preference_counts", {})
    if rp:
        xs = [f"Race {rid}" for rid in sorted(int(k) for k in rp.keys())]
        ys = [rp[str(int(x.split()[-1]))] for x in xs]
        plt.figure(figsize=(9, 4.5))
        plt.bar(xs, ys)
        plt.title("阵营偏好（按局数）")
        plt.ylabel("局数")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "race_preference.png"), dpi=220)
        plt.close()

    # 4) Technical summary table (top features by absolute value)
    tech = report.get("tech_summary", {})
    if tech:
        # show a selected subset more meaningful
        pick = [
            "f_cap_eff", "f_cap_pressure_per_min",
            "f_trade_eff_kd", "f_dmg_per_kill",
            "f_dmg_per_fuel", "f_vkill_per_fuel", "f_vloss_per_fuel",
            "f_squad_loss_ratio",
            "f_fuelearn_per_min", "f_fuel_spend_ratio",
        ]
        rows = []
        for k in pick:
            if k in tech:
                rows.append((k, tech[k]))
        if rows:
            df = pd.DataFrame(rows, columns=["指标(特征名)", "数值"])
            df.to_csv(os.path.join(out_dir, "tech_summary.csv"), index=False, encoding="utf-8-sig")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--player_jsonl", required=True)
    ap.add_argument("--model_dir", default="models/task2_tier20")
    ap.add_argument("--out_dir", default="models/task2_tier20/player_reports")
    ap.add_argument("--min_games_per_race", type=int, default=10)
    ap.add_argument("--w1", type=float, default=0.7)
    ap.add_argument("--w2", type=float, default=0.3)
    ap.add_argument("--tau_games", type=float, default=25.0)
    ap.add_argument("--min_gt_min", type=float, default=2.0)
    args = ap.parse_args()

    cfg = InferConfig(
        model_dir=args.model_dir,
        out_dir=args.out_dir,
        min_games_per_race=args.min_games_per_race,
        w_1v1=args.w1,
        w_2v2=args.w2,
    )
    cfg.agg_cfg.recency_tau_games = args.tau_games
    cfg.agg_cfg.min_gt_min = args.min_gt_min

    report = infer_player(args.player_jsonl, cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)

    # save json
    base = os.path.splitext(os.path.basename(args.player_jsonl))[0]
    json_path = os.path.join(cfg.out_dir, f"{base}_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # plots
    plot_dir = os.path.join(cfg.out_dir, base)
    plot_outputs(report, plot_dir)

    print(f"[task2_infer_and_viz] saved report: {json_path}")
    print(f"[task2_infer_and_viz] saved plots: {plot_dir}")


if __name__ == "__main__":
    main()