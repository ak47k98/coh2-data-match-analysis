import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import torch

from task2_features import (
    AggConfig, parse_player_file, RACE_IDS,
    MODE_1V1, MODE_2V2, MODE_TEAM,
)
# 用训练时的网络结构进行必要的 skill 重算
from task2_train_calibrator import CalibMLP

# -------------------------
# 阵营中文名映射
# -------------------------
RACE_NAME = {
    0: "东德",
    1: "苏联",
    2: "西德",
    3: "美国",
    4: "英国",
}

def race_label(rid: int) -> str:
    return RACE_NAME.get(int(rid), f"Race {rid}")

# -------------------------
# 字体
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
# 中文指标名
# -------------------------
FEATURE_CN = {
    "gt_min": "对局时长(分钟)",
    "dmg_per_min": "每分钟伤害输出",
    "ekills_per_min": "每分钟击杀数",
    "edeaths_per_min": "每分钟阵亡数",
    "pcap_per_min": "每分钟占点数",
    "plost_per_min": "每分钟丢点数",
    "fuelearn_per_min": "每分钟燃料获取",
    "manearn_per_min": "每分钟人力获取",
    "munearn_per_min": "每分钟弹药获取",
    "vkill_per_min": "每分钟载具击杀数",
    "vlost_per_min": "每分钟载具损失数",
    "sqprod_per_min": "每分钟小队生产数",
    "sqlost_per_min": "每分钟小队损失数",
    "sqkilled_per_min": "每分钟击杀敌方小队数",
    "cap_pressure_per_min": "每分钟控图压力",
    "cap_eff": "控图效率",
    "man_spend_ratio": "人力消耗比",
    "mun_spend_ratio": "弹药消耗比",
    "fuel_spend_ratio": "燃料消耗比",
    "trade_eff_kd": "交换效率(K/D)",
    "dmg_per_kill": "每击杀伤害",
    "dmg_per_death": "每阵亡伤害",
    "dmg_per_man": "人力输出效率",
    "kills_per_man": "人力击杀效率",
    "dmg_per_fuel": "燃料输出效率",
    "vkill_per_fuel": "燃料换载具击杀效率",
    "vloss_per_fuel": "燃料载具损失强度",
    "squad_loss_ratio": "小队损失率",
    "cabil": "指挥官技能使用",
    "squad_net_ratio": "小队净收益比例",
    "bprod": "建筑生产数量",
    "man_float_ratio": "人力储备率",
}

# -------------------------
# 配置
# -------------------------
@dataclass
class VizConfig:
    out_dir: str = "models/task2_calib/player_viz"
    topk_features: int = 12
    sort_by_abs_diff: bool = True
    min_gt_min: float = 3.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # 可视化尺度
    scale_mode: str = "ratio"   # 'ratio' | 'diff' | 'raw'
    annotate_values: bool = True
    decimals: int = 2
    use_log_for_raw: bool = False  # raw 模式下是否使用对数轴

# -------------------------
# IO
# -------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_player_id_from_jsonl(player_jsonl: str) -> Optional[int]:
    base = os.path.splitext(os.path.basename(player_jsonl))[0]
    return int(base) if base.isdigit() else None

# -------------------------
# 全体均值
# -------------------------
def get_global_mean_by_race(global_means: Dict[str, Any], race_id: int) -> Dict[str, float]:
    feature_keys = global_means["feature_keys"]
    for row in global_means.get("global_by_race", []):
        if int(row.get("race_id")) == int(race_id):
            return {k: float(row.get(k, 0.0)) for k in feature_keys}
    return {k: float(global_means["global_all"].get(k, 0.0)) for k in feature_keys}

# -------------------------
# 玩家均值（从 jsonl 重算）
# -------------------------
def compute_player_feature_means_by_race(player_jsonl: str, cfg: VizConfig, target_pid: Optional[int], feature_keys: List[str]) -> Dict[int, Dict[str, float]]:
    df = parse_player_file(
        player_jsonl,
        AggConfig(min_gt_min=cfg.min_gt_min, min_games_per_race=10),
        target_profile_id=target_pid,
    )
    if df.empty:
        return {}
    # 重命名 f_* -> 裸名
    rename = {c: c[2:] for c in df.columns if c.startswith("f_")}
    df = df.rename(columns=rename)
    cols = ["race_id"] + [k for k in feature_keys if k in df.columns]
    df = df[cols].copy()
    out = {}
    for rid, sub in df.groupby("race_id"):
        out[int(rid)] = {k: float(sub[k].mean()) for k in feature_keys if k in sub.columns}
    return out

# -------------------------
# 权重/分位
# -------------------------
def load_weights(weights_json: Optional[str], w1: float, w2: float) -> Tuple[float, float]:
    if weights_json and os.path.exists(weights_json):
        try:
            data = load_json(weights_json)
            return float(data.get("w_1v1", w1)), float(data.get("w_2v2", w2))
        except Exception:
            pass
    return w1, w2

def load_skill_bins(skill_bins_json: Optional[str]) -> Optional[List[float]]:
    if not skill_bins_json or not os.path.exists(skill_bins_json):
        return None
    data = load_json(skill_bins_json)
    if isinstance(data, dict):
        if "bins" in data and isinstance(data["bins"], list):
            return list(sorted(float(x) for x in data["bins"]))
        if "overall" in data and "bins" in data["overall"]:
            return list(sorted(float(x) for x in data["overall"]["bins"]))
    return None

def tier_from_skill(skill: float, bins: Optional[List[float]]) -> int:
    if bins:
        cnt = sum(1 for b in bins if skill > b)
        return int(np.clip(1 + cnt, 1, 20))
    return int(np.clip(np.floor(skill * 20.0) + 1, 1, 20))

# -------------------------
# 必要时重算每局 win_prob -> 每阵营 skill_score
# -------------------------
def load_race_model(model_dir: str, race_id: int, device: str):
    meta_path = os.path.join(model_dir, f"calib_race{race_id}.json")
    pt_path = os.path.join(model_dir, f"calib_race{race_id}.pt")
    if not (os.path.exists(meta_path) and os.path.exists(pt_path)):
        return None
    meta = load_json(meta_path)
    feat_cols = meta["feature_cols"]  # 裸名
    mu = np.array(meta["scaler_mean"], dtype=np.float32)
    sd = np.array(meta["scaler_std"], dtype=np.float32)
    hidden = tuple(meta["config"]["hidden_sizes"])
    dropout = float(meta["config"]["dropout"])
    model = CalibMLP(in_dim=len(feat_cols), hidden=hidden, dropout=dropout).to(device)
    state = torch.load(pt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "feat_cols": feat_cols, "mu": mu, "sd": sd, "device": device}

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

def predict_win_prob(bundle, row_dict: Dict[str, float]) -> float:
    feat_cols = bundle["feat_cols"]
    mu = bundle["mu"]; sd = bundle["sd"]
    device = bundle["device"]
    x = np.array([float(row_dict.get(col, 0.0)) for col in feat_cols], dtype=np.float32)
    x = (x - mu) / (sd + 1e-8)
    xb = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logit = bundle["model"](xb).detach().cpu().numpy()[0]
    return float(sigmoid(np.array([logit]))[0])

def recompute_skill_from_models(player_jsonl: str, cfg: VizConfig, model_dir: str, target_pid: Optional[int], w1: float, w2: float, min_games_per_race: int = 10) -> Dict[int, float]:
    """
    从玩家 jsonl 重算每阵营 skill_score：
    - 优先 1v1/2v2，按 7:3 融合
    - 不足则 team fallback
    """
    df = parse_player_file(
        player_jsonl,
        AggConfig(min_gt_min=cfg.min_gt_min, min_games_per_race=min_games_per_race),
        target_profile_id=target_pid,
    )
    if df.empty:
        return {}

    # 重命名 f_* -> 裸名，避免取值失败
    rename = {c: c[2:] for c in df.columns if c.startswith("f_")}
    df = df.rename(columns=rename)

    skills = {}
    for rid in RACE_IDS:
        sub = df[df["race_id"] == rid].copy()
        if len(sub) < min_games_per_race:
            continue
        bundle = load_race_model(model_dir, rid, cfg.device)
        if bundle is None:
            continue

        probs_1v1, probs_2v2, probs_team = [], [], []
        for _, row in sub.iterrows():
            row_dict = build_row_dict(row.to_dict(), bundle["feat_cols"])
            p = predict_win_prob(bundle, row_dict)
            if row["mode"] == MODE_1V1:
                probs_1v1.append(p)
            elif row["mode"] == MODE_2V2:
                probs_2v2.append(p)
            else:
                probs_team.append(p)

        if (len(probs_1v1) + len(probs_2v2)) >= min_games_per_race and (len(probs_1v1) + len(probs_2v2)) > 0:
            parts, ws = [], []
            if probs_1v1:
                parts.append(float(np.mean(probs_1v1))); ws.append(w1)
            if probs_2v2:
                parts.append(float(np.mean(probs_2v2))); ws.append(w2)
            ws = [w / sum(ws) for w in ws]
            skill = float(sum(p * w for p, w in zip(parts, ws)))
        else:
            skill = float(np.mean(probs_team)) if probs_team else 0.0
        skills[rid] = skill
    return skills

# -------------------------
# 绘图（每阵营 组合图），支持尺度选择与绝对值标注，使用阵营中文名
# -------------------------
def plot_combined_per_race(
    race_id: int,
    feature_keys: List[str],
    player_mean: Dict[str, float],
    global_mean: Dict[str, float],
    skill_score: Optional[float],
    pred_tier: Optional[int],
    out_dir: str,
    cfg: VizConfig,
):
    rows = []
    for k in feature_keys:
        if k in player_mean and k in global_mean:
            gm = global_mean.get(k, 0.0)
            pm = player_mean.get(k, 0.0)
            ratio = pm / gm if gm != 0 else np.nan
            diff = pm - gm
            rows.append({
                "feature": k,
                "指标中文": FEATURE_CN.get(k, k),
                "玩家均值": pm,
                "全体均值": gm,
                "ratio": ratio,
                "diff": diff,
                "abs_diff": abs(diff),
                "abs_ratio_diff": abs(ratio - 1.0) if np.isfinite(ratio) else np.nan,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # 排序依据
    sort_key = "abs_ratio_diff" if cfg.scale_mode == "ratio" else "abs_diff"
    df = df.sort_values(sort_key, ascending=False)
    df_top = df.head(cfg.topk_features).copy()

    # 选择绘制值
    if cfg.scale_mode == "ratio":
        plot_vals_global = np.ones(len(df_top))  # 全体=1
        plot_vals_player = df_top["ratio"].to_numpy()
        x_label = "玩家/全体 比值"
    elif cfg.scale_mode == "diff":
        plot_vals_global = np.zeros(len(df_top))  # 全体=0
        plot_vals_player = df_top["diff"].to_numpy()
        x_label = "玩家-全体 差值"
    else:  # raw
        plot_vals_global = df_top["全体均值"].to_numpy()
        plot_vals_player = df_top["玩家均值"].to_numpy()
        x_label = "均值（原始）"

    # 绘图
    plt.figure(figsize=(11, 7))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    df_plot = df_top.iloc[::-1]  # 倒序，顶部为差异最大
    y = np.arange(len(df_plot))

    ax1.barh(y - 0.18, plot_vals_global[::-1], height=0.35, label="全体", alpha=0.75, color="#999999")
    ax1.barh(y + 0.18, plot_vals_player[::-1], height=0.35, label="玩家", alpha=0.85, color="#4c72b0")

    ax1.set_yticks(y)
    ax1.set_yticklabels(df_plot["指标中文"])
    ax1.set_title(f"{race_label(race_id)} | 技术指标对比（Top{len(df_plot)}差异） | 模式={cfg.scale_mode}")
    ax1.set_xlabel(x_label)
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", linestyle=":", alpha=0.3)

    # raw 模式下支持对数轴，缓解量级不一致
    if cfg.scale_mode == "raw" and cfg.use_log_for_raw:
        ax1.set_xscale("symlog", linthresh=1e-3)

    # 绝对值标注：条形右侧标玩家/全体原始均值
    if cfg.annotate_values:
        for i, (_, row) in enumerate(df_plot.iterrows()):
            pm = row["玩家均值"]
            gm = row["全体均值"]
            # 取可见长度的最大值，并乘以1.02偏移
            vis_max = max(plot_vals_global[::-1][i], plot_vals_player[::-1][i]) if cfg.scale_mode != "raw" else max(pm, gm)
            ax1.text(
                x=vis_max * 1.02,
                y=i + 0.18,
                s=f"玩家={pm:.{cfg.decimals}f}",
                va="center", ha="left", fontsize=9, color="#4c72b0",
            )
            ax1.text(
                x=vis_max * 1.02,
                y=i - 0.18,
                s=f"全体={gm:.{cfg.decimals}f}",
                va="center", ha="left", fontsize=9, color="#555555",
            )

    # 下半：skill/tier
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    if skill_score is not None:
        ax2.bar([f"{race_label(race_id)} skill_score"], [skill_score], color="#4c72b0")
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis="y", linestyle=":", alpha=0.3)
    txt = f"{race_label(race_id)} 预测段位: {pred_tier if pred_tier is not None else 'N/A'}"
    if skill_score is not None:
        txt += f" | skill={skill_score:.{cfg.decimals}f}"
    ax2.text(0.02, 0.8, txt, transform=ax2.transAxes, fontsize=12, bbox=dict(facecolor="#f0f0f0", edgecolor="none"))
    ax2.set_title(f"{race_label(race_id)} | 段位预测与skill_score")
    ax2.set_xticks([])

    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout()
    # 文件名保留原样，但在CSV和图标题中已使用中文名
    plt.savefig(os.path.join(out_dir, f"race{race_id}_combined.png"), dpi=240)
    plt.close()

    # CSV 输出，附带 ratio 与 diff，并含 race_name
    df_top_out = df_top[["feature", "指标中文", "玩家均值", "全体均值", "ratio", "diff"]].copy()
    df_top_out.insert(0, "race_id", int(race_id))
    df_top_out.insert(1, "race_name", race_label(race_id))
    df_top_out.to_csv(
        os.path.join(out_dir, f"race{race_id}_feature_compare_top{len(df_top_out)}.csv"),
        index=False, encoding="utf-8-sig"
    )

# -------------------------
# 主流程
# -------------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_json", required=True)
    ap.add_argument("--player_jsonl", required=True)
    ap.add_argument("--global_means_json", default="models/task2_calib/global_feature_means.json")
    ap.add_argument("--skill_bins_json", default="")
    ap.add_argument("--weights_json", default="")
    ap.add_argument("--w1", type=float, default=0.7)
    ap.add_argument("--w2", type=float, default=0.3)
    ap.add_argument("--out_dir", default="models/task2_calib/player_viz")
    ap.add_argument("--topk", type=int, default=12)
    ap.add_argument("--min_gt_min", type=float, default=3.0)
    ap.add_argument("--scale_mode", choices=["ratio", "diff", "raw"], default="ratio")
    ap.add_argument("--annotate_values", type=int, default=1)  # 1/0
    ap.add_argument("--decimals", type=int, default=2)
    ap.add_argument("--use_log_for_raw", type=int, default=0)  # 1/0
    args = ap.parse_args()

    setup_chinese_font_ubuntu()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = VizConfig(
        out_dir=args.out_dir,
        topk_features=args.topk,
        min_gt_min=args.min_gt_min,
        scale_mode=args.scale_mode,
        annotate_values=bool(args.annotate_values),
        decimals=args.decimals,
        use_log_for_raw=bool(args.use_log_for_raw),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    infer_rep = load_json(args.infer_json)
    # 权重
    if args.weights_json and os.path.exists(args.weights_json):
        w1, w2 = load_weights(args.weights_json, args.w1, args.w2)
    else:
        w1, w2 = args.w1, args.w2
    # skill->tier bins
    bins = load_skill_bins(args.skill_bins_json)

    # 全体均值与特征集
    global_means = load_json(args.global_means_json)
    feature_keys = global_means["feature_keys"]

    # 目标玩家 id
    target_pid = infer_rep.get("target_profile_id", None)
    if target_pid is None:
        target_pid = get_player_id_from_jsonl(args.player_jsonl)

    # 玩家 per-race 均值
    player_means_by_race = compute_player_feature_means_by_race(
        args.player_jsonl, cfg, target_pid, feature_keys
    )

    # 读 skill_score；若缺失或异常小，则重算
    per_race_infer = infer_rep.get("per_race", {}) or {}
    used_skill: Dict[int, float] = {}
    for rid in RACE_IDS:
        info = per_race_infer.get(str(rid), {})
        if info.get("status") == "ok" and (info.get("skill_score") is not None):
            used_skill[rid] = float(info["skill_score"])

    need_recompute = False
    for rid in RACE_IDS:
        if (rid in player_means_by_race) and (rid not in used_skill):
            need_recompute = True
        elif rid in used_skill and used_skill[rid] < 1e-6:  # 异常极小值触发重算
            need_recompute = True

    if need_recompute:
        recomputed_skills = recompute_skill_from_models(
            args.player_jsonl, cfg, model_dir=os.path.dirname(args.global_means_json),
            target_pid=target_pid, w1=w1, w2=w2, min_games_per_race=10
        )
        for rid in recomputed_skills:
            used_skill[rid] = recomputed_skills[rid]

    # 绘图与汇总
    summary_rows = []
    for rid in RACE_IDS:
        if rid not in player_means_by_race:
            continue
        player_mean = player_means_by_race[rid]
        global_mean = get_global_mean_by_race(global_means, rid)

        skill = used_skill.get(rid, None)
        pred_tier = tier_from_skill(skill, bins) if skill is not None else None

        plot_combined_per_race(
            race_id=rid,
            feature_keys=feature_keys,
            player_mean=player_mean,
            global_mean=global_mean,
            skill_score=skill,
            pred_tier=pred_tier,
            out_dir=args.out_dir,
            cfg=cfg,
        )

        summary_rows.append({
            "race_id": rid,
            "race_name": race_label(rid),
            "pred_tier": int(pred_tier) if pred_tier is not None else None,
            "skill_score": float(skill) if skill is not None else None,
            "source": "infer_json" if (rid in per_race_infer and per_race_infer[str(rid)].get("skill_score") is not None) else "recomputed",
            "scale_mode": cfg.scale_mode,
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(args.out_dir, "pred_tier_summary.csv"),
            index=False, encoding="utf-8-sig"
        )

    # debug & 偏好（含中文阵营名）
    debug_obj = {
        "w_1v1": w1, "w_2v2": w2,
        "bins_loaded": bool(bins),
        "used_skill": {race_label(k): v for k, v in used_skill.items()},
        "target_profile_id": target_pid,
        "scale_mode": cfg.scale_mode,
        "annotate_values": cfg.annotate_values,
        "decimals": cfg.decimals,
        "use_log_for_raw": cfg.use_log_for_raw,
    }
    save_json(os.path.join(args.out_dir, "viz_debug.json"), debug_obj)

    # 将偏好也输出中文阵营名映射
    race_pref = infer_rep.get("race_preference", {})
    race_pref_named = {race_label(int(k)): v for k, v in race_pref.items()}

    pref_out = {
        "mode_preference": infer_rep.get("mode_preference", {}),
        "race_preference": race_pref,  # 原始
        "race_preference_named": race_pref_named,  # 中文名
    }
    save_json(os.path.join(args.out_dir, "player_preferences.json"), pref_out)

    print("[task2_viz_player_report] done:", args.out_dir)


if __name__ == "__main__":
    main()