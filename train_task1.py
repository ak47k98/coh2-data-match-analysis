import os
import glob
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
)
import joblib


# ---------- Config ----------
RACE_IDS = [0, 1, 2, 3, 4]  # user要求先按 0~4 处理
MODE_1V1 = "1v1"
MODE_NVN = "nvn"


@dataclass
class TrainConfig:
    data_glob: str = "data/players/*.jsonl"
    out_dir: str = "models/task1"
    random_state: int = 42
    test_size: float = 0.2
    min_samples_per_race: int = 300  # 太少会不稳定，你可以按实际调
    rf_n_estimators: int = 400


# ---------- Parsing ----------
def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


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


def match_mode(maxplayers: int) -> str:
    return MODE_1V1 if maxplayers == 2 else MODE_NVN


# ---------- Feature Engineering ----------
def build_features_from_counters(c: Dict[str, Any]) -> Dict[str, float]:
    # raw
    gt = float(c.get("gt", 0.0))  # seconds
    minutes = max(gt / 60.0, 1e-6)

    def per_min(key: str) -> float:
        return float(c.get(key, 0.0)) / minutes

    def ratio(num_key: str, den_key: str) -> float:
        return float(c.get(num_key, 0.0)) / (float(c.get(den_key, 0.0)) + 1.0)

    # core per-min
    feats = {
        "gt_min": minutes,
        "dmg_per_min": per_min("dmgdone"),
        "ekills_per_min": per_min("ekills"),
        "edeaths_per_min": per_min("edeaths"),
        "pcap_per_min": per_min("pcap"),
        "plost_per_min": per_min("plost"),
        "fuelearn_per_min": per_min("fuelearn"),
        "manearn_per_min": per_min("manearn"),
        "munearn_per_min": per_min("munearn"),
        "vkill_per_min": per_min("vkill"),
        "vlost_per_min": per_min("vlost"),
        "sqprod_per_min": per_min("sqprod"),
        "sqlost_per_min": per_min("sqlost"),
        "sqkilled_per_min": per_min("sqkilled"),
    }

    # capture intensity & stability
    pcap = float(c.get("pcap", 0.0))
    plost = float(c.get("plost", 0.0))
    feats.update({
        "cap_pressure_per_min": (pcap + plost) / minutes,
        "cap_eff": pcap / (pcap + plost + 1.0),  # 控图效率
    })

    # spend ratios (resource usage)
    feats.update({
        "man_spend_ratio": ratio("manspnt", "manearn"),
        "mun_spend_ratio": ratio("munspnt", "munearn"),
        "fuel_spend_ratio": ratio("fuelspnt", "fuelearn"),
        # float (advantage/overflow proxy)
        "man_float_ratio": ratio("manmax", "manearn"),
        "mun_float_ratio": ratio("munmax", "munearn"),
        "fuel_float_ratio": ratio("fuelmax", "fuelearn"),
    })

    # trade efficiency & "damage but no kill"
    feats.update({
        "trade_eff_kd": ratio("ekills", "edeaths"),          # 交换效率，不直接等同进攻性
        "dmg_per_kill": ratio("dmgdone", "ekills"),          # 伤害转化为击杀的效率（越低越“有效击杀”）
        "dmg_per_death": ratio("dmgdone", "edeaths"),        # 输出/损耗
    })

    # resource-exchange metrics (重点：你要求的资源交换比)
    feats.update({
        "dmg_per_man": ratio("dmgdone", "manearn"),
        "kills_per_man": ratio("ekills", "manearn"),
        "dmg_per_fuel": ratio("dmgdone", "fuelspnt"),        # 燃料投入的输出效率（含“装甲换血”倾向）
        "vkill_per_fuel": ratio("vkill", "fuelspnt"),        # 燃料换来的“重资产击杀 proxy”
        "vloss_per_fuel": ratio("vlost", "fuelspnt"),
    })

    # squad survival proxy
    feats.update({
        "squad_loss_ratio": ratio("sqlost", "sqprod"),
        "squad_net_ratio": (float(c.get("sqprod", 0.0)) - float(c.get("sqlost", 0.0))) / (float(c.get("sqprod", 0.0)) + 1.0),
    })

    # tech / composition
    feats.update({
        "utypes": float(c.get("utypes", 0.0)),
        "upg": float(c.get("upg", 0.0)),
        "abil": float(c.get("abil", 0.0)),
        "cabil": float(c.get("cabil", 0.0)),
        "bprod": float(c.get("bprod", 0.0)),
        "blost": float(c.get("blost", 0.0)),
    })

    # veterancy proxies (注意：优势局满星会封顶 -> 仍可作为一个维度，但不要当唯一)
    feats.update({
        "svetxp_per_min": per_min("svetxp"),
        "vvetxp_per_min": per_min("vvetxp"),
        "svetrank": float(c.get("svetrank", 0.0)),
        "vvetrank": float(c.get("vvetrank", 0.0)),
    })

    return feats


def build_dataset(cfg: TrainConfig) -> pd.DataFrame:
    rows = []
    for fp in glob.glob(cfg.data_glob):
        for match in iter_matches_from_jsonl(fp):
            maxplayers = int(match.get("maxplayers", 0) or 0)
            m_mode = match_mode(maxplayers)

            for r in match.get("matchhistoryreportresults", []):
                counters = r.get("counters", "{}")
                c = safe_json_loads(counters) if isinstance(counters, str) else (counters or {})

                row = {
                    "match_id": match.get("id"),
                    "mode": m_mode,
                    "maxplayers": maxplayers,
                    "profile_id": r.get("profile_id"),
                    "teamid": r.get("teamid"),
                    "race_id": r.get("race_id"),
                    "y_win": 1 if int(r.get("resulttype", 0)) == 1 else 0,
                }
                row.update(build_features_from_counters(c))
                rows.append(row)

    df = pd.DataFrame(rows)
    # basic cleaning
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # remove obviously broken rows (gt too small)
    df = df[df["gt_min"] >= 1.0].copy()
    return df


# ---------- Training per (mode, race) ----------
def train_one_group(
    df: pd.DataFrame, cfg: TrainConfig, mode: str, race_id: int
) -> Dict[str, Any]:
    g = df[(df["mode"] == mode) & (df["race_id"] == race_id)].copy()

    if len(g) < cfg.min_samples_per_race:
        return {"skipped": True, "reason": f"too few samples: {len(g)}"}

    # features: drop identifiers + label
    drop_cols = ["match_id", "mode", "maxplayers", "profile_id", "teamid", "race_id", "y_win"]
    feature_cols = [c for c in g.columns if c not in drop_cols]
    X = g[feature_cols].astype(float)
    y = g["y_win"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    # Logistic Regression (explainable)
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000,
            n_jobs=None,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])
    lr.fit(X_train, y_train)
    proba_lr = lr.predict_proba(X_test)[:, 1]
    pred_lr = (proba_lr >= 0.5).astype(int)

    # Random Forest (robust reference)
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_n_estimators,
        random_state=cfg.random_state,
        class_weight="balanced_subsample",
        n_jobs=-1,
        max_depth=None
    )
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    pred_rf = (proba_rf >= 0.5).astype(int)

    # metrics
    metrics = {
        "n_samples": int(len(g)),
        "pos_rate": float(y.mean()),
        "lr_acc": float(accuracy_score(y_test, pred_lr)),
        "lr_f1": float(f1_score(y_test, pred_lr)),
        "lr_auc": float(roc_auc_score(y_test, proba_lr)),
        "rf_acc": float(accuracy_score(y_test, pred_rf)),
        "rf_f1": float(f1_score(y_test, pred_rf)),
        "rf_auc": float(roc_auc_score(y_test, proba_rf)),
        "lr_confusion_matrix": confusion_matrix(y_test, pred_lr).tolist(),
        "rf_confusion_matrix": confusion_matrix(y_test, pred_rf).tolist(),
        "lr_report": classification_report(y_test, pred_lr, output_dict=True),
        "rf_report": classification_report(y_test, pred_rf, output_dict=True),
    }

    # coefficients for "winning formula"
    # Pipeline: scaler + clf
    coef = lr.named_steps["clf"].coef_[0]  # shape (n_features,)
    intercept = float(lr.named_steps["clf"].intercept_[0])
    coef_table = pd.DataFrame({
        "feature": feature_cols,
        "coef": coef,
        "abs_coef": np.abs(coef),
    }).sort_values("abs_coef", ascending=False)

    # RF importance
    rf_imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": rf.feature_importances_,
    }).sort_values("importance", ascending=False)

    # Save models
    os.makedirs(cfg.out_dir, exist_ok=True)
    lr_path = os.path.join(cfg.out_dir, f"lr_{mode}_race{race_id}.joblib")
    rf_path = os.path.join(cfg.out_dir, f"rf_{mode}_race{race_id}.joblib")
    joblib.dump({"model": lr, "feature_cols": feature_cols}, lr_path)
    joblib.dump({"model": rf, "feature_cols": feature_cols}, rf_path)

    # Save report json (Top5 + formula)
    top5_lr = coef_table.head(5)[["feature", "coef"]].to_dict(orient="records")
    top5_rf = rf_imp.head(5)[["feature", "importance"]].to_dict(orient="records")

    out = {
        "skipped": False,
        "mode": mode,
        "race_id": race_id,
        "metrics": metrics,
        "logistic": {
            "intercept": intercept,
            "top5": top5_lr,
            "coef_table": coef_table[["feature", "coef"]].to_dict(orient="records"),
        },
        "random_forest": {
            "top5": top5_rf,
            "importance_table": rf_imp.to_dict(orient="records"),
        },
        "model_paths": {"lr": lr_path, "rf": rf_path},
    }

    json_path = os.path.join(cfg.out_dir, f"report_{mode}_race{race_id}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return out


def main():
    cfg = TrainConfig()
    print(f"[Task1] Loading dataset from: {cfg.data_glob}")
    df = build_dataset(cfg)
    print(f"[Task1] Total player-samples: {len(df)}")
    print(df[["mode", "race_id", "y_win"]].groupby(["mode", "race_id"]).size())

    results = []
    for mode in [MODE_1V1, MODE_NVN]:
        for race_id in RACE_IDS:
            print(f"\n[Task1] Training mode={mode}, race_id={race_id} ...")
            res = train_one_group(df, cfg, mode, race_id)
            if res.get("skipped"):
                print("  -> skipped:", res.get("reason"))
            else:
                m = res["metrics"]
                print(f"  -> LR AUC={m['lr_auc']:.3f}, RF AUC={m['rf_auc']:.3f}, n={m['n_samples']}")
                print("  -> LR Top5:", [x["feature"] for x in res["logistic"]["top5"]])
                print("  -> RF Top5:", [x["feature"] for x in res["random_forest"]["top5"]])
            results.append(res)

    summary_path = os.path.join(cfg.out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[Task1] Done. Reports/models saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()