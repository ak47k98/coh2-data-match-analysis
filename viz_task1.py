import os
import glob
import json
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import seaborn as sns

# ======================
# 常量定义
# ======================

RACE_IDS = [0, 1, 2, 3, 4]
MODE_1V1 = "1v1"
MODE_NVN = "nvn"

RACE_NAME = {
    0: "东德",
    1: "苏联",
    2: "西德",
    3: "美国",
    4: "英国",
}

MODE_NAME = {
    MODE_1V1: "单挑(1v1)",
    MODE_NVN: "团战(nvn)",
}

METRIC_CN = {
    "n_samples": "样本数",
    "pos_rate": "正样本比例",
    "lr_auc": "逻辑回归AUC",
    "rf_auc": "随机森林AUC",
    "lr_acc": "逻辑回归准确率",
    "rf_acc": "随机森林准确率",
}

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
    "bprod" : "建筑生产数量",
    "man_float_ratio" : "人力储备率",
    
}

FALLBACK_TO_RAW_FEATURE_NAME = True

# ======================
# 基础工具
# ======================

def race_label(race_id: int) -> str:
    return f"{RACE_NAME.get(race_id, '未知')}({race_id})"

def mode_label(mode: str) -> str:
    return MODE_NAME.get(mode, mode)

def feature_cn(raw: str) -> str:
    if raw in FEATURE_CN:
        return FEATURE_CN[raw]
    return raw if FALLBACK_TO_RAW_FEATURE_NAME else "未命名特征"

def setup_chinese_font_ubuntu():
    candidates = [
        "Noto Sans CJK SC",
        "WenQuanYi Micro Hei",
        "Source Han Sans SC",
        "AR PL UMing CN",
        "DejaVu Sans",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    print("[警告] 未检测到中文字体，图片可能乱码")
    return None

def load_reports(report_dir: str) -> List[Dict[str, Any]]:
    reports = []
    for fp in glob.glob(os.path.join(report_dir, "report_*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            reports.append(json.load(f))
    return reports

# ======================
# 原有 Top5 图（未破坏）
# ======================

def plot_top5_bar(items, value_key, title, xlabel, save_path):
    if not items:
        return
    feats = [feature_cn(x["feature"]) for x in items][::-1]
    vals = [x[value_key] for x in items][::-1]
    plt.figure(figsize=(10, 4))
    plt.barh(feats, vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(save_path, dpi=220)
    plt.close()

# ======================
# 新增增强可视化
# ======================

def plot_auc_acc_samples(df, mode, out_dir):
    sub = df[df["_mode_raw"] == mode]
    if sub.empty:
        return

    sub = sub.sort_values("_race_id_raw")
    x = np.arange(len(sub))
    labels = [race_label(x) for x in sub["_race_id_raw"]]
    samples = sub[METRIC_CN["n_samples"]]

    plt.figure(figsize=(11, 4.5))
    plt.plot(x, sub[METRIC_CN["lr_auc"]], marker="o", label="LR AUC")
    plt.plot(x, sub[METRIC_CN["rf_auc"]], marker="o", label="RF AUC")
    plt.plot(x, sub[METRIC_CN["lr_acc"]], "--", label="LR ACC", alpha=0.7)
    plt.plot(x, sub[METRIC_CN["rf_acc"]], "--", label="RF ACC", alpha=0.7)
    plt.scatter(x, sub[METRIC_CN["lr_auc"]],
                s=samples / samples.max() * 800,
                alpha=0.25, color="gray", label="样本量")

    plt.xticks(x, labels)
    plt.ylim(0.4, 1.02)
    plt.title(f"模型性能综合对比 | {mode_label(mode)}")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"模型性能综合_{mode_label(mode)}.png"), dpi=260)
    plt.close()

def plot_metric_heatmap(df, mode, out_dir):
    sub = df[df["_mode_raw"] == mode]
    if sub.empty:
        return

    sub = sub.sort_values("_race_id_raw")
    mat = sub[
        [
            METRIC_CN["lr_auc"],
            METRIC_CN["rf_auc"],
            METRIC_CN["lr_acc"],
            METRIC_CN["rf_acc"],
        ]
    ]
    mat.index = [race_label(x) for x in sub["_race_id_raw"]]

    plt.figure(figsize=(8, 4))
    sns.heatmap(mat, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title(f"模型指标热力图 | {mode_label(mode)}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"指标热力图_{mode_label(mode)}.png"), dpi=260)
    plt.close()

# ======================
# 主函数（接口不变）
# ======================

def main():
    report_dir = "models/task1"
    out_dir = "models/task1/viz"
    os.makedirs(out_dir, exist_ok=True)

    setup_chinese_font_ubuntu()

    reports = load_reports(report_dir)
    if not reports:
        print(f"未找到报告文件：{report_dir}")
        return

    rows = []
    for r in reports:
        m = r["metrics"]
        rows.append({
            "对战模式": mode_label(r["mode"]),
            "阵营/国家": race_label(r["race_id"]),
            METRIC_CN["n_samples"]: m["n_samples"],
            METRIC_CN["pos_rate"]: m["pos_rate"],
            METRIC_CN["lr_auc"]: m["lr_auc"],
            METRIC_CN["rf_auc"]: m["rf_auc"],
            METRIC_CN["lr_acc"]: m["lr_acc"],
            METRIC_CN["rf_acc"]: m["rf_acc"],
            "_mode_raw": r["mode"],
            "_race_id_raw": r["race_id"],
        })

    df = pd.DataFrame(rows).sort_values(["_mode_raw", "_race_id_raw"])
    df.drop(columns=["_mode_raw", "_race_id_raw"]).to_csv(
        os.path.join(out_dir, "汇总指标表_summary_metrics.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # 原有 Top5
    for r in reports:
        mode_cn = mode_label(r["mode"])
        race_cn = race_label(r["race_id"])

        plot_top5_bar(
            r["logistic"]["top5"],
            "coef",
            f"Top5 致胜因素（逻辑回归）| {mode_cn} | {race_cn}",
            "系数",
            os.path.join(out_dir, f"Top5_LR_{mode_cn}_{race_cn}.png"),
        )

        plot_top5_bar(
            r["random_forest"]["top5"],
            "importance",
            f"Top5 致胜因素（随机森林）| {mode_cn} | {race_cn}",
            "重要性",
            os.path.join(out_dir, f"Top5_RF_{mode_cn}_{race_cn}.png"),
        )

    # 新增增强图
    for mode in [MODE_1V1, MODE_NVN]:
        plot_auc_acc_samples(df, mode, out_dir)
        plot_metric_heatmap(df, mode, out_dir)

    print(f"[viz_task1] 完成：输出已保存至 {out_dir}")

if __name__ == "__main__":
    main()

