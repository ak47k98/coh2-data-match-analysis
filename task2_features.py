import json
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

MODE_1V1 = "1v1"
MODE_2V2 = "2v2"
MODE_TEAM = "team"

RACE_IDS = [0, 1, 2, 3, 4]


def match_mode(maxplayers: Optional[int]) -> str:
    mp = int(maxplayers) if maxplayers is not None else 0
    if mp == 2:
        return MODE_1V1
    if mp == 4:
        return MODE_2V2
    return MODE_TEAM


def safe_json_loads(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return {}


def iter_matches_from_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def build_features_from_counters(c: Dict[str, Any]) -> Dict[str, float]:
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

        "vkill_per_min": per_min("vkill"),
        "vlost_per_min": per_min("vlost"),

        "sqprod_per_min": per_min("sqprod"),
        "sqlost_per_min": per_min("sqlost"),
        "sqkilled_per_min": per_min("sqkilled"),

        "man_spend_ratio": ratio("manspnt", "manearn"),
        "mun_spend_ratio": ratio("munspnt", "munearn"),
        "fuel_spend_ratio": ratio("fuelspnt", "fuelearn"),

        "man_float_ratio": ratio("manmax", "manearn"),
        "mun_float_ratio": ratio("munmax", "munearn"),
        "fuel_float_ratio": ratio("fuelmax", "fuelearn"),

        "trade_eff_kd": ratio("ekills", "edeaths"),
        "dmg_per_kill": ratio("dmgdone", "ekills"),
        "dmg_per_death": ratio("dmgdone", "edeaths"),

        "dmg_per_man": ratio("dmgdone", "manearn"),
        "kills_per_man": ratio("ekills", "manearn"),
        "dmg_per_fuel": ratio("dmgdone", "fuelspnt"),
        "vkill_per_fuel": ratio("vkill", "fuelspnt"),
        "vloss_per_fuel": ratio("vlost", "fuelspnt"),

        "squad_loss_ratio": ratio("sqlost", "sqprod"),
        "squad_net_ratio": (float(c.get("sqprod", 0.0) or 0.0) - float(c.get("sqlost", 0.0) or 0.0)) / (float(c.get("sqprod", 0.0) or 0.0) + 1.0),

        "utypes": float(c.get("utypes", 0.0) or 0.0),
        "upg": float(c.get("upg", 0.0) or 0.0),
        "abil": float(c.get("abil", 0.0) or 0.0),
        "cabil": float(c.get("cabil", 0.0) or 0.0),
        "bprod": float(c.get("bprod", 0.0) or 0.0),
        "blost": float(c.get("blost", 0.0) or 0.0),

        "svetxp_per_min": per_min("svetxp"),
        "vvetxp_per_min": per_min("vvetxp"),
        "svetrank": float(c.get("svetrank", 0.0) or 0.0),
        "vvetrank": float(c.get("vvetrank", 0.0) or 0.0),
    }
    return feats


@dataclass
class AggConfig:
    min_gt_min: float = 3.0
    min_games_per_race: int = 10


def parse_player_file(
    player_jsonl: str,
    agg_cfg: AggConfig,
    *,
    target_profile_id: Optional[int],
) -> pd.DataFrame:
    rows = []
    for match in iter_matches_from_jsonl(player_jsonl):
        maxplayers_raw = match.get("maxplayers", None)
        maxplayers = int(maxplayers_raw) if maxplayers_raw is not None else 0
        mode = match_mode(maxplayers)
        start_raw = match.get("startgametime", None)
        start = int(start_raw) if start_raw is not None else 0

        for r in match.get("matchhistoryreportresults", []) or []:
            pid_raw = r.get("profile_id", None)
            pid = int(pid_raw) if pid_raw is not None else -1
            if target_profile_id is not None and pid != int(target_profile_id):
                continue

            race_raw = r.get("race_id", None)
            race_id = int(race_raw) if race_raw is not None else -1  # FIX: do NOT use "or -1"
            if race_id not in RACE_IDS:
                continue

            counters = r.get("counters", "{}")
            c = safe_json_loads(counters) if isinstance(counters, str) else (counters or {})
            feats = build_features_from_counters(c)

            if feats["gt_min"] < agg_cfg.min_gt_min:
                continue

            row = {
                "profile_id": pid,
                "race_id": race_id,
                "mode": mode,
                "startgametime": start,
            }
            for k, v in feats.items():
                row[f"f_{k}"] = float(v)
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df