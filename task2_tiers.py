import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

MODE_1V1 = "1v1"
MODE_2V2 = "2v2"
MODE_TEAM = "team"

BOTTOM_PERCENT = {
    15: 0.95,
    14: 0.90,
    13: 0.85,
    12: 0.80,
    11: 0.75,
    10: 0.69,
    9:  0.65,
    8:  0.60,
    7:  0.55,
    6:  0.50,
    5:  0.40,
    4:  0.30,
    3:  0.20,
    2:  0.10,
}


@dataclass
class TierConfig:
    """
    Tier assignment per mode (global leaderboard approximation from your dataset).
    """
    top200_enabled: bool = True
    bottom_percent: Dict[int, float] = None

    def __post_init__(self):
        if self.bottom_percent is None:
            self.bottom_percent = dict(BOTTOM_PERCENT)


def assign_tiers_per_mode(players: pd.DataFrame, cfg: TierConfig) -> pd.DataFrame:
    """
    players: columns: [mode, profile_id, oldrating, startgametime]
    Returns: same + [rank, tier]
    """
    out = []

    for mode, g in players.groupby("mode"):
        g = g.sort_values("oldrating", ascending=False).reset_index(drop=True).copy()
        g["rank"] = np.arange(len(g)) + 1
        g["tier"] = np.nan

        if cfg.top200_enabled:
            g.loc[g["rank"].between(1, 2), "tier"] = 20
            g.loc[g["rank"].between(3, 13), "tier"] = 19
            g.loc[g["rank"].between(14, 36), "tier"] = 18
            g.loc[g["rank"].between(37, 80), "tier"] = 17
            g.loc[g["rank"].between(81, 200), "tier"] = 16

        remaining = g[g["tier"].isna()].copy()
        remaining_count = len(remaining)

        if remaining_count > 0:
            # bottom-based assignment needs ascending
            remaining = remaining.sort_values("oldrating", ascending=True)

            for tier, pct in cfg.bottom_percent.items():
                cutoff = int(math.floor(remaining_count * pct))
                if cutoff <= 0:
                    continue
                idx = remaining.iloc[:cutoff].index
                g.loc[idx, "tier"] = tier

            g["tier"] = g["tier"].fillna(1)

        g["tier"] = g["tier"].astype(int)
        out.append(g)

    return pd.concat(out, ignore_index=True)


def build_player_latest_table(matches: pd.DataFrame) -> pd.DataFrame:
    """
    matches: match-level rows, must include [profile_id, mode, startgametime, oldrating]
    Returns per (mode, profile_id) latest oldrating (by time)
    """
    players = (
        matches.sort_values("startgametime")
        .groupby(["mode", "profile_id"], as_index=False)
        .last()
    )
    return players


def build_tier_lookup(matches: pd.DataFrame, cfg: TierConfig) -> Dict[Tuple[str, int], int]:
    """
    Build lookup: (mode, profile_id) -> tier
    """
    players = build_player_latest_table(matches)
    tiers = assign_tiers_per_mode(players, cfg)
    lookup = {(row["mode"], int(row["profile_id"])): int(row["tier"]) for _, row in tiers.iterrows()}
    return lookup