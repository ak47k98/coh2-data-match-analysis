import os
import pandas as pd
import numpy as np
import argparse

from task2_features import AggConfig, parse_player_file

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--player_jsonl", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--min_gt_min", type=float, default=3.0)
    ap.add_argument("--target_profile_id", type=int, default=None)
    args = ap.parse_args()

    target_pid = args.target_profile_id
    if target_pid is None:
        base = os.path.splitext(os.path.basename(args.player_jsonl))[0]
        if base.isdigit():
            target_pid = int(base)
    if target_pid is None:
        raise ValueError("Cannot infer target_profile_id from filename; please pass --target_profile_id")

    df = parse_player_file(
        args.player_jsonl,
        AggConfig(min_gt_min=args.min_gt_min),
        target_profile_id=target_pid,
    )
    if df.empty:
        raise ValueError("No valid matches after filter")

    # convert f_* -> raw feature name for CSV readability
    rename = {c: c[2:] for c in df.columns if c.startswith("f_")}
    df = df.rename(columns=rename)

    df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print("[task2_export_player_matches] saved:", args.out_csv, "rows=", len(df))

if __name__ == "__main__":
    main()