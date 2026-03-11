import argparse
import datetime as dt
import json
import os
import sys
from typing import Iterable, List

import requests
import ijson
import io

BASE_URL = "https://storage.coh2stats.com/matches/matches-{ts}.json"


def utc_midnight_ts(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())


def build_timestamps(start: dt.date, end: dt.date) -> List[int]:
    cur = start
    tss = []
    while cur <= end:
        tss.append(utc_midnight_ts(cur))
        cur += dt.timedelta(days=1)
    return tss


def iter_day_matches(ts: int) -> Iterable[dict]:
    """
    获取某一天的全部 matches，自动处理 gzip 压缩。
    返回一个可迭代对象（使用 ijson 在内存缓冲区上迭代，不再直接读取 resp.raw）。
    """
    url = BASE_URL.format(ts=ts)
    headers = {
        "Accept": "application/json",
        # 服务器会返回 gzip；这里允许 gzip，之后用 resp.content 做解压后的字节缓冲
        "Accept-Encoding": "gzip, deflate",
        "User-Agent": "coh2stats-scraper/1.0",
    }
    resp = requests.get(url, stream=True, timeout=60, headers=headers)
    if resp.status_code == 404:
        # 当天没有文件
        return []
    resp.raise_for_status()

    # requests 会根据 Content-Encoding 自动解压到 resp.content
    # 用 BytesIO 包一层，再交给 ijson 迭代 'matches.item'
    buf = io.BytesIO(resp.content)
    try:
        return ijson.items(buf, "matches.item")
    except ijson.common.IncompleteJSONError as e:
        # 退回到一次性解析（极端情况下）
        data = resp.json()
        return data.get("matches", [])


def fetch_for_profile(profile_id: int, timestamps: List[int], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    seen_ids = set()
    count = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for ts in timestamps:
            try:
                for match in iter_day_matches(ts):
                    pids = match.get("profile_ids") or []
                    if profile_id not in pids:
                        continue
                    mid = match.get("id")
                    if mid in seen_ids:
                        continue
                    seen_ids.add(mid)
                    out.write(json.dumps(match, ensure_ascii=False) + "\n")
                    count += 1
            except requests.RequestException as e:
                print(f"[error] ts={ts} {e}", file=sys.stderr)
            except Exception as e:
                print(f"[error] ts={ts} unexpected: {e}", file=sys.stderr)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Fetch all matches for a COH2 profile_id over a date range (daily archives)."
    )
    parser.add_argument("--profile-id", type=int, required=True, help="COH2 profile_id, e.g. 5905567 for ak47k98")
    parser.add_argument("--out", default="data/players/{pid}.jsonl", help="Output JSONL path (use {pid} placeholder)")
    parser.add_argument("--days", type=int, default=60, help="How many past days to scan (max ~60)")
    parser.add_argument("--start", help="UTC start date YYYY-MM-DD")
    parser.add_argument("--end", help="UTC end date YYYY-MM-DD")
    args = parser.parse_args()

    today_utc = dt.datetime.now(dt.timezone.utc).date()
    if args.start and args.end:
        start = dt.date.fromisoformat(args.start)
        end = dt.date.fromisoformat(args.end)
    else:
        end = today_utc - dt.timedelta(days=1)  # 昨天
        start = end - dt.timedelta(days=args.days - 1)

    ts_list = build_timestamps(start, end)
    out_path = args.out.format(pid=args.profile_id)
    n = fetch_for_profile(args.profile_id, ts_list, out_path)
    print(f"[done] profile_id={args.profile_id} saved {n} matches to {out_path} across {len(ts_list)} days")


if __name__ == "__main__":
    main()
