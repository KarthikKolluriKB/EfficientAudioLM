import argparse
import json
import os
import random
import sys
import torchaudio

from pathlib import Path
from typing import List, Dict, Any, Tuple


def audio_duration_seconds(path: str) -> float:
    """Return duration in seconds using torchaudio.info without decoding."""
    try:
        sinfo = torchaudio.info(path) 
        # Some containers may not expose num_frames; handle defensively
        if sinfo.num_frames is not None and sinfo.sample_rate and sinfo.sample_rate > 0:
            return float(sinfo.num_frames) / float(sinfo.sample_rate)
    except Exception:
        pass
    # Fallback: open and derive via length if needed (slower)
    try:
        wav, sr = torchaudio.load(path)
        return wav.size(-1) / float(sr)
    except Exception:
        return 0.0


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"Skipping malformed line: {e}", file=sys.stderr)
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


def summarize(rows: List[Dict[str, Any]]) -> Tuple[int, float]:
    n = len(rows)
    total_sec = 0.0
    for r in rows:
        total_sec += float(r.get("_duration_sec", 0.0))
    return n, total_sec


def make_subset(
    src_jsonl: str,
    dst_jsonl: str,
    target_hours: float = 20.0,
    seed: int = 42,
    min_dur_sec: float = 0.2,
    max_dur_sec: float = 45.0,
    verify_paths: bool = True,
) -> None:
    rows = read_jsonl(src_jsonl)
    if not rows:
        print("No rows found in source JSONL; aborting.", file=sys.stderr)
        sys.exit(1)

    # Filter rows with missing files or extreme durations
    eligible: List[Dict[str, Any]] = []
    for r in rows:
        src = r.get("source")
        if not isinstance(src, str):
            continue
        if verify_paths and not os.path.isfile(src):
            continue
        dur = audio_duration_seconds(src)
        if dur <= 0.0:
            continue
        if dur < min_dur_sec or dur > max_dur_sec:
            continue
        r["_duration_sec"] = dur
        eligible.append(r)

    if not eligible:
        print("No eligible rows after filtering; check paths and durations.", file=sys.stderr)
        sys.exit(1)

    random.seed(seed)
    random.shuffle(eligible)

    target_sec = target_hours * 3600.0
    picked: List[Dict[str, Any]] = []
    acc = 0.0
    for r in eligible:
        picked.append(r)
        acc += float(r["_duration_sec"])
        if acc >= target_sec:
            break

    # Strip helper field before writing
    for r in picked:
        r.pop("_duration_sec", None)

    write_jsonl(picked, dst_jsonl)

    n_full, total_full = summarize(eligible)
    n_sel, total_sel = summarize([{**r, "_duration_sec": audio_duration_seconds(r["source"])} for r in picked])

    print(f"Source: {src_jsonl}")
    print(f"Eligible clips: {n_full}, total ≈ {total_full/3600.0:.2f} h")
    print(f"Wrote subset: {dst_jsonl}")
    print(f"Subset clips: {n_sel}, total ≈ {total_sel/3600.0:.2f} h (target {target_hours:.1f} h)")


def main():
    ap = argparse.ArgumentParser(description="Create subset JSONL from a larger JSONL.")
    ap.add_argument("--src_jsonl", type=str, required=True, help="Path to source train JSONL")
    ap.add_argument("--dst_jsonl", type=str, required=True, help="Output subset JSONL path")
    ap.add_argument("--hours", type=float, default=20.0, help="Target hours (default: 20)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--min_dur", type=float, default=0.5, help="Min clip duration (sec) to include")
    ap.add_argument("--max_dur", type=float, default=30.0, help="Max clip duration (sec) to include")
    ap.add_argument("--no_verify", action="store_true", help="Skip file existence checks for speed")
    args = ap.parse_args()

    make_subset(
        src_jsonl=args.src_jsonl,
        dst_jsonl=args.dst_jsonl,
        target_hours=args.hours,
        seed=args.seed,
        min_dur_sec=args.min_dur,
        max_dur_sec=args.max_dur,
        verify_paths=not args.no_verify,
    )


if __name__ == "__main__":
    main()
