import os
import json
import argparse
from pathlib import Path
import torchaudio

SPLITS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

def parse_args():
    p = argparse.ArgumentParser(description="Create JSONL for LibriSpeech split")  # basic CLI parser [2][1]
    p.add_argument("--root", type=str, default="data", help="Dataset root containing LibriSpeech/")  # [2]
    p.add_argument("--split", type=str, default="test-clean", choices=SPLITS, help="LibriSpeech subset to use")  # [21]
    p.add_argument("--out", type=str, default=None, help="Output JSONL; default: <root>/<split>.jsonl")  # [2]
    p.add_argument("--absolute", action="store_true", help="Store absolute audio paths in JSONL")  # [2]
    p.add_argument("--limit", type=int, default=None, help="Limit number of examples for quick tests")  # [2]
    return p.parse_args()  # returns argparse.Namespace [2]

def get_librispeech_dataset(root, split):
    # Download/load only the requested split
    return torchaudio.datasets.LIBRISPEECH(root=root, url=split, download=True)  # split list in docs [21]

def get_base_dir(root):
    # Base dir for joining rel paths returned by get_metadata
    return Path(root) / "LibriSpeech"  # root/LibriSpeech [21]

def resolve_output_path(args):
    # Resolve output path and ensure parent exists (safe even if just a filename) [10]
    out_path = Path(args.out) if args.out else (Path(args.root) / f"{args.split}.jsonl")  # default <root>/<split>.jsonl [2]
    out_path.parent.mkdir(parents=True, exist_ok=True)  # robust mkdir of parent [10]
    return out_path

def write_jsonl(ds, base, out_path, absolute, limit):
    n_total = len(ds)
    n_write = n_total if limit is None else min(limit, n_total)
    print(f"[INFO] root={base.parent} split={ds._url} total={n_total} writing={n_write}")  # debug print [2]

    with out_path.open("w", encoding="utf-8") as f:
        for idx in range(n_write):
            # get_metadata: (rel_path, sample_rate, transcript, speaker_id, chapter_id, utterance_id) [21]
            rel_path, sample_rate, transcript, speaker_id, chapter_id, utterance_id = ds.get_metadata(idx)  # [21]
            full_path = (base / rel_path)  # join to root/LibriSpeech/<split>/<spk>/<chap>/<utt>.flac [21]
            if absolute:
                full_path = full_path.resolve()  # produce absolute path if requested [10]

            entry = {
                "source": str(full_path),
                "target": transcript,
                "key": f"{speaker_id}-{chapter_id}-{utterance_id}",
            }
            f.write(json.dumps(entry) + "\n")

            # Debug: show first few lines
            if idx < 3:
                print(f"[DEBUG] idx={idx} rel={rel_path} -> source={entry['source']}")  # quick sanity [2]

    print(f"[DONE] Wrote {n_write} lines to {out_path}")  # completion message [2]

def main():
    args = parse_args()  # parse CLI args [2]
    ds = get_librispeech_dataset(args.root, args.split)
    base = get_base_dir(args.root)
    out_path = resolve_output_path(args)
    write_jsonl(ds, base, out_path, args.absolute, args.limit)

if __name__ == "__main__":
    main()  # standard CLI entrypoint [2]
