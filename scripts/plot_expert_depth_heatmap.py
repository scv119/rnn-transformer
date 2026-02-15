#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot expert activation heatmap by depth from JSONL logs")
    p.add_argument("--jsonl", required=True, help="Path to depth heatmap JSONL")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--step", type=int, default=None, help="Use record at or before this optimizer step")
    p.add_argument("--mode", choices=["last", "mean"], default="last", help="last record or mean over all records")
    return p.parse_args()


def load_records(path: Path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No records in {path}")
    return records


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.jsonl))

    if args.step is not None:
        records = [r for r in records if int(r.get("opt_step", -1)) <= args.step]
        if not records:
            raise RuntimeError(f"No records at or before step {args.step}")

    if args.mode == "mean":
        mats = [np.array(r["load_by_layer"], dtype=np.float32) for r in records]
        mat = np.mean(np.stack(mats, axis=0), axis=0)
        title = f"Expert activation load by depth (mean over {len(records)} records)"
    else:
        rec = records[-1]
        mat = np.array(rec["load_by_layer"], dtype=np.float32)
        title = f"Expert activation load by depth @ step {rec.get('opt_step')}"

    plt.figure(figsize=(10, 5))
    im = plt.imshow(mat, aspect="auto", interpolation="nearest", cmap="magma")
    plt.colorbar(im, label="Routed load")
    plt.xlabel("Expert id")
    plt.ylabel("Depth (layer index)")
    plt.title(title)
    plt.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=180)
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
