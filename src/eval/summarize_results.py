import argparse
import json
import math
import os
import re
from pathlib import Path


def check_inf_nan(value):
    if math.isinf(value) or math.isnan(value):
        return 0
    return value


def analyze_checkpoint(path):
    log_dir = os.path.join(path, "log")
    if not os.path.isdir(log_dir):
        return None

    json_files = [
        f for f in os.listdir(log_dir) if f.endswith(".json")
    ]
    if not json_files:
        return None

    succ = 0
    for j in json_files:
        with open(os.path.join(log_dir, j)) as f:
            try:
                data = json.load(f)
                succ += check_inf_nan(int(data["success"]))
            except Exception:
                continue

    total = len(json_files)
    return {
        "success": succ,
        "total": total,
        "success_rate": succ / total,
    }


def checkpoint_sort_key(name):
    match = re.search(r"(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    return (1, name)


def find_checkpoints(root):
    root = Path(root)
    checkpoints = []

    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for ckpt_dir in sorted(exp_dir.iterdir(), key=lambda p: checkpoint_sort_key(p.name)):
            if not ckpt_dir.is_dir():
                continue
            if not (ckpt_dir / "log").is_dir():
                continue
            checkpoints.append((f"{exp_dir.name}/{ckpt_dir.name}", str(ckpt_dir)))

    return checkpoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="eval_log",
        help="Root directory containing experiment/checkpoint results",
    )
    args = parser.parse_args()

    checkpoints = find_checkpoints(args.path)
    if not checkpoints:
        print(f"No checkpoints found under {args.path}")
        return

    rows = []
    for name, ckpt_path in checkpoints:
        result = analyze_checkpoint(ckpt_path)
        if result is None:
            continue
        rows.append((name, result))

    if not rows:
        print(f"No valid results found under {args.path}")
        return

    name_width = max(len(name) for name, _ in rows)
    print(f"{'Run':<{name_width}}  Success rate")
    print("-" * (name_width + 20))
    for name, result in rows:
        succ = result["success"]
        total = result["total"]
        rate = result["success_rate"]
        print(f"{name:<{name_width}}  {succ}/{total} ({rate:.3f})")


if __name__ == "__main__":
    main()
