import argparse
import json
import math
import os


def is_success(value):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return not (math.isnan(v) or math.isinf(v)) and v >= 0.5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Default: <path>/failed_episodes.txt",
    )
    args = parser.parse_args()

    log_dir = os.path.join(args.path, "log")
    output_path = args.output or os.path.join(args.path, "failed_episodes.txt")

    failed_ids = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.startswith("stats_") or not filename.endswith(".json"):
            continue
        with open(os.path.join(log_dir, filename)) as f:
            data = json.load(f)
        if not is_success(data.get("success")):
            failed_ids.append(data.get("id", filename[6:-5]))

    failed_ids.sort(key=lambda x: int(x))
    with open(output_path, "w") as f:
        f.write("\n".join(str(i) for i in failed_ids) + ("\n" if failed_ids else ""))

    print(f"Failed: {len(failed_ids)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
