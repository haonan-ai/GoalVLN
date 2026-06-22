import argparse
import gzip
import json
import os
import random


def load_episode_ids(data_path):
    with gzip.open(data_path, "rt") as f:
        data = json.load(f)
    return [str(ep["episode_id"]) for ep in data["episodes"]]


def main():
    parser = argparse.ArgumentParser(
        description="Sample episode ids from a VLN dataset split for SELECTED_EPISODES_FILE."
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="val_unseen",
        help="Dataset split to sample from (default: val_unseen)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Path to dataset json.gz; default: "
            "data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=200,
        help="Number of episodes to sample (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/selected_episodes_val_unseen_200.txt",
        help="Output path, one episode id per line",
    )
    args = parser.parse_args()

    split = args.dataset_split
    data_path = args.data_path or (
        f"data/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz"
    )
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    all_ids = load_episode_ids(data_path)
    if args.num_episodes > len(all_ids):
        raise ValueError(
            f"Requested {args.num_episodes} episodes but split "
            f"'{split}' only has {len(all_ids)}"
        )

    rng = random.Random(args.seed)
    selected = sorted(rng.sample(all_ids, args.num_episodes), key=int)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(selected) + "\n")

    print(f"Split: {split}")
    print(f"Dataset: {data_path}")
    print(f"Total episodes in split: {len(all_ids)}")
    print(f"Sampled: {len(selected)} (seed={args.seed})")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
