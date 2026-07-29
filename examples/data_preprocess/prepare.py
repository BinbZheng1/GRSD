#!/usr/bin/env python3
"""Create the lightweight Parquet schedules used by agent environments.

ALFWorld and WebShop obtain their actual task prompts from their environment
assets at reset time. The veRL dataloader still needs train/test Parquet files
to determine sampling and evaluation cardinality; this script creates those
files without downloading an unrelated placeholder dataset.
"""

import argparse
from pathlib import Path

from datasets import Dataset


TASK_DEFAULTS = {
    "alfworld": {"train": 16, "test": 128},
    "webshop": {"train": 16, "test": 128},
}


def build_schedule(task: str, split: str, size: int) -> Dataset:
    rows = []
    for index in range(size):
        rows.append(
            {
                "data_source": "text",
                "prompt": [{"role": "user", "content": ""}],
                "ability": "agent",
                "extra_info": {"task": task, "split": split, "index": index},
            }
        )
    return Dataset.from_list(rows)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Create ALFWorld or WebShop train/test Parquet schedules."
    )
    parser.add_argument("--task", required=True, choices=sorted(TASK_DEFAULTS))
    parser.add_argument(
        "--local-dir",
        "--local_dir",
        type=Path,
        help="Output directory (default: <repo>/data/<task>/metadata).",
    )
    parser.add_argument("--train-data-size", "--train_data_size", type=int)
    parser.add_argument("--val-data-size", "--val_data_size", type=int)
    args = parser.parse_args()
    if args.local_dir is None:
        args.local_dir = repo_root / "data" / args.task / "metadata"
    return args


def main() -> None:
    args = parse_args()
    defaults = TASK_DEFAULTS[args.task]
    train_size = defaults["train"] if args.train_data_size is None else args.train_data_size
    val_size = defaults["test"] if args.val_data_size is None else args.val_data_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError("Dataset sizes must be positive integers.")

    output_dir = args.local_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_schedule(args.task, "train", train_size).to_parquet(output_dir / "train.parquet")
    build_schedule(args.task, "test", val_size).to_parquet(output_dir / "test.parquet")
    print(
        f"Created {args.task} schedules in {output_dir} "
        f"(train={train_size}, test={val_size})."
    )


if __name__ == "__main__":
    main()
