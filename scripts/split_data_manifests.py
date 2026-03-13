#!/usr/bin/env python3
import argparse
import math
import random
from collections import defaultdict
from pathlib import Path


def read_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")


def split_grouped(items, key_fn, val_ratio, test_ratio, seed):
    groups = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    group_keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    n_groups = len(group_keys)
    n_val = math.floor(n_groups * val_ratio)
    n_test = math.floor(n_groups * test_ratio)

    val_keys = set(group_keys[:n_val])
    test_keys = set(group_keys[n_val:n_val + n_test])

    train_items, val_items, test_items = [], [], []
    for key in group_keys:
        target = train_items
        if key in val_keys:
            target = val_items
        elif key in test_keys:
            target = test_items
        target.extend(groups[key])

    # Keep file order deterministic and easy to diff.
    train_items.sort()
    val_items.sort()
    test_items.sort()
    return train_items, val_items, test_items


def denoise_key(item: str):
    # One clean image per entry; grouping by filename avoids cross-split leakage.
    return Path(item).name


def rain_key(item: str):
    # rain13K uses input/xxxx.jpg <-> target/xxxx.jpg, key by basename stem.
    return Path(item).stem


def haze_key(item: str):
    # synthetic/OTS/0025_0.8_0.04.jpg -> group by clean source id 0025
    name = Path(item).name
    return name.split("_")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file_dir", type=str, default="data_dir")
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    if args.val_ratio < 0 or args.test_ratio < 0 or args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Require 0 <= val_ratio, test_ratio and val_ratio + test_ratio < 1")

    data_root = Path(args.data_file_dir)

    specs = [
        (Path("noisy/denoise_airnet.txt"), denoise_key),
        (Path("rainy/rainTrain.txt"), rain_key),
        (Path("hazy/hazy_outside.txt"), haze_key),
    ]

    for rel_path, key_fn in specs:
        src = data_root / rel_path
        if not src.exists():
            raise FileNotFoundError(f"Manifest not found: {src}")

        items = read_lines(src)
        train_items, val_items, test_items = split_grouped(
            items, key_fn=key_fn, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed
        )

        stem = src.stem
        write_lines(src.with_name(f"{stem}_train.txt"), train_items)
        write_lines(src.with_name(f"{stem}_val.txt"), val_items)
        write_lines(src.with_name(f"{stem}_test.txt"), test_items)

        print(f"[{rel_path}] total={len(items)} train={len(train_items)} val={len(val_items)} test={len(test_items)}")


if __name__ == "__main__":
    main()
