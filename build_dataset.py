"""
build_dataset.py – Orchestration entry point for the DeepBee to YOLO pipeline.

Reads CSV annotations, partitions images, tiles them with optional density
filtering, and writes audit reports.  Expects the scripts/ package in the
same directory.
"""

import shutil
from pathlib import Path
import random

from scripts.config import (
    RAW_IMAGES_TRAIN, RAW_IMAGES_TEST,
    TRAIN_CSV, TEST_CSV, OUTPUT_DIR,
    EVAL_IMAGE_NAMES, RANDOM_SEED, VAL_SPLIT,
    AUDIT_SAMPLES_PER_SPLIT,
    TILE_SIZE, OVERLAP, DEFAULT_RADIUS_TRAIN, DENSITY_TIERS,
)
from scripts.parsers import parse_train_csv, parse_test_csv
from scripts.splits import split_train_val
from scripts.tiling import tile_and_write
from scripts.audit import write_audit_report, write_sample_mosaic, _imbalance_ratio


def main() -> None:
    # --- validate inputs ---
    missing = [str(p) for p in (RAW_IMAGES_TRAIN, RAW_IMAGES_TEST, TRAIN_CSV, TEST_CSV) if not p.exists()]
    if missing:
        print("Missing paths (edit config.py):")
        for p in missing:
            print(" ", p)
        return

    if OUTPUT_DIR.exists():
        print(f"Removing existing {OUTPUT_DIR} ...")
        shutil.rmtree(OUTPUT_DIR)

    rng = random.Random(RANDOM_SEED)

    # --- parse annotations ---
    print("Parsing CSVs ...")
    sparse_all = parse_train_csv(TRAIN_CSV, rng)
    exhaustive = parse_test_csv(TEST_CSV)
    print(f"  sparse images : {len(sparse_all)}")
    print(f"  exhaustive images : {len(exhaustive)}")

    # --- partition exhaustive set ---
    exh_eval  = {k: v for k, v in exhaustive.items() if     k in EVAL_IMAGE_NAMES}
    exh_train = {k: v for k, v in exhaustive.items() if k not in EVAL_IMAGE_NAMES}
    print(f"  eval (held-out) : {len(exh_eval)}")
    print(f"  train/val pool  : {len(exh_train)}")

    # --- split sparse + merge exhaustive ---
    sparse_train, sparse_val = split_train_val(sparse_all, VAL_SPLIT, RANDOM_SEED)
    exh_t, exh_v = split_train_val(exh_train, VAL_SPLIT, RANDOM_SEED + 10)
    sparse_train.update(exh_t)
    sparse_val.update(exh_v)

    print(f"\nSource images -> train: {len(sparse_train)}, val: {len(sparse_val)}, eval: {len(exh_eval)}")

    # --- tiling ---
    rng = random.Random(RANDOM_SEED)
    print("\nTiling ...")
    print("Density tiers (sparse sources):")
    for lo, hi, prob in DENSITY_TIERS:
        hi_str = str(hi) if hi < 999 else "inf"
        print(f"  {lo}-{hi_str} annotations/tile -> keep {prob:.0%}")

    # train (sparse + exhaustive)
    n_tr1, ctr_tr1, den_tr1, lbl_tr1 = tile_and_write(
        {k: v for k, v in sparse_train.items() if k not in exh_t},
        RAW_IMAGES_TRAIN,
        OUTPUT_DIR / "images/train", OUTPUT_DIR / "labels/train",
        "train (sparse)", rng, use_density_filter=True, prefix="sparse_"
    )
    n_tr2, ctr_tr2, den_tr2, lbl_tr2 = tile_and_write(
        exh_t, RAW_IMAGES_TEST,
        OUTPUT_DIR / "images/train", OUTPUT_DIR / "labels/train",
        "train (exhaustive)", rng, use_density_filter=False, prefix="exh_"
    )
    n_tr   = n_tr1 + n_tr2
    ctr_tr = ctr_tr1 + ctr_tr2
    den_tr = den_tr1 + den_tr2
    lbl_tr = lbl_tr1 + lbl_tr2

    # val (sparse + exhaustive)
    n_vl1, ctr_vl1, den_vl1, lbl_vl1 = tile_and_write(
        {k: v for k, v in sparse_val.items() if k not in exh_v},
        RAW_IMAGES_TRAIN,
        OUTPUT_DIR / "images/val", OUTPUT_DIR / "labels/val",
        "val (sparse)", rng, use_density_filter=True, prefix="sparse_"
    )
    n_vl2, ctr_vl2, den_vl2, lbl_vl2 = tile_and_write(
        exh_v, RAW_IMAGES_TEST,
        OUTPUT_DIR / "images/val", OUTPUT_DIR / "labels/val",
        "val (exhaustive)", rng, use_density_filter=False, prefix="exh_"
    )
    n_vl   = n_vl1 + n_vl2
    ctr_vl = ctr_vl1 + ctr_vl2
    den_vl = den_vl1 + den_vl2
    lbl_vl = lbl_vl1 + lbl_vl2

    # eval (exhaustive only)
    n_ev, ctr_ev, den_ev, lbl_ev = tile_and_write(
        exh_eval, RAW_IMAGES_TEST,
        OUTPUT_DIR / "images/eval", OUTPUT_DIR / "labels/eval",
        "eval (exhaustive)", rng, use_density_filter=False, prefix="exh_"
    )

    # --- audit ---
    print("\nWriting audit files ...")
    audit_dir = OUTPUT_DIR / "audit"
    split_data = {
        "train": (n_tr, ctr_tr, den_tr),
        "val":   (n_vl, ctr_vl, den_vl),
        "eval":  (n_ev, ctr_ev, den_ev),
    }
    write_audit_report(OUTPUT_DIR, split_data, config={
        "TILE_SIZE": TILE_SIZE,
        "OVERLAP": OVERLAP,
        "DEFAULT_RADIUS_TRAIN": DEFAULT_RADIUS_TRAIN,
        "VAL_SPLIT": VAL_SPLIT,
        "RANDOM_SEED": RANDOM_SEED,
    })

    rng_audit = random.Random(RANDOM_SEED + 99)
    for split, lbl_paths, lbl_dir in [
        ("train", lbl_tr, OUTPUT_DIR / "labels/train"),
        ("val",   lbl_vl, OUTPUT_DIR / "labels/val"),
        ("eval",  lbl_ev, OUTPUT_DIR / "labels/eval"),
    ]:
        write_sample_mosaic(
            lbl_paths, lbl_dir,
            audit_dir / f"sample_tiles_{split}.jpg",
            AUDIT_SAMPLES_PER_SPLIT, rng_audit,
        )

    # --- final summary ---
    print("\n--- Final Summary ---")
    for split, (n_tiles, ctr, _) in split_data.items():
        total = sum(ctr.values())
        ratio, maj, mino = _imbalance_ratio(ctr)
        flag = f"  imbalance {ratio:.0f}x" if ratio > 5 else ""
        print(f"  {split:<6} : {n_tiles} tiles, {total} annotations  {flag}")

    print(f"\nSparse images used: {len(sparse_train)} train + {len(sparse_val)} val")
    print(f"Exhaustive images used: {len(exh_t)+len(exh_v)} train/val + {len(exh_eval)} eval")
    print(f"\nOutput directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()