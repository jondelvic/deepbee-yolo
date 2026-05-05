"""
build_dataset.py – Orchestration entry point for the DeepBee → YOLO pipeline.

Reads CSV annotations, partitions images, tiles them with optional density
filtering, writes audit reports, and generates dataset.yaml.

Directory layout produced:
    OUTPUT_DIR/
        images/
            train/   ← sparse + non-eval exhaustive tiles
            val/     ← sparse + non-eval exhaustive tiles (15% holdout)
            eval/    ← exhaustive-only tiles (pinned benchmark images)
        labels/
            train/
            val/
            eval/
        dataset.yaml
        audit/
            summary.txt
            class_counts.csv
            density_histogram.csv
            sample_tiles_*.jpg
"""

import shutil
import yaml
from pathlib import Path
import random

from scripts.config import (
    RAW_IMAGES_TRAIN, RAW_IMAGES_TEST,
    TRAIN_CSV, TEST_CSV, OUTPUT_DIR,
    EVAL_IMAGE_NAMES, RANDOM_SEED, VAL_SPLIT,
    AUDIT_SAMPLES_PER_SPLIT,
    TILE_SIZE, OVERLAP, DEFAULT_RADIUS_TRAIN, DENSITY_TIERS,
    CLASS_NAMES,
)
from scripts.parsers import parse_train_csv, parse_test_csv
from scripts.splits import split_train_val
from scripts.tiling import tile_and_write
from scripts.audit import write_audit_report, write_sample_mosaic, _imbalance_ratio


def write_dataset_yaml(output_dir: Path) -> None:
    """
    Write dataset.yaml next to the images/ and labels/ folders.

    Paths use forward slashes and are relative to the dataset root so the
    file is portable across machines and Kaggle environments.

    Splits:
        train → images/train   (sparse + non-eval exhaustive)
        val   → images/val     (15% holdout of train pool)
        test  → images/eval    (pinned exhaustive benchmark — never seen during training)
    """
    yaml_path = output_dir / "dataset.yaml"
    content = {
        "path":  str(output_dir).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/eval",   # exhaustive benchmark; used for final metrics only
        "nc":    len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(content, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    print(f"    📄  dataset.yaml → {yaml_path}")


def main() -> None:
    # ── validate inputs ────────────────────────────────────────────────────────
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

    # ── parse annotations ──────────────────────────────────────────────────────
    print("Parsing CSVs ...")
    sparse_all = parse_train_csv(TRAIN_CSV, rng)
    exhaustive = parse_test_csv(TEST_CSV)
    print(f"  sparse images     : {len(sparse_all)}")
    print(f"  exhaustive images : {len(exhaustive)}")

    # ── partition exhaustive set ───────────────────────────────────────────────
    exh_eval  = {k: v for k, v in exhaustive.items() if     k in EVAL_IMAGE_NAMES}
    exh_train = {k: v for k, v in exhaustive.items() if k not in EVAL_IMAGE_NAMES}
    print(f"  eval (pinned)     : {len(exh_eval)}")
    print(f"  train/val pool    : {len(exh_train)}")

    if not exh_eval:
        print("\n  ⚠  WARNING: no eval images found — check EVAL_IMAGE_NAMES in config.py")

    # ── split sparse + merge exhaustive into train/val ─────────────────────────
    sparse_train, sparse_val = split_train_val(sparse_all, VAL_SPLIT, RANDOM_SEED)
    exh_t, exh_v             = split_train_val(exh_train,  VAL_SPLIT, RANDOM_SEED + 10)
    sparse_train.update(exh_t)
    sparse_val.update(exh_v)

    print(f"\nSource images → train: {len(sparse_train)}, val: {len(sparse_val)}, eval: {len(exh_eval)}")

    # ── tiling ─────────────────────────────────────────────────────────────────
    rng = random.Random(RANDOM_SEED)
    print("\nTiling ...")
    print("Density tiers (sparse sources):")
    for lo, hi, prob in DENSITY_TIERS:
        hi_str = str(hi) if hi < 999 else "inf"
        print(f"  {lo}–{hi_str} annotations/tile → keep {prob:.0%}")

    print("\nCreating directory structure ...")
    for split in ["train", "val", "eval"]:
        for data_type in ["images", "labels"]:
            dir_path = OUTPUT_DIR / data_type / split
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / ".gitkeep").touch()

    # train — sparse sources (density filtered)
    n_tr1, ctr_tr1, den_tr1, lbl_tr1 = tile_and_write(
        {k: v for k, v in sparse_train.items() if k not in exh_t},
        RAW_IMAGES_TRAIN,
        OUTPUT_DIR / "images/train", OUTPUT_DIR / "labels/train",
        "train (sparse)", rng, use_density_filter=True, prefix="sparse_",
    )
    # train — exhaustive sources (no density filter; complete annotations)
    n_tr2, ctr_tr2, den_tr2, lbl_tr2 = tile_and_write(
        exh_t, RAW_IMAGES_TEST,
        OUTPUT_DIR / "images/train", OUTPUT_DIR / "labels/train",
        "train (exhaustive)", rng, use_density_filter=False, prefix="exh_",
    )
    n_tr   = n_tr1 + n_tr2
    ctr_tr = ctr_tr1 + ctr_tr2
    den_tr = den_tr1 + den_tr2
    lbl_tr = lbl_tr1 + lbl_tr2

    # val — sparse sources (density filtered)
    n_vl1, ctr_vl1, den_vl1, lbl_vl1 = tile_and_write(
        {k: v for k, v in sparse_val.items() if k not in exh_v},
        RAW_IMAGES_TRAIN,
        OUTPUT_DIR / "images/val", OUTPUT_DIR / "labels/val",
        "val (sparse)", rng, use_density_filter=True, prefix="sparse_",
    )
    # val — exhaustive sources (no density filter)
    n_vl2, ctr_vl2, den_vl2, lbl_vl2 = tile_and_write(
        exh_v, RAW_IMAGES_TEST,
        OUTPUT_DIR / "images/val", OUTPUT_DIR / "labels/val",
        "val (exhaustive)", rng, use_density_filter=False, prefix="exh_",
    )
    n_vl   = n_vl1 + n_vl2
    ctr_vl = ctr_vl1 + ctr_vl2
    den_vl = den_vl1 + den_vl2
    lbl_vl = lbl_vl1 + lbl_vl2

    # eval — exhaustive only, pinned images, no density filter
    n_ev, ctr_ev, den_ev, lbl_ev = tile_and_write(
        exh_eval, RAW_IMAGES_TEST,
        OUTPUT_DIR / "images/eval", OUTPUT_DIR / "labels/eval",
        "eval (exhaustive)", rng, use_density_filter=False, prefix="exh_",
    )

    # ── write dataset.yaml ─────────────────────────────────────────────────────
    print("\nWriting dataset.yaml ...")
    write_dataset_yaml(OUTPUT_DIR)

    # ── audit ──────────────────────────────────────────────────────────────────
    print("\nWriting audit files ...")
    split_data = {
        "train": (n_tr, ctr_tr, den_tr),
        "val":   (n_vl, ctr_vl, den_vl),
        "eval":  (n_ev, ctr_ev, den_ev),
    }
    write_audit_report(OUTPUT_DIR, split_data, config={
        "TILE_SIZE":            TILE_SIZE,
        "OVERLAP":              OVERLAP,
        "DEFAULT_RADIUS_TRAIN": DEFAULT_RADIUS_TRAIN,
        "VAL_SPLIT":            VAL_SPLIT,
        "RANDOM_SEED":          RANDOM_SEED,
    })

    rng_audit = random.Random(RANDOM_SEED + 99)
    for split, lbl_paths, lbl_dir in [
        ("train", lbl_tr, OUTPUT_DIR / "labels/train"),
        ("val",   lbl_vl, OUTPUT_DIR / "labels/val"),
        ("eval",  lbl_ev, OUTPUT_DIR / "labels/eval"),
    ]:
        write_sample_mosaic(
            lbl_paths, lbl_dir,
            OUTPUT_DIR / "audit" / f"sample_tiles_{split}.jpg",
            AUDIT_SAMPLES_PER_SPLIT, rng_audit,
        )

    # ── final summary ──────────────────────────────────────────────────────────
    print("\n─── Final Summary ───────────────────────────────────")
    for split, (n_tiles, ctr, _) in split_data.items():
        total = sum(ctr.values())
        ratio, maj, mino = _imbalance_ratio(ctr)
        flag = f"  ⚠  imbalance {ratio:.0f}× ({maj}/{mino})" if ratio > 5 else ""
        print(f"  {split:<6}: {n_tiles:>5} tiles, {total:>6} annotations{flag}")

    print(f"\n  Sparse  : {len(sparse_train)} train + {len(sparse_val)} val source images")
    print(f"  Exhaustive : {len(exh_t)+len(exh_v)} train/val + {len(exh_eval)} eval source images")
    print(f"\n  Output → {OUTPUT_DIR}")
    print(f"  YAML   → {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()