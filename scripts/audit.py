from pathlib import Path
import random
import math
from collections import Counter
import csv
import textwrap
import cv2
import numpy as np
from scripts.config import CLASS_COLORS_BGR, CLASS_NAMES, TILE_SIZE, DENSITY_TIERS

# claude generated for dataset debugging purposes

def _draw_boxes_on_tile(tile_path: Path, label_dir: Path) -> np.ndarray | None:
    """
    Load a tile image and draw its YOLO bounding boxes for visual inspection.

    Returns the annotated BGR image, a plain image if no label file exists
    (background tile), or None if the image cannot be read.
    """
    img = cv2.imread(str(tile_path))
    if img is None:
        return None

    lbl_path = label_dir / (tile_path.stem + ".txt")
    if not lbl_path.exists():
        return img  # background tile — valid, just no labels

    h, w = img.shape[:2]
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue

        cid = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])
        # Denormalize to pixel coordinates
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        color = CLASS_COLORS_BGR[cid % len(CLASS_COLORS_BGR)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, CLASS_NAMES[cid], (x1, max(y1 - 4, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )
    return img


def write_sample_mosaic(
    labeled_paths: list[Path],
    label_dir:     Path,
    out_path:      Path,
    n:             int,
    rng:           random.Random,
) -> None:
    """
    Sample up to n labeled tiles at random, draw their bounding boxes,
    arrange them in a square grid, and save as a JPEG mosaic for visual QA.

    Use the mosaic to verify: boxes are approximately cell-sized (~10–15% of
    tile width), centers land on actual cells, and class labels look correct.
    """
    sample = rng.sample(labeled_paths, min(n, len(labeled_paths)))
    drawn  = [_draw_boxes_on_tile(p, label_dir) for p in sample]
    drawn  = [d for d in drawn if d is not None]

    if not drawn:
        print(f"    ⚠  No tiles to draw for {out_path.name}")
        return

    cols   = math.ceil(math.sqrt(len(drawn)))
    rows   = math.ceil(len(drawn) / cols)
    cell   = TILE_SIZE
    canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

    for idx, img in enumerate(drawn):
        r, c = divmod(idx, cols)
        resized = cv2.resize(img, (cell, cell))
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = resized

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"    🖼   sample mosaic → {out_path}  ({len(drawn)} tiles)")


# ══════════════════════════════════════════════════════════════════════════════
#  AUDIT — text + CSV reports
# ══════════════════════════════════════════════════════════════════════════════

def _class_table(ctr: Counter, total_tiles: int) -> str:
    """Format a per-class annotation count table with inline bar chart."""
    total_annots = sum(ctr.values())
    lines = [f"  {'ID':<4} {'Class':<10} {'Annots':>8} {'%':>7}  Bar"]
    lines.append("  " + "-" * 55)
    for i, name in enumerate(CLASS_NAMES):
        n   = ctr.get(i, 0)
        pct = 100 * n / max(total_annots, 1)
        bar = "█" * min(30, int(30 * pct / 100))
        lines.append(f"  {i:<4} {name:<10} {n:>8}  {pct:6.1f}%  {bar}")
    lines.append(f"\n  Total annotations : {total_annots}")
    lines.append(f"  Total tiles        : {total_tiles}")
    lines.append(f"  Tiles with labels  : (see density histogram)")
    return "\n".join(lines)


def _density_table(density_ctr: Counter) -> str:
    """Format a density histogram table (pre-filter counts)."""
    lines = [f"  {'Annots/tile':<14} {'Tiles (pre-filter)':>20}"]
    lines.append("  " + "-" * 36)
    for k in sorted(density_ctr):
        lines.append(f"  {k:<14} {density_ctr[k]:>20}")
    return "\n".join(lines)


def _imbalance_ratio(ctr: Counter) -> tuple[float, str, str]:
    """
    Return (max/min ratio, majority class name, minority class name) for
    non-zero classes. Returns (1.0, "", "") if fewer than 2 classes present.
    """
    counts  = [ctr.get(i, 0) for i in range(len(CLASS_NAMES))]
    nonzero = [c for c in counts if c > 0]
    if len(nonzero) < 2:
        return 1.0, "", ""
    ratio    = max(nonzero) / min(nonzero)
    majority = CLASS_NAMES[counts.index(max(counts))]
    minority = CLASS_NAMES[counts.index(min(c for c in counts if c > 0))]
    return ratio, majority, minority


def write_audit_report(
    output_dir: Path,
    split_data: dict,   # {split_name: (n_tiles, class_ctr, density_ctr)}
    config:     dict,
) -> None:
    """
    Write three audit files to OUTPUT_DIR/audit/:

      summary.txt           Human-readable run report with config, per-split
                            class tables, imbalance warnings, and density data.
      class_counts.csv      Per-class annotation counts and percentages per split.
      density_histogram.csv Distribution of annotations-per-tile per split,
                            recorded before density filtering was applied.
    """
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # ── summary.txt ───────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 62)
    lines.append("  DeepBee → YOLO  build_dataset.py  audit summary")
    lines.append("=" * 62)
    lines.append("")
    lines.append("CONFIG")
    lines.append(f"  TILE_SIZE             : {config['TILE_SIZE']}")
    lines.append(f"  OVERLAP               : {config['OVERLAP']}")
    lines.append(f"  DEFAULT_RADIUS_TRAIN  : {config['DEFAULT_RADIUS_TRAIN']} px")
    lines.append(f"  VAL_SPLIT             : {config['VAL_SPLIT']}")
    lines.append(f"  RANDOM_SEED           : {config['RANDOM_SEED']}")
    lines.append("")
    lines.append("DENSITY FILTER TIERS  (sparse train/val only)")
    for lo, hi, prob in DENSITY_TIERS:
        hi_str = str(hi) if hi < 999 else "∞"
        lines.append(f"  {lo}–{hi_str} annots/tile → keep prob {prob:.0%}")

    for split, (n_tiles, class_ctr, density_ctr) in split_data.items():
        lines.append("")
        lines.append("─" * 62)
        lines.append(f"  SPLIT: {split.upper()}")
        lines.append("─" * 62)
        lines.append(_class_table(class_ctr, n_tiles))
        ratio, maj, mino = _imbalance_ratio(class_ctr)
        if ratio > 5:
            lines.append(f"\n  ⚠  Imbalance: {maj} vs {mino} = {ratio:.1f}×")
            lines.append(     "     Use  cls=2.0 label_smoothing=0.1  when training.")
        lines.append("")
        lines.append("  ANNOTATION DENSITY (pre-filter counts for this split)")
        lines.append(_density_table(density_ctr))

    lines.append("")
    lines.append("=" * 62)
    lines.append("WHAT TO CHECK")
    lines.append("=" * 62)
    lines.append(textwrap.dedent("""
  1. CLASS BALANCE
     Ideally no class is >10× another in train. If it is, increase cls= in
     your YOLO training command. The eggs class is typically the minority.

  2. EVAL SPLIT SIZE
     eval/images/ should contain tiles from exactly the images listed in
     EVAL_IMAGE_NAMES. These have real per-cell radii → reliable mAP signal.

  3. SAMPLE MOSAICS  (audit/sample_tiles_*.jpg)
     Open each mosaic and verify:
       a. Boxes are cell-sized (roughly 10-15% of tile width).
       b. Box centres land on actual cells, not gaps.
       c. Class labels look correct (e.g. capped = sealed dark cells).
     If boxes look too large/small → adjust DEFAULT_RADIUS_TRAIN.

  4. DENSITY HISTOGRAM
     The "0 annots → N tiles" row shows background-only tiles retained
     (≈2% of empty tiles). A very large number can hurt training.
     Reduce the 0-tier keep probability in DENSITY_TIERS if needed.

  5. EXHAUSTIVE vs SPARSE MIX
     train/ contains tiles from both exhaustive and sparse source images.
     Exhaustive tiles carry reliable, complete annotations and are the
     primary classification signal. If their count is small relative to
     sparse tiles, consider moving more exhaustive images from eval to
     the train pool by editing EVAL_IMAGE_NAMES.
    """).rstrip())

    (audit_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"    📄  audit/summary.txt")

    # ── class_counts.csv ──────────────────────────────────────────────────────
    rows = [["split", "class_id", "class_name", "annotations", "pct"]]
    for split, (n_tiles, class_ctr, _) in split_data.items():
        total = max(sum(class_ctr.values()), 1)
        for i, name in enumerate(CLASS_NAMES):
            n = class_ctr.get(i, 0)
            rows.append([split, i, name, n, f"{100 * n / total:.2f}"])
    with open(audit_dir / "class_counts.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"    📄  audit/class_counts.csv")

    # ── density_histogram.csv ─────────────────────────────────────────────────
    all_n = sorted(set().union(*[d.keys() for _, _, d in split_data.values()]))
    rows  = [["n_annotations_per_tile"] + list(split_data.keys())]
    for n in all_n:
        rows.append([n] + [split_data[s][2].get(n, 0) for s in split_data])
    with open(audit_dir / "density_histogram.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"    📄  audit/density_histogram.csv")