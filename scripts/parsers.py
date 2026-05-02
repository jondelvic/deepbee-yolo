from pathlib import Path
from collections import defaultdict
import random
import pandas as pd
from scripts.config import CLASS_MAP, DEFAULT_RADIUS_TRAIN   # or relative import

# parse the sparse train csv (the one with no header and no radius column)
# Because no per-cell radius is available, DEFAULT_RADIUS_TRAIN is used with a ±4 px uniform jitter per annotation. See module config for full rationale.

# Returns a dict mapping image_nam to list of annotation dicts: {"x": float, "y": float, "r": float, "class_id": int}
def parse_train_csv(csv_path: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["id", "x", "y", "class_name", "image_name"],
        dtype=str,
    )
    by_image: dict[str, list[dict]] = defaultdict(list)
    unknown: set[str] = set()

    for _, row in df.iterrows():
        cls_raw  = row["class_name"].strip().lower()
        class_id = CLASS_MAP.get(cls_raw)
        if class_id is None:
            unknown.add(cls_raw)
            continue

        # jitter (for reproducibility) based from natural CHT variance from deepbee source code
        jitter = random.uniform(-4.0, 4.0)
        by_image[row["image_name"].strip()].append({
            "x":        float(row["x"]),
            "y":        float(row["y"]),
            "r":        DEFAULT_RADIUS_TRAIN + jitter,
            "class_id": class_id,
        })

    if unknown:
        print(f"Unknown train class names (skipped): {unknown}")
    return dict(by_image)

# parse test csv (w/ header and per cell radius)
# column names are also normalized just in case for formatting differences
# uses the direct measured radius, no default/jitter compared to parse train csv
# Returns a dict mapping image_name → list of annotation dicts: {"x": float, "y": float, "r": float, "class_id": int}
def parse_test_csv(csv_path: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # locate columns by keyword rather than position
    img_col  = next(c for c in df.columns if "image"  in c)
    name_col = next(c for c in df.columns if "name"   in c and "image" not in c)
    rad_col  = next(c for c in df.columns if "radius" in c)

    by_image: dict[str, list[dict]] = defaultdict(list)
    unknown: set[str] = set()

    for _, row in df.iterrows():
        cls_raw  = str(row[name_col]).strip().lower()
        class_id = CLASS_MAP.get(cls_raw)
        if class_id is None:
            unknown.add(cls_raw)
            continue
        by_image[row[img_col].strip()].append({
            "x":        float(row["x"]),
            "y":        float(row["y"]),
            "r":        float(row[rad_col]),
            "class_id": class_id,
        })

    if unknown:
        print(f"Unknown test class names (skipped): {unknown}")
    return dict(by_image)