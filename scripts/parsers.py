from pathlib import Path
from collections import defaultdict
import random
import pandas as pd
from scripts.config import CLASS_MAP, DEFAULT_RADIUS_TRAIN

# parse the sparse train csv (no header, no radius column)
# DEFAULT_RADIUS_TRAIN is used with ±4 px uniform jitter per annotation.
#
# CLASS_MAP entries with value None are intentionally excluded classes
# (honey, nectar, pollen). These are silently skipped — they are NOT
# reported as unknown. Only strings absent from CLASS_MAP entirely are
# flagged, as those indicate a genuine data/mapping problem.
#
# Returns: {image_name: [{"x", "y", "r", "class_id"}, ...]}

def parse_train_csv(csv_path: Path, rng: random.Random) -> dict[str, list[dict]]:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["id", "x", "y", "class_name", "image_name"],
        dtype=str,
    )
    by_image: dict[str, list[dict]] = defaultdict(list)
    unknown: set[str] = set()    # strings not in CLASS_MAP at all
    excluded: set[str] = set()   # strings in CLASS_MAP but mapped to None

    for _, row in df.iterrows():
        cls_raw = row["class_name"].strip().lower()

        if cls_raw not in CLASS_MAP:
            unknown.add(cls_raw)
            continue

        class_id = CLASS_MAP[cls_raw]
        if class_id is None:
            excluded.add(cls_raw)   # intentionally dropped — no warning
            continue

        jitter = rng.uniform(-4.0, 4.0)
        by_image[row["image_name"].strip()].append({
            "x":        float(row["x"]),
            "y":        float(row["y"]),
            "r":        DEFAULT_RADIUS_TRAIN + jitter,
            "class_id": class_id,
        })

    if excluded:
        print(f"  Excluded classes (intentional, not an error): {sorted(excluded)}")
    if unknown:
        print(f"  ⚠  Unknown class names not in CLASS_MAP (check config): {unknown}")

    return dict(by_image)


# parse test csv (has header and per-cell radius column)
# Uses the measured radius directly — no default/jitter.
#
# Returns: {image_name: [{"x", "y", "r", "class_id"}, ...]}

def parse_test_csv(csv_path: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path, dtype=str)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    img_col  = next(c for c in df.columns if "image"  in c)
    name_col = next(c for c in df.columns if "name"   in c and "image" not in c)
    rad_col  = next(c for c in df.columns if "radius" in c)

    by_image: dict[str, list[dict]] = defaultdict(list)
    unknown:  set[str] = set()
    excluded: set[str] = set()

    for _, row in df.iterrows():
        cls_raw = str(row[name_col]).strip().lower()

        if cls_raw not in CLASS_MAP:
            unknown.add(cls_raw)
            continue

        class_id = CLASS_MAP[cls_raw]
        if class_id is None:
            excluded.add(cls_raw)
            continue

        by_image[row[img_col].strip()].append({
            "x":        float(row["x"]),
            "y":        float(row["y"]),
            "r":        float(row[rad_col]),
            "class_id": class_id,
        })

    if excluded:
        print(f"  Excluded classes (intentional, not an error): {sorted(excluded)}")
    if unknown:
        print(f"  ⚠  Unknown class names not in CLASS_MAP (check config): {unknown}")

    return dict(by_image)