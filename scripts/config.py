from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_IMAGES_TRAIN = PROJECT_ROOT / "dataset" / "raw" / "train" / "images"
RAW_IMAGES_TEST  = PROJECT_ROOT / "dataset" / "raw" / "test" / "images"
TRAIN_CSV        = PROJECT_ROOT / "dataset" / "raw" / "labels_train.csv"
TEST_CSV         = PROJECT_ROOT / "dataset" / "raw" / "labels_test.csv"
OUTPUT_DIR       = PROJECT_ROOT / "dataset" / "deepbee-processed-yolo-final"

# handpicked images for final benchmark
EVAL_IMAGE_NAMES = {
    "DSC_0819.JPG",   # most balanced image; covers all classes well
    "DSC_0818.JPG",   # strong larves representation
    "DSC_0840.JPG",   # strong capped representation
}

# tiling parameters
TILE_SIZE = 640   # px to match YOLOv11 default input resolution
OVERLAP   = 0.20  # 20% overlap; ensures cells near tile boundaries appear in at least two tiles

# annotation density filter for visible cells that have no label
# 0 annotations: keep 2% to avoid background-class dominance
# 1-3 annotations: keep 25%
# 4-7 annotations: keep 70%
# 8+ annotations: keep 100%
# this keeps just enough background tiles to teach "nothing here" something liek dat
DENSITY_TIERS = [
    # (min_annotations_inclusive, max_annotations_exclusive, keep_probability)
    (0,   1,   0.02), # empty tiles: keep 2%
    (1,   4,   0.25), # very sparse: keep 25%
    (4,   8,   0.70), # moderate: keep 70%
    (8,   999, 1.00), # dense: keep all
]

# BOUNDING BOX RADIUS
DEFAULT_RADIUS_TRAIN = 34 # midpoint of DeepBee CHT range

VAL_SPLIT   = 0.15  # 15% of source images held out for validation
RANDOM_SEED = 42

AUDIT_SAMPLES_PER_SPLIT = 16  # FOR AUDIT: tiles drawn into each sample mosaic

# CLASS MAPPING
CLASS_NAMES = ["capped", "eggs", "larves", "other"] # for yaml output

# raw label strings from both train/test csv for unified class id yaml
CLASS_MAP = {
    # kept classes — contiguous IDs
    "capped":       0, "capped brood": 0,
    "eggs":         1, "egg":          1,
    "larves":       2, "larva":        2, "larvae":      2,
    "other":        3,
    # excluded classes — parsers.py skips any entry where class_id is None
    "honey":        None,
    "nectar":       None,
    "pollen":       None,
}


# BGR colors for audit mosaic visualization only (one per kept class)
CLASS_COLORS_BGR = [
    (255, 180,   0),   # 0 capped – gold
    (  0, 255, 255),   # 1 eggs   – yellow
    (255,  80, 200),   # 2 larves – pink
    (180, 180, 180),   # 3 other  – grey
]
