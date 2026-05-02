from pathlib import Path
from collections import Counter
import random
import cv2
from tqdm import tqdm
from scripts.config import TILE_SIZE, OVERLAP, DENSITY_TIERS

# DENSITY FILTER
# return the keep probability for a tile with _ annotations
def _keep_prob(n_annots: int) -> float:
    for lo, hi, prob in DENSITY_TIERS:
        if lo <= n_annots < hi:
            return prob
    return 1.0  # fallback: keep if somehow outside all defined tiers

# TILING AND WRITING
# note: claude generated. i understand nothing of this
def tile_and_write(
    annotations:        dict[str, list[dict]],
    image_folder:       Path,
    out_img_folder:     Path,
    out_lbl_folder:     Path,
    split_name:         str,
    rng:                random.Random,
    use_density_filter: bool = True,
) -> tuple[int, Counter, Counter, list[Path]]:
    """
    Slide a TILE_SIZE×TILE_SIZE window over each source image, convert
    annotations to YOLO format, apply optional density filtering, and write
    tile images (.jpg) and label files (.txt) to the output directories.

    Tiling strategy:
        step = TILE_SIZE * (1 - OVERLAP) = 512 px for default settings.
        Grid origin points are computed first; if the final column/row does not
        reach the image edge, an extra tile anchored to the boundary is added
        so no region of the image is left uncovered.

    Bounding box conversion:
        Each annotation is a point (x, y) + radius r in original image pixel
        coordinates. The box is converted to YOLO normalized format:
            cx = (x - tile_x_min) / TILE_SIZE   [0, 1]
            cy = (y - tile_y_min) / TILE_SIZE   [0, 1]
            bw = bh = (r * 2) / TILE_SIZE        [0, 1]
        Boxes that extend past tile boundaries are clipped symmetrically.
        Boxes clipped below 1% of tile width/height are discarded.

    Density filter (use_density_filter=True):
        Applied AFTER annotation collection but BEFORE writing. If a tile is
        rejected, its class_ctr increments are rolled back to keep counts
        accurate. Exhaustive-source images call this function with
        use_density_filter=False, bypassing the filter entirely.

    Args:
        annotations:        {image_name: [annotation_dict, ...]}
        image_folder:       directory containing the source image files
        out_img_folder:     destination for tile .jpg files
        out_lbl_folder:     destination for YOLO .txt label files
        split_name:         label for tqdm progress bar
        rng:                seeded Random instance for density filter decisions
        use_density_filter: if False, all tiles are written regardless of count

    Returns:
        tiles_total        — total tiles written to disk
        class_ctr          — {class_id: annotation_count} across written tiles
        density_ctr        — {n_annots: tile_count} recorded BEFORE filtering
                             (used in the audit density histogram)
        labeled_tile_paths — paths of written tiles that contain ≥1 annotation
                             (used to draw the audit sample mosaic)
    """
    out_img_folder.mkdir(parents=True, exist_ok=True)
    out_lbl_folder.mkdir(parents=True, exist_ok=True)

    step               = int(TILE_SIZE * (1 - OVERLAP))
    class_ctr          = Counter()
    density_ctr        = Counter()
    tiles_total        = 0
    imgs_missing       = 0
    labeled_tile_paths: list[Path] = []

    for img_name, annots in tqdm(annotations.items(), desc=f"  {split_name}", unit="img"):

        # ── locate image ──────────────────────────────────────────────────────
        # Try the exact filename first; fall back to common extension variants
        # to tolerate case mismatches (e.g. .JPG vs .jpg on Linux).
        img_path = image_folder / img_name
        if not img_path.exists():
            stem  = Path(img_name).stem
            found = next(
                (image_folder / (stem + ext)
                 for ext in (".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG")
                 if (image_folder / (stem + ext)).exists()),
                None,
            )
            if found is None:
                imgs_missing += 1
                if imgs_missing <= 5:
                    print(f"    ✗ not found: {img_name}")
                continue
            img_path = found

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"    ✗ corrupted: {img_name}")
            continue
        img_h, img_w = img.shape[:2]
        stem = img_path.stem

        pts = [(a["x"], a["y"], a["r"], a["class_id"]) for a in annots]

        # ── compute tile grid ─────────────────────────────────────────────────
        y_origins = list(range(0, img_h - TILE_SIZE + 1, step))
        x_origins = list(range(0, img_w - TILE_SIZE + 1, step))

        # Append boundary-anchored tile if the grid doesn't reach the edge
        if not y_origins or y_origins[-1] + TILE_SIZE < img_h:
            y_origins.append(max(0, img_h - TILE_SIZE))
        if not x_origins or x_origins[-1] + TILE_SIZE < img_w:
            x_origins.append(max(0, img_w - TILE_SIZE))

        for y0 in y_origins:
            for x0 in x_origins:
                x_min, y_min = x0, y0
                x_max, y_max = x0 + TILE_SIZE, y0 + TILE_SIZE

                # ── collect annotations for this tile ─────────────────────────
                tile_lines: list[str] = []
                for (ax, ay, ar, cid) in pts:
                    if not (x_min <= ax < x_max and y_min <= ay < y_max):
                        continue

                    # Normalized center within the tile
                    cx = max(0.0, min(1.0, (ax - x_min) / TILE_SIZE))
                    cy = max(0.0, min(1.0, (ay - y_min) / TILE_SIZE))

                    # Normalized diameter; square box because radius is isotropic
                    raw_w = (ar * 2) / TILE_SIZE
                    raw_h = (ar * 2) / TILE_SIZE

                    # Clip box symmetrically to tile boundary.
                    # Derivation: distance from center to left edge = cx,
                    # so max half-width = cx; similarly for right, top, bottom.
                    bw_clipped = min(raw_w, 2 * cx, 2 * (1 - cx))
                    bh_clipped = min(raw_h, 2 * cy, 2 * (1 - cy))

                    # Discard boxes clipped to less than 1% of tile dimension
                    if bw_clipped > 0.01 and bh_clipped > 0.01:
                        tile_lines.append(
                            f"{cid} {cx:.6f} {cy:.6f} {bw_clipped:.6f} {bh_clipped:.6f}"
                        )
                        class_ctr[cid] += 1

                n = len(tile_lines)
                density_ctr[n] += 1  # record pre-filter count for audit histogram

                # ── density filter ────────────────────────────────────────────
                if use_density_filter:
                    if rng.random() > _keep_prob(n):
                        # Roll back class counts for the rejected tile.
                        # Note: counts were incremented inside the loop above,
                        # so the rollback must happen here, after collection.
                        for line in tile_lines:
                            class_ctr[int(line.split()[0])] -= 1
                        continue

                # ── write tile to disk ────────────────────────────────────────
                tile_fname = f"{stem}_x{x0}_y{y0}"
                tile_img   = img[y_min:y_max, x_min:x_max]
                tile_path  = out_img_folder / f"{tile_fname}.jpg"
                cv2.imwrite(str(tile_path), tile_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if tile_lines:
                    # YOLO label format: one line per object,
                    # "class_id cx cy bw bh" (all values normalized 0-1)
                    (out_lbl_folder / f"{tile_fname}.txt").write_text(
                        "\n".join(tile_lines) + "\n"
                    )
                    labeled_tile_paths.append(tile_path)

                tiles_total += 1

    if imgs_missing > 5:
        print(f"    ✗ … and {imgs_missing - 5} more missing images")

    return tiles_total, class_ctr, density_ctr, labeled_tile_paths