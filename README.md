# DeepBee with YOLO

Implementing the [DeepBee](https://avsthiago.github.io/DeepBee/) project using the YOLO framework for a lightweight deployment approach.

[DeepBee YOLO Dataset](https://www.kaggle.com/datasets/jondelvic/)

## Findings
After converting the original annotations to YOLO, and *n* number of wasted kaggle and colab free training hours, we found out that YOLO bounding-box regression fails on sparsely annotated textures (e.g., honey, nectar, pollen) and strutures (e.g., eggs, larvae, capped brood).

DeepBee to YOLO Dataset (Brood Focus)

Original data:
  - DeepBee cell annotation CSVs (train.csv, test.csv)
  - Beehive frame images (train/ and test/ folders)

Retained classes
  0: capped      – sealed brood cells (dark, flat caps)
  1: eggs        – tiny white eggs at cell bottom
  2: larves      – open cells containing larvae
  3: other       – empty cells, nectar, honey, pollen, shadows
  (honey, nectar, pollen excluded; 'dontcare' labels dropped entirely)

Splits
  train   : 3,158 tiles, 51,231 annotations (sparse + exhaustive, density filtered)
  val     :   652 tiles, 13,507 annotations (15% holdout, same construction)
  eval    :   165 tiles, 14,130 annotations (3 hand‑picked exhaustive images,
             full manual labels, no density filter)

Eval composition (balanced across all four classes)
  capped  23.0%   eggs   19.7%   larves  25.1%   other   32.2%

What we did:

1. Class selection
   - Retained: capped, eggs, larves, other
   - Excluded: honey, nectar, pollen (those labels are dropped)

2. Train/val/eval split
   - Sparse images (from train.csv) were split by image ID: 85% train, 15% val.
   - Exhaustive images (from test.csv) were split the same way, except two hand‑pinned images reserved for a final eval benchmark.
   - Eval contains only exhaustive‑source tiles from the pinned images.

3. Converting from point + radius to YOLO-normalized square boxes
   - The original annotation format from DeepBee annotates each cell as a point (center coordinates in image pixels) and a radius that represents the cell size, typically obtained from a Circular Hough Transform (CHT).

   - Tiling & Normalization
        - The image is sliced into 640×640 px tiles (20% overlap). For each tile:
            1. Collect all annotations whose centre falls inside the tile.
            2. Create a square bounding box around the centre: width = height = 2 × radius (the diameter). This box is considered the cell’s extent.
            3. Normalise to the tile:
                * cx = (cell_x – tile_x_min) / 640
                * cy = (cell_y – tile_y_min) / 640
                * bw = bh = (2 × radius) / 640
                - All values are clamped to the range [0, 1] – these are the YOLO‑format coordinates and sizes.

    - Clipping and Discarding
        - If a box extends beyond the tile boundary, it is clipped symmetrically. For example, if the centre is near the left edge, the box width is reduced so it doesn’t go outside the tile. Boxes that end up smaller than 1% of the tile width or height are discarded since they would be too tiny to be a meaningful training signal.

    - Radius Handling from labels_train.csv and labels_test.csv
        - Sparse images (original labels_train.csv) do not include a radius column. The pipeline assigns a fixed default radius (DEFAULT_RADIUS_TRAIN = 34 px), and adds a small random jitter (~4 px) per annotation to simulate natural cell size variation.
        - Exhaustive images (original labels_test.csv) contain a per‑cell true radius, measured manually. That radius is used directly – no default or jitter.
        
        * Sparse vs Exhaustive in the context of this Dataset
            - SPARSE (labels_train.csv): 
                - not all cells may be labelled, many cells are missing so labels are less reliable
                - the output tiles are mixed into /train and /val splits, also has a file prefix sparse_ (density filter was applied to prevent too many near-empty tiles)
            - EXHAUSTIVE (labels_test.csv): 
                - complete manual verification of every cell since it also includes the radius for each cell already 
                - output tiles are mixed into /train and /val with a prefix (exh_) and all tiles for /eval come from two handpicked exhaustive images which contained all the classes
                - density filter was not applied since the annotations/labels are reliable/complete

4. Density filtering (sparse sources only)
   - To prevent too many empty or near‑empty tiles from dominating training:
       * 0 annotations: keep 2%
       * 1‑3 annotations: keep 25%
       * 4‑7 annotations: keep 70%
       * 8+ annotations: keep all
   - Exhaustive‑source tiles (complete labels) are always kept.

5. Output layout
   dataset/
     images/    train/  val/  eval/
     labels/    train/  val/  eval/
     dataset.yaml
     audit/
       summary.txt
       class_counts.csv
       density_histogram.csv
       sample_tiles_*.jpg

Audit files give a transparent, pre‑training view of class balance,
annotation density, and visual tile samples.