# DeepBee with YOLO

Implementing the [DeepBee](https://avsthiago.github.io/DeepBee/) project using the YOLO framework for a lightweight deployment approach.

[DeepBee YOLO Dataset](https://www.kaggle.com/datasets/jondelvic/honey-bee-colony-dataset-deepbee-to-yolo)

CSV formats
-----------
**labels_train.csv  (NO header row)**
- col 0 : annotation id
- col 1 : x  (cell centre, pixels)
- col 2 : y  (cell centre, pixels)
- col 3 : class name  (e.g. "eggs", "larves", "capped" …)
- col 4 : image file name

**labels_test.csv** (id,x,y,radius,class (int),class name,image name)

YOLO label format (per image, one .txt file)
--------------------------------------------
`<class_id>  <cx_norm>  <cy_norm>  <w_norm>  <h_norm>`

All values are normalised to [0, 1] by image width / height. 

For a circular cell at (x, y) with radius r:
<br>
cx = x / W <br>
cy = y / H <br>
bw = 2r / W <br>
bh = 2r / H <br>

## NOTES
- The `labels_train.csv` file lacks bounding box dimensions.
- While `labels_test.csv` includes a `radius` column (often around 15-18px), this value is too small for YOLO. It only captures the cell interior, which might miss the wax walls for distinguishing capped vs empty
- From the notebook, it shows that cell centers are around 73px apart in both train and test set so optical scale will be the same?
- Overlap would be theoretically fine for YOLO using its non-maximum supperssion (NMS).