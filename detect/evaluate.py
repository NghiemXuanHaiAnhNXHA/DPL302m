"""
Input: predictions.csv + video_0{?}.csv
"""

import os
import cv2
import pandas as pd
import numpy as np

from scipy.optimize import linear_sum_assignment

from iou_utils import compute_iou

# Config
# -----------------------------------------------
# Notice to change it
VIDEO_ID = "video_"

# Notice to change it
METHOD_DIR = "DPL302m/detect/detect_yolo_bytetrack_cooldown"

WINDOW_SIZE = 2

IOU_THRESHOLD = 0.5

DATASET_DIR = "DPL302m/dataset"
# -----------------------------------------------

# Load predictions.csv
pred_path = os.path.join(
    METHOD_DIR,
    VIDEO_ID,
    "predictions.csv"
)

pred_df = pd.read_csv(pred_path)

# Load ground truth
gt_path = os.path.join(
    DATASET_DIR,
    VIDEO_ID,
    f"{VIDEO_ID}.csv"
)

gt_df = pd.read_csv(gt_path)
gt_ids = sorted(
    gt_df["gt_pothole_id"]
    .unique()
)

gt_to_idx = {
    gt_id: idx
    for idx, gt_id
    in enumerate(gt_ids)
}

# Build cost matrix
num_pred = len(pred_df)

num_gt = len(gt_ids)

iou_matrix = np.zeros(
    (num_pred, num_gt),
    dtype=np.float32
)

matched_gt_frame = {}

# Fill matrix
for pred_idx, pred_row in pred_df.iterrows():

    pred_frame = pred_row["frame_id"]

    pred_box = [
        pred_row["x1"],
        pred_row["y1"],
        pred_row["x2"],
        pred_row["y2"]
    ]

    candidate_gt = gt_df[
        (
            gt_df["frame_id"]
            >= pred_frame - WINDOW_SIZE
        )
        &
        (
            gt_df["frame_id"]
            <= pred_frame + WINDOW_SIZE
        )
    ]

    for gt_id in candidate_gt[
        "gt_pothole_id"
    ].unique():

        gt_rows = candidate_gt[
            candidate_gt[
                "gt_pothole_id"
            ] == gt_id
        ]

        best_iou = 0
        best_frame = None

        for _, gt_row in gt_rows.iterrows():

            gt_box = [
                gt_row["x1"],
                gt_row["y1"],
                gt_row["x2"],
                gt_row["y2"]
            ]

            iou = compute_iou(
                pred_box,
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_frame = gt_row[
                    "frame_id"
                ]

        if best_iou >= IOU_THRESHOLD:

            col = gt_to_idx[gt_id]

            iou_matrix[
                pred_idx,
                col
            ] = best_iou

            matched_gt_frame[
                (
                    pred_idx,
                    col
                )
            ] = best_frame

# Hungarian
cost_matrix = 1.0 - iou_matrix
rows, cols = linear_sum_assignment(
    cost_matrix
)

# Extract valid matches
matches = []

for r, c in zip(rows, cols):

    iou = iou_matrix[r, c]

    if iou < IOU_THRESHOLD:
        continue

    pred_row = pred_df.iloc[r]

    gt_id = gt_ids[c]

    gt_frame = matched_gt_frame[
        (r, c)
    ]

    matches.append([
        pred_row["detection_id"],
        pred_row["frame_id"],

        pred_row["x1"],
        pred_row["y1"],
        pred_row["x2"],
        pred_row["y2"],

        gt_id,
        gt_frame,
        iou
    ])

# Save matched_frame.csv
match_df = pd.DataFrame(
    matches,
    columns=[
        "detection_id",
        "predicted_frame_id",

        "x1_predicted",
        "y1_predicted",
        "x2_predicted",
        "y2_predicted",

        "gt_id",
        "gt_frame_id",
        "iou"
    ]
)

match_df.to_csv(
    os.path.join(
        METHOD_DIR,
        VIDEO_ID,
        "matched_frame.csv"
    ),
    index=False
)

# Summary
number_frame_records = (
    pred_df["frame_id"]
    .nunique()
)

number_predicted_records = (
    len(pred_df)
)

number_potholes_matched = (
    match_df["gt_id"]
    .nunique()
)

number_duplicates = (
    number_predicted_records
    - number_potholes_matched
)

number_gt = (
    gt_df["gt_pothole_id"]
    .nunique()
)

# Save video_01_summary.csv
summary_df = pd.DataFrame([
    {
        "video_id": VIDEO_ID,

        "number_frame_records":
        number_frame_records,

        "number_predicted_records":
        number_predicted_records,

        "number_potholes_matched":
        number_potholes_matched,

        "number_duplicates":
        number_duplicates,

        "number_gt":
        number_gt
    }
])

summary_df.to_csv(
    os.path.join(
        METHOD_DIR,
        VIDEO_ID,
        f"{VIDEO_ID}_summary.csv"
    ),
    index=False
)