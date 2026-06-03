"""
Input:
detect/METHOD_NAME/
│
├── video_01/
│   └── video_01_summary.csv
│
├── video_02/
│   └── video_02_summary.csv
...
Output: detect/METHOD_NAME/detect_yolo_results.csv
"""

import os
import pandas as pd

# =====================================
# CONFIG
# =====================================

METHOD_NAME = "detect_yolo"

METHOD_DIR = os.path.join(
    "detect",
    METHOD_NAME
)

# =====================================
# FIND ALL SUMMARY FILES
# =====================================

summary_files = []

for item in os.listdir(METHOD_DIR):

    item_path = os.path.join(
        METHOD_DIR,
        item
    )

    if not os.path.isdir(item_path):
        continue

    summary_file = os.path.join(
        item_path,
        f"{item}_summary.csv"
    )

    if os.path.exists(summary_file):
        summary_files.append(summary_file)

# =====================================
# LOAD ALL SUMMARIES
# =====================================

all_summaries = []

for summary_file in summary_files:

    df = pd.read_csv(summary_file)

    all_summaries.append(df)

if len(all_summaries) == 0:
    raise ValueError(
        "No summary files found."
    )

summary_df = pd.concat(
    all_summaries,
    ignore_index=True
)

# =====================================
# AGGREGATE
# =====================================

total_frame_records = (
    summary_df["number_frame_records"]
    .sum()
)

total_predicted_records = (
    summary_df["number_predicted_records"]
    .sum()
)

total_potholes_matched = (
    summary_df["number_potholes_matched"]
    .sum()
)

total_duplicates = (
    summary_df["number_duplicates"]
    .sum()
)

total_gt = (
    summary_df["number_gt"]
    .sum()
)

# =====================================
# METRICS
# =====================================

duplicate_rate = 0.0
recall = 0.0
precision = 0.0
f1_score = 0.0

if total_predicted_records > 0:

    duplicate_rate = (
        total_duplicates
        / total_predicted_records
    )

    precision = (
        total_potholes_matched
        / total_predicted_records
    )

if total_gt > 0:

    recall = (
        total_potholes_matched
        / total_gt
    )

if (precision + recall) > 0:

    f1_score = (
        2
        * precision
        * recall
        / (precision + recall)
    )

# =====================================
# SAVE RESULT
# =====================================

result_df = pd.DataFrame([
    {
        "method": METHOD_NAME,

        "total_frame_records":
        total_frame_records,

        "total_predicted_records":
        total_predicted_records,

        "total_potholes_matched":
        total_potholes_matched,

        "total_duplicates":
        total_duplicates,

        "total_gt":
        total_gt,

        "duplicate_rate":
        round(duplicate_rate, 4),

        "recall":
        round(recall, 4),

        "precision":
        round(precision, 4),

        "f1_score":
        round(f1_score, 4)
    }
])

output_file = os.path.join(
    METHOD_DIR,
    f"{METHOD_NAME}_results.csv"
)

result_df.to_csv(
    output_file,
    index=False
)

print(
    f"[DONE] Results saved to: "
    f"{output_file}"
)

print(result_df)