"""
Input: video
Output: predictions.csv
"""

from ultralytics import YOLO
import cv2
import pandas as pd
import os

# Config
VIDEO_ID = "video_01"

BASE_DIR = "dataset"
METHOD_DIR = "detect/detect_yolo"

MODEL_PATH = "../best.pt"

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# Run YOLO
model = YOLO(MODEL_PATH)

video_path = os.path.join(
    BASE_DIR,
    VIDEO_ID,
    f"{VIDEO_ID}.mp4"
)

cap = cv2.VideoCapture(video_path)

predictions = []

detection_counter = 1
frame_id = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(
        frame,
        conf=CONF_THRESHOLD,
        verbose=False
    )

    boxes = results[0].boxes

    if boxes is not None:

        for box in boxes:

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            confidence = float(
                box.conf[0]
            )

            predictions.append([
                f"D{detection_counter}",
                frame_id,
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                confidence
            ])

            detection_counter += 1

    frame_id += 1

cap.release()

# Save predictions.csv
pred_df = pd.DataFrame(
    predictions,
    columns=[
        "detection_id",
        "frame_id",
        "x1",
        "y1",
        "x2",
        "y2",
        "confidence"
    ]
)

output_dir = os.path.join(
    METHOD_DIR,
    VIDEO_ID
)

os.makedirs(
    output_dir,
    exist_ok=True
)

pred_df.to_csv(
    os.path.join(
        output_dir,
        "predictions.csv"
    ),
    index=False
)