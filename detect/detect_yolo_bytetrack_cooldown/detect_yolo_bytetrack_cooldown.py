import os
import cv2
import pandas as pd

from ultralytics import YOLO

# =====================================
# CONFIG
# =====================================

VIDEO_ID = "video_01"

MODEL_PATH = "DPL302m/detect/best.pt"

DATASET_DIR = "DPL302m/dataset"

BASE_DIR = "DPL302m/detect"

METHOD = "detect_yolo_bytetrack_cooldown"

METHOD_DIR = os.path.join(
    BASE_DIR,
    METHOD
)

CONF_THRESHOLD = 0.5

# -------------------------------------
# Hyperparameter
# -------------------------------------

COOLDOWN_FRAMES = 20

# =====================================
# PATHS
# =====================================

video_path = os.path.join(
    DATASET_DIR,
    VIDEO_ID,
    f"{VIDEO_ID}.mp4"
)

output_dir = os.path.join(
    METHOD_DIR,
    VIDEO_ID
)

os.makedirs(
    output_dir,
    exist_ok=True
)

# =====================================
# LOAD MODEL
# =====================================

model = YOLO(MODEL_PATH)

# =====================================
# VIDEO
# =====================================

cap = cv2.VideoCapture(video_path)

predictions = []

frame_id = 0

detection_counter = 1

# -------------------------------------
# Track memory
# -------------------------------------

saved_tracks = set()

last_saved_frame = -999999

# =====================================
# DETECT + TRACK + COOLDOWN
# =====================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        conf=CONF_THRESHOLD,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    boxes = results[0].boxes

    if (
        boxes is None
        or len(boxes) == 0
        or boxes.id is None
    ):
        frame_id += 1
        continue

    for box in boxes:

        if box.id is None:
            continue

        track_id = int(
            box.id.cpu().item()
        )

        # ---------------------
        # Track already saved
        # ---------------------

        if track_id in saved_tracks:
            continue

        # ---------------------
        # Global cooldown
        # ---------------------

        if (
            frame_id
            - last_saved_frame
            < COOLDOWN_FRAMES
        ):
            continue

        saved_tracks.add(track_id)

        last_saved_frame = frame_id

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

            track_id,
            confidence
        ])

        detection_counter += 1

    frame_id += 1

cap.release()

# =====================================
# SAVE CSV
# =====================================

pred_df = pd.DataFrame(
    predictions,
    columns=[
        "detection_id",
        "frame_id",

        "x1",
        "y1",
        "x2",
        "y2",

        "track_id",
        "confidence"
    ]
)

pred_path = os.path.join(
    output_dir,
    "predictions.csv"
)

pred_df.to_csv(
    pred_path,
    index=False
)

print(
    f"[DONE] Saved predictions: "
    f"{pred_path}"
)

print(
    f"Unique tracks: "
    f"{len(pred_df)}"
)