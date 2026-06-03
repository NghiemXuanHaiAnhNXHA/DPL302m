"""
detect_yolo.py
------------------------
Script phát hiện ổ gà (pothole) trong video bằng YOLO (không tracking).
- Nhập đường dẫn video (hoặc kéo thả).
- Chạy model best.pt trên từng frame.
- Frame nào phát hiện ổ gà → lưu vào folder "frame_by_frame/" kèm bounding box.

Cách dùng:
    python detect_yolo.py
    python detect_yolo.py --video path/to/video.mp4
    python detect_yolo.py --video path/to/video.mp4 --conf 0.5
"""

import os
import sys
import argparse
import cv2
from ultralytics import YOLO

# ── Cấu hình ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "best.pt")
FRAME_DIR = os.path.join(BASE_DIR, "frame_by_frame")


def load_model(weights: str) -> YOLO:
    """Tải YOLO model từ file weights."""
    if not os.path.exists(weights):
        print(f"[LỖI] Không tìm thấy file weights: {weights}")
        sys.exit(1)
    print(f"[INFO] Đang tải model từ {weights} ...")
    model = YOLO(weights)
    print("[INFO] Tải model thành công!")
    return model


def detect_potholes_in_video(
    video_path: str,
    model: YOLO,
    output_dir: str,
    conf: float = 0.6,
    skip: int = 1,
) -> None:
    """
    Đọc video, chạy YOLO trên từng frame.
    Nếu frame có ổ gà → lưu vào output_dir.

    Args:
        video_path:  Đường dẫn tới video đầu vào.
        model:       YOLO model đã load.
        output_dir:  Thư mục lưu frame (mặc định: "frame_by_frame").
        conf:        Ngưỡng confidence (0‑1).
        skip:        Xử lý mỗi `skip` frame (1 = mọi frame, 2 = cách 1 frame …).
    """
    if not os.path.isfile(video_path):
        print(f"[LỖI] Không tìm thấy video: {video_path}")
        return

    # Tạo folder lưu frame nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[LỖI] Không mở được video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Video: {video_path}")
    print(f"[INFO] Tổng số frame: {total_frames} | FPS: {fps:.1f}")
    print(f"[INFO] Confidence threshold: {conf}")
    print(f"[INFO] Xử lý mỗi {skip} frame")
    print(f"[INFO] Thư mục lưu frame: {output_dir}")
    print("-" * 60)

    frame_idx = 0          # chỉ số frame đang đọc
    saved_count = 0        # số frame đã lưu

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Bỏ qua frame nếu cần
        if frame_idx % skip != 0:
            continue

        # ── Chạy YOLO ────────────────────────────────────────────────
        results = model(frame, imgsz=640, conf=conf, verbose=False)

        # Kiểm tra có detection nào không
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            # Không phát hiện ổ gà → bỏ qua
            _print_progress(frame_idx, total_frames)
            continue

        # ── Có ổ gà → vẽ bbox và lưu ảnh ─────────────────────────
        annotated = results[0].plot()

        # Tên file: frame_<số thứ tự>_<số ổ gà>.jpg
        num_detections = len(boxes)
        filename = f"frame_{frame_idx:06d}_{num_detections}potholes.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, annotated)

        saved_count += 1
        _print_progress(frame_idx, total_frames, detected=num_detections)

    cap.release()
    print()
    print("=" * 60)
    print(f"[HOÀN TẤT] Đã xử lý {frame_idx} frame.")
    print(f"[HOÀN TẤT] Lưu {saved_count} frame có ổ gà vào: {output_dir}")
    print("=" * 60)


def _print_progress(current: int, total: int, detected: int = 0) -> None:
    """In tiến trình xử lý (ghi đè dòng)."""
    pct = (current / total * 100) if total > 0 else 0
    det_str = f" | Phát hiện {detected} ổ gà ✓" if detected else ""
    print(f"\r[{pct:5.1f}%] Frame {current}/{total}{det_str}    ", end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phát hiện ổ gà trong video bằng YOLO và lưu các frame có ổ gà."
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Đường dẫn tới file video. Nếu không truyền sẽ hỏi nhập từ bàn phím.",
    )
    parser.add_argument(
        "--conf", "-c",
        type=float,
        default=0.6,
        help="Ngưỡng confidence cho YOLO (mặc định: 0.6).",
    )
    parser.add_argument(
        "--skip", "-s",
        type=int,
        default=1,
        help="Xử lý mỗi N frame (mặc định: 1 = tất cả frame).",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=FRAME_DIR,
        help=f'Thư mục lưu frame (mặc định: "{FRAME_DIR}").',
    )
    parser.add_argument(
        "--weights", "-w",
        type=str,
        default=WEIGHTS_PATH,
        help=f'File weights YOLO (mặc định: "{WEIGHTS_PATH}").',
    )

    args = parser.parse_args()

    # Nếu không truyền --video thì hỏi nhập
    video_path = args.video
    if video_path is None:
        video_path = input("Nhập đường dẫn video: ").strip().strip('"').strip("'")

    if not video_path:
        print("[LỖI] Chưa nhập đường dẫn video!")
        sys.exit(1)

    # Load model & chạy detect
    model = load_model(args.weights)
    detect_potholes_in_video(
        video_path=video_path,
        model=model,
        output_dir=args.output,
        conf=args.conf,
        skip=args.skip,
    )


if __name__ == "__main__":
    main()
