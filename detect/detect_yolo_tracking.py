"""
detect_yolo_tracking.py
---------------
Script phát hiện ổ gà (pothole) trong video bằng YOLO + ByteTrack.
- Nhập đường dẫn video.
- Chạy model best.pt với ByteTrack tracking trên từng frame.
- Mỗi ổ gà được gán 1 track ID duy nhất.
- Chỉ lưu frame đại diện tốt nhất (confidence cao nhất) cho mỗi track ID
→ tránh 1 ổ gà bị lưu thành nhiều frame trùng lặp.

Cách dùng:
    python detect_yolo_tracking.py
    python detect_yolo_tracking.py --video path/to/video.mp4
    python detect_yolo_tracking.py --video path/to/video.mp4 --conf 0.5
"""

import os
import sys
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# ── Cấu hình ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(BASE_DIR, "best.pt")
FRAME_DIR = os.path.join(BASE_DIR, "frame")


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
    Đọc video, chạy YOLO + ByteTrack trên từng frame.
    Mỗi ổ gà (track ID) chỉ lưu 1 frame đại diện có confidence cao nhất.

    Args:
        video_path:  Đường dẫn tới video đầu vào.
        model:       YOLO model đã load.
        output_dir:  Thư mục lưu frame (mặc định: "frame").
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
    print(f"[INFO] Tracker: ByteTrack")
    print(f"[INFO] Thư mục lưu frame: {output_dir}")
    print("-" * 60)

    frame_idx = 0          # chỉ số frame đang đọc
    unique_tracks = 0      # số track ID duy nhất đã phát hiện

    # ── Tracking state ────────────────────────────────────────────
    # Lưu frame tốt nhất cho mỗi track ID
    # best_frames[track_id] = {
    #     "frame_idx": int,
    #     "confidence": float,
    #     "annotated": np.ndarray,   # frame đã vẽ bbox
    #     "num_detections": int,     # tổng số detection trong frame đó
    # }
    best_frames: dict = {}
    seen_track_ids: set = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Bỏ qua frame nếu cần
        if frame_idx % skip != 0:
            continue

        # ── Chạy YOLO + ByteTrack ────────────────────────────────
        results = model.track(
            frame,
            imgsz=640,
            conf=conf,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False,
        )

        # Kiểm tra có detection nào không
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            _print_progress(frame_idx, total_frames)
            continue

        # Lấy track IDs và confidences
        track_ids = boxes.id  # tensor hoặc None nếu tracker chưa gán
        confidences = boxes.conf  # tensor

        if track_ids is None:
            # Tracker chưa gán ID (frame đầu tiên hoặc lỗi) → bỏ qua
            _print_progress(frame_idx, total_frames)
            continue

        track_ids_list = track_ids.cpu().numpy().astype(int).tolist()
        conf_list = confidences.cpu().numpy().tolist()

        # ── Kiểm tra xem có track ID mới không ───────────────────
        new_ids_in_frame = []
        for tid, c in zip(track_ids_list, conf_list):
            if tid not in seen_track_ids:
                # Track ID hoàn toàn mới → cần lưu
                seen_track_ids.add(tid)
                new_ids_in_frame.append(tid)
                best_frames[tid] = {
                    "frame_idx": frame_idx,
                    "confidence": c,
                    "annotated": None,  # sẽ vẽ sau
                    "num_detections": len(boxes),
                }
            elif c > best_frames[tid]["confidence"]:
                # Track ID đã thấy nhưng frame này có confidence cao hơn
                # → cập nhật frame đại diện
                best_frames[tid] = {
                    "frame_idx": frame_idx,
                    "confidence": c,
                    "annotated": None,
                    "num_detections": len(boxes),
                }

        # Vẽ bbox cho các track cần lưu/cập nhật ở frame này
        annotated = results[0].plot()
        for tid in track_ids_list:
            if tid in best_frames and best_frames[tid]["frame_idx"] == frame_idx:
                best_frames[tid]["annotated"] = annotated.copy()

        if new_ids_in_frame:
            unique_tracks = len(seen_track_ids)
            _print_progress(
                frame_idx, total_frames,
                detected=len(boxes),
                new_ids=len(new_ids_in_frame),
                total_tracks=unique_tracks,
            )
        else:
            _print_progress(frame_idx, total_frames)

    cap.release()

    # ── Lưu frame đại diện cho từng track ID ─────────────────────
    print()
    print("-" * 60)
    print(f"[INFO] Đang lưu frame đại diện cho {len(best_frames)} ổ gà ...")

    saved_count = 0
    for tid, info in sorted(best_frames.items()):
        if info["annotated"] is not None:
            filename = f"pothole_track{tid:04d}_frame{info['frame_idx']:06d}_conf{info['confidence']:.2f}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, info["annotated"])
            saved_count += 1
            print(f"  ✓ Track #{tid}: frame {info['frame_idx']}, conf={info['confidence']:.3f} → {filename}")

    print("=" * 60)
    print(f"[HOÀN TẤT] Đã xử lý {frame_idx} frame.")
    print(f"[HOÀN TẤT] Phát hiện {len(seen_track_ids)} ổ gà duy nhất (track IDs).")
    print(f"[HOÀN TẤT] Lưu {saved_count} frame đại diện vào: {output_dir}")
    print("=" * 60)


def _print_progress(
    current: int, total: int,
    detected: int = 0,
    new_ids: int = 0,
    total_tracks: int = 0,
) -> None:
    """In tiến trình xử lý (ghi đè dòng)."""
    pct = (current / total * 100) if total > 0 else 0
    if detected:
        det_str = f" | {detected} detect, +{new_ids} mới, tổng {total_tracks} track"
    else:
        det_str = ""
    print(f"\r[{pct:5.1f}%] Frame {current}/{total}{det_str}    ", end="", flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Phát hiện ổ gà trong video bằng YOLO + ByteTrack, lưu frame đại diện."
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
