import cv2
import os
import argparse

"""
Standalone video playback + save script.
Input: video file.
Output: saved video file with optional overlays.
Also displays each frame while processing.
"""

DEFAULT_VIDEO = 'video_01.mp4'


def make_output_path(video_path, output_path=None):
    if output_path:
        return output_path
    base = os.path.splitext(os.path.basename(video_path))[0]
    return f"{base}_output.mp4"


def run_video_display_save(video_path, output_path=None, overlay_text=True):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path = make_output_path(video_path, output_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    window_name = 'Video Display'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_id = 0
    # Use a minimal delay so the script processes the entire video as fast as possible
    delay = 1

    print(f"Processing and saving video: {video_path}")
    print(f"Output video will be: {output_path}")
    print("Press 'q' to stop early. Otherwise the script will process the full video immediately.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if overlay_text:
            text = f"Frame: {frame_id + 1}/{total_frames}"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow(window_name, frame)

        key = cv2.waitKey(delay) & 0xFF
        if key == ord('q'):
            print("Playback stopped by user.")
            break

        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Saved output video: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description='Play a video and save a copy while displaying frames.')
    parser.add_argument('--video', '-v', type=str, default=None, help='Path to input video file. Uses DEFAULT_VIDEO if omitted.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video path. Defaults to <input>_output.mp4.')
    parser.add_argument('--no-overlay', action='store_true', help='Do not draw frame text overlay on saved/displayed video.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    video_path = args.video if args.video else DEFAULT_VIDEO
    run_video_display_save(video_path, args.output, overlay_text=not args.no_overlay)
