import cv2
import pandas as pd
import os
import argparse

"""
Create a single annotated video with all labeled potholes.

Reads a CSV with columns: gt_pothole_id, frame_id, x1, y1, x2, y2
and draws bounding boxes directly on the source video.
Saves the result as a new video file with all annotations overlaid.
"""

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_VIDEO = os.path.join(SCRIPT_DIR, "video_02.mp4")
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "video_02.csv")


def create_annotated_video(csv_path, video_path, output_path=None, colors=None):
    """
    Create annotated video with bounding boxes from CSV.
    
    Args:
        csv_path: Path to CSV with gt_pothole_id, frame_id, x1, y1, x2, y2
        video_path: Path to input video
        output_path: Path to output video (default: video_annotated.mp4)
        colors: Dict mapping pothole_id to color tuple (B, G, R)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    required_cols = {'gt_pothole_id', 'frame_id', 'x1', 'y1', 'x2', 'y2'}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain: {required_cols}")
    
    # Output path
    if output_path is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(os.path.dirname(video_path), f"{base}_annotated.mp4")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Cannot create output video: {output_path}")
    
    # Generate colors for each pothole ID
    if colors is None:
        colors = {}
        unique_ids = df['gt_pothole_id'].unique()
        color_list = [
            (0, 255, 0),    # Green
            (0, 165, 255),  # Orange
            (255, 0, 0),    # Blue
            (0, 255, 255),  # Cyan
            (255, 255, 0),  # Magenta
            (255, 0, 255),  # Purple
            (0, 128, 255),  # Gold
            (128, 0, 255),  # Pink
            (255, 128, 0),  # Dark Blue
            (0, 255, 128),  # Spring Green
        ]
        for i, pid in enumerate(unique_ids):
            colors[pid] = color_list[i % len(color_list)]
    
    # Group annotations by frame
    frame_annotations = df.groupby('frame_id')
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}")
    print(f"Unique potholes: {df['gt_pothole_id'].nunique()}")
    print(f"Annotations: {len(df)}")
    print(f"Output: {output_path}")
    
    # Process frames
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw annotations for this frame
        if frame_id in frame_annotations.groups:
            annotations = frame_annotations.get_group(frame_id)
            for _, row in annotations.iterrows():
                x1 = int(row['x1'])
                y1 = int(row['y1'])
                x2 = int(row['x2'])
                y2 = int(row['y2'])
                pothole_id = row['gt_pothole_id']
                
                color = colors.get(pothole_id, (0, 255, 0))
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{pothole_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Draw frame number overlay (top-left) for easier tracking
        try:
            frame_label = f"Frame: {frame_id}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(frame_label, font, font_scale, font_thickness)
            margin = 8
            x0, y0 = margin, margin
            rect_tl = (x0 - 4, y0 - 4)
            rect_br = (x0 + text_w + 4, y0 + text_h + 4)

            # Use an overlay for semi-transparent background
            overlay = frame.copy()
            cv2.rectangle(overlay, rect_tl, rect_br, (0, 0, 0), -1)
            alpha = 0.45
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # Put white text over the background
            text_pos = (x0, y0 + text_h - 2)
            cv2.putText(frame, frame_label, text_pos, font, font_scale, (255, 255, 255), font_thickness)
        except Exception:
            # If anything goes wrong with overlay, continue without crashing
            pass
        
        # Write frame
        out.write(frame)
        frame_id += 1
        
        if frame_id % 100 == 0:
            print(f"  Processed {frame_id}/{frame_count} frames...")
    
    cap.release()
    out.release()
    
    print(f"✓ Saved annotated video: {output_path}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description='Create video with all labeled potholes annotated.')
    parser.add_argument('--csv', '-c', type=str, default=DEFAULT_CSV, help='Input CSV file with annotations.')
    parser.add_argument('--video', '-v', type=str, default=DEFAULT_VIDEO, help='Input video file.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output video path (default: video_annotated.mp4).')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_annotated_video(args.csv, args.video, args.output)
