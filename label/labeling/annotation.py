import cv2
import pandas as pd
import os

# ==============================================================================
# --- CONFIGURATION ---
# Change these values for each video segment/pothole you want to process.
# ==============================================================================
VIDEO_PATH = "video_01.mp4"
OUTPUT_CSV = "video_01.csv"
POTHOLE_ID = "P12"  # Unique ID for the pothole (e.g., P1, P2, P3...)
START_FRAME = 991   # Frame number where the pothole first appears
END_FRAME = 1005 # Frame number where the pothole disappears or leaves the view
# ==============================================================================

def main():
    # Verify video exists
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file '{VIDEO_PATH}' not found. Please check your path.")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    data_list = []

    # 1. Jump directly to the starting frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read the starting frame. Check your START_FRAME setting.")
        cap.release()
        return

    # 2. Manual initialization for the very first frame
    print(f"\n[>>>] Initializing {POTHOLE_ID} at frame {START_FRAME}.")
    print("Action Required: Draw a bounding box around the pothole, then press ENTER or SPACE.")
    
    clone_init = frame.copy()
    cv2.putText(clone_init, f"INIT: {POTHOLE_ID} | Frame: {START_FRAME}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    bbox = cv2.selectROI("Initial Annotation", clone_init, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Initial Annotation")
    
    # If the user closes the window without selecting anything
    if bbox[2] == 0 or bbox[3] == 0:
        print("Initialization cancelled by user. Exiting.")
        cap.release()
        return

    # Initialize the tracking algorithm
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Convert OpenCV format (x, y, w, h) to absolute IoU-ready coordinates (x1, y1, x2, y2)
    x1, y1, w, h = [int(v) for v in bbox]
    data_list.append({
        "gt_pothole_id": POTHOLE_ID,
        "frame_id": START_FRAME,
        "x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h
    })

    # 3. Track automatically with Human Fallback
    frame_id = START_FRAME + 1
    terminate_early = False

    while frame_id <= END_FRAME:
        ret, frame = cap.read()
        if not ret:
            print(f"Reached end of video stream at frame {frame_id - 1}.")
            break

        # Attempt automated tracking update
        success, bbox = tracker.update(frame)

        if success:
            # Algorithmic tracking succeeded
            x1, y1, w, h = [int(v) for v in bbox]
            x2, y2 = x1 + w, y1 + h
        else:
            # --- HUMAN INTERVENTION FALLBACK ---
            print(f"\n[!] Tracking lost at frame {frame_id}.")
            clone_fallback = frame.copy()
            cv2.putText(clone_fallback, f"LOST TRACKING! Frame: {frame_id}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(clone_fallback, "Press 'c' to correct box, 'e' to exit and save.", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Show options and wait for user decision
            while True:
                cv2.imshow("Human Correction Required", clone_fallback)
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyWindow("Human Correction Required")

                # Continue and specify bounding box manually
                if key == ord('c'):
                    bbox = cv2.selectROI("Human Correction Required - Draw Box then ENTER/SPACE", clone_fallback, fromCenter=False, showCrosshair=True)
                    cv2.destroyWindow("Human Correction Required - Draw Box then ENTER/SPACE")

                    # If user didn't draw a box, return to options
                    if bbox[2] == 0 or bbox[3] == 0:
                        print("No box drawn. Returning to options.")
                        continue

                    x1, y1, w, h = [int(v) for v in bbox]
                    x2, y2 = x1 + w, y1 + h

                    # Re-initialize the tracker on this frame using the human-corrected data
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, bbox)
                    print(f"[+] Tracker successfully re-anchored at frame {frame_id}.")
                    break

                # Exit and save current results
                elif key == ord('e') or key == ord('s') or key == ord('q'):
                    print(f"[i] Exiting at frame {frame_id} and saving collected results.")
                    terminate_early = True
                    break

                else:
                    print("Invalid key. Press 'c' to correct or 'q' to exit and save.")

            if terminate_early:
                break

        # Store the precise data coordinates
        data_list.append({
            "gt_pothole_id": POTHOLE_ID,
            "frame_id": frame_id,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2
        })

        # Render visual validation box on screen (Live review)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Tracking: {POTHOLE_ID} | Frame: {frame_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Ground Truth Annotation Tool", frame)
        
        # Press 'q' key on keyboard to abort processing current sequence early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nProcessing aborted early by user.")
            break
            
        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    # --- 4. CSV EXPORT & APPEND PIPELINE ---
    if data_list:
        df_new = pd.DataFrame(data_list)
        
        # If dataset already exists from previous runs, append without destroying old work
        if os.path.exists(OUTPUT_CSV):
            df_existing = pd.read_csv(OUTPUT_CSV)
            df_final = pd.concat([df_existing, df_new], ignore_index=True)
            print(f"\n[+] Appended tracking data for {POTHOLE_ID} to existing file.")
        else:
            df_final = df_new
            print(f"\n[+] Created brand new dataset file: {OUTPUT_CSV}")

        # Save out to CSV
        df_final.to_csv(OUTPUT_CSV, index=False)
        print(f"[✓] Successfully recorded frames {START_FRAME} to {frame_id - 1} into '{OUTPUT_CSV}'.")
    else:
        print("\n[!] Processing completed. No data points collected.")

if __name__ == "__main__":
    main()