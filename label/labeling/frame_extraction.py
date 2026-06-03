import cv2
import numpy as np
import os

# --- CONFIGURATION ---
VIDEO_PATH = "video_01.mp4"
# ---------------------

def on_trackbar(val):
    # This function updates the video frame when you slide the trackbar
    global cap, frame_count
    if val >= frame_count:
        val = frame_count - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, val)
    ret, frame = cap.read()
    if ret:
        display_frame(frame)
        display_controls(val)

def display_frame(frame):
    # Display only the video frame without toolbar overlay
    cv2.imshow("Frame Extractor Window", frame)


def display_controls(current_frame):
    # Show toolbar information in a dedicated control window
    toolbar_w = 520
    toolbar_h = 80
    toolbar = 255 * np.ones((toolbar_h, toolbar_w, 3), dtype=np.uint8)

    cv2.putText(toolbar, f"FRAME: {current_frame} / {frame_count - 1}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(toolbar, "D/A = +/-1 | G = jump frame | Q = quit", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 1)
    cv2.imshow("Frame Extractor Controls", toolbar)


def ask_frame_number_via_gui(current_frame):
    global cap, frame_count
    frame_input = ""
    prompt_message = "Enter frame number and press ENTER. ESC to cancel."

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            return current_frame

        display_frame(frame)
        display_controls(current_frame)

        prompt_w = 520
        prompt_h = 80
        prompt_window = 255 * np.ones((prompt_h, prompt_w, 3), dtype=np.uint8)
        cv2.putText(prompt_window, prompt_message, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(prompt_window, frame_input if frame_input else "_", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.imshow("Frame Extractor Prompt", prompt_window)

        key = cv2.waitKey(0) & 0xFF
        if key == 13:  # Enter
            cv2.destroyWindow("Frame Extractor Prompt")
            if frame_input == "":
                return current_frame
            try:
                chosen_frame = int(frame_input)
                if chosen_frame < 0 or chosen_frame >= frame_count:
                    frame_input = ""
                    prompt_message = f"Invalid range. Enter 0 to {frame_count - 1}."
                    continue
                return chosen_frame
            except ValueError:
                frame_input = ""
                prompt_message = "Invalid number. Enter digits only."
                continue
        elif key == 27:  # ESC
            cv2.destroyWindow("Frame Extractor Prompt")
            return current_frame
        elif key in range(ord('0'), ord('9') + 1):
            if len(frame_input) < 8:
                frame_input += chr(key)
        elif key in (8, 127):
            frame_input = frame_input[:-1]
        else:
            prompt_message = "Digits, ENTER, or ESC only."

if not os.path.exists(VIDEO_PATH):
    print(f"Error: Video file '{VIDEO_PATH}' not found.")
    exit()

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow("Frame Extractor Window")
# Create a slider tied to the total frame count
cv2.createTrackbar("Frame", "Frame Extractor Window", 0, frame_count - 1, on_trackbar)

# Show the first frame initially
ret, first_frame = cap.read()
if ret:
    display_frame(first_frame)
    display_controls(0)

print("\n--- INSTRUCTIONS ---")
print("1. Drag the slider to skim through the video.")
print("2. Click on the video window and use keyboard hotkeys for fine tuning:")
print("   - Press 'D' to move FORWARD 1 frame")
print("   - Press 'A' to move BACKWARD 1 frame")
print("   - Press 'G' to jump to a specific frame number")
print("3. Note your START_FRAME and END_FRAME from the top left text display.")
print("4. Press 'Q' to exit.")

current_f = 0
while True:
    key = cv2.waitKey(0) & 0xFF
    
    # 'd' key to step forward
    if key == ord('d'):
        current_f = min(current_f + 1, frame_count - 1)
        cv2.setTrackbarPos("Frame", "Frame Extractor Window", current_f)
        on_trackbar(current_f)
        
    # 'a' key to step backward
    elif key == ord('a'):
        current_f = max(current_f - 1, 0)
        cv2.setTrackbarPos("Frame", "Frame Extractor Window", current_f)
        on_trackbar(current_f)
        
    # 'g' key to jump to a specific frame
    elif key == ord('g'):
        jump_frame = ask_frame_number_via_gui(current_f)
        if jump_frame != current_f:
            current_f = jump_frame
            cv2.setTrackbarPos("Frame", "Frame Extractor Window", current_f)
            on_trackbar(current_f)
        else:
            cv2.setTrackbarPos("Frame", "Frame Extractor Window", current_f)
            on_trackbar(current_f)
        
    # 'q' key to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()