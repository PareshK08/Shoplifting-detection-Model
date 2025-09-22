import cv2
import os
import pandas as pd
from ultralytics import YOLO
import torch

# Select device automatically (GPU if available)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load YOLOv8 pose model
model = YOLO("yolov8n-pose.pt").to(device)

# Input video
video_path = r"Videos/vid2.mp4"
video_name = os.path.basename(video_path)  # keep track of which video data came from

# Output dataset directories
dataset_path = "Asset/Dataset"
normal_path = os.path.join(dataset_path, "Normal")
suspicious_path = os.path.join(dataset_path, "Suspicious")
os.makedirs(normal_path, exist_ok=True)
os.makedirs(suspicious_path, exist_ok=True)

# CSV to store keypoints + labels
csv_file = "Asset\\Dataset\\dataset_keypoints.csv"
data = []

# --- Control how often to save frames ---
SAVE_EVERY = 3  # Save every 5th frame (adjust as needed)
# ----------------------------------------

# Open video
cap = cv2.VideoCapture(video_path)
frame_count = 0

# Person ID to track in suspicious mode
TARGET_ID = 1   # change this to your target ID

# IDs to exclude from Normal dataset
EXCLUDED_IDS = [2]  # <-- Add IDs you don't want in Normal mode

# Current active label
current_label = None

# Track all unique IDs seen (independent of mode)
unique_ids = set()

print("Controls:")
print("Press 'n' → switch to Normal mode")
print("Press 's' → switch to Suspicious mode")
print("Press 'q' → quit and save dataset")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Run YOLO with tracking
    results = model.track(frame, persist=True, verbose=False, device=device)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int).tolist()
        keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else []

        # --- Always update unique IDs set ---
        unique_ids.update(ids)

        # Show frame with IDs
        display_frame = frame.copy()
        for idx, person_id in enumerate(ids):
            x1, y1, x2, y2 = boxes[idx]
            cv2.putText(display_frame, f"ID:{person_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show current mode on screen
        if current_label is not None:
            cv2.putText(display_frame, f"MODE: {current_label}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Tracking", display_frame)

        # Non-blocking key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            current_label = "Normal"
            print("Switched to Normal mode")
        elif key == ord('s'):
            current_label = "Suspicious"
            print("Switched to Suspicious mode")
        elif key == ord('q'):
            break

        # --- Save only when mode is set ---
        if current_label is not None and frame_count % SAVE_EVERY == 0:
            for idx, person_id in enumerate(ids):

                # Skip excluded IDs in Normal mode
                if current_label == "Normal" and person_id in EXCLUDED_IDS:
                    continue

                # Only save TARGET_ID in Suspicious mode
                if current_label == "Suspicious" and person_id != TARGET_ID:
                    continue

                # Clip bounding box to frame
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = boxes[idx]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    continue

                # Save cropped image
                save_dir = normal_path if current_label == "Normal" else suspicious_path
                save_path = os.path.join(save_dir, f"{video_name}_frame{frame_count}_id{person_id}.jpg")
                cv2.imwrite(save_path, cropped)

                # Save keypoints
                if len(keypoints) > idx:
                    kpts = keypoints[idx].flatten().tolist()
                else:
                    kpts = [0] * 34
                data.append([video_name, frame_count, person_id] + kpts + [current_label])

    else:
        # Still show frame even if no IDs
        display_frame = frame.copy()
        cv2.putText(display_frame, "No IDs detected", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Tracking", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Save dataset safely
df = pd.DataFrame(data, columns=["video", "frame", "person_id"] + [f"kpt_{i}" for i in range(34)] + ["label"])

if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
    try:
        old_df = pd.read_csv(csv_file)
        df = pd.concat([old_df, df], ignore_index=True)
    except pd.errors.EmptyDataError:
        print("⚠️ Existing CSV was empty, creating a new one.")

df.to_csv(csv_file, index=False)

# --- Print unique IDs seen, always ---
print(f"👥 Unique IDs detected: {sorted(unique_ids)}")
print(f"🔢 Total unique IDs: {len(unique_ids)}")
# -------------------------------------

print(f"✅ Cropped dataset created with {len(df)} total samples")
print(f"Images saved under: {dataset_path}")
print(f"CSV saved as: {csv_file}")
