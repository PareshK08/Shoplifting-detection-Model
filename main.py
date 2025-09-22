
import os
import cv2
import time
import math
import queue
import joblib
import threading
import numpy as np
import datetime as dt
from collections import defaultdict, deque

import torch
from ultralytics import YOLO

# ===================== CONFIG =====================
# Video source: 0 (webcam) | path to file | RTSP URL like "rtsp://user:pass@IP:554/..."
VIDEO_SOURCE = 0  # e.g., 0 or "rtsp://..."
WINDOW_NAME = "CCTV Shoplifting Detector"

# Models
POSE_WEIGHTS = "yolo11n-pose.pt"                       # YOLOv11 Pose
CLS_WEIGHTS  = "runs/classify/train/weights/best.pt"   # YOLOv11 Classify (Normal/Suspicious)
KPTS_CLF_PKL = "keypoints_rf.pkl"
KPTS_LE_PKL  = "keypoints_label_encoder.pkl"

# Classes expected in the image classifier
IMG_EXPECTED_CLASSES = ["Normal", "Suspicious"]

# Alert logic
ALERT_THRESHOLD   = 0.5     # Suspicious probability threshold after fusion
ALERT_COOLDOWN_S  = 30       # Per-ID cooldown
ALERT_BURST_N     = 3        # Collect N crops before sending email (also used as evidence pack)

# Smoothing / debouncing
EMA_ALPHA         = 0.3      # EMA for per-ID suspicious prob smoothing
VOTE_WINDOW       = 12       # Sliding window size per ID
VOTE_MIN_SUSP_FR  = 6        # Require at least this many "suspicious" frames in the window

# Fusion weights
W_IMG             = 0.6      # Image classifier weight
W_KPT             = 0.4      # Keypoints classifier weight

# Performance
TARGET_MAX_WIDTH  = 1280     # Downscale incoming frames to this width (keep aspect)
PROCESS_EVERY_N   = 1        # Process every Nth frame (>=2 to skip frames on weak hardware)
POSE_IMGSZ        = 640      # Pose inference size
CLS_IMGSZ         = 224      # Classifier inference size
DRAW_THICK        = 2
SHOW_WINDOW       = True     # Set False on headless servers

# Email (read from env; safer than hardcoding)
SENDER_EMAIL      = os.getenv("SHOPLIFT_EMAIL", "")
APP_PASSWORD      = os.getenv("SHOPLIFT_APP_PASSWORD", "")
RECIPIENT_EMAIL   = os.getenv("SHOPLIFT_RECIPIENT", "")

# Output
ALERT_DIR         = "alerts"
os.makedirs(ALERT_DIR, exist_ok=True)

# ===================== DEVICE & MODELS =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {DEVICE.upper()}")

model_pose = YOLO(POSE_WEIGHTS)
model_pose.to(DEVICE)

# half precision on GPU for speed
POSE_KW = {"imgsz": POSE_IMGSZ, "device": DEVICE, "verbose": False, "persist": True}
if DEVICE == "cuda":
    POSE_KW["half"] = True

# Image classifier (YOLO-CLS)
model_cls = YOLO(CLS_WEIGHTS)
model_cls.to(DEVICE)

# Resolve class index mapping robustly
IMG_CLASS_NAMES = model_cls.names  # dict like {0:'Normal',1:'Suspicious'} or list
if isinstance(IMG_CLASS_NAMES, dict):
    inv_names = {v: k for k, v in IMG_CLASS_NAMES.items()}
else:
    inv_names = {name: i for i, name in enumerate(IMG_CLASS_NAMES)}

missing = [c for c in IMG_EXPECTED_CLASSES if c not in inv_names]
if missing:
    raise SystemExit(f"[ERROR] Your classification model is missing classes: {missing}. Found: {IMG_CLASS_NAMES}")

idx_normal = inv_names["Normal"]
idx_susp   = inv_names["Suspicious"]

# Keypoints classifier
clf_kpts = joblib.load(KPTS_CLF_PKL)
le       = joblib.load(KPTS_LE_PKL)
try:
    kpts_idx_normal = list(le.classes_).index("Normal")
    kpts_idx_susp   = list(le.classes_).index("Suspicious")
except ValueError:
    # fallback to binary 0/1 predicted order
    print(f"[WARN] 'Normal'/'Suspicious' not found in keypoints label encoder. Classes: {list(le.classes_)}")
    kpts_idx_normal, kpts_idx_susp = 0, 1

# ===================== UTILITIES =====================
def safe_resize(frame, max_w):
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / float(w)
    nh = int(h * scale)
    return cv2.resize(frame, (max_w, nh), interpolation=cv2.INTER_AREA)

def get_img_probs(crop_bgr):
    # Ultralytics YOLO classify can take numpy directly
    r = model_cls.predict(crop_bgr, imgsz=CLS_IMGSZ, device=DEVICE, verbose=False)
    probs = r[0].probs.data
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    p_norm = float(probs[idx_normal]) if len(probs) > idx_normal else 0.5
    p_susp = float(probs[idx_susp])   if len(probs) > idx_susp   else 0.5
    s = p_norm + p_susp
    if s <= 0:
        return np.array([0.5, 0.5], dtype=np.float32)
    return np.array([p_norm / s, p_susp / s], dtype=np.float32)

def get_kpt_probs(kpts_xy):
  
    if kpts_xy is None or len(kpts_xy) != 17:
        return np.array([0.5, 0.5], dtype=np.float32)

    row = np.asarray(kpts_xy, dtype=np.float32).reshape(-1)
    if not np.isfinite(row).all():
        return np.array([0.5, 0.5], dtype=np.float32)

    proba = clf_kpts.predict_proba([row])[0]
    # Map to Normal/Suspicious using label encoder indices if available
    if len(proba) == 2 and max(kpts_idx_normal, kpts_idx_susp) < len(proba):
        p_norm = float(proba[kpts_idx_normal])
        p_susp = float(proba[kpts_idx_susp])
    else:
        # single-class or unexpected order
        p_susp = float(proba[-1]) if len(proba) else 0.5
        p_norm = 1.0 - p_susp

    s = p_norm + p_susp
    if s <= 0:
        return np.array([0.5, 0.5], dtype=np.float32)
    return np.array([p_norm / s, p_susp / s], dtype=np.float32)

def fuse_probs(p_img, p_kpt, w_img=W_IMG, w_kpt=W_KPT):
    fused = w_img * p_img + w_kpt * p_kpt
    s = fused.sum()
    return fused / s if s > 0 else np.array([0.5, 0.5], dtype=np.float32)

def draw_label(img, text, pt, bg=(0,0,0), fg=(255,255,255)):
    x, y = pt
    fx, fy = 0.6, 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fx, fy)
    cv2.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 6), cv2.FONT_HERSHEY_SIMPLEX, fx, fg, fy, cv2.LINE_AA)

def now_str():
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

# ===================== EMAIL (ASYNC) =====================
def send_email_async(subject, body, attachments):
    if not (SENDER_EMAIL and APP_PASSWORD and RECIPIENT_EMAIL):
        print("[WARN] Email env vars not set. Skipping email.")
        return

    def _run():
        try:
            import yagmail
            yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
            yag.send(to=RECIPIENT_EMAIL, subject=subject, contents=body, attachments=attachments)
            print("[INFO] Email sent.")
        except Exception as e:
            print(f"[ERROR] Email failed: {e}")

    threading.Thread(target=_run, daemon=True).start()

# ===================== FRAME GRABBER THREAD =====================
class FrameGrabber(threading.Thread):
    def __init__(self, src, max_q=2):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"[ERROR] Could not open source: {src}")
        self.q = queue.Queue(maxsize=max_q)
        self.stopped = False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0

    def run(self):
        while not self.stopped:
            ok, frame = self.cap.read()
            if not ok:
                self.stop()
                break
            if not self.q.full():
                self.q.put(frame)
            else:
                # Drop oldest frame to reduce latency
                try:
                    _ = self.q.get_nowait()
                except queue.Empty:
                    pass
                self.q.put(frame)

    def read(self, timeout=1.0):
        try:
            return True, self.q.get(timeout=timeout)
        except queue.Empty:
            return False, None

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except Exception:
            pass

# ===================== STATE (per-ID) =====================
ema_prob   = defaultdict(lambda: 0.0)      # EMA of suspicious prob
vote_buf   = defaultdict(lambda: deque(maxlen=VOTE_WINDOW))
last_alert = defaultdict(lambda: 0.0)
pack_imgs  = defaultdict(list)             # store paths of crops until we send

# ===================== MAIN LOOP =====================
def main():
    grabber = FrameGrabber(VIDEO_SOURCE)
    grabber.start()
    print("[INFO] Stream started.")

    frame_idx = 0
    t0 = time.time()
    fps_disp = 0.0

    while True:
        ok, frame = grabber.read(timeout=2.0)
        if not ok or frame is None:
            print("[WARN] No frame read; ending.")
            break

        frame_idx += 1
        # Optional downscale to keep real-time
        frame = safe_resize(frame, TARGET_MAX_WIDTH)

        # Optionally skip frames for slow machines
        if frame_idx % PROCESS_EVERY_N != 0:
            if SHOW_WINDOW:
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        # Pose + tracking
        # Using persist=True to keep ByteTrack IDs by default
        results = model_pose.track(
            frame,
            **POSE_KW  # imgsz, device, half, persist, verbose
        )

        out = frame.copy()
        r0 = results[0]

        ids   = r0.boxes.id
        boxes = r0.boxes.xyxy if r0.boxes is not None else None
        kpts  = r0.keypoints.xy if r0.keypoints is not None else None

        ids   = ids.detach().cpu().numpy().astype(int).tolist() if ids is not None else []
        boxes = boxes.detach().cpu().numpy().astype(int).tolist() if boxes is not None else []
        kpts  = kpts.detach().cpu().numpy().tolist() if kpts is not None else []

        for i, pid in enumerate(ids):
            x1, y1, x2, y2 = boxes[i]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(0, x2), max(0, y2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Per-stream predictions
            p_img = get_img_probs(crop)
            k_xy  = kpts[i] if i < len(kpts) else None
            p_kpt = get_kpt_probs(k_xy)

            # Fuse + smooth
            p_final = fuse_probs(p_img, p_kpt, W_IMG, W_KPT)
            p_norm, p_susp = float(p_final[0]), float(p_final[1])

            # EMA smoothing
            ema_prob[pid] = EMA_ALPHA * p_susp + (1 - EMA_ALPHA) * ema_prob[pid]
            p_susp_smooth = ema_prob[pid]

            # Voting buffer
            vote_buf[pid].append(p_susp_smooth >= ALERT_THRESHOLD)
            voted_susp = sum(vote_buf[pid]) >= VOTE_MIN_SUSP_FR

            # Decide label/color
            label = "Suspicious" if p_susp_smooth >= ALERT_THRESHOLD or voted_susp else "Normal"
            color = (0, 0, 255) if label == "Suspicious" else (0, 200, 0)

            # Draw
            cv2.rectangle(out, (x1, y1), (x2, y2), color, DRAW_THICK)
            draw_label(out, f"ID:{pid} {label}  p={p_susp_smooth:.2f}", (x1, y1 - 8),
                       bg=(20, 20, 20), fg=(255, 255, 255))

            # Evidence + alert (debounced)
            if label == "Suspicious":
                ts = now_str()
                crop_path = os.path.join(ALERT_DIR, f"crop_ID{pid}_{ts}.jpg")
                cv2.imwrite(crop_path, crop)
                pack_imgs[pid].append(crop_path)

                # When we have N evidence crops, and cooldown passed, send alert + scene
                now = time.time()
                if len(pack_imgs[pid]) >= ALERT_BURST_N and (now - last_alert[pid] >= ALERT_COOLDOWN_S):
                    scene_path = os.path.join(ALERT_DIR, f"scene_ID{pid}_{ts}.jpg")
                    cv2.imwrite(scene_path, frame)
                    subject = f"🚨 Shoplifting Alert • ID {pid} • {ts}"
                    body = (f"Suspicious activity detected for track ID {pid} at {dt.datetime.now()}.\n"
                            f"Attachments: {ALERT_BURST_N} crops + full scene.")
                    send_email_async(subject, body, pack_imgs[pid][:ALERT_BURST_N] + [scene_path])
                    last_alert[pid] = now
                    pack_imgs[pid].clear()

        # FPS
        t1 = time.time()
        dt_frame = t1 - t0
        if dt_frame > 0:
            fps_disp = 1.0 / dt_frame
        t0 = t1

        draw_label(out, f"FPS: {fps_disp:.1f}", (10, 30), bg=(0, 0, 0), fg=(255, 255, 255))

        if SHOW_WINDOW:
            cv2.imshow(WINDOW_NAME, out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    grabber.stop()
    if SHOW_WINDOW:
        cv2.destroyAllWindows()
    print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
