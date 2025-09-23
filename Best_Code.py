import os, cv2, numpy as np, joblib, datetime, yagmail, threading, time, torch
from ultralytics import YOLO
from collections import defaultdict

# ===================== CONFIG =====================
VIDEO = r"Videos/vid2.mp4"   # or 0 for webcam
YOLO_CLS_WEIGHTS = r"runs/classify/train/weights/best.pt"  # update path
ALERT_THRESHOLD = 0.75
ALERT_COOLDOWN = 30  # seconds between alerts for same ID

# Email settings
SENDER_EMAIL = ""
APP_PASSWORD = ""
RECIPIENT_EMAIL = ""

# ===================== DEVICE =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {'GPU' if DEVICE=='cuda' else 'CPU'}")

# ===================== LOAD MODELS =====================
model_pose = YOLO("yolo11n-pose.pt")
model_pose.to(DEVICE)

model_cls = YOLO(YOLO_CLS_WEIGHTS)
model_cls.to(DEVICE)

clf_kpts = joblib.load("keypoints_rf.pkl")
le = joblib.load("keypoints_label_encoder.pkl")
IMG_CLASS_NAMES = model_cls.names

# Try to find Normal and Suspicious indices in keypoints classifier
try:
    normal_idx = list(le.classes_).index("Normal")
    suspicious_idx = list(le.classes_).index("Suspicious")
except ValueError:
    print(f"⚠ Warning: Could not auto-detect 'Normal' and 'Suspicious' in label encoder.\n"
          f"Available classes: {list(le.classes_)}")
    normal_idx, suspicious_idx = None, None

# ===================== TRACKERS =====================
suspicious_images = defaultdict(list)
last_alert_time = defaultdict(float)

# ===================== FUNCTIONS =====================
def get_img_probs(crop_bgr):
    r = model_cls.predict(crop_bgr, imgsz=224, device=DEVICE, verbose=False)
    probs = r[0].probs.data.cpu().numpy()
    p_norm = probs[list(IMG_CLASS_NAMES.values()).index("Normal")]
    p_susp = probs[list(IMG_CLASS_NAMES.values()).index("Suspicious")]
    return np.array([p_norm, p_susp])

def get_kpt_probs(kpts_xy):
    if kpts_xy is None or len(kpts_xy) != 17:
        return np.array([0.5, 0.5])
    row = np.array(kpts_xy).reshape(-1)
    if not np.isfinite(row).all():
        return np.array([0.5, 0.5])

    probs = clf_kpts.predict_proba([row])[0]

    if len(probs) == 1:  # Single-class case
        p_susp = probs[0]
        p_norm = 1 - p_susp
    else:  # Normal two-class case
        p_norm = probs[normal_idx] if normal_idx is not None else 0.5
        p_susp = probs[suspicious_idx] if suspicious_idx is not None else 0.5

    return np.array([p_norm, p_susp])

def fuse(p_img, p_kpt, w_img=0.6, w_kpt=0.4):
    fused = w_img * p_img + w_kpt * p_kpt
    return fused / fused.sum()

def send_alert_async(person_id, image_paths, full_frame):
    """Send email with suspicious images + full scene snapshot."""
    def _send():
        try:
            os.makedirs("alerts", exist_ok=True)
            full_frame_path = f"alerts/full_scene_ID{person_id}_{int(time.time())}.jpg"
            cv2.imwrite(full_frame_path, full_frame)

            all_attachments = image_paths + [full_frame_path]
            subject = f"🚨 Shoplifting Alert: Person ID {person_id}"
            body = (
                f"Suspicious activity detected for Person ID {person_id} "
                f"at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.\n"
                f"Attached: cropped images + full scene snapshot."
            )

            yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
            yag.send(
                to=RECIPIENT_EMAIL,
                subject=subject,
                contents=body,
                attachments=all_attachments
            )
            print(f"📧 Alert email sent for ID {person_id} with full frame")
        except Exception as e:
            print(f"❌ Email failed: {e}")
    threading.Thread(target=_send, daemon=True).start()

# ===================== VIDEO LOOP =====================
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit("Could not open video")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    results = model_pose.track(frame, persist=True, verbose=False, device=DEVICE)
    r0 = results[0]
    ids = r0.boxes.id.cpu().numpy().astype(int).tolist() if r0.boxes.id is not None else []
    boxes = r0.boxes.xyxy.cpu().numpy().astype(int).tolist() if r0.boxes is not None else []
    kpts = r0.keypoints.xy.cpu().numpy().tolist() if r0.keypoints is not None else []

    out = frame.copy()

    for i, pid in enumerate(ids):
        x1, y1, x2, y2 = boxes[i]
        crop = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if crop.size == 0:
            continue

        p_img = get_img_probs(crop)
        k = kpts[i] if i < len(kpts) else None
        p_kpt = get_kpt_probs(k)
        p_final = fuse(p_img, p_kpt)

        p_norm, p_susp = float(p_final[0]), float(p_final[1])
        label = "Suspicious" if p_susp >= p_norm else "Normal"
        color = (0, 0, 255) if label == "Suspicious" else (0, 255, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, f"ID:{pid} {label} {p_susp:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if p_susp >= ALERT_THRESHOLD:
            os.makedirs("alerts", exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"alerts/alert_ID{pid}_{ts}.jpg"
            cv2.imwrite(img_path, crop)
            suspicious_images[pid].append(img_path)

            if len(suspicious_images[pid]) >= 3:
                now = time.time()
                if now - last_alert_time[pid] > ALERT_COOLDOWN:
                    send_alert_async(pid, suspicious_images[pid][:3], frame.copy())
                    last_alert_time[pid] = now
                suspicious_images[pid].clear()

    cv2.imshow("Fusion CCTV", out)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
