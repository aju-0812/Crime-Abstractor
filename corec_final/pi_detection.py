"""
Corec — Pi Standalone Weapon Detection with Bounding Box
Runs on Raspberry Pi 4B.

Approach:
  1. Motion gate (MOG2) — only run model when something new enters frame
  2. Sliding window grid — find WHERE in frame the weapon is
  3. Draw precise bounding box with label (Weapon-Knife / Weapon-Gun / Weapon)
  4. Buzzer fires on confirmed detection
"""

import cv2
import numpy as np
import datetime
import time
import os
import threading

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── Config ────────────────────────────────────────────────────────────────────
RTSP_SUB  = "rtsp://192.168.50.86:554/user=admin_password=tlJwpbo6_channel=0_stream=1&onvif=0.sdp?real_stream"
RTSP_MAIN = "rtsp://192.168.50.86:554/user=admin_password=tlJwpbo6_channel=0_stream=0&onvif=0.sdp?real_stream"
MODEL_PATH     = os.path.expanduser("~/corec_final/models/weapon_detection_model.h5")

WEAPON_THRESHOLD = 0.85   # confidence to call a crop "weapon"
MOTION_MIN_PX    = 2000   # motion pixels to activate model
INFER_EVERY_N    = 6      # run detection every N frames
CONFIRM_NEEDED   = 3      # consecutive detections to confirm alert
WEAPON_COOLDOWN  = 10     # seconds between buzzer alerts
GRID_ROWS        = 3      # sliding window rows
GRID_COLS        = 4      # sliding window cols

# ── Load model ────────────────────────────────────────────────────────────────
print("[AI] Loading weapon_detection_model.h5 ...")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[AI] Model ready ✅  input={model.input_shape}")

# ── Predict on a single crop ─────────────────────────────────────────────────
def predict_crop(crop):
    """Returns confidence (0.0 – 1.0) for a single image crop."""
    try:
        img = cv2.resize(crop, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        return float(model.predict(img, verbose=0)[0][0])
    except Exception:
        return 0.0

# ── Sliding window localization ───────────────────────────────────────────────
def find_weapon_box(frame):
    """
    Scan frame in GRID_ROWS x GRID_COLS grid.
    Return (best_box, best_conf, weapon_label) for highest-confidence cell.
    """
    H, W = frame.shape[:2]
    best_conf  = 0.0
    best_box   = None

    # Also scan 2 overlapping offsets for better coverage
    offsets = [(0, 0), (W//(GRID_COLS*2), H//(GRID_ROWS*2))]

    for ox, oy in offsets:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1 = min(int(c * W / GRID_COLS) + ox, W-2)
                y1 = min(int(r * H / GRID_ROWS) + oy, H-2)
                x2 = min(int((c+1) * W / GRID_COLS) + ox, W)
                y2 = min(int((r+1) * H / GRID_ROWS) + oy, H)

                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue

                crop = frame[y1:y2, x1:x2]
                conf = predict_crop(crop)

                if conf > best_conf:
                    best_conf = conf
                    best_box  = (x1, y1, x2, y2)

    return best_box, best_conf

def get_weapon_label(box):
    """
    Heuristic label based on bounding box aspect ratio:
      Tall & narrow  → Weapon-Knife  (blade shape)
      Wide & flat    → Weapon-Gun    (pistol shape)
      Other          → Weapon
    """
    if box is None:
        return "Weapon"
    x1, y1, x2, y2 = box
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = h / w
    if ratio > 1.4:
        return "Weapon-Knife"
    elif ratio < 0.75:
        return "Weapon-Gun"
    else:
        return "Weapon"

# ── Draw weapon bounding box + label ─────────────────────────────────────────
def draw_weapon_detection(frame, box, label, conf):
    H, W = frame.shape[:2]

    # Red border around entire frame
    cv2.rectangle(frame, (3, 3), (W-3, H-3), (0, 0, 255), 3)

    if box is not None:
        x1, y1, x2, y2 = box
        # Main bounding box — thick red
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Corner accents (tactical style)
        corner = 14
        for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
            dx = corner if cx == x1 else -corner
            dy = corner if cy == y1 else -corner
            cv2.line(frame, (cx, cy), (cx + dx, cy), (0, 0, 255), 4)
            cv2.line(frame, (cx, cy), (cx, cy + dy), (0, 0, 255), 4)

        # Label background + text
        label_text = f"{label}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx, ly = x1, max(y1 - 10, th + 6)
        cv2.rectangle(frame, (lx, ly - th - 6), (lx + tw + 8, ly + 2), (0, 0, 200), -1)
        cv2.putText(frame, label_text, (lx + 4, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Top alert banner
    cv2.rectangle(frame, (0, 0), (W, 44), (0, 0, 180), -1)
    cv2.putText(frame, f"!! WEAPON DETECTED !!  {conf:.0%}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

# ── Buzzer ────────────────────────────────────────────────────────────────────
_last_buzz = 0.0
def buzz_weapon():
    global _last_buzz
    if time.time() - _last_buzz < WEAPON_COOLDOWN:
        return
    _last_buzz = time.time()
    try:
        import RPi.GPIO as GPIO
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(27, GPIO.OUT)
        for on_t, off_t in [(0.35, 0.25), (0.35, 0.25), (0.35, 1.0)]:
            GPIO.output(27, GPIO.HIGH); time.sleep(on_t)
            GPIO.output(27, GPIO.LOW);  time.sleep(off_t)
    except Exception as e:
        print(f"[BUZZ] {e}")

# ── Camera connection ─────────────────────────────────────────────────────────
def connect_camera():
    for url, name in [(RTSP_SUB, "Sub-Stream"), (RTSP_MAIN, "Main Stream")]:
        print(f"[CAM] Trying {name}...")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, f = cap.read()
            if ret and f is not None:
                print(f"[CAM] Connected ✅  {name}")
                return cap
            cap.release()
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[CAM] Webcam connected.")
        return cap
    return None

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = connect_camera()
if cap is None:
    print("[ERROR] No camera. Exiting.")
    exit()

bg_sub  = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=50, detectShadows=False)
kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

frame_idx     = 0
confirm_count = 0
best_box      = None
weapon_conf   = 0.0
weapon_label  = "Clear"

print("\n[INFO] Window opening... Press Q to quit.\n")
cv2.namedWindow("Corec — Weapon Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Corec — Weapon Detection", 860, 500)

while True:
    for _ in range(2):
        cap.grab()
    ret, frame = cap.retrieve()

    if not ret or frame is None:
        print("[CAM] Frame lost. Reconnecting...")
        cap.release(); time.sleep(1)
        cap = connect_camera()
        if cap is None: break
        continue

    frame_idx += 1
    frame = cv2.resize(frame, (640, 360))
    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Motion detection ──────────────────────────────────────────────────────
    fg          = bg_sub.apply(frame)
    fg          = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    motion_area = int(np.sum(fg > 0))
    has_motion  = motion_area > MOTION_MIN_PX

    # ── Weapon detection: sliding window (only when motion) ───────────────────
    if has_motion and frame_idx % INFER_EVERY_N == 0:
        box, conf = find_weapon_box(frame)
        if conf >= WEAPON_THRESHOLD:
            best_box      = box
            weapon_conf   = conf
            weapon_label  = get_weapon_label(box)
            confirm_count += 1
            print(f"[AI] {weapon_label}  {conf:.2%}  box={box}")
        else:
            confirm_count = max(0, confirm_count - 1)
            if confirm_count == 0:
                weapon_conf  = 0.0
                weapon_label = "Clear"
                best_box     = None
    elif not has_motion:
        confirm_count = 0
        weapon_label  = "Clear"
        weapon_conf   = 0.0
        best_box      = None

    confirmed = (confirm_count >= CONFIRM_NEEDED)

    # ── Draw result ───────────────────────────────────────────────────────────
    H, W = frame.shape[:2]

    if confirmed:
        draw_weapon_detection(frame, best_box, weapon_label, weapon_conf)
        threading.Thread(target=buzz_weapon, daemon=True).start()
    else:
        # Status bar at bottom
        if has_motion:
            color  = (0, 140, 255)
            status = f"MONITORING — Motion Detected ({motion_area}px)"
        else:
            color  = (0, 140, 0)
            status = "CLEAR — No Threat Detected"

        cv2.rectangle(frame, (0, H-36), (W, H), color, -1)
        cv2.putText(frame, status, (10, H-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    # ── HUD ───────────────────────────────────────────────────────────────────
    if not confirmed:
        cv2.putText(frame, f"COREC | {ts}", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame,
                    f"AI: {weapon_label} {weapon_conf:.0%}" if has_motion else "AI: IDLE (still)",
                    (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    cv2.imshow("Corec — Weapon Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Quit.")
        break

cap.release()
cv2.destroyAllWindows()
