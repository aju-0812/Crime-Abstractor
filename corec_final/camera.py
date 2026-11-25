"""
Corec — Crime Detection System
camera.py: COFE RTSP Camera + Haar Face Detection + h5 Weapon Model

Weapon detection strategy:
  1. Background subtraction (MOG2) detects motion/new objects in frame
  2. Only run weapon_detection_model.h5 when significant motion detected
  3. Empty static room = no motion = always CLEAR (solves false positive)
  4. Weapon shown in frame = motion + high model confidence = ALERT

Model: ~/corec_final/models/weapon_detection_model.h5 (134MB)
Trained on: Sohas_weapon-Detection dataset (knife, gun images)
Input: 224x224 RGB, normalized /255, sigmoid output (0=clear, 1=weapon)
"""

import cv2
import os
import threading
import time
import datetime
import numpy as np

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# ── RTSP stream URLs ──────────────────────────────────────────────────────────
RTSP_SUB  = "rtsp://192.168.50.86:554/user=admin_password=tlJwpbo6_channel=0_stream=1&onvif=0.sdp?real_stream"
RTSP_MAIN = "rtsp://192.168.50.86:554/user=admin_password=tlJwpbo6_channel=0_stream=0&onvif=0.sdp?real_stream"

# ── Weapon model path ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "weapon_detection_model.h5")

# ── Detection settings ────────────────────────────────────────────────────────
WEAPON_THRESHOLD   = 0.88   # sigmoid confidence to count as weapon
MOTION_THRESHOLD   = 2500   # min pixel area of motion to trigger model
CONFIRM_NEEDED     = 4      # consecutive weapon frames needed before alert
WEAPON_COOLDOWN    = 10     # seconds between buzzer alerts
INFER_EVERY_N      = 4      # run model every N frames when motion detected

# ── Shared global state ───────────────────────────────────────────────────────
global_frame         = None
frame_lock           = threading.Lock()
latest_threat_status = {
    "level":      "CLEAR",
    "message":    "System Monitoring — Normal",
    "label":      "",
    "confidence": 0.0,
    "timestamp":  "",
}

# ── Weapon alert callback (set by app.py) ─────────────────────────────────────
weapon_alert_callback = None

# ── Weapon model ──────────────────────────────────────────────────────────────
weapon_model    = None
_confirm_count  = 0
_last_alert_t   = 0.0

def _load_weapon_model():
    global weapon_model
    if not os.path.exists(MODEL_PATH):
        print(f"[AI] Model NOT found: {MODEL_PATH}")
        return
    try:
        import tensorflow as tf
        weapon_model = tf.keras.models.load_model(MODEL_PATH)
        shape = weapon_model.input_shape
        print(f"[AI] weapon_detection_model.h5 loaded ✅  input={shape}")
    except Exception as e:
        print(f"[AI] Model load failed: {e}")

def _predict(frame):
    """
    Run model on frame. Returns (label, confidence).
    Preprocessing matches training code in 1.FinalModel-MITHUN.ipynb:
      - BGR → RGB
      - resize 224x224
      - /255.0 normalize
      - sigmoid output: >0.5 = weapon
    """
    if weapon_model is None:
        return "No Model", 0.0
    try:
        img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        conf = float(weapon_model.predict(img, verbose=0)[0][0])
        label = "Weapon" if conf >= WEAPON_THRESHOLD else "Clear"
        return label, conf
    except Exception as e:
        return "Err", 0.0

# ── Face cascade ──────────────────────────────────────────────────────────────
def _load_cascade():
    try:
        clf = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if not clf.empty():
            return clf
    except Exception:
        pass
    local = os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(local) if os.path.exists(local) else None

# ── Placeholder frame ─────────────────────────────────────────────────────────
def create_placeholder_frame(msg="Connecting to COFE Camera…"):
    f = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(f, "COREC SURVEILLANCE ENGINE", (130, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (59, 130, 246), 2)
    cv2.putText(f, msg, (60, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(f, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (200, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 130, 150), 1)
    return f

# ── Camera connection ─────────────────────────────────────────────────────────
def _connect_camera():
    for url, label in [(RTSP_SUB, "Sub-Stream"), (RTSP_MAIN, "Main Stream")]:
        print(f"[CAM] Trying {label}…")
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[CAM] Connected — {label} ✅")
                return cap, label
            cap.release()
    print("[CAM] RTSP failed. Trying webcam…")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap, "Webcam"
    return None, "No Source"

# ── Main surveillance loop ────────────────────────────────────────────────────
def _surveillance_loop():
    global global_frame, latest_threat_status, _confirm_count, _last_alert_t

    with frame_lock:
        global_frame = create_placeholder_frame("Starting up…")

    cascade = _load_cascade()
    cap, source = _connect_camera()

    if cap is None:
        with frame_lock:
            global_frame = create_placeholder_frame("Camera Offline")
        return

    # ── Background subtractor (MOG2) for motion gating ──────────────────────
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=120, varThreshold=50, detectShadows=False)

    failures  = 0
    frame_idx = 0
    weapon_label = "Clear"
    weapon_conf  = 0.0
    motion_area  = 0

    print(f"[CORE] Running on [{source}] | Motion gate: {MOTION_THRESHOLD}px | Threshold: {WEAPON_THRESHOLD}")

    while True:
        for _ in range(2):
            cap.grab()

        ret, raw = cap.retrieve()

        if not ret or raw is None:
            failures += 1
            if failures > 25:
                print("[CAM] Stream lost. Reconnecting…")
                cap.release()
                time.sleep(1.5)
                cap, source = _connect_camera()
                if cap is None:
                    with frame_lock:
                        global_frame = create_placeholder_frame("Reconnecting…")
                    time.sleep(2)
                    continue
                failures = 0
            time.sleep(0.05)
            continue

        failures = 0
        frame_idx += 1
        frame = cv2.resize(raw, (640, 360))
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Motion detection via background subtraction ───────────────────────
        fg_mask    = bg_sub.apply(frame)
        # Remove noise with morphological ops
        kernel     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask    = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        motion_area = int(np.sum(fg_mask > 0))
        motion_detected = motion_area > MOTION_THRESHOLD

        # ── Face / person detection ───────────────────────────────────────────
        faces = []
        if cascade is not None:
            try:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=4, minSize=(30, 30))
            except Exception:
                faces = []
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "PERSON", (x, y-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ── Weapon model: ONLY runs when motion detected ──────────────────────
        if motion_detected and frame_idx % INFER_EVERY_N == 0:
            weapon_label, weapon_conf = _predict(frame)
            print(f"[AI] Motion={motion_area}px | {weapon_label} {weapon_conf:.2%}")

            if weapon_label == "Weapon":
                _confirm_count += 1
            else:
                _confirm_count = 0
        elif not motion_detected:
            # Static/empty room → force CLEAR, reset counter
            _confirm_count = 0
            weapon_label   = "Clear"
            weapon_conf    = 0.0

        # ── Threat level decision ─────────────────────────────────────────────
        is_weapon_confirmed = (_confirm_count >= CONFIRM_NEEDED)
        now = time.time()

        if is_weapon_confirmed:
            level = "ALERT"
            msg   = f"WEAPON DETECTED — {weapon_conf:.0%} confidence"
            cv2.putText(frame, "!! WEAPON DETECTED !!", (120, 345),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if weapon_alert_callback and (now - _last_alert_t) > WEAPON_COOLDOWN:
                _last_alert_t = now
                threading.Thread(target=weapon_alert_callback, daemon=True).start()

        elif motion_detected and len(faces) > 0:
            level = "WARNING"
            msg   = f"Person Detected ({len(faces)})"
            cv2.putText(frame, f"PERSON IN CORRIDOR ({len(faces)})", (10, 345),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        else:
            level = "CLEAR"
            msg   = "Corridor Clear — No Threat"
            cv2.putText(frame, "Status: CLEAR", (10, 345),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

        # ── HUD overlays ──────────────────────────────────────────────────────
        cv2.putText(frame, f"COFE CAM | {ts}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        motion_txt = f"Motion: {motion_area}px" if motion_detected else "Motion: STILL"
        cv2.putText(frame, motion_txt, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
        ai_txt = f"AI: {weapon_label} {weapon_conf:.0%}" if motion_detected else "AI: IDLE (still)"
        cv2.putText(frame, ai_txt, (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

        with frame_lock:
            global_frame = frame.copy()
            latest_threat_status.update({
                "level":      level,
                "message":    msg,
                "label":      weapon_label,
                "confidence": round(weapon_conf, 3),
                "timestamp":  ts,
            })

        time.sleep(0.025)

    cap.release()

# ── Public start function ─────────────────────────────────────────────────────
def start_surveillance():
    _load_weapon_model()
    t = threading.Thread(target=_surveillance_loop, daemon=True)
    t.start()
    return t
