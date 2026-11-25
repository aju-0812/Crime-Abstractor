"""
Corec — Local Weapon Detection Test
Runs on Windows Laptop.

- Pulls MJPEG video from Pi's Flask stream (http://172.16.142.171:5000/video_feed)
- Loads weapon_detection_model.h5 locally
- Shows live OpenCV window with bounding box + detection overlay
- Press 'Q' to quit
"""

import cv2
import numpy as np
import urllib.request
import datetime
import time
import os

# ── Config ────────────────────────────────────────────────────────────────────
PI_STREAM_URL  = "http://172.16.142.171:5000/video_feed"   # Pi MJPEG stream
MODEL_PATH     = r"d:\sathish sir\1.Crime Abstractor\Project Expo-Main\weapon_detection_model.h5"
THRESHOLD      = 0.88    # sigmoid confidence to call "Weapon"
MOTION_MIN_PX  = 2000    # ignore frames with less motion than this
INFER_EVERY_N  = 5       # run model every N frames
CONFIRM_NEEDED = 3       # consecutive "Weapon" frames to confirm alert

# ── Load model ────────────────────────────────────────────────────────────────
print("[AI] Loading weapon_detection_model.h5 ...")
import tensorflow as tf
model = tf.keras.models.load_model(MODEL_PATH)
print(f"[AI] Model loaded ✅  input={model.input_shape}")

# ── Face cascade ──────────────────────────────────────────────────────────────
cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ── MJPEG stream reader ───────────────────────────────────────────────────────
def open_stream(url):
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        print(f"[CAM] Connected to Pi stream ✅  {url}")
        return cap
    print(f"[CAM] Could not connect to {url}")
    print("      Make sure 'python3 app.py' is running on Pi!")
    return None

# ── Weapon prediction ─────────────────────────────────────────────────────────
def predict(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    conf = float(model.predict(img, verbose=0)[0][0])
    return ("Weapon" if conf >= THRESHOLD else "Clear"), conf

# ── Draw bounding box on full frame when weapon detected ─────────────────────
def draw_weapon_box(frame, conf):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (4, 4), (w-4, h-4), (0, 0, 255), 3)
    label = f"WEAPON DETECTED  {conf:.0%}"
    cv2.rectangle(frame, (0, 0), (w, 38), (0, 0, 200), -1)
    cv2.putText(frame, label, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

# ── Main loop ─────────────────────────────────────────────────────────────────
cap = open_stream(PI_STREAM_URL)
if cap is None:
    input("Press Enter to exit.")
    exit()

bg_sub        = cv2.createBackgroundSubtractorMOG2(
                    history=120, varThreshold=50, detectShadows=False)
kernel        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

frame_idx     = 0
confirm_count = 0
weapon_label  = "Clear"
weapon_conf   = 0.0
motion_area   = 0

print("\n[INFO] OpenCV window opening... Press Q to quit.\n")
cv2.namedWindow("Corec — Weapon Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Corec — Weapon Detection", 900, 540)

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print("[CAM] Frame read failed. Retrying...")
        time.sleep(0.5)
        cap = open_stream(PI_STREAM_URL)
        continue

    frame_idx += 1
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Motion detection ──────────────────────────────────────────────────────
    fg      = bg_sub.apply(frame)
    fg      = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
    motion_area = int(np.sum(fg > 0))
    has_motion  = motion_area > MOTION_MIN_PX

    # ── Face detection ────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.2, 4, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, "PERSON", (x, y-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

    # ── Weapon model (only when motion) ──────────────────────────────────────
    if has_motion and frame_idx % INFER_EVERY_N == 0:
        weapon_label, weapon_conf = predict(frame)
        print(f"[AI] Motion={motion_area:5d}px | {weapon_label:6s} {weapon_conf:.2%}")
        confirm_count = confirm_count + 1 if weapon_label == "Weapon" else 0
    elif not has_motion:
        confirm_count = 0
        weapon_label  = "Clear"
        weapon_conf   = 0.0

    confirmed = (confirm_count >= CONFIRM_NEEDED)

    # ── Draw overlays ─────────────────────────────────────────────────────────
    if confirmed:
        draw_weapon_box(frame, weapon_conf)
    else:
        # Normal status bar at bottom
        if has_motion and len(faces) > 0:
            bar_color = (0, 140, 255)
            status    = f"WARNING — Person in corridor ({len(faces)})"
        else:
            bar_color = (0, 160, 0)
            status    = "CLEAR — No Threat Detected"

        h_f = frame.shape[0]
        cv2.rectangle(frame, (0, h_f-34), (frame.shape[1], h_f), bar_color, -1)
        cv2.putText(frame, status, (10, h_f-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── HUD (top-right) ───────────────────────────────────────────────────────
    cv2.putText(frame, f"COREC | {ts}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    motion_txt = f"Motion: {motion_area}px" if has_motion else "Motion: STILL"
    cv2.putText(frame, motion_txt, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    ai_txt = f"AI: {weapon_label} {weapon_conf:.0%}" if has_motion else "AI: IDLE"
    cv2.putText(frame, ai_txt, (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.imshow("Corec — Weapon Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] Quit by user.")
        break

cap.release()
cv2.destroyAllWindows()
