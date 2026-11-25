"""
Corec — Crime Detection System
app.py: Flask Web Server + API Gateway

Starts:
  • Camera + AI surveillance thread (camera.py)
  • RFID scanner + buzzer thread    (rfid_buzzer.py)
  • Flask web dashboard on port 5000
"""

import datetime
import threading
import time
import cv2

from flask import Flask, render_template, jsonify, request, Response

# ── Import camera engine ──────────────────────────────────────────────────────
from camera import (
    start_surveillance,
    global_frame,
    frame_lock,
    latest_threat_status,
    create_placeholder_frame,
    weapon_alert_callback,
)
import camera   # need mutable reference for callback assignment

# ── Import RFID / buzzer ──────────────────────────────────────────────────────
from rfid_buzzer import run_loop as rfid_loop, trigger_weapon_alarm

# Set weapon alert callback so camera.py can fire buzzer
camera.weapon_alert_callback = trigger_weapon_alarm

# ── MongoDB (optional — graceful fallback to memory) ─────────────────────────
MONGODB_AVAILABLE = False
db = None
try:
    import pymongo
    _client = pymongo.MongoClient("mongodb://localhost:27017/",
                                  serverSelectionTimeoutMS=1000)
    _client.server_info()
    db = _client["corec"]
    MONGODB_AVAILABLE = True
    print("[DB] MongoDB connected ✅")
except Exception as e:
    print(f"[DB] MongoDB offline — using in-memory store. ({e})")

# ── In-memory stores ──────────────────────────────────────────────────────────
_rfid_logs    = []   # list of dicts
_weapon_logs  = []   # list of dicts (weapon detections)

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ══════════════════════════════ ROUTES ════════════════════════════════════════

@app.route("/")
def index():
    return render_template("dashboard.html")

# ── MJPEG live video feed ─────────────────────────────────────────────────────
def _gen_frames():
    while True:
        with frame_lock:
            if camera.global_frame is None:
                f = create_placeholder_frame()
            else:
                f = camera.global_frame.copy()
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if ok:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.033)   # ~30 FPS

@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ── RFID scan received from rfid_buzzer.py ───────────────────────────────────
@app.route("/api/rfid_alert", methods=["POST"])
def rfid_alert():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"status": "error", "message": "No JSON"}), 400

    data["received_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Memory store
    _rfid_logs.insert(0, {k: v for k, v in data.items() if k != "_id"})
    if len(_rfid_logs) > 100:
        _rfid_logs.pop()

    # MongoDB (if available)
    if MONGODB_AVAILABLE and db is not None:
        try:
            db.rfid_logs.insert_one(dict(data))
        except Exception:
            pass

    print(f"[API] 💳 RFID — Tag: {data.get('rfid_tag')} | {data.get('node_id')}")
    return jsonify({"status": "success"}), 200

# ── Fetch RFID logs for dashboard ─────────────────────────────────────────────
@app.route("/api/rfid_logs", methods=["GET"])
def get_rfid_logs():
    try:
        if MONGODB_AVAILABLE and db is not None:
            logs = list(db.rfid_logs.find({}, {"_id": 0}).sort("_id", -1).limit(20))
        else:
            logs = _rfid_logs[:20]
        return jsonify({"status": "success", "count": len(logs), "logs": logs}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ── Threat status from camera AI ─────────────────────────────────────────────
@app.route("/api/threat_status", methods=["GET"])
def threat_status():
    # Keep a weapon detection log
    ts = camera.latest_threat_status
    if ts.get("level") == "ALERT":
        entry = dict(ts)
        if not _weapon_logs or _weapon_logs[0].get("timestamp") != entry.get("timestamp"):
            _weapon_logs.insert(0, entry)
            if len(_weapon_logs) > 50:
                _weapon_logs.pop()
            # Persist to MongoDB
            if MONGODB_AVAILABLE and db is not None:
                try:
                    db.weapon_alerts.insert_one(dict(entry))
                except Exception:
                    pass
    return jsonify({"status": "success", "threat": ts}), 200

# ── Weapon alert history ──────────────────────────────────────────────────────
@app.route("/api/weapon_logs", methods=["GET"])
def weapon_logs():
    try:
        if MONGODB_AVAILABLE and db is not None:
            logs = list(db.weapon_alerts.find({}, {"_id": 0}).sort("_id", -1).limit(20))
        else:
            logs = _weapon_logs[:20]
        return jsonify({"status": "success", "count": len(logs), "logs": logs}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ══════════════════════════════ STARTUP ═══════════════════════════════════════
if __name__ == "__main__":
    # 1. Camera + AI
    print("[MAIN] 🎥 Starting camera & AI surveillance…")
    start_surveillance()

    # 2. RFID + Buzzer hardware thread
    print("[MAIN] 💳 Starting RFID reader thread…")
    threading.Thread(target=rfid_loop, daemon=True).start()

    # 3. Flask
    print("[MAIN] 🌐 Dashboard → http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
