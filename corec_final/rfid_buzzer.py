"""
Corec — Crime Detection System
rfid_buzzer.py: MFRC522 RFID Reader + Active Buzzer

GPIO27 (Pin 13) — Active Buzzer
SPI              — MFRC522 RFID Module

Buzzer patterns:
  Entry beep   → 1 short beep (0.2 s on)
  Weapon alarm → 3 sharp pulses over 3 seconds (different modulation)
"""

import time
import threading
import requests
from datetime import datetime, timezone

import RPi.GPIO as GPIO
GPIO.setwarnings(False)

from mfrc522 import SimpleMFRC522

# ── Configuration ─────────────────────────────────────────────────────────────
NODE_ID      = "corridor-1"
BUZZER_PIN   = 27           # GPIO27  →  Physical Pin 13
DEBOUNCE_S   = 3            # ignore same tag within N seconds
SERVER_URL   = "http://127.0.0.1:5000/api/rfid_alert"

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUZZER_PIN, GPIO.OUT)
GPIO.output(BUZZER_PIN, GPIO.LOW)

# ── Buzzer patterns ───────────────────────────────────────────────────────────
def beep_entry():
    """Short single beep → RFID entry confirmed."""
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.2)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

def beep_weapon_alarm():
    """
    Weapon alert: 3 sharp rapid pulses, spread over 3 seconds.
    Clearly different from the short entry beep.
    Pattern: ON-OFF-ON-OFF-ON  with gaps, total ~3 s
    """
    pattern = [
        (0.35, 0.25),   # pulse 1: 0.35s on, 0.25s off
        (0.35, 0.25),   # pulse 2
        (0.35, 1.45),   # pulse 3: long silence at end to total ~3 s
    ]
    for on_t, off_t in pattern:
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
        time.sleep(on_t)
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        time.sleep(off_t)

def trigger_weapon_alarm():
    """Called from camera.py weapon_alert_callback in a daemon thread."""
    print("[BUZZER] 🚨 WEAPON ALARM — 3-pulse modulated alert!")
    beep_weapon_alarm()

# ── RFID reader init ──────────────────────────────────────────────────────────
try:
    reader = SimpleMFRC522()
    RFID_READY = True
    print("[RFID] MFRC522 reader ready ✅")
except Exception as e:
    reader = None
    RFID_READY = False
    print(f"[RFID] Init failed (SPI disabled?): {e}")

# ── HTTP alert to Flask server ────────────────────────────────────────────────
def _post_alert(tag_id, text):
    payload = {
        "node_id":    NODE_ID,
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "event_type": "rfid_scan",
        "rfid_tag":   str(tag_id),
        "tag_text":   text.strip() if text else None,
    }
    try:
        r = requests.post(SERVER_URL, json=payload, timeout=2)
        print(f"[RFID] ✅ Posted to server ({r.status_code})")
    except Exception as e:
        print(f"[RFID] ⚠️  Server post failed: {e}")

# ── Main RFID loop ────────────────────────────────────────────────────────────
def run_loop():
    if not RFID_READY:
        print("[RFID] Hardware not ready. Loop standing by.")
        return

    last_id   = None
    last_time = 0.0

    print(f"[RFID] Scanning… (server: {SERVER_URL})")

    try:
        while True:
            tag_id, text = reader.read()
            now = time.time()

            # Debounce: skip same tag within DEBOUNCE_S seconds
            if tag_id == last_id and (now - last_time) < DEBOUNCE_S:
                continue

            last_id, last_time = tag_id, now
            print(f"[RFID] 📌 Tag: {tag_id} | Text: '{text.strip() if text else ''}'")

            # 1. Short entry beep
            beep_entry()

            # 2. Send log to Flask API in background (non-blocking)
            threading.Thread(target=_post_alert, args=(tag_id, text), daemon=True).start()

    except KeyboardInterrupt:
        print("[RFID] Stopped.")
    finally:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.cleanup()

if __name__ == "__main__":
    run_loop()
