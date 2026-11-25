# 🚨 Corec Final — Real-Time Crime Surveillance & Threat Tracking Engine

> ⭐ **PRIMARY PRODUCTION DIRECTORY** ⭐  
> This directory contains the latest stabilized and actively tested implementation for the **Crime Abstractor** project, integrating real-time weapon detection, actor-object spatial suspect tracking, RFID-based access control, buzzer modulation, and a unified Flask web dashboard.

---

## 📌 Architecture & Modules

| Module / Script | Description |
| :--- | :--- |
| **`threat_suspect_tracker.py`** | **Core Threat & Suspect Tracking Engine**. Uses YOLOv8 + specialized threat detection weights + ByteTrack to detect weapons (`Gun`, `Knife`, `Scissors`), detect people, perform spatial association (identifies who is armed), and assigns a dynamic **🔴 RED SUSPECT** bounding box to the armed actor while maintaining **🟢 GREEN NORMAL** for bystanders. |
| **`app.py`** | Full-featured Flask Surveillance Server with live MJPEG camera streaming (`/video_feed`), REST APIs for threat and RFID event logs, and an administrative dashboard interface. |
| **`camera.py`** | RTSP stream capture handler with TCP fallback, MOG2 background motion gating, multi-class threat evaluation, and automatic alert dispatching. |
| **`rfid_buzzer.py`** | Hardware interface for MFRC522 RFID reader and GPIO buzzer alarm with multi-pulse modulation (short chime for authorized access vs. rapid alarm for weapon threats). |
| **`pi_detection.py`** | Standalone Raspberry Pi 4B edge inference script for running locally on device with HDMI/X11 output. |
| **`run_live_test.bat`** | 1-Click Windows launcher to start the live suspect tracker with your local webcam. |
| **`run_video_test.bat`** | 1-Click Windows launcher to run the tracker against sample knife/threat test clips. |
| **`threat_weights.pt`** | Specialized fine-tuned YOLOv8 weights for firearm, knife, and grenade detection. |
| **`yolov8n.pt`** | YOLOv8 Nano base model for high-speed person tracking and multi-object tracking. |

---

## 🚀 Quick Start Guide

### 1. Requirements
Ensure Python 3.8+ and dependencies are installed:
```bash
pip install ultralytics opencv-python numpy flask pymongo pytz
```

### 2. Live Surveillance Mode (Webcam)
```bash
python threat_suspect_tracker.py 0
```
*Or double-click `run_live_test.bat` on Windows.*

### 3. IP Camera Mode (RTSP)
```bash
python threat_suspect_tracker.py 2
```

### 4. Flask Web Dashboard Mode
```bash
python app.py
```
*Access dashboard in browser at `http://localhost:5000`.*

---

## 🎯 Detection Logic
* **Weapon Detection**: Detects weapons with individual bounding boxes and confidence scores.
* **Suspect Association**: If a weapon's bounding box is held within or proximal to a person's upper body / hand region, that individual is dynamically marked as **`SUSPECT: Robber (Armed: <Weapon>)`**.
* **Attribution Link**: Draws tactical attribution lines connecting the weapon to the armed individual's center.
* **Persistent Tracking**: ByteTrack preserves ID tracking even across rapid body movement or temporary occlusions.
