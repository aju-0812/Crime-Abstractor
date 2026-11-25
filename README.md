# Corec - Abstracting Crimes in Corridors

[![Project Status: Active](https://img.shields.io/badge/Project%20Status-Active-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()
[![AI/ML: YOLO-TensorFlow](https://img.shields.io/badge/AI/ML-YOLO--TensorFlow-red.svg)]()

> "Empowering urban safety through real-time intelligence, computer vision, and the next generation of wireless technologies."

> ⭐ **ACTIVE & PRIMARY PRODUCTION MODULE: [`/corec_final`](./corec_final)** ⭐  
> **Note for Evaluators & Developers**: All recent updates, including real-time multi-weapon detection (`threat_suspect_tracker.py`), spatial suspect-weapon association, Raspberry Pi edge deployment, RFID buzzer modulation, and live dashboard are located inside the [`corec_final/`](./corec_final) directory.

---

## 📖 Proper Explanation

**Corec - Abstracting Crimes in Corridors** is a holistic security framework designed to transform public streets, private corridors, and industrial zones into proactive safety environments. By moving beyond traditional passive recording, the system utilizes **Deep Learning** to detect threats (like weapons or suspicious posture) and **Facial Biometrics** to identify and track individuals in real-time.

This project bridges the gap between physical infrastructure and digital intelligence, incorporating cutting-edge concepts like **Digital Twins**, **Li-Fi communication**, and **SAP API integration** for industrial-grade reporting and resource management. Instead of waiting for a crime to be discovered, Corec actively abstracts and prevents incidents as they unfold.

---

## 🔄 Workflow Architecture

The system operates on an event-driven pipeline where every frame captured is processed through multiple intelligence layers before being stored or triggering an alert.

1. **Live Feed Acquisition**: A surveillance camera or webcam captures a live stream.
2. **Frame Pre-processing**: Images are cleaned and optimized for analysis.
3. **Detection Engine**:
   - **Face Detection**: Uses Haar Cascades, Dlib, and DeepFace to identify faces.
   - **Weapon Detection**: Utilizes custom-trained YOLO (You Only Look Once) models to spot knives, guns, and other threats.
4. **Threat Assessment & Logic**: Combines identification logic with threat assessment to determine the severity.
5. **Event Handling**: 
   - On positive detection, it triggers video recording via an FFmpeg buffer.
   - For an immediate threat, it triggers localized SMS alerts or notifications.
6. **Data Storage & Review**: Saved footage is encoded and uploaded as binary chunks to MongoDB using GridFS. Security personnel review this evidence via the Flask Dashboard UI.

---

## 🚀 Key Features

### 🕵️ Real-Time Intelligence
- **Intelligent Face Tracking**: Seamlessly detects and identifies individuals across frames using custom Haar Cascade configurations and Dlib integration.
- **Weapon & Threat Detection**: Custom-trained models specifically tuned for dangerous objects.
- **Posture Analysis**: Detecting suspicious behaviors and unusual movements in corridors.

### 📼 Smart Recording & Evidence
- **Trigger-Based Recording**: The system only records when a face or weapon is detected, saving massive amounts of storage space.
- **MongoDB GridFS Integration**: Videos are stored as binary chunks in a database, ensuring scalability and security.
- **Automated Video Conversion**: Integrated FFmpeg pipeline to convert raw streams into high-compression MP4 formats.

### 🔔 Integrated Alert System
- **SMS Notifications**: Automated alerts via integrated SMS gateways when a weapon is detected.
- **Phone Tracking (Experimental)**: Ability to locate and track nearby devices for localized alerts.
- **Proximity Alerts**: Li-Fi and WiMAX integration for high-speed, localized data broadcasting in corridors.

---

## 📂 Project Modules Breakdown

The project contains several folders with distinct code logic:

1. **`Project Expo-Main`**: The R&D laboratory containing Jupyter notebooks for training models, datasets, and Digital Twin mapping logic.
2. **`final Datacf` / `dataCFnew`**: The deployable version of the system featuring the Flask web interface (`facedjango.py`), the detection backend (`face.py`), and background MP4 conversation logics.
3. **`Face recognition-basics`**: Isolated feature tests including age detection (`Agedect.py`), SMS logic, and Dlib face trackers.
4. **`dataCF-winning in innova`**: A snapshot version specifically tuned and optimized for the Innova competition.
5. **`Grand-challenge-top finals` & `Docs and required`**: The hardware specifications, circuit diagrams, and project presentation modules.

---

## ⚙️ Installation & Setup

### Prerequisites
- **Python 3.8 - 3.10**
- **MongoDB**: Installed and running on `localhost:27017`.
- **FFmpeg**: Must be installed on your system and added to your environmental PATH.

### 1. Clone & Prepare Environment
```bash
git clone https://github.com/Mithun017/Corec---Abstracting-Crimes-in-Corridors.git
cd Corec---Abstracting-Crimes-in-Corridors
```

### 2. Install Dependencies
```bash
pip install opencv-python dlib flask pymongo pytz tensorflow
```

### 3. Configure Paths
Update the `ffmpeg_path` in `final Datacf/dataCFnew/face.py` (Line 38) to point to your local installation.
```python
ffmpeg_path = 'C:/Path/To/Your/ffmpeg.exe'
```

### 4. Database Setup
Ensure MongoDB is running locally. The system will automatically construct the required database upon the first execution.

---

## 🎮 Usage Guide

### Launching the Full System
1. **Start the Detection Engine**:
   ```bash
   cd "final Datacf/dataCFnew"
   python face.py
   ```
   *This initiates camera monitoring for faces and weapons.*

2. **Start the Dashboard**:
   Open a new terminal and run:
   ```bash
   cd "final Datacf/dataCFnew"
   python facedjango.py
   ```
   Access the security dashboard at `http://127.0.0.1:5000`.

---

## 🏆 Conclusion

**Corec - Abstracting Crimes in Corridors** serves as a vital proof-of-concept that standard surveillance cameras can be empowered with sophisticated intelligence. By uniting independent modules—from facial detection and age estimation in "Face recognition-basics", to neural network model training in "Project Expo-Main", and culminating in the real-time deployable suite in "final Datacf"—the project stands at the forefront of urban and corridor safety engineering.

---

**Developed & Maintained by Mithun**
