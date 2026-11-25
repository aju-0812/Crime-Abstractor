import cv2
import numpy as np
import time
import os
import sys

DEFAULT_WEAPON_CLASSES = {
    43: "Knife",
    76: "Scissors",
}

CUSTOM_WEAPON_NAMES = ["gun", "pistol", "rifle", "knife", "scissors", "weapon", "blade"]

CONFIDENCE_THRESHOLD = 0.35
WEAPON_CONFIDENCE    = 0.28
PROXIMITY_MARGIN     = 55

COFE_RTSP_URL = "rtsp://192.168.50.86:554/user=admin_password=tlJwpbo6_channel=0_stream=1&onvif=0.sdp?real_stream"

def box_intersection_or_proximity(p_box, w_box, margin=PROXIMITY_MARGIN):
    px1, py1, px2, py2 = p_box
    wx1, wy1, wx2, wy2 = w_box
    exp_px1 = px1 - margin
    exp_py1 = py1 - margin
    exp_px2 = px2 + margin
    exp_py2 = py2 + margin
    ix1 = max(exp_px1, wx1)
    iy1 = max(exp_py1, wy1)
    ix2 = min(exp_px2, wx2)
    iy2 = min(exp_py2, wy2)
    if ix1 < ix2 and iy1 < iy2:
        overlap_area = (ix2 - ix1) * (iy2 - iy1)
        weapon_area = (wx2 - wx1) * (wy2 - wy1)
        return (overlap_area / max(weapon_area, 1)) > 0.25
    return False

def draw_styled_box(img, box, color, label, track_id=None, is_suspect=False):
    x1, y1, x2, y2 = map(int, box)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    thickness = 3 if is_suspect else 2
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    c_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
    accent_thick = thickness + 2
    cv2.line(img, (x1, y1), (x1 + c_len, y1), color, accent_thick)
    cv2.line(img, (x1, y1), (x1, y1 + c_len), color, accent_thick)
    cv2.line(img, (x2, y1), (x2 - c_len, y1), color, accent_thick)
    cv2.line(img, (x2, y1), (x2, y1 + c_len), color, accent_thick)
    cv2.line(img, (x1, y2), (x1 + c_len, y2), color, accent_thick)
    cv2.line(img, (x1, y2), (x1, y2 - c_len), color, accent_thick)
    cv2.line(img, (x2, y2), (x2 - c_len, y2), color, accent_thick)
    cv2.line(img, (x2, y2), (x2, y2 - c_len), color, accent_thick)
    full_text = f"{label}"
    if track_id is not None:
        full_text += f" [ID:{track_id}]"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55 if not is_suspect else 0.65
    font_thick = 2
    (tw, th), baseline = cv2.getTextSize(full_text, font, font_scale, font_thick)
    ly1 = max(0, y1 - th - 8)
    ly2 = y1
    lx2 = min(w, x1 + tw + 10)
    cv2.rectangle(img, (x1, ly1), (lx2, ly2), color, -1)
    cv2.putText(img, full_text, (x1 + 5, y1 - 5), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

def run_tracker(source_id=0):
    print("=" * 60)
    print("[INIT] Initializing YOLOv8 Threat & Suspect Tracking Engine...")
    print("=" * 60)
    from ultralytics import YOLO
    print("[AI] Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")
    weapon_weights_path = os.path.join(os.path.dirname(__file__), "threat_weights.pt")
    weapon_model = None
    if os.path.exists(weapon_weights_path):
        print(f"[AI] Loading Specialized Weapon Model ({weapon_weights_path})...")
        weapon_model = YOLO(weapon_weights_path)
    print("[OK] Models Loaded Successfully!")
    print(f"[VIDEO] Connecting to Video Source: {source_id} ...")
    cap = cv2.VideoCapture(source_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[ERR] Failed to open video source: {source_id}")
        return
    print("=" * 60)
    print("[CAM] Surveillance Active. Controls:")
    print("   - Press 'q' or ESC to quit.")
    print("   - Press 's' to save screenshot.")
    print("=" * 60)
    cv2.namedWindow("Corec — Threat & Suspect Tracker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Corec — Threat & Suspect Tracker", 1100, 680)
    fps_history = []
    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            # If playing a video file, loop back to start
            if isinstance(source_id, str) and os.path.exists(source_id):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            print("[WARN] Stream empty or disconnected. Retrying...")
            time.sleep(0.1)
            continue
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 0.001)
        prev_time = curr_time
        fps_history.append(fps)
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = sum(fps_history) / len(fps_history)
        results = model.track(frame, persist=True, verbose=False, conf=WEAPON_CONFIDENCE, device="cpu")
        persons = []
        weapons = []
        if results and len(results) > 0:
            boxes_data = results[0].boxes
            if boxes_data is not None and len(boxes_data) > 0:
                xyxy = boxes_data.xyxy.cpu().numpy()
                confs = boxes_data.conf.cpu().numpy()
                cls_ids = boxes_data.cls.cpu().numpy().astype(int)
                track_ids = boxes_data.id.cpu().numpy().astype(int) if boxes_data.id is not None else [None] * len(xyxy)
                for box, conf, cid, tid in zip(xyxy, confs, cls_ids, track_ids):
                    class_name = model.names.get(cid, str(cid)).lower()
                    if cid == 0 or class_name == "person":
                        if conf >= CONFIDENCE_THRESHOLD:
                            persons.append({
                                "box": box,
                                "conf": conf,
                                "id": tid,
                                "is_suspect": False,
                                "linked_weapons": []
                            })
                    elif cid in DEFAULT_WEAPON_CLASSES or any(w in class_name for w in CUSTOM_WEAPON_NAMES):
                        w_label = DEFAULT_WEAPON_CLASSES.get(cid, class_name.capitalize())
                        weapons.append({
                            "box": box,
                            "conf": conf,
                            "label": w_label,
                            "id": tid
                        })

        # Run specialized weapon model if available
        if weapon_model is not None:
            w_res = weapon_model(frame, verbose=False, conf=0.25, device="cpu")
            if w_res and len(w_res) > 0 and w_res[0].boxes is not None:
                for wb in w_res[0].boxes:
                    w_cname = weapon_model.names[int(wb.cls[0])].capitalize()
                    w_cconf = float(wb.conf[0])
                    if w_cname.lower() in ["knife", "gun", "grenade"]:
                        weapons.append({
                            "box": wb.xyxy[0].cpu().numpy(),
                            "conf": w_cconf,
                            "label": w_cname,
                            "id": None
                        })
        suspects_count = 0
        for w in weapons:
            w_box = w["box"]
            matched_person = None
            for p in persons:
                if box_intersection_or_proximity(p["box"], w_box):
                    p["is_suspect"] = True
                    p["linked_weapons"].append(w["label"])
                    matched_person = p
                    break
            if matched_person is not None:
                p_center = (int((matched_person["box"][0] + matched_person["box"][2]) / 2),
                            int((matched_person["box"][1] + matched_person["box"][3]) / 2))
                w_center = (int((w_box[0] + w_box[2]) / 2),
                            int((w_box[1] + w_box[3]) / 2))
                cv2.line(frame, p_center, w_center, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, w_center, 4, (0, 255, 255), -1)
        for p in persons:
            if p["is_suspect"]:
                suspects_count += 1
                color = (0, 0, 255)
                label = f"SUSPECT: Robber (Armed: {','.join(p['linked_weapons'])})"
                draw_styled_box(frame, p["box"], color, label, track_id=p["id"], is_suspect=True)
            else:
                color = (0, 230, 0)
                label = "NORMAL PERSON"
                draw_styled_box(frame, p["box"], color, label, track_id=p["id"], is_suspect=False)
        for w in weapons:
            w_color = (0, 165, 255)
            w_label = f"WEAPON: {w['label']} ({w['conf']:.0%})"
            draw_styled_box(frame, w["box"], w_color, w_label, is_suspect=False)
        H, W = frame.shape[:2]
        hud_bg_color = (0, 0, 180) if suspects_count > 0 else (30, 30, 30)
        cv2.rectangle(frame, (0, 0), (W, 46), hud_bg_color, -1)
        if suspects_count > 0:
            status_text = f"[ALERT] THREAT LEVEL: HIGH | SUSPECTS ARMED: {suspects_count} | WEAPONS: {len(weapons)}"
        else:
            status_text = f"[NORMAL] STATUS: NORMAL | PERSONS: {len(persons)} | NO WEAPON THREAT"
        cv2.putText(frame, status_text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (W - 130, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.imshow("Corec — Threat & Suspect Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            print("\n[INFO] Exiting on user request.")
            break
        elif key == ord("s"):
            filename = f"capture_threat_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"[SAVED] Screenshot saved to: {filename}")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    source = 0
    if len(sys.argv) > 1:
        arg = sys.argv[1].strip()
        if arg == "2":
            source = COFE_RTSP_URL
        elif arg.isdigit():
            source = int(arg)
        else:
            source = arg
    run_tracker(source)
