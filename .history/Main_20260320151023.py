import cv2
import numpy as np
import json
import os
import time
from ultralytics import YOLO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
from datetime import datetime
from collections import deque

# ---------------------- Configuration ----------------------
EMAIL_SENDER = "khaloudakamoun@gmail.com"
EMAIL_PASSWORD = "jlkz rasj glbp qiup"  # Gmail app password
EMAIL_RECEIVER = "faditrani@gmail.com"
TEST_MODE = False  # Set True for debug without sending emails

WALL_FILE = "wall_config.json"
frame_buffer = deque(maxlen=180)  # ~3 seconds at 30 FPS, adjust if needed

track_history = {}
thief_ids = set()
email_sent_ids = set()

# ---------------------- Helpers ----------------------
def buffer_current_frame(raw_frame):
    ts = time.time()
    frame_buffer.append((ts, raw_frame.copy()))

def _find_closest_frame(target_ts):
    if not frame_buffer:
        return None
    closest = None
    best_diff = None
    for ts, frame in frame_buffer:
        diff = abs(ts - target_ts)
        if best_diff is None or diff < best_diff:
            best_diff = diff
            closest = (ts, frame)
    return closest

def save_frame_to_path(frame, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, frame)

def send_email_with_images(image_paths):
    if TEST_MODE:
        print("TEST MODE - would send email with:", image_paths)
        return True

    msg = MIMEMultipart()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg['Subject'] = f"⚠ Security Alert: Intruder Detected at {current_time}"
    msg['From'] = EMAIL_SENDER
    msg['To'] = EMAIL_RECEIVER
    msg.attach(MIMEText(
        f"⚠ Intruder detected by your security camera at {current_time}!\n"
        "See the 3 attached images ."
    ))

    for idx, path in enumerate(image_paths):
        try:
            with open(path, 'rb') as f:
                img_data = f.read()
            name = ["intruder_before.jpg", "intruder.jpg", "intruder_after.jpg"][idx] if idx < 3 else os.path.basename(path)
            msg.attach(MIMEImage(img_data, name=name))
        except Exception as e:
            print(f"Warning: failed to attach {path}: {e}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully.")
        return True
    except Exception as e:
        print("Failed to send email:", e)
        return False

def prepare_and_send_three_images(track_id, current_frame, current_ts, wait_time=1.0):
    base = os.path.join("intruder_captures", f"intruder_{track_id}_{int(current_ts)}")
    path_during = f"{base}.jpg"
    path_before = f"{base}_before.jpg"
    path_after = f"{base}_after.jpg"

    save_frame_to_path(current_frame, path_during)

    # Before frame (~1.5 second before)
    target_before_ts = current_ts - 1.5
    found = _find_closest_frame(target_before_ts)
    if found:
        _, before_frame = found
        save_frame_to_path(before_frame, path_before)
    else:
        save_frame_to_path(current_frame, path_before)

    # Thread for after frame and email
    def after_and_send():
        time.sleep(wait_time)
        target_after_ts = current_ts + 2
        found_after = _find_closest_frame(target_after_ts)
        if found_after:
            _, after_frame = found_after
            save_frame_to_path(after_frame, path_after)
        else:
            if frame_buffer:
                _, latest_frame = frame_buffer[-1]
                save_frame_to_path(latest_frame, path_after)
            else:
                save_frame_to_path(current_frame, path_after)

        send_email_with_images([path_before, path_during, path_after])

    t = threading.Thread(target=after_and_send, daemon=True)
    t.start()

# ---------------------- Load YOLOv8 ----------------------
model = YOLO('yolov8n.pt')

# Open video file
cap = cv2.VideoCapture('C:/Users/khaled/Documents/v2.mp4')
if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# ---------------------- Wall Selection ----------------------
# ---------------------- Wall Selection ----------------------
ret, frame = cap.read()
if not ret:
    print("Error: Could not read first frame")
    exit()

if os.path.exists(WALL_FILE):
    with open(WALL_FILE, "r") as f:
        points = json.load(f)
    print("Loaded saved wall points:", points)
else:
    points = []

    def get_coordinates(event, x, y, flags, param):
        global points
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)} selected: X={x}, Y={y}")

    cv2.namedWindow("Select Wall")
    cv2.setMouseCallback("Select Wall", get_coordinates)

    while True:
        frame_copy = frame.copy()
        for p in points:
            cv2.circle(frame_copy, p, 5, (0,0,255), -1)
        if len(points) > 1:
            cv2.polylines(frame_copy, [np.array(points, np.int32)], len(points) == 4, (255,0,0), 2)

        cv2.imshow("Select Wall", frame_copy)
        key = cv2.waitKey(1) & 0xFF
        if key == 13 and len(points) == 4:
            with open(WALL_FILE, "w") as f:
                json.dump(points, f)
            print("Wall points saved:", points)
            break
        elif key == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Select Wall")

wall_polygon = np.array(points, np.int32).reshape((-1,1,2))
# ---------------------- Detection Loop ----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    buffer_current_frame(frame)  # keep buffer updated

    results = model.track(frame, tracker="ultralytics/cfg/trackers/bytetrack.yaml",
                          persist=True, classes=[0], conf=0.6)

    annotated_frame = frame.copy()
    cv2.polylines(annotated_frame, [wall_polygon], True, (0,0,255), 2)

    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy() if r.boxes.id is not None else []

        for box, track_id in zip(boxes, ids):
            x1, y1, x2, y2 = box
            cx, cy = int((x1+x2)/2), int((y1+y2)/2)

            track_history.setdefault(track_id, []).append((cx, cy))

            if cv2.pointPolygonTest(wall_polygon, (cx, cy), False) >= 0:
                thief_ids.add(track_id)

            # Draw trajectory
            for i in range(1, len(track_history[track_id])):
                cv2.line(annotated_frame, track_history[track_id][i-1], track_history[track_id][i], (0,255,255), 2)

            # Draw box and label
            is_thief = track_id in thief_ids
            color = (0,0,255) if is_thief else (0,255,0)
            label = f"THIEF {track_id}!" if is_thief else f"ID {track_id}"
            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(annotated_frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Capture and send email once per thief
            if is_thief and track_id not in email_sent_ids:
                ts = time.time()
                raw_frame_to_save = frame.copy()
                prepare_and_send_three_images(track_id, raw_frame_to_save, ts)
                email_sent_ids.add(track_id)

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
