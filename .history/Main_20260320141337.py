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

def send_email_alert_async(image_path,image_path_after,image_path_before):
    thread = threading.Thread(target=send_email_alert, args=(image_path,image_path_after, image_path_before),)
    thread.start()


WALL_FILE = "wall_config.json"  # File to store wall coordinates

# --- Email Alert Function with Image ---
def send_email_alert(image_path,image_path_before,image_path_after):
    sender = "khaloudakamoun@gmail.com"
    password = "jlkz rasj glbp qiup"  # Gmail app password
    receiver = "faditrani@gmail.com"
    current_time = datetime.now().strftime("%H:%M:%S")
    msg = MIMEMultipart()
    msg['Subject'] = f"⚠ Security Alert: Intruder Detected at {current_time}"
    msg['From'] = sender
    msg['To'] = receiver

    msg.attach(MIMEText("⚠ Intruder detected by your security camera!\nSee attached image."))

    with open(image_path, 'rb') as f:
        
        msg.attach(MIMEImage(f.read(), name="intruder1.jpg"))
    with open(image_path_before, 'rb') as f:
        msg.attach(MIMEImage(f.read(), name="intruder0.jpg"))
    with open(image_path_after, 'rb') as f:
        msg.attach(MIMEImage(f.read(), name="intruder2.jpg"))
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)

    print("Email with images sent!")

# --- Load YOLOv8 ---
model = YOLO('yolov8n.pt')

# Open video file
cap = cv2.VideoCapture('C:\Users\khaled\Desktop\PROJECT_AKK/v2.mp4')
if not cap.isOpened():
    print("Error: Could not open video")
    exit()

# --- Step 1: Load or Select Wall Points ---

if True:
    # Capture first frame for wall selection
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read video frame")
        exit()

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
        if key == 13 and len(points) == 4:  # Enter to confirm
            # Save wall points to file
            with open(WALL_FILE, "w") as f:
                json.dump(points, f)
            print("Wall points saved:", points)
            break
        elif key == 27:  # ESC to cancel
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Select Wall")

wall_polygon = np.array(points, np.int32).reshape((-1,1,2))

# --- Step 2: YOLOv8 Detection & Thief Tracking ---
track_history = {}
thief_ids = set()
email_sent_ids = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

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

            # Check if inside the polygon
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
                t = int(time.time())
                image_path = f"intruder_{track_id}_{t}.jpg"
                image_path_after = f"intruder_{track_id}_{t+1}.jpg"
                image_path_before = f"intruder_{track_id}_{t-1}.jpg"
                raw_frame_to_save = frame.copy()  # Copy the original frame (no drawings yet)
                cv2.imwrite(image_path, raw_frame_to_save)
                cv2.imwrite(image_path_after, raw_frame_to_save)
                cv2.imwrite(image_path_before, raw_frame_to_save)
                send_email_alert_async(image_path,image_path_before,image_path_after)
                email_sent_ids.add(track_id)

    cv2.imshow("YOLOv8 Detection", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()