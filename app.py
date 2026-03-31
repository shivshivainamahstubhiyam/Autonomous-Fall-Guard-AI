import cv2
import numpy as np
import smtplib
import asyncio
import threading
import queue
import time
from datetime import datetime
from typing import List
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from ultralytics import YOLO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ==========================================
# Configuration (FastAPI Base)
# ==========================================
GMAIL_SENDER    = "parmarjatin5565@gmail.com"
GMAIL_APP_PASS  = "syas rlba fbzo ulwv" 
ALERT_RECIPIENT = "parmarjatin5565@gmail.com"
SMTP_PORT       = 587
MODEL_PATH      = "yolo11n.pt"
VIDEO_SOURCE    = 0

# ==========================================
# Async-friendly Alerting
# ==========================================
class GmailSender:
    def __init__(self):
        self._connected = False
        self.server = None

    def _connect(self):
        try:
            self.server = smtplib.SMTP("smtp.gmail.com", SMTP_PORT, timeout=15)
            self.server.ehlo()
            self.server.starttls()
            self.server.ehlo()
            self.server.login(GMAIL_SENDER, GMAIL_APP_PASS)
            self._connected = True
            print(f"[SMTP] Connected securely as {GMAIL_SENDER}")
        except Exception as e:
            print(f"[SMTP BUSY/ERROR] {e}")
            self._connected = False

    def send_alert(self, track_id, conf, ts, frame):
        if not self._connected: self._connect()
        if not self._connected: return

        msg = MIMEMultipart()
        msg["From"] = GMAIL_SENDER
        msg["To"]   = ALERT_RECIPIENT
        msg["Subject"] = f"🔔 AI FALL ALERT: ID {track_id}"
        
        body = f"FALL DETECTED\nID: {track_id}\nConf: {conf:.2%}\nTime: {ts}"
        msg.attach(MIMEText(body, "plain"))

        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            msg.attach(MIMEImage(buf.tobytes(), _subtype="jpeg"))

        try:
            self.server.sendmail(GMAIL_SENDER, ALERT_RECIPIENT, msg.as_string())
            print(f"[SUCCESS] Email sent for ID {track_id}")
        except:
            self._connect()

alert_queue = queue.Queue()

def email_worker():
    sender = GmailSender()
    while True:
        job = alert_queue.get()
        if job is None: break
        sender.send_alert(*job)
        alert_queue.task_done()

# ==========================================
# Optimized Detection Core
# ==========================================
class DetectionEngine:
    def __init__(self, model_path, source):
        self.model = YOLO(model_path)
        self.source = source
        self.cap = None
        self._init_camera()
        self.current_frame = None
        self.is_running = True
        self.last_alerts = {}

    def _init_camera(self):
        # Using CAP_DSHOW for better compatibility on Windows
        for idx in [0, 1, 2]:
            print(f"[CAMERA] Attempting index {idx} (DSHOW)...")
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if self.cap.isOpened():
                self.source = idx
                print(f"[CAMERA] Success! Using index {idx}")
                return
            self.cap.release()
        print("[CAMERA ERROR] No active camera device found.")

    def run_inference(self):
        if not self.cap or not self.cap.isOpened():
            print("[CORE] Detection engine running in DUMMY MODE (No Camera).")
            while self.is_running:
                # Generate a "Camera Not Found" static placeholder
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(dummy, "CAMERA NOT FOUND OR BUSY", (50, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(dummy, "Check your hardware and close other apps.", (50, 280), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                self.current_frame = dummy
                time.sleep(0.5)
            return
        
        print(f"[CORE] Running FastAPI-Linked Detection Engine on {self.source}")
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            results = self.model.track(frame, verbose=False, conf=0.5, persist=True)[0]
            
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy().astype(int)
                ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy().astype(int)

                for box, tid, conf, cls in zip(boxes, ids, confs, clss):
                    if cls == 0: 
                        now = time.time()
                        if tid not in self.last_alerts or (now - self.last_alerts[tid] > 60):
                            ts = datetime.now().strftime("%H:%M:%S")
                            alert_queue.put((tid, float(conf), ts, frame.copy()))
                            self.last_alerts[tid] = now

                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
                        cv2.putText(frame, f"FALL ID:{tid}", (box[0], box[1]-10), 
                                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 255), 2)

            self.current_frame = frame

# ==========================================
# FastAPI Implementation
# ==========================================
app = FastAPI(title="Fall Guard FastAPI")
templates = Jinja2Templates(directory="templates")
engine = DetectionEngine(MODEL_PATH, VIDEO_SOURCE)

@app.on_event("startup")
async def startup_event():
    # Start threads on app initialization 
    threading.Thread(target=email_worker, daemon=True).start()
    threading.Thread(target=engine.run_inference, daemon=True).start()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

async def video_streamer():
    while True:
        if engine.current_frame is not None:
            _, buffer = cv2.imencode('.jpg', engine.current_frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        await asyncio.sleep(0.04) # ~25 FPS

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_streamer(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    return {
        "status": "Online",
        "ids_alerted": [int(i) for i in engine.last_alerts.keys()],
        "time": datetime.now().strftime("%H:%M:%S")
    }

if __name__ == "__main__":
    import uvicorn
    print("\n🚀 FAST-API DETECTOR HUB STARTING...")
    print("🔗 ACCESS AT: http://127.0.0.1:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
