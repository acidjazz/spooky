from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import tempfile
import threading
import time
import urllib.request
import cv2
import mediapipe as mp

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "efficientdet_lite2.tflite")


class VisionTracker:
    def __init__(self, max_targets=5, mirror_x=False, solo_test=False, sensitivity=1.4, y_offset=0.22, port=8080):
        self.max_targets = max_targets
        self.mirror_x = mirror_x
        self.solo_test = solo_test
        self.sensitivity = sensitivity
        self.y_offset = y_offset
        self.port = port
        self.latest_targets = []
        self.latest_jpeg = None
        self.running = True

        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.28,
            category_allowlist=["person"],
        )
        self.detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

        threading.Thread(target=self._start_server, daemon=True).start()
        self.worker = threading.Thread(target=self._vision_worker, daemon=True)
        self.worker.start()

    def get_targets(self):
        return list(self.latest_targets)

    def stop(self):
        self.running = False
        self.worker.join(timeout=1.0)
        self.detector.close()

    def _open_camera(self):
        c = cv2.VideoCapture(0)
        if c.isOpened():
            c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return c

    def _start_server(self):
        tracker = self

        class CamHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                if self.path == "/flip":
                    tracker.mirror_x = not tracker.mirror_x
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return
                if self.path == "/solo":
                    tracker.solo_test = not tracker.solo_test
                    self.send_response(302)
                    self.send_header("Location", "/")
                    self.end_headers()
                    return
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Age", "0")
                    self.send_header("Cache-Control", "no-cache, private")
                    self.send_header("Pragma", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while tracker.running:
                            if tracker.latest_jpeg is not None:
                                self.wfile.write(b"--frame\r\n")
                                self.send_header("Content-Type", "image/jpeg")
                                self.send_header("Content-Length", str(len(tracker.latest_jpeg)))
                                self.end_headers()
                                self.wfile.write(tracker.latest_jpeg)
                                self.wfile.write(b"\r\n")
                            time.sleep(0.08)
                    except Exception:
                        pass
                else:
                    m_state = "ON" if tracker.mirror_x else "OFF"
                    s_state = "ON" if tracker.solo_test else "OFF"
                    html = (
                        f"<html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                        f"<title>SpookySpy Aim</title></head>"
                        f"<body style='margin:0;background:black;color:red;font-family:sans-serif;text-align:center;'>"
                        f"<h3 style='margin:10px;'>SpookySpy Live Aiming View</h3>"
                        f"<img src='/stream' style='width:100%;max-width:960px;border:2px solid red;'/>"
                        f"<div style='margin:14px;'>"
                        f"<a href='/flip' style='color:white;background:darkred;padding:10px 16px;text-decoration:none;border-radius:6px;margin:6px;display:inline-block;'>Flip Left/Right ({m_state})</a>"
                        f"<a href='/solo' style='color:white;background:darkred;padding:10px 16px;text-decoration:none;border-radius:6px;margin:6px;display:inline-block;'>3-Eye Crowd Test ({s_state})</a>"
                        f"</div></body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(html)))
                    self.end_headers()
                    self.wfile.write(html)

        try:
            server = ThreadingHTTPServer(("0.0.0.0", self.port), CamHandler)
            server.serve_forever()
        except Exception:
            pass

    def _vision_worker(self):
        cap = self._open_camera()
        last_retry = time.time()
        last_seen = 0.0

        while self.running:
            now = time.time()
            if not cap.isOpened():
                if now - last_retry > 3.0:
                    last_retry = now
                    cap = self._open_camera()
                time.sleep(0.1)
                continue

            success, image = cap.read()
            if not success or image is None:
                cap.release()
                continue

            img_h, img_w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(mp_image)

            debug_frame = cv2.flip(image, 1)

            if results.detections:
                sorted_dets = sorted(
                    results.detections,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                    reverse=True,
                )
                new_targets = []
                for det in sorted_dets[: self.max_targets]:
                    bbox = det.bounding_box
                    fx = (bbox.origin_x + (bbox.width * 0.5)) / img_w
                    fy = (bbox.origin_y + (bbox.height * 0.22)) / img_h

                    dx1 = int((1.0 - (bbox.origin_x + bbox.width) / img_w) * img_w)
                    dx2 = int((1.0 - bbox.origin_x / img_w) * img_w)
                    dy1, dy2 = int(bbox.origin_y), int(bbox.origin_y + bbox.height)
                    cv2.rectangle(debug_frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
                    cv2.circle(debug_frame, (int((1.0 - fx) * img_w), int(fy * img_h)), 8, (0, 255, 0), -1)

                    raw_x = (1.0 - (fx * 2.0)) if self.mirror_x else ((fx * 2.0) - 1.0)
                    raw_y = ((fy + self.y_offset) * 2.0) - 1.0
                    nx = max(-1.0, min(1.0, raw_x * self.sensitivity))
                    ny = max(-1.0, min(1.0, raw_y * self.sensitivity))
                    new_targets.append((nx, ny))

                if self.solo_test and len(new_targets) == 1:
                    bx, by = new_targets[0]
                    new_targets.extend([
                        (max(-1.0, min(1.0, bx - 0.25)), by),
                        (max(-1.0, min(1.0, bx + 0.25)), by),
                    ])
                self.latest_targets = new_targets
                last_seen = now
            elif now - last_seen > 0.6:
                self.latest_targets = []

            cx, cy = img_w // 2, img_h // 2
            cv2.circle(debug_frame, (cx, cy), 45, (80, 80, 80), 2)
            if self.latest_targets:
                gx, gy = self.latest_targets[0]
                tx = int(cx - gx * 40)
                ty = int(cy + gy * 40)
                cv2.line(debug_frame, (cx, cy), (tx, ty), (0, 255, 255), 3)
                cv2.circle(debug_frame, (tx, ty), 8, (0, 255, 255), -1)

            preview = cv2.resize(debug_frame, (640, 360))
            ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                self.latest_jpeg = buf.tobytes()

        if cap.isOpened():
            cap.release()