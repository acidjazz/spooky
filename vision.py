from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import tempfile
import threading
import time
import urllib.request
import cv2
import mediapipe as mp
import numpy as np

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "efficientdet_lite2.tflite")


class VisionTracker:
    def __init__(
        self,
        max_targets=5,
        mirror_x=False,
        solo_test=False,
        sensitivity=0.90,
        x_offset=-0.09,
        y_offset=0.30,
        min_score=0.18,
        port=8080,
    ):
        self.max_targets = max_targets
        self.mirror_x = mirror_x
        self.solo_test = solo_test
        self.sensitivity = sensitivity
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.min_score = min_score
        self.port = port
        self.latest_targets = []
        self.latest_jpeg = None
        self.running = True

        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        inv_gamma = 1.0 / 1.65
        self.gamma_lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
        ).astype("uint8")

        if not os.path.exists(MODEL_PATH):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

        base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.15,
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
            c.set(cv2.CAP_PROP_GAIN, 255)
            c.set(cv2.CAP_PROP_BRIGHTNESS, 150)
        return c

    def _enhance_night(self, bgr_frame):
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_clahe = self.clahe.apply(l)
        l_boosted = cv2.LUT(l_clahe, self.gamma_lut)
        merged = cv2.merge((l_boosted, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _start_server(self):
        tracker = self

        class CamHandler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
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
                    html = (
                        "<html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                        "<title>SpookySpy Aim</title></head>"
                        "<body style='margin:0;background:black;color:red;font-family:sans-serif;text-align:center;'>"
                        "<h3 style='margin:10px;'>SpookySpy Live Aiming View</h3>"
                        "<img src='/stream' style='width:100%;max-width:960px;border:2px solid red;'/>"
                        "</body></html>"
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
        slot_first_seen = [0.0] * self.max_targets
        slot_last_seen = [0.0] * self.max_targets
        slot_confirmed_until = [0.0] * self.max_targets
        slot_pos = [(0.0, 0.0)] * self.max_targets

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
            enhanced_bgr = self._enhance_night(image)
            image_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = self.detector.detect(mp_image)

            debug_frame = cv2.flip(enhanced_bgr, 1)
            valid_dets = [
                d for d in (results.detections or [])
                if d.categories and d.categories[0].score >= self.min_score
            ]

            raw_targets = []
            if valid_dets:
                sorted_dets = sorted(
                    valid_dets,
                    key=lambda d: d.bounding_box.width * d.bounding_box.height,
                    reverse=True,
                )
                accepted_fx = []
                for det in sorted_dets:
                    bbox = det.bounding_box
                    fx = (bbox.origin_x + (bbox.width * 0.5)) / img_w
                    if any(abs(fx - afx) < 0.14 for afx in accepted_fx):
                        continue
                    accepted_fx.append(fx)

                    score_pct = int(round(det.categories[0].score * 100))
                    fy = (bbox.origin_y + (bbox.height * 0.22)) / img_h

                    dx1 = int((1.0 - (bbox.origin_x + bbox.width) / img_w) * img_w)
                    dx2 = int((1.0 - bbox.origin_x / img_w) * img_w)
                    dy1, dy2 = int(bbox.origin_y), int(bbox.origin_y + bbox.height)
                    cv2.rectangle(debug_frame, (dx1, dy1), (dx2, dy2), (0, 0, 255), 3)
                    cv2.circle(debug_frame, (int((1.0 - fx) * img_w), int(fy * img_h)), 8, (0, 255, 0), -1)
                    cv2.putText(
                        debug_frame,
                        f"{score_pct}%",
                        (dx1, max(24, dy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                    adj_x = fx + self.x_offset
                    raw_x = (1.0 - (adj_x * 2.0)) if self.mirror_x else ((adj_x * 2.0) - 1.0)
                    raw_y = ((fy + self.y_offset) * 2.0) - 1.0
                    nx = max(-1.0, min(1.0, raw_x * self.sensitivity))
                    ny = max(-1.0, min(1.0, raw_y * self.sensitivity))
                    raw_targets.append((nx, ny))

                    if len(raw_targets) >= self.max_targets:
                        break

                if self.solo_test and len(raw_targets) == 1:
                    bx, by = raw_targets[0]
                    raw_targets.extend([
                        (max(-1.0, min(1.0, bx - 0.25)), by),
                        (max(-1.0, min(1.0, bx + 0.25)), by),
                    ])

            active_targets = []
            for i in range(self.max_targets):
                if i < len(raw_targets):
                    if slot_first_seen[i] == 0.0:
                        slot_first_seen[i] = now
                    slot_last_seen[i] = now
                    slot_pos[i] = raw_targets[i]
                    req_delay = 0.0 if i == 0 else 0.45
                    if (now - slot_first_seen[i]) >= req_delay:
                        slot_confirmed_until[i] = now + 1.5
                else:
                    if (now - slot_last_seen[i]) > 0.25:
                        slot_first_seen[i] = 0.0

                if now < slot_confirmed_until[i]:
                    active_targets.append(slot_pos[i])

            self.latest_targets = active_targets

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