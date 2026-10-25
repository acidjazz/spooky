from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import math
import os
import random
import sys
import tempfile
import threading
import time
import urllib.request
import cv2
import mediapipe as mp
import pygame

MIRROR_X = True
SOLO_CROWD_TEST = False
SENSITIVITY = 1.4
STREAM_PORT = 8080

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/1/efficientdet_lite2.tflite"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "efficientdet_lite2.tflite")

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp.tasks.vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.28,
    category_allowlist=["person"],
)
detector = mp.tasks.vision.ObjectDetector.create_from_options(options)

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_w, screen_h = screen.get_size()
pygame.mouse.set_visible(False)

SLOT_CONFIGS = [
    (0.50, 0.50, 1.05, "slit", 0.38),
    (0.22, 0.20, 0.52, "round", 0.42),
    (0.78, 0.22, 0.54, "slit", 0.34),
    (0.24, 0.80, 0.50, "slit", 0.44),
    (0.76, 0.78, 0.52, "round", 0.36),
]

class EyePair:
    def __init__(self, idx, nx, ny, scale, style, slant):
        self.idx = idx
        self.cx = int(screen_w * nx)
        self.cy = int(screen_h * ny)
        self.base_scale = scale
        self.scale = scale
        self.target_scale = scale
        self.style = style
        self.slant = slant
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.alert = 0.0
        self.target_alert = 0.0
        self.openness = 1.0 if idx == 0 else 0.0
        self.target_openness = 1.0 if idx == 0 else 0.0
        self.last_active = time.time() if idx == 0 else 0.0
        self.is_blinking = False
        self.blink_start = 0.0
        self.blink_dur = 0.16
        self.next_blink = time.time() + random.uniform(2.0, 6.0)
        self.next_saccade = time.time() + random.uniform(0.5, 2.0)
        self.breath_phase = random.uniform(0, math.tau)

    def update(self, now, assigned_target, crowd_count, idle_mode):
        if assigned_target is not None:
            self.last_active = now
            self.target_openness = 1.0
            self.target_alert = 1.0
            self.target_x, self.target_y = assigned_target
        elif self.idx == 0:
            self.target_openness = 1.0
            self.target_alert = 0.0
            if now >= self.next_saccade:
                if random.random() < 0.25:
                    self.target_x, self.target_y = 0.0, 0.0
                else:
                    ang = random.uniform(0, math.tau)
                    mag = random.uniform(0.3, 0.95)
                    self.target_x = math.cos(ang) * mag
                    self.target_y = math.sin(ang) * mag * 0.6
                self.next_saccade = now + random.uniform(0.6, 2.4)
        else:
            self.target_alert = 0.0
            if idle_mode:
                if self.target_openness < 0.5 and random.random() < 0.002:
                    self.last_active = now + random.uniform(2.5, 5.0)
                    self.target_openness = 1.0
                if now < self.last_active:
                    if now >= self.next_saccade:
                        ang = random.uniform(0, math.tau)
                        mag = random.uniform(0.2, 0.9)
                        self.target_x = math.cos(ang) * mag
                        self.target_y = math.sin(ang) * mag * 0.6
                        self.next_saccade = now + random.uniform(0.5, 1.8)
                else:
                    self.target_openness = 0.0
            else:
                if now - self.last_active > 1.2:
                    self.target_openness = 0.0

        if self.idx == 0:
            self.target_scale = 1.15 if crowd_count <= 1 else 0.88

        self.scale += (self.target_scale - self.scale) * 0.1
        self.openness += (self.target_openness - self.openness) * 0.18
        self.alert += (self.target_alert - self.alert) * 0.15
        self.curr_x += (self.target_x - self.curr_x) * 0.25
        self.curr_y += (self.target_y - self.curr_y) * 0.25

        if self.openness > 0.1 and not self.is_blinking and now >= self.next_blink:
            self.is_blinking = True
            self.blink_start = now

        blink_cut = 0.0
        if self.is_blinking:
            el = now - self.blink_start
            if el >= self.blink_dur:
                self.is_blinking = False
                self.next_blink = now + (0.12 if random.random() < 0.2 else random.uniform(2.5, 6.5))
            else:
                blink_cut = math.sin((el / self.blink_dur) * math.pi)

        effective_open = max(0.0, self.openness * (1.0 - blink_cut))
        if effective_open > 0.02:
            self.draw(effective_open, now)

    def draw(self, eff_open, now):
        eye_r = int(screen_h * 0.19 * self.scale)
        spacing = int(eye_r * 1.35)
        max_shift_x = eye_r * 0.48
        max_shift_y = eye_r * 0.32
        bob_y = int(math.sin(now * 2.2 + self.breath_phase) * (eye_r * 0.05))

        for side in (-1, 1):
            ex = self.cx + (side * spacing)
            ey = self.cy + bob_y
            px = int(ex + self.curr_x * max_shift_x)
            py = int(ey + self.curr_y * max_shift_y)

            steps = 10
            for i in range(steps):
                ratio = i / float(steps - 1)
                r = int(eye_r * (1.0 - ratio * 0.25))
                red = int(165 + 90 * ratio)
                green = int(5 + 18 * ratio)
                pygame.draw.circle(screen, (red, green, 12), (ex, ey), r)

            if self.style == "round":
                pinch = 1.0 - (0.35 * self.alert)
                p_rad = max(6, int(eye_r * 0.42 * pinch))
                pygame.draw.circle(screen, (0, 0, 0), (px, py), p_rad)
            else:
                pinch = 1.0 - (0.55 * self.alert)
                slit_w = max(3, int(eye_r * 0.20 * pinch))
                slit_h = int(eye_r * (0.76 + 0.08 * self.alert))
                pts = [
                    (px, py - slit_h),
                    (px + slit_w, py - int(slit_h * 0.25)),
                    (px + int(slit_w * 0.7), py + int(slit_h * 0.35)),
                    (px, py + slit_h),
                    (px - int(slit_w * 0.7), py + int(slit_h * 0.35)),
                    (px - slit_w, py - int(slit_h * 0.25)),
                ]
                pygame.draw.polygon(screen, (0, 0, 0), pts)

            close_amt = 1.0 - eff_open
            slant_dir = -side
            tilt = eye_r * (self.slant + 0.06 * self.alert) * slant_dir

            base_top = ey - int(eye_r * 0.22)
            curr_top = base_top + int(eye_r * 1.15 * close_amt)

            upper_mask = [
                (ex - eye_r - 15, ey - eye_r - 15),
                (ex + eye_r + 15, ey - eye_r - 15),
                (ex + eye_r + 15, int(curr_top + tilt)),
                (ex - eye_r - 15, int(curr_top - tilt)),
            ]
            pygame.draw.polygon(screen, (0, 0, 0), upper_mask)

            base_bot = ey + int(eye_r * 0.72)
            curr_bot = base_bot - int(eye_r * 0.85 * close_amt)
            bot_tilt = -tilt * 0.35

            lower_mask = [
                (ex - eye_r - 15, ey + eye_r + 15),
                (ex + eye_r + 15, ey + eye_r + 15),
                (ex + eye_r + 15, int(curr_bot + bot_tilt)),
                (ex - eye_r - 15, int(curr_bot - bot_tilt)),
            ]
            pygame.draw.polygon(screen, (0, 0, 0), lower_mask)

            ring_thick = max(20, int(eye_r * 0.35))
            pygame.draw.circle(screen, (0, 0, 0), (ex, ey), eye_r + ring_thick, ring_thick)

def match_sticky_targets(eye_pairs, raw_targets):
    assigned = [None] * len(eye_pairs)
    count = min(len(eye_pairs), len(raw_targets))
    if count == 0:
        return assigned

    candidates = []
    for p_idx in range(count):
        p = eye_pairs[p_idx]
        for t_idx in range(count):
            tx, ty = raw_targets[t_idx]
            dist = (p.curr_x - tx) ** 2 + (p.curr_y - ty) ** 2
            candidates.append((dist, p_idx, t_idx))

    candidates.sort(key=lambda item: item[0])
    used_pairs = set()
    used_targets = set()

    for _, p_idx, t_idx in candidates:
        if p_idx not in used_pairs and t_idx not in used_targets:
            assigned[p_idx] = raw_targets[t_idx]
            used_pairs.add(p_idx)
            used_targets.add(t_idx)

    return assigned

pairs = [EyePair(i, *cfg) for i, cfg in enumerate(SLOT_CONFIGS)]
latest_targets = []
latest_jpeg = None
running = True

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
                while running:
                    if latest_jpeg is not None:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(latest_jpeg)))
                        self.end_headers()
                        self.wfile.write(latest_jpeg)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.08)
            except Exception:
                pass
        else:
            html = (
                b"<html><head><meta name='viewport' content='width=device-width, initial-scale=1.0'>"
                b"<title>SpookySpy Aim</title></head>"
                b"<body style='margin:0;background:black;color:red;font-family:sans-serif;text-align:center;'>"
                b"<h3 style='margin:10px;'>SpookySpy Live Aiming View</h3>"
                b"<img src='/stream' style='width:100%;max-width:960px;border:2px solid red;'/>"
                b"</body></html>"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)

def start_server():
    try:
        server = ThreadingHTTPServer(("0.0.0.0", STREAM_PORT), CamHandler)
        server.serve_forever()
    except Exception:
        pass

def open_camera():
    c = cv2.VideoCapture(0)
    if c.isOpened():
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return c

def vision_worker():
    global latest_targets, latest_jpeg, running
    cap = open_camera()
    last_retry = time.time()
    last_seen = 0.0

    while running:
        now = time.time()
        if not cap.isOpened():
            if now - last_retry > 3.0:
                last_retry = now
                cap = open_camera()
            time.sleep(0.1)
            continue

        success, image = cap.read()
        if not success or image is None:
            cap.release()
            continue

        img_h, img_w = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

        debug_frame = image.copy()

        if results.detections:
            sorted_dets = sorted(
                results.detections,
                key=lambda d: d.bounding_box.width * d.bounding_box.height,
                reverse=True,
            )
            new_targets = []
            for det in sorted_dets[: len(pairs)]:
                bbox = det.bounding_box
                x1, y1 = int(bbox.origin_x), int(bbox.origin_y)
                x2, y2 = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                fx = (bbox.origin_x + (bbox.width * 0.5)) / img_w
                fy = (bbox.origin_y + (bbox.height * 0.22)) / img_h
                cv2.circle(debug_frame, (int(fx * img_w), int(fy * img_h)), 8, (0, 255, 0), -1)

                raw_x = (1.0 - (fx * 2.0)) if MIRROR_X else ((fx * 2.0) - 1.0)
                nx = max(-1.0, min(1.0, raw_x * SENSITIVITY))
                ny = max(-1.0, min(1.0, ((fy * 2.0) - 1.0) * SENSITIVITY))
                new_targets.append((nx, ny))

            if SOLO_CROWD_TEST and len(new_targets) == 1:
                bx, by = new_targets[0]
                new_targets.extend([
                    (max(-1.0, min(1.0, bx - 0.25)), by),
                    (max(-1.0, min(1.0, bx + 0.25)), by),
                ])
            latest_targets = new_targets
            last_seen = now
        elif now - last_seen > 0.6:
            latest_targets = []

        if MIRROR_X:
            debug_frame = cv2.flip(debug_frame, 1)
        preview = cv2.resize(debug_frame, (640, 360))
        ok, buf = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if ok:
            latest_jpeg = buf.tobytes()

    if cap.isOpened():
        cap.release()

threading.Thread(target=start_server, daemon=True).start()
worker = threading.Thread(target=vision_worker, daemon=True)
worker.start()

clock = pygame.time.Clock()

try:
    while True:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        targets = list(latest_targets)
        assigned_list = match_sticky_targets(pairs, targets)

        screen.fill((0, 0, 0))
        idle_mode = len(targets) == 0
        active_crowd = len(targets) if not idle_mode else 1 + sum(1 for p in pairs[1:] if p.openness > 0.2)

        for i, pair in enumerate(pairs):
            pair.update(now, assigned_list[i], active_crowd, idle_mode)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    running = False
    worker.join(timeout=1.0)
    detector.close()
    pygame.quit()
    sys.exit()