import math
import os
import random
import sys
import tempfile
import time
import urllib.request
import cv2
import mediapipe as mp
import pygame

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
MODEL_PATH = os.path.join(tempfile.gettempdir(), "blaze_face_short_range.tflite")

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = mp.tasks.BaseOptions(model_asset_path=MODEL_PATH)
options = mp.tasks.vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.45,
)
detector = mp.tasks.vision.FaceDetector.create_from_options(options)

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
            self.draw(effective_open)

    def draw(self, eff_open):
        eye_r = int(screen_h * 0.19 * self.scale)
        spacing = int(eye_r * 1.35)
        max_shift_x = eye_r * 0.48
        max_shift_y = eye_r * 0.32

        for side in (-1, 1):
            ex = self.cx + (side * spacing)
            ey = self.cy
            px = int(ex + self.curr_x * max_shift_x)
            py = int(ey + self.curr_y * max_shift_y)

            steps = 10
            for i in range(steps):
                ratio = i / float(steps - 1)
                r = int(eye_r * (1.0 - ratio * 0.25))
                red = int(185 + 70 * ratio)
                idle_green = 140 + 85 * ratio
                track_green = 5 + 20 * ratio
                green = int(idle_green * (1.0 - self.alert) + track_green * self.alert)
                blue = int(15 * self.alert)
                pygame.draw.circle(screen, (red, green, blue), (ex, ey), r)

            if self.style == "round":
                p_rad = int(eye_r * 0.42)
                pygame.draw.circle(screen, (0, 0, 0), (px, py), p_rad)
            else:
                slit_w = max(6, int(eye_r * 0.18))
                slit_h = int(eye_r * 0.78)
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
            tilt = eye_r * self.slant * slant_dir

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

pairs = [EyePair(i, *cfg) for i, cfg in enumerate(SLOT_CONFIGS)]
cap = cv2.VideoCapture(0)
last_cam_retry = time.time()
sensitivity = 1.65
clock = pygame.time.Clock()

try:
    while True:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        targets = []
        if not cap.isOpened():
            if now - last_cam_retry > 3.0:
                last_cam_retry = now
                cap.open(0)
        else:
            success, image = cap.read()
            if not success or image is None:
                cap.release()
            else:
                img_h, img_w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = detector.detect(mp_image)

                if results.detections:
                    sorted_dets = sorted(
                        results.detections,
                        key=lambda d: d.bounding_box.width * d.bounding_box.height,
                        reverse=True,
                    )
                    for det in sorted_dets[: len(pairs)]:
                        bbox = det.bounding_box
                        fx = (bbox.origin_x + (bbox.width / 2.0)) / img_w
                        fy = (bbox.origin_y + (bbox.height / 2.0)) / img_h
                        nx = max(-1.0, min(1.0, (1.0 - (fx * 2.0)) * sensitivity))
                        ny = max(-1.0, min(1.0, ((fy * 2.0) - 1.0) * sensitivity))
                        targets.append((nx, ny))

        screen.fill((0, 0, 0))
        idle_mode = len(targets) == 0
        active_crowd = len(targets) if not idle_mode else 1 + sum(1 for p in pairs[1:] if p.openness > 0.2)

        for i, pair in enumerate(pairs):
            assigned = targets[i] if i < len(targets) else None
            pair.update(now, assigned, active_crowd, idle_mode)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    cap.release()
    detector.close()
    pygame.quit()
    sys.exit()