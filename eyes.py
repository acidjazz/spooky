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
    min_detection_confidence=0.5,
)
detector = mp.tasks.vision.FaceDetector.create_from_options(options)

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_w, screen_h = screen.get_size()
pygame.mouse.set_visible(False)

eye_radius = int(screen_h * 0.26)
iris_radius = int(eye_radius * 0.48)
base_pupil_radius = int(iris_radius * 0.52)
max_travel = eye_radius - iris_radius - 8

left_eye_center = (int(screen_w * 0.32), int(screen_h * 0.5))
right_eye_center = (int(screen_w * 0.68), int(screen_h * 0.5))

random.seed(42)
veins = []
for _ in range(18):
    angle = random.uniform(0, math.tau)
    segs = []
    curr_a = angle
    for s in range(5):
        curr_a += random.uniform(-0.18, 0.18)
        segs.append(curr_a)
    veins.append(segs)

cap = cv2.VideoCapture(0)

sensitivity = 1.5
current_x, current_y = 0.0, 0.0
target_x, target_y = 0.0, 0.0

next_saccade_time = time.time() + 1.0
next_blink_time = time.time() + random.uniform(2.0, 5.0)
blink_start = 0.0
blink_duration = 0.18
is_blinking = False

pupil_scale = 1.0
target_pupil_scale = 1.0
squint_amount = 0.15
target_squint = 0.15

def draw_eye(center, offset_x, offset_y, is_left, blink_progress, squint, p_scale):
    cx, cy = center
    ix = int(cx + offset_x)
    iy = int(cy + offset_y)

    for i in range(15):
        ratio = i / 14.0
        r = int(eye_radius * (1.0 - ratio * 0.18))
        val = int(140 + 110 * ratio)
        pygame.draw.circle(screen, (val, val - 10, val - 15), (cx, cy), r)

    for segs in veins:
        pts = []
        for idx, a in enumerate(segs):
            dist_ratio = 1.0 - (idx * 0.12)
            r = eye_radius * dist_ratio
            px = cx + math.cos(a) * r + (offset_x * (1.0 - dist_ratio) * 0.5)
            py = cy + math.sin(a) * r + (offset_y * (1.0 - dist_ratio) * 0.5)
            pts.append((int(px), int(py)))
        pygame.draw.lines(screen, (190, 40, 40), False, pts, 2)

    pygame.draw.circle(screen, (60, 0, 0), (ix, iy), iris_radius + 3)
    for i in range(12):
        ratio = i / 11.0
        r = int(iris_radius * (1.0 - ratio * 0.45))
        red = int(180 + 75 * ratio)
        green = int(20 + 140 * (ratio ** 2))
        pygame.draw.circle(screen, (red, green, 0), (ix, iy), r)

    p_rad = int(base_pupil_radius * p_scale)
    pygame.draw.circle(screen, (0, 0, 0), (ix, iy), p_rad)

    glint_x = ix - int(iris_radius * 0.28)
    glint_y = iy - int(iris_radius * 0.28)
    pygame.draw.circle(screen, (255, 255, 255), (glint_x, glint_y), max(4, int(eye_radius * 0.06)))
    pygame.draw.circle(screen, (255, 220, 200), (ix + int(iris_radius * 0.22), iy + int(iris_radius * 0.22)), max(2, int(eye_radius * 0.025)))

    close_factor = max(squint, blink_progress)
    top_y = (cy - eye_radius) + (eye_radius * 1.35 * close_factor)
    tilt = eye_radius * 0.32 * (1 if is_left else -1)

    upper_lid = [
        (cx - eye_radius - 20, cy - eye_radius - 20),
        (cx + eye_radius + 20, cy - eye_radius - 20),
        (cx + eye_radius + 20, int(top_y + tilt)),
        (cx - eye_radius - 20, int(top_y - tilt)),
    ]
    pygame.draw.polygon(screen, (0, 0, 0), upper_lid)

    bot_y = (cy + eye_radius) - (eye_radius * 0.75 * close_factor)
    lower_lid = [
        (cx - eye_radius - 20, cy + eye_radius + 20),
        (cx + eye_radius + 20, cy + eye_radius + 20),
        (cx + eye_radius + 20, int(bot_y)),
        (cx - eye_radius - 20, int(bot_y)),
    ]
    pygame.draw.polygon(screen, (0, 0, 0), lower_lid)

    pygame.draw.circle(screen, (0, 0, 0), (cx, cy), eye_radius + 40, 40)

clock = pygame.time.Clock()

try:
    while True:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        face_found = False
        if cap.isOpened():
            success, image = cap.read()
            if success and image is not None:
                img_h, img_w = image.shape[:2]
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                results = detector.detect(mp_image)

                if results.detections:
                    face_found = True
                    bbox = results.detections[0].bounding_box
                    face_x = (bbox.origin_x + (bbox.width / 2.0)) / img_w
                    face_y = (bbox.origin_y + (bbox.height / 2.0)) / img_h

                    norm_x = 1.0 - (face_x * 2.0)
                    norm_y = (face_y * 2.0) - 1.0

                    target_x = max(-max_travel, min(max_travel, norm_x * max_travel * sensitivity))
                    target_y = max(-max_travel, min(max_travel, norm_y * max_travel * sensitivity))
                    target_pupil_scale = 1.25
                    target_squint = 0.22

        if not face_found:
            if now >= next_saccade_time:
                if random.random() < 0.25:
                    target_x, target_y = 0.0, 0.0
                else:
                    angle = random.uniform(0, math.tau)
                    dist = random.uniform(0.2, 0.95) * max_travel
                    target_x = math.cos(angle) * dist
                    target_y = math.sin(angle) * dist * 0.6
                target_pupil_scale = random.uniform(0.75, 1.3)
                target_squint = random.uniform(0.05, 0.3)
                next_saccade_time = now + random.uniform(0.4, 2.6)

        if not is_blinking and now >= next_blink_time:
            is_blinking = True
            blink_start = now

        blink_val = 0.0
        if is_blinking:
            elapsed = now - blink_start
            if elapsed >= blink_duration:
                is_blinking = False
                if random.random() < 0.25:
                    next_blink_time = now + 0.08
                else:
                    next_blink_time = now + random.uniform(2.5, 6.5)
            else:
                blink_val = math.sin((elapsed / blink_duration) * math.pi)

        current_x += (target_x - current_x) * 0.22
        current_y += (target_y - current_y) * 0.22
        pupil_scale += (target_pupil_scale - pupil_scale) * 0.1
        squint_amount += (target_squint - squint_amount) * 0.1

        screen.fill((0, 0, 0))

        draw_eye(left_eye_center, current_x, current_y, True, blink_val, squint_amount, pupil_scale)
        draw_eye(right_eye_center, current_x, current_y, False, blink_val, squint_amount, pupil_scale)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    cap.release()
    detector.close()
    pygame.quit()
    sys.exit()