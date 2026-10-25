import os
import sys
import tempfile
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

eye_radius = int(screen_h * 0.25)
pupil_radius = int(eye_radius * 0.4)
max_travel = eye_radius - pupil_radius - 10

left_eye_center = (int(screen_w * 0.35), int(screen_h * 0.5))
right_eye_center = (int(screen_w * 0.65), int(screen_h * 0.5))

cap = cv2.VideoCapture(0)

sensitivity = 1.5
current_x, current_y = 0.0, 0.0
target_x, target_y = 0.0, 0.0
clock = pygame.time.Clock()

try:
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        success, image = cap.read()
        if success and image is not None:
            img_h, img_w = image.shape[:2]
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            results = detector.detect(mp_image)

            if results.detections:
                bbox = results.detections[0].bounding_box
                face_x = (bbox.origin_x + (bbox.width / 2.0)) / img_w
                face_y = (bbox.origin_y + (bbox.height / 2.0)) / img_h

                norm_x = 1.0 - (face_x * 2.0)
                norm_y = (face_y * 2.0) - 1.0

                target_x = max(-max_travel, min(max_travel, norm_x * max_travel * sensitivity))
                target_y = max(-max_travel, min(max_travel, norm_y * max_travel * sensitivity))
            else:
                target_x, target_y = 0.0, 0.0

        current_x += (target_x - current_x) * 0.15
        current_y += (target_y - current_y) * 0.15

        screen.fill((0, 0, 0))

        pygame.draw.circle(screen, (255, 0, 255), left_eye_center, eye_radius)
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (left_eye_center[0] + int(current_x), left_eye_center[1] + int(current_y)),
            pupil_radius,
        )

        pygame.draw.circle(screen, (255, 0, 0), right_eye_center, eye_radius)
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (right_eye_center[0] + int(current_x), right_eye_center[1] + int(current_y)),
            pupil_radius,
        )

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    cap.release()
    detector.close()
    pygame.quit()
    sys.exit()