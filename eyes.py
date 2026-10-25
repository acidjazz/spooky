import sys
import cv2
import mediapipe as mp
import pygame

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_w, screen_h = screen.get_size()
pygame.mouse.set_visible(False)

eye_radius = int(screen_h * 0.25)
pupil_radius = int(eye_radius * 0.4)
max_travel = eye_radius - pupil_radius - 10

left_eye_center = (int(screen_w * 0.35), int(screen_h * 0.5))
right_eye_center = (int(screen_w * 0.65), int(screen_h * 0.5))

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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
        if success:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image_rgb)

            if results.detections:
                bbox = results.detections[0].location_data.relative_bounding_box
                face_x = bbox.xmin + (bbox.width / 2)
                face_y = bbox.ymin + (bbox.height / 2)

                norm_x = 1.0 - (face_x * 2)
                norm_y = (face_y * 2) - 1.0

                target_x = max(-max_travel, min(max_travel, norm_x * max_travel * sensitivity))
                target_y = max(-max_travel, min(max_travel, norm_y * max_travel * sensitivity))
            else:
                target_x, target_y = 0.0, 0.0

        current_x += (target_x - current_x) * 0.15
        current_y += (target_y - current_y) * 0.15

        screen.fill((0, 0, 0))

        pygame.draw.circle(screen, (255, 255, 255), left_eye_center, eye_radius)
        pygame.draw.circle(
            screen,
            (0, 0, 0),
            (left_eye_center[0] + int(current_x), left_eye_center[1] + int(current_y)),
            pupil_radius,
        )

        pygame.draw.circle(screen, (255, 255, 255), right_eye_center, eye_radius)
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
    pygame.quit()
    sys.exit()
