import random
import sys
import time
import pygame
from renderer import EyePair, match_sticky_targets
from vision import VisionTracker

MIRROR_X = False
SOLO_CROWD_TEST = False
SENSITIVITY = 1.15
X_OFFSET = -0.09
Y_OFFSET = 0.30
MIN_SCORE = 0.25
STREAM_PORT = 8080

SLOT_CONFIGS = [
    (0.50, 0.44, 1.05, 0.40),
    (0.23, 0.22, 0.52, 0.42),
    (0.77, 0.22, 0.54, 0.36),
    (0.24, 0.70, 0.50, 0.44),
    (0.76, 0.70, 0.52, 0.38),
]

DETECTED_LAYOUTS = {
    1: [
        (0.50, 0.46, 1.45),
    ],
    2: [
        (0.27, 0.44, 0.84),
        (0.73, 0.48, 0.84),
    ],
    3: [
        (0.50, 0.64, 0.78),
        (0.26, 0.28, 0.72),
        (0.74, 0.28, 0.72),
    ],
    4: [
        (0.27, 0.68, 0.68),
        (0.26, 0.26, 0.68),
        (0.74, 0.26, 0.68),
        (0.73, 0.68, 0.68),
    ],
    5: [
        (0.50, 0.46, 0.85),
        (0.23, 0.22, 0.58),
        (0.77, 0.22, 0.58),
        (0.24, 0.72, 0.56),
        (0.76, 0.72, 0.56),
    ],
}

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
pygame.mouse.set_visible(False)

pairs = [EyePair(screen, i, *cfg) for i, cfg in enumerate(SLOT_CONFIGS)]
vision = VisionTracker(
    max_targets=len(pairs),
    mirror_x=MIRROR_X,
    solo_test=SOLO_CROWD_TEST,
    sensitivity=SENSITIVITY,
    x_offset=X_OFFSET,
    y_offset=Y_OFFSET,
    min_score=MIN_SCORE,
    port=STREAM_PORT,
)

clock = pygame.time.Clock()

try:
    while True:
        now = time.time()
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                raise KeyboardInterrupt

        targets = vision.get_targets()
        assigned_list = match_sticky_targets(pairs, targets)

        screen.fill((0, 0, 0))
        idle_mode = len(targets) == 0

        if idle_mode:
            awake_others = [p for p in pairs[1:] if now < p.last_active]
            if len(awake_others) < 2:
                sleeping = [p for p in pairs[1:] if now >= p.last_active]
                sleeping.sort(key=lambda p: p.openness, reverse=True)
                for p in sleeping[: 2 - len(awake_others)]:
                    p.last_active = now + random.uniform(6.0, 14.0)
                    p.target_openness = 1.0

            for i, pair in enumerate(pairs):
                nx, ny, sc, _ = SLOT_CONFIGS[i]
                pair.set_layout(nx, ny, sc)

            active_crowd = 1 + sum(1 for p in pairs[1:] if p.openness > 0.2)
        else:
            active_crowd = min(len(pairs), len(targets))
            layout = DETECTED_LAYOUTS.get(active_crowd, DETECTED_LAYOUTS[5])
            for i, pair in enumerate(pairs):
                if i < len(layout):
                    nx, ny, sc = layout[i]
                    pair.set_layout(nx, ny, sc)
                else:
                    nx, ny, sc, _ = SLOT_CONFIGS[i]
                    pair.set_layout(nx, ny, sc)

        for i, pair in enumerate(pairs):
            pair.update(now, assigned_list[i], active_crowd, idle_mode)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    vision.stop()
    pygame.quit()
    sys.exit()