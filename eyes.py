import sys
import time
import pygame
from renderer import EyePair, match_sticky_targets
from vision import VisionTracker

MIRROR_X = False
SOLO_CROWD_TEST = False
SENSITIVITY = 2.1
X_OFFSET = -0.08
Y_OFFSET = 0.30
MIN_SCORE = 0.40
STREAM_PORT = 8080

SLOT_CONFIGS = [
    (0.50, 0.44, 1.05, 0.40),
    (0.23, 0.22, 0.52, 0.42),
    (0.77, 0.22, 0.54, 0.36),
    (0.24, 0.70, 0.50, 0.44),
    (0.76, 0.70, 0.52, 0.38),
]

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
        active_crowd = len(targets) if not idle_mode else 1 + sum(1 for p in pairs[1:] if p.openness > 0.2)

        for i, pair in enumerate(pairs):
            pair.update(now, assigned_list[i], active_crowd, idle_mode)

        pygame.display.flip()
        clock.tick(60)

except KeyboardInterrupt:
    vision.stop()
    pygame.quit()
    sys.exit()