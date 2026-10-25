import math
import random
import time
import pygame


class EyePair:
    def __init__(self, screen, idx, nx, ny, scale, slant):
        self.screen = screen
        self.screen_w, self.screen_h = screen.get_size()
        self.idx = idx
        self.cx = int(self.screen_w * nx)
        self.cy = int(self.screen_h * ny)
        self.base_scale = scale
        self.scale = scale
        self.target_scale = scale
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
        self.alert += (self.target_alert - self.alert) * 0.14
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
        eye_r = int(self.screen_h * 0.19 * self.scale)
        spacing = int(eye_r * 1.35)
        max_shift_x = eye_r * 0.48
        max_shift_y = eye_r * 0.32
        bob_y = int(math.sin(now * 2.2 + self.breath_phase) * (eye_r * 0.05))

        for side in (-1, 1):
            ex = self.cx + (side * spacing)
            ey = self.cy + bob_y
            px = ex + self.curr_x * max_shift_x
            py = ey + self.curr_y * max_shift_y

            steps = 10
            for i in range(steps):
                ratio = i / float(steps - 1)
                r = int(eye_r * (1.0 - ratio * 0.25))
                red = int(165 + 90 * ratio)
                green = int(5 + 18 * ratio)
                pygame.draw.circle(self.screen, (red, green, 12), (ex, ey), r)

            pts = []
            num_verts = 32
            c_rad = eye_r * 0.42
            s_w = eye_r * 0.13
            s_h = eye_r * 0.82
            for k in range(num_verts):
                t = (k / float(num_verts)) * math.tau
                u = math.sin(t)
                c_val = math.cos(t)
                sign_x = 1.0 if c_val >= 0.0 else -1.0

                cw = abs(c_val) * c_rad
                ch = u * c_rad

                sw = (1.0 - (abs(u) ** 1.35)) * s_w
                sh = u * s_h

                vx = px + sign_x * (cw * (1.0 - self.alert) + sw * self.alert)
                vy = py + (ch * (1.0 - self.alert) + sh * self.alert)
                pts.append((int(vx), int(vy)))

            pygame.draw.polygon(self.screen, (0, 0, 0), pts)

            u1 = -0.36
            edge_x1 = (math.sqrt(1.0 - u1 * u1) * c_rad) * (1.0 - self.alert) + ((1.0 - (abs(u1) ** 1.35)) * s_w) * self.alert
            edge_y1 = u1 * (c_rad * (1.0 - self.alert) + s_h * self.alert)
            r1 = max(3, int(eye_r * (0.105 - 0.025 * self.alert)))
            pygame.draw.circle(self.screen, (255, 255, 255), (int(px + edge_x1), int(py + edge_y1)), r1)

            u2 = -0.04
            edge_x2 = (math.sqrt(1.0 - u2 * u2) * c_rad) * (1.0 - self.alert) + ((1.0 - (abs(u2) ** 1.35)) * s_w) * self.alert
            edge_y2 = u2 * (c_rad * (1.0 - self.alert) + s_h * self.alert)
            r2 = max(2, int(eye_r * (0.05 - 0.012 * self.alert)))
            pygame.draw.circle(self.screen, (255, 255, 255), (int(px + edge_x2 + r2 * 0.5), int(py + edge_y2)), r2)

            close_amt = 1.0 - eff_open
            slant_dir = -side
            active_slant = self.slant * (0.45 + 0.70 * self.alert)
            tilt = eye_r * active_slant * slant_dir

            base_top = ey - int(eye_r * 0.22)
            curr_top = base_top + int(eye_r * 1.15 * close_amt)

            upper_mask = [
                (ex - eye_r - 15, ey - eye_r - 15),
                (ex + eye_r + 15, ey - eye_r - 15),
                (ex + eye_r + 15, int(curr_top + tilt)),
                (ex - eye_r - 15, int(curr_top - tilt)),
            ]
            pygame.draw.polygon(self.screen, (0, 0, 0), upper_mask)

            base_bot = ey + int(eye_r * 0.72)
            curr_bot = base_bot - int(eye_r * 0.85 * close_amt)
            bot_tilt = -tilt * 0.35

            lower_mask = [
                (ex - eye_r - 15, ey + eye_r + 15),
                (ex + eye_r + 15, ey + eye_r + 15),
                (ex + eye_r + 15, int(curr_bot + bot_tilt)),
                (ex - eye_r - 15, int(curr_bot - bot_tilt)),
            ]
            pygame.draw.polygon(self.screen, (0, 0, 0), lower_mask)

            ring_thick = max(20, int(eye_r * 0.35))
            pygame.draw.circle(self.screen, (0, 0, 0), (ex, ey), eye_r + ring_thick, ring_thick)


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