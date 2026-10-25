# 👁️ SpookySpy

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-60_FPS-crimson?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-EfficientDet--Lite2-orange?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-720p_Vision-green?logo=opencv&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows_Mini_PC-0078D6?logo=windows&logoColor=white)

**SpookySpy** is an interactive, AI-powered Halloween window projection system. It renders giant, glowing crimson eyes at 60 FPS that scan the yard in the dark and dynamically lock onto trick-or-treaters walking down the sidewalk—morphing their pupils into razor-sharp cat slits and waking up additional lurking eyes when crowds approach.

---

## ✨ Features

* **🩸 Procedural 60 FPS Vector Eyes:** Pure-black background disappears completely in a window projection, leaving only glowing crimson gradients, eyelid masks, autonomous blinking, natural saccades, and subtle sine-wave breathing motion.
* **🐱 Predator Lock-On Morphing:** Eyes patrol in an idle state with relaxed brows and round pupils. The instant a person is detected, the pupils smoothly constrict into vertical cat slits while the upper eyelids tilt down into an aggressive scowl.
* **👥 Multi-Person Crowd Spawning:** Tracks up to **5 people simultaneously**. A single visitor gets the giant center pair of eyes; as more trick-or-treaters enter the frame, up to 4 additional pairs of lurking eyes blink open in the surrounding darkness to track each person independently.
* **🎯 Sticky Target Matching:** Uses frame-to-frame Euclidean gaze matching so multiple eye pairs stay locked onto their assigned person without jittering or swapping targets when people walk past each other.
* **🔭 Long-Range Costume-Proof Vision:** Runs MediaPipe's `efficientdet_lite2` full-body person detector at 1280x720 in a dedicated background thread—reliably spotting silhouettes from 30–50+ feet away regardless of Halloween masks, costumes, or viewing angle.
* **📱 Live Phone Aiming Server:** Built-in multithreaded MJPEG web server on port `8080` streams the live camera view with bounding boxes and target reticles directly to a smartphone browser for easy sidewalk calibration.
* **🔄 Zero-Write Hot Reloading:** A lightweight `watch.py` daemon polls `git ls-remote` in memory every 30 seconds and automatically pulls and restarts the projection whenever changes are pushed to GitHub.

---

## 🗂️ Architecture

```text
spookyspy/
├── eyes.py        # Main 60 FPS Pygame loop, eye slot layout, and tuning constants
├── renderer.py    # EyePair procedural rendering, pupil morph math, and sticky target matching
├── vision.py      # Threaded 720p OpenCV capture, MediaPipe person detection, and :8080 web stream
└── watch.py       # Headless background Git poller and process supervisor
```

---

## ⚙️ Configuration

All field-tuning settings live at the top of **`eyes.py`**:

| Constant | Default | Description |
| :--- | :--- | :--- |
| `MIRROR_X` | `True` | Flips horizontal tracking & preview stream to match rear/front window projection. |
| `SOLO_CROWD_TEST` | `False` | Simulates 3 people when 1 person is detected so you can test multi-eye spawning solo. |
| `SENSITIVITY` | `1.4` | Multiplier for how far the pupils travel toward the edges of the eye sockets. |
| `STREAM_PORT` | `8080` | Port for the live mobile aiming web server (`http://<MINI_PC_IP>:8080`). |
| `SLOT_CONFIGS` | `5 slots` | Tuples of `(norm_x, norm_y, scale, brow_slant)` defining each eye pair's position. |

---

## 🛠️ Hardware Setup

* **Compute:** Windows Mini PC running headless at `C:\spookyspy`
* **Camera:** Anker PowerConf C200 webcam mounted flush against the window glass to eliminate interior projector glare
* **Display:** Window rear-projection screen driven via long-throw HDMI

---

## 🚀 Deployment & Remote Operations

The mini PC boots straight into the projection via a Windows Scheduled Task (`spookyspy`) triggered on user logon.

### Mobile Aiming View
Stand out on the sidewalk and open your phone browser to align the camera:
```text
[http://192.168.1.251:8080](http://192.168.1.251:8080)
```
* **Red Box:** Detected person silhouette
* **Green Dot:** Active upper-torso/head gaze target (`22%` down from top of bounding box)

### Global Windows SSH Commands
SSH into the mini PC (`ssh k@192.168.1.251`) and run:

* **`sync`** — Immediately pulls the latest code from GitHub and restarts `watch.py` and `eyes.py`.
* **`logs`** — Prints the latest runtime output and any Python tracebacks.

### Power & Process Control via SSH

Stop the projection without turning off the PC:
```cmd
taskkill /IM python.exe /F
```

Reboot the mini PC cleanly:
```cmd
shutdown /r /t 0
```

Power down the mini PC completely:
```cmd
shutdown /s /t 0
```