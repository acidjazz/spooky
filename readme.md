# 👀 spookyspy

Giant, unblinking eyes projected onto the front window that lock onto trick-or-treaters and follow them down the sidewalk. 

Built to run headless on a dedicated mini PC connected to a projector and a window-mounted webcam.

## How It Works

* **The Gaze (`pygame`):** Renders fullscreen high-contrast eyes in the dark so only the sclera and pupils shine through the window.
* **The Brain (`mediapipe` + `opencv`):** Scans the street at 60 FPS, locks onto the nearest face, and smoothly interpolates pupil movement to track passersby in real time.
* **The Ghost in the Machine:** Runs automatically on boot via Windows Task Scheduler (`spookyspy`) to hijack the physical display straight from a remote SSH session.

## Stack

* Python 3.11
* OpenCV (`opencv-python`)
* MediaPipe (`mediapipe`)
* Pygame (`pygame`)
