import subprocess
import time

REPO = r"C:\spookyspy"
PY = r"C:\Users\k\AppData\Local\Programs\Python\Python311\python.exe"
LOG = r"C:\spookyspy\spooky.log"
NO_WIN = subprocess.CREATE_NO_WINDOW

def git(*args):
    return subprocess.check_output(
        ["git", "-C", REPO, *args], creationflags=NO_WIN
    ).decode().strip()

def start_eyes():
    log_file = open(LOG, "w")
    return subprocess.Popen(
        [PY, "-u", r"C:\spookyspy\eyes.py"],
        cwd=REPO,
        stdout=log_file,
        stderr=log_file,
        creationflags=NO_WIN,
    )

proc = start_eyes()
last_hash = git("rev-parse", "HEAD")

while True:
    time.sleep(10)
    try:
        git("fetch", "origin", "main")
        remote_hash = git("rev-parse", "origin/main")
        if remote_hash != last_hash:
            if proc.poll() is None:
                proc.terminate()
                proc.wait(timeout=5)
            git("pull")
            last_hash = remote_hash
            proc = start_eyes()
    except Exception as e:
        with open(LOG, "a") as f:
            f.write(f"\nWatcher error: {e}\n")