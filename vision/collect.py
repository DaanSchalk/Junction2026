"""Browser app to build a labelled dataset of the minis.  (UNO Q or laptop with a camera)

    python collect.py                 # camera 0, http://<this-machine>:8000
    python collect.py --camera 1 --port 8080

Open the address in a browser (laptop or phone on the same network), then:
  1. Clear the scene and press "Set empty background" (B).
  2. Pick the mini (keys 1-4), put it in the scene, hands out of view.
  3. Press "Capture" (space). The box is found automatically by comparing
     with the empty background. If it's wrong, drag a new box over the photo,
     or mark the photo "no animal" / delete it.
  4. Move the mini (other spot, angle, distance) and capture again.
     Re-set the background whenever the light changes.

Photos go to data/raw/images, YOLO labels to data/raw/labels, named like
capture.py does, so split_dataset.py and train.py work on them directly.
"""

import argparse
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_file, send_from_directory

from common import RAW_IMAGES, RAW_LABELS, ROOT, list_images, load_config, write_classes_txt

EMPTY = "empty"  # pseudo-class: photo of the scene without any mini (background example)


# ---- camera ---------------------------------------------------------------

class Camera:
    """Reads frames in a background thread so we always have the newest one."""

    def __init__(self, source: str, width: int, height: int):
        self.source = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera '{source}'. Try --camera 1, or check ls /dev/video*")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame, self.count = None, 0
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if isinstance(self.source, str):  # video file: loop it (handy for testing)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.03)
                continue
            with self.lock:
                self.frame, self.count = frame, self.count + 1

    def latest(self, newer_than: int = -1, timeout: float = 2.0):
        """Newest frame (and its number), waiting for one newer than `newer_than`."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self.lock:
                if self.frame is not None and self.count > newer_than:
                    return self.frame.copy(), self.count
            time.sleep(0.01)
        raise RuntimeError("camera delivers no frames")


# ---- automatic box --------------------------------------------------------

def auto_box(background, frame, min_area=0.002):
    """Box around whatever differs from the empty background, normalised [x1, y1, x2, y2], or None."""
    if background is None or background.shape != frame.shape:
        return None
    a = cv2.GaussianBlur(background, (7, 7), 0)
    b = cv2.GaussianBlur(frame, (7, 7), 0)
    diff = cv2.absdiff(a, b).max(axis=2)              # biggest change over the 3 colour channels
    mask = (diff > 35).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))   # remove speckle noise
    merged = cv2.dilate(mask, np.ones((25, 25), np.uint8))                     # join legs/ears to body
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape
    contours = sorted((c for c in contours if cv2.contourArea(c) > min_area * w * h), key=cv2.contourArea, reverse=True)
    if not contours:
        return None
    if len(contours) > 1 and cv2.contourArea(contours[1]) > 0.3 * cv2.contourArea(contours[0]):
        # changed in two places: a hand in view, or the background wasn't empty. Don't guess.
        return None
    x, y, bw, bh = cv2.boundingRect(contours[0])
    ys, xs = np.nonzero(mask[y:y + bh, x:x + bw])     # tighten to the real (undilated) pixels
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = x + xs.min(), x + xs.max() + 1, y + ys.min(), y + ys.max() + 1
    return [round(float(v), 5) for v in (x1 / w, y1 / h, x2 / w, y2 / h)]


# ---- labels ---------------------------------------------------------------

def write_label(stem: str, cls_id: int | None, box):
    """box = normalised [x1, y1, x2, y2] or None (= empty label file, a background example)."""
    path = RAW_LABELS / f"{stem}.txt"
    if cls_id is None or box is None:
        path.write_text("")
        return
    x1, y1, x2, y2 = box
    path.write_text(f"{cls_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")


def read_label(stem: str, classes):
    path = RAW_LABELS / f"{stem}.txt"
    if not path.exists():
        return None, None
    rows = [r.split() for r in path.read_text().splitlines() if r.strip()]
    if not rows:
        return EMPTY, None
    c, x, y, w, h = int(rows[0][0]), *map(float, rows[0][1:5])
    return classes[c], [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


# ---- web app --------------------------------------------------------------

def create_app(camera: Camera) -> Flask:
    classes = load_config()["classes"]
    write_classes_txt(classes)
    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    state = {"background": None, "burst": 0}
    lock = threading.Lock()
    app = Flask(__name__)

    def info(stem: str) -> dict:
        animal, box = read_label(stem, classes)
        return {"name": stem, "animal": animal, "box": box}

    @app.get("/")
    def index():
        return send_from_directory(ROOT, "collect_ui.html")

    @app.get("/api/config")
    def config():
        return jsonify(classes=classes, empty=EMPTY, has_background=state["background"] is not None)

    @app.get("/stream")
    def stream():
        def gen():
            last = -1
            while True:
                try:
                    frame, last = camera.latest(last)
                except RuntimeError:
                    continue
                small = cv2.resize(frame, (640, int(640 * frame.shape[0] / frame.shape[1])))
                jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(0.08)  # ~12 fps is plenty and keeps the UNO Q CPU free
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/background")
    def set_background():
        state["background"], _ = camera.latest()
        return jsonify(ok=True)

    @app.post("/api/capture")
    def capture():
        body = request.get_json(force=True)
        animal = body.get("animal")
        if animal != EMPTY and animal not in classes:
            abort(400, f"unknown animal {animal!r}")
        frames = max(1, min(int(body.get("frames", 1)), 10))
        gap = max(0.0, min(float(body.get("gap", 0.4)), 5.0))
        with lock:
            state["burst"] += 1
            burst = state["burst"]
        out, last = [], -1
        for i in range(frames):
            if i:
                time.sleep(gap)
            frame, last = camera.latest(last)
            stem = f"{animal}-{session}_b{burst:04d}_f{i}"
            cv2.imwrite(str(RAW_IMAGES / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if animal == EMPTY:
                write_label(stem, None, None)
            else:
                box = auto_box(state["background"], frame)
                if box is None:
                    (RAW_LABELS / f"{stem}.txt").unlink(missing_ok=True)  # no label yet = needs a box
                else:
                    write_label(stem, classes.index(animal), box)
            item = info(stem)
            item["wanted"] = animal
            out.append(item)
        return jsonify(images=out)

    @app.post("/api/label")
    def label():
        body = request.get_json(force=True)
        stem = Path(body["name"]).name
        if not (RAW_IMAGES / f"{stem}.jpg").exists():
            abort(404)
        animal, box = body.get("animal"), body.get("box")
        if animal == EMPTY or box is None:
            write_label(stem, None, None)
        elif animal in classes:
            box = [min(max(float(v), 0.0), 1.0) for v in box]
            write_label(stem, classes.index(animal), box)
        else:
            abort(400, f"unknown animal {animal!r}")
        return jsonify(info(stem))

    @app.post("/api/delete")
    def delete():
        stem = Path(request.get_json(force=True)["name"]).name
        (RAW_IMAGES / f"{stem}.jpg").unlink(missing_ok=True)
        (RAW_LABELS / f"{stem}.txt").unlink(missing_ok=True)
        return jsonify(ok=True)

    @app.get("/api/image/<name>")
    def image(name):
        path = RAW_IMAGES / f"{Path(name).name}.jpg"
        if not path.exists():
            abort(404)
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/api/stats")
    def stats():
        counts = {c: 0 for c in classes} | {EMPTY: 0, "unlabelled": 0}
        for p in list_images(RAW_IMAGES):
            animal, _ = read_label(p.stem, classes)
            counts[animal or "unlabelled"] += 1
        return jsonify(counts)

    @app.get("/api/recent")
    def recent():
        imgs = sorted(list_images(RAW_IMAGES), key=lambda p: p.stat().st_mtime)[-int(request.args.get("n", 12)):]
        return jsonify(images=[info(p.stem) for p in reversed(imgs)])

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index, /dev/videoN or a video file")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--host", default="0.0.0.0", help="0.0.0.0 = reachable from other devices on the network")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    app = create_app(Camera(args.camera, args.width, args.height))
    print(f"\nOpen http://localhost:{args.port}  (or http://<ip-of-this-machine>:{args.port} from another device)\n")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
