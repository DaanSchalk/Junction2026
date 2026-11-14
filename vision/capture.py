"""Collect training photos with your own camera. Runs on the UNO Q (or any laptop).

Two modes:

  motion   (default) waits for movement, then takes a burst of frames, the same
           way your deterrent will. These are the photos worth labelling.
  interval one photo every N seconds, motion or not. Use this to collect empty
           "background" shots (rain, wind in plants, night) so the model learns
           what is NOT an animal.

Examples:
  python capture.py                                  # USB camera 0, motion bursts
  python capture.py --frames 5 --gap 0.4             # bursts like the real deterrent
  python capture.py --mode interval --every 60       # one photo a minute
  python capture.py --camera test.mp4                # try it on a video file

Photos land in data/raw/images as  <session>_b<burst>_f<frame>.jpg
"""

import argparse
import time
from datetime import datetime

import cv2

from common import RAW_IMAGES


def open_camera(source: str, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera '{source}'. Try --camera 1, or check ls /dev/video*")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # we want the newest frame, not a queued old one
    return cap


class MotionDetector:
    """Cheap frame-difference motion check. Good enough to decide *when* to take photos."""

    def __init__(self, threshold: float):
        self.threshold = threshold  # fraction of pixels that must change
        self.background = None

    def __call__(self, frame) -> float:
        small = cv2.resize(frame, (160, 120))
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype("float32")
        if self.background is None:
            self.background = gray
            return 0.0
        diff = cv2.absdiff(gray, self.background)
        cv2.accumulateWeighted(gray, self.background, 0.05)  # slowly adapt to light changes
        return float((diff > 25).mean())

    def triggered(self, frame) -> bool:
        return self(frame) > self.threshold


def save(frame, session: str, burst: int, idx: int) -> None:
    path = RAW_IMAGES / f"{session}_b{burst:04d}_f{idx}.jpg"
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])


def take_burst(cap, session, burst, frames, gap) -> int:
    """Take `frames` photos `gap` seconds apart. Keeps reading in between so frames stay fresh."""
    saved, next_shot = 0, time.monotonic()
    while saved < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if time.monotonic() >= next_shot:
            save(frame, session, burst, saved)
            saved += 1
            next_shot += gap
    return saved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", default="0", help="camera index, /dev/videoN, video file or stream URL")
    ap.add_argument("--mode", choices=["motion", "interval"], default="motion")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--frames", type=int, default=5, help="frames per burst (motion mode)")
    ap.add_argument("--gap", type=float, default=0.4, help="seconds between burst frames")
    ap.add_argument("--cooldown", type=float, default=3.0, help="seconds to wait after a burst")
    ap.add_argument("--sensitivity", type=float, default=0.01,
                    help="fraction of the image that must change to count as motion (raise if wind triggers it)")
    ap.add_argument("--every", type=float, default=30.0, help="seconds between photos (interval mode)")
    ap.add_argument("--max", type=int, default=0, help="stop after this many bursts/photos (0 = run forever)")
    args = ap.parse_args()

    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height)
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    motion = MotionDetector(args.sensitivity)
    count, last_event = 0, 0.0

    # let auto-exposure settle and build a background
    for _ in range(15):
        ok, frame = cap.read()
        if ok:
            motion(frame)

    print(f"Capturing ({args.mode}) to {RAW_IMAGES}  -  Ctrl+C to stop")
    try:
        while not args.max or count < args.max:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped delivering frames.")
                break
            now = time.monotonic()

            if args.mode == "interval":
                if now - last_event >= args.every:
                    count += 1
                    save(frame, session, count, 0)
                    last_event = now
                    print(f"photo {count}")
                continue

            if motion.triggered(frame) and now - last_event >= args.cooldown:
                count += 1
                n = take_burst(cap, session, count, args.frames, args.gap)
                last_event = time.monotonic()
                print(f"burst {count}: {n} frames")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
    print(f"Done: {count} {'bursts' if args.mode == 'motion' else 'photos'} in {RAW_IMAGES}")


if __name__ == "__main__":
    main()
