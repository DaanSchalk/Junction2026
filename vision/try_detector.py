"""Try detector.py on photos or the live camera, and see how fast it runs.  (UNO Q or PC)

  python try_detector.py --images data/raw/images          # every burst in a folder
  python try_detector.py --camera 0                        # take one live burst (5 frames, 0.4 s apart)
  python try_detector.py --camera 0 --loop                 # keep taking bursts
  python try_detector.py --model yolo26n.pt --only cat dog bird --images some/folder
  python try_detector.py --images data/raw/images --save   # also write annotated frames to runs/try/
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2

from common import ROOT, burst_id, list_images
from detector import AnimalDetector


def grab_burst(cap, frames: int, gap: float):
    out, next_shot = [], time.monotonic()
    while len(out) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        if time.monotonic() >= next_shot:
            out.append(frame)
            next_shot += gap
    return out


def report(det: AnimalDetector, label: str, frames, save: bool):
    t0 = time.perf_counter()
    animals = det.analyze_burst(frames)
    ms = (time.perf_counter() - t0) * 1000 / max(len(frames), 1)
    print(f"\n{label}: {len(frames)} frames, {ms:.0f} ms/frame")
    if not animals:
        print("  no animals")
    for a in animals:
        print(f"  {a.name:10} conf {a.confidence:.2f}  seen {a.frames_seen}/{len(frames)}  "
              f"{a.direction:7} {a.approach:11} (dx {a.dx:+.2f}, dy {a.dy:+.2f} body lengths)")
    if save:
        out = ROOT / "runs" / "try"
        out.mkdir(parents=True, exist_ok=True)
        for i, f in enumerate(frames):
            img = cv2.imread(str(f)) if isinstance(f, (str, Path)) else f.copy()
            for n, a in enumerate(animals):
                d = a.boxes[i]
                if d is None:
                    continue
                p1 = (int(d.x - d.w / 2), int(d.y - d.h / 2))
                p2 = (int(d.x + d.w / 2), int(d.y + d.h / 2))
                cv2.rectangle(img, p1, p2, (0, 200, 255), 2)
                cv2.putText(img, f"#{n} {a.name} {a.direction}", (p1[0], max(p1[1] - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.imwrite(str(out / f"{label}_f{i}.jpg"), img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", type=Path, help="folder of photos named like capture.py does")
    src.add_argument("--camera", help="camera index / device / video file")
    ap.add_argument("--model", help="default: model_path from config.yaml")
    ap.add_argument("--only", nargs="*", help="only keep these class names")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--gap", type=float, default=0.4)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    det = AnimalDetector(args.model, conf=args.conf, only=args.only)

    if args.images:
        bursts = defaultdict(list)
        for p in list_images(args.images):
            bursts[burst_id(p)].append(p)
        for name, paths in bursts.items():
            report(det, name, paths, args.save)
        return

    cap = cv2.VideoCapture(int(args.camera) if args.camera.isdigit() else args.camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(15):  # let exposure settle
        cap.read()
    n = 0
    try:
        while True:
            frames = grab_burst(cap, args.frames, args.gap)
            if not frames:
                break
            n += 1
            report(det, f"live{n:03d}", frames, args.save)
            if not args.loop:
                break
    except KeyboardInterrupt:
        pass
    cap.release()


if __name__ == "__main__":
    main()
