"""The one file your own code needs: find animals in a burst and work out where they're going.

    from detector import AnimalDetector

    det = AnimalDetector()                     # loads the model from config.yaml
    animals = det.analyze_burst(frames)        # frames: list of BGR images (cv2) or file paths
    for a in animals:
        print(a.name, a.direction, a.approach) # e.g. "fox right approaching"

    # after the deterrent:
    if det.is_gone(after_frames):
        ...

Before you've trained your own model you can use the stock COCO model:
    det = AnimalDetector("yolo26n.pt", only=["cat", "dog", "bird"])

Directions are in IMAGE terms: left/right/up/down as seen in the picture, plus
"approaching"/"leaving" from the box getting bigger/smaller. How that maps to
the real world depends on where you point the camera.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ultralytics import YOLO

from common import ROOT, load_config


@dataclass
class Detection:
    name: str
    conf: float
    x: float  # box centre, pixels
    y: float
    w: float  # box size, pixels
    h: float
    at_edge: bool = False  # box touches the image border, so it's probably cut off
    position: float = 0.5  # where the box centre is across the view: 0 = left edge, 1 = right edge

    @property
    def size(self) -> float:
        return max(self.w, self.h)


@dataclass
class Animal:
    """One animal followed through a burst."""
    name: str                     # most common class over the burst
    confidence: float             # average detector confidence
    frames_seen: int
    direction: str                # left / right / up / down / still / unknown (seen once)
    approach: str                 # approaching / leaving / same / unknown
    dx: float                     # movement first->last sighting, in body lengths (+ = right)
    dy: float                     # (+ = down)
    position: float = 0.5         # where it was last seen across the view: 0 = left edge, 1 = right edge
    boxes: list[Detection | None] = field(repr=False, default_factory=list)  # one entry per frame


class AnimalDetector:
    def __init__(self, model: str | Path | None = None, conf: float = 0.4, imgsz: int | None = None,
                 only: list[str] | None = None, move_threshold: float = 0.25, max_jump: float = 2.0):
        """
        model           path to weights; default = model_path from config.yaml
        conf            ignore detections below this confidence
        only            keep only these class names (handy with the stock COCO model)
        move_threshold  movement (in body lengths) below this counts as "still"
        max_jump        a box further than this (in body lengths) from the previous one is a different animal
        """
        cfg = load_config()
        model = Path(model or cfg["model_path"])
        if not model.is_absolute() and (ROOT / model).exists():
            model = ROOT / model
        if not model.exists() and model.suffix != ".pt":
            raise FileNotFoundError(f"No model at {model}. Train one (train.py) or use AnimalDetector('yolo26n.pt').")
        self.model = YOLO(str(model), task="detect")
        self.conf = conf
        self.imgsz = imgsz or cfg["imgsz"]
        self.only = set(only) if only else None
        self.move_threshold = move_threshold
        self.max_jump = max_jump

    # ---- single frames -------------------------------------------------

    def detect(self, frame) -> list[Detection]:
        return self.detect_many([frame])[0]

    def detect_many(self, frames, conf: float | None = None) -> list[list[Detection]]:
        """`conf` overrides the confidence cut-off for this call only."""
        conf = self.conf if conf is None else conf
        out = []
        for r in self.model.predict(list(frames), imgsz=self.imgsz, conf=conf, verbose=False, stream=True):
            H, W = r.orig_shape
            dets = []
            for c, p, (x, y, w, h) in zip(r.boxes.cls, r.boxes.conf, r.boxes.xywh.tolist()):
                edge = x - w / 2 <= 2 or y - h / 2 <= 2 or x + w / 2 >= W - 2 or y + h / 2 >= H - 2
                dets.append(Detection(r.names[int(c)], float(p), x, y, w, h, edge, x / W))
            dets = [d for d in dets if self.only is None or d.name in self.only]
            out.append(self._drop_duplicates(dets))
        return out

    @staticmethod
    def _drop_duplicates(dets: list[Detection]) -> list[Detection]:
        """The model sometimes puts two boxes with different names on the same animal
        (fox AND wolf). Keep only the more confident one, or it would count as two animals."""
        kept: list[Detection] = []
        for d in sorted(dets, key=lambda d: -d.conf):
            for k in kept:
                ix = max(0.0, min(d.x + d.w / 2, k.x + k.w / 2) - max(d.x - d.w / 2, k.x - k.w / 2))
                iy = max(0.0, min(d.y + d.h / 2, k.y + k.h / 2) - max(d.y - d.h / 2, k.y - k.h / 2))
                if ix * iy > 0.6 * min(d.w * d.h, k.w * k.h):
                    break
            else:
                kept.append(d)
        return kept

    # ---- bursts --------------------------------------------------------

    def analyze_burst(self, frames, conf: float | None = None) -> list[Animal]:
        """Detect in every frame, link detections into animals, and compute their movement.
        `conf` overrides the confidence cut-off for this call, e.g. lower while you are following
        an animal you already identified: a weak sighting still tells you it hasn't left."""
        per_frame = self.detect_many(frames, conf)
        tracks: list[list[Detection | None]] = []

        for i, dets in enumerate(per_frame):
            # pair each detection with the nearest existing track (greedy, closest pairs first)
            pairs = []
            for ti, t in enumerate(tracks):
                last = next(d for d in reversed(t) if d is not None)
                for di, d in enumerate(dets):
                    dist = ((d.x - last.x) ** 2 + (d.y - last.y) ** 2) ** 0.5 / max(last.size, 1.0)
                    # same animal = close by AND roughly the same size (unless cut off by the border);
                    # otherwise a leaving animal gets linked to some other object in view
                    similar = d.at_edge or last.at_edge or 0.5 <= d.size / max(last.size, 1.0) <= 2.0
                    if dist <= self.max_jump and similar:
                        pairs.append((dist, ti, di))
            used_t, used_d = set(), set()
            for _, ti, di in sorted(pairs):
                if ti not in used_t and di not in used_d:
                    tracks[ti].append(dets[di])
                    used_t.add(ti)
                    used_d.add(di)
            for ti, t in enumerate(tracks):
                if ti not in used_t:
                    t.append(None)  # missed in this frame
            for di, d in enumerate(dets):
                if di not in used_d:
                    tracks.append([None] * i + [d])

        return [self._summarise(t) for t in tracks]

    def is_gone(self, frames, empty_in_a_row: int = 2) -> bool:
        """True if the LAST `empty_in_a_row` frames contain no animal (one empty frame could be a miss)."""
        per_frame = self.detect_many(frames)
        tail = per_frame[-empty_in_a_row:]
        return len(tail) == empty_in_a_row and not any(tail)

    # ---- helpers -------------------------------------------------------

    def _summarise(self, track: list[Detection | None]) -> Animal:
        seen = [d for d in track if d is not None]
        # the name with the most total confidence wins (sorted first, so ties never depend on chance)
        name = max(sorted({d.name for d in seen}), key=lambda n: sum(d.conf for d in seen if d.name == n))
        conf = sum(d.conf for d in seen) / len(seen)
        if len(seen) < 2:
            return Animal(name, conf, 1, "unknown", "unknown", 0.0, 0.0, seen[0].position, track)

        first, last = seen[0], seen[-1]
        body = sum(d.size for d in seen) / len(seen)
        dx, dy = (last.x - first.x) / body, (last.y - first.y) / body
        if max(abs(dx), abs(dy)) < self.move_threshold:
            direction = "still"
        elif abs(dx) >= abs(dy):
            direction = "right" if dx > 0 else "left"
        else:
            direction = "down" if dy > 0 else "up"

        # compare box sizes, skipping boxes cut off by the image border (they look smaller than they are)
        whole = [d for d in seen if not d.at_edge]
        if len(whole) >= 2:
            growth = (whole[-1].w * whole[-1].h) / max(whole[0].w * whole[0].h, 1.0)
            approach = "approaching" if growth > 1.25 else "leaving" if growth < 0.8 else "same"
        else:
            approach = "unknown"
        return Animal(name, conf, len(seen), direction, approach, round(dx, 2), round(dy, 2), last.position, track)

