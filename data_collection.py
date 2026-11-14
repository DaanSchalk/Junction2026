"""One record per animal encounter: what came, which way it went, what we did, what happened.

Records are appended as one JSON line each to encounters.jsonl, which is easy to
write on the UNO Q and survives a crash halfway through a demo.

Typical use in the main loop (det = vision.detector.AnimalDetector):

    before = det.analyze_burst(frames_before)
    animal = main_animal(before)
    if animal:
        deterrent = pick_deterrent()
        ...fire it, then take frames_after...
        after = det.analyze_burst(frames_after)
        enc = build_encounter(animal, deterrent, after,
                              gone=det.is_gone(frames_after),
                              seconds_until_gone=1.2,
                              images=[...paths of saved frames...])
        EncounterLog().append(enc)

`python data_collection.py` prints a summary of the log: how often each
deterrent made the animal leave.

This file does not import the detector on purpose: it only needs objects with
.name, .confidence, .direction, .approach and .dx (detector.Animal has those).
"""

from __future__ import annotations

import json
import random
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

Animal = Literal["fox", "bear", "wolf", "deer"]  # the 3D-printed minis
Deterrent = Literal["none", "sound", "light"]     # "none" is the control group - keep it
Direction = Literal["left", "right", "up", "down", "still", "unknown"]  # as seen in the image
Approach = Literal["approaching", "leaving", "same", "unknown"]       # box growing / shrinking

# The setup has the critical area on the RIGHT side of the camera view.
# If the camera moves, change this instead of the stored data.
CRITICAL_SIDE: Direction = "right"

LOG_FILE = Path(__file__).resolve().parent / "encounters.jsonl"


@dataclass
class Encounter:
    animal: Animal
    confidence: float                  # detector confidence, 0..1
    direction_before: Direction
    approach_before: Approach
    dx_before: float                   # sideways movement in body lengths (+ = right)
    deterrent: Deterrent
    direction_after: Direction         # "unknown" if it was not seen after the deterrent
    approach_after: Approach
    gone: bool                         # no animal in the last frames after the deterrent
    seconds_until_gone: float | None   # None if it didn't leave
    temperature_c: float | None = None # optional: the UNO Q has no sensor on board
    images: list[str] = field(default_factory=list)  # saved frames -> reuse as training data!
    model: str = ""                    # which model made the detection
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def toward_critical_area(self) -> bool:
        return self.direction_before == CRITICAL_SIDE

    @property
    def turned_around(self) -> bool:
        """Was heading toward the critical area, and went the other way after the deterrent."""
        opposite = {"left": "right", "right": "left", "up": "down", "down": "up"}
        return self.toward_critical_area and self.direction_after == opposite.get(CRITICAL_SIDE)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["toward_critical_area"] = self.toward_critical_area  # stored for convenience, derived on read
        d["turned_around"] = self.turned_around
        return d


# ---- helpers -----------------------------------------------------------

def pick_deterrent(rng: random.Random = random) -> Deterrent:
    """Pick at random, so a comparison between deterrents is fair (no human bias)."""
    return rng.choice(get_args(Deterrent))


def main_animal(animals: list):
    """The animal to react to: the most confident one that we know. None if nothing usable."""
    known = [a for a in animals if a.name in get_args(Animal)]
    return max(known, key=lambda a: (a.frames_seen, a.confidence), default=None)


def build_encounter(before, deterrent: Deterrent, after: list, gone: bool,
                    seconds_until_gone: float | None = None, images: list[str] | None = None,
                    temperature_c: float | None = None, model: str = "") -> Encounter:
    """`before` is one detector.Animal, `after` the analyze_burst() result of the frames after the deterrent."""
    same = [a for a in after if a.name == before.name] or after
    a2 = max(same, key=lambda a: (a.frames_seen, a.confidence), default=None)
    return Encounter(
        animal=before.name,
        confidence=round(before.confidence, 3),
        direction_before=before.direction,
        approach_before=before.approach,
        dx_before=before.dx,
        deterrent=deterrent,
        direction_after=a2.direction if a2 else "unknown",
        approach_after=a2.approach if a2 else "unknown",
        gone=gone,
        seconds_until_gone=seconds_until_gone if gone else None,
        temperature_c=temperature_c,
        images=list(images or []),
        model=model,
    )


class EncounterLog:
    def __init__(self, path: Path | str = LOG_FILE):
        self.path = Path(path)

    def append(self, enc: Encounter) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(enc.to_dict()) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def summary(self) -> str:
        rows = self.read()
        if not rows:
            return f"No encounters in {self.path} yet."
        by_det = defaultdict(list)
        for r in rows:
            by_det[r["deterrent"]].append(r)
        lines = [f"{len(rows)} encounters in {self.path.name}",
                 f"{'deterrent':10} {'n':>4} {'left area':>10} {'turned':>7} {'avg s to leave':>15}"]
        for d in get_args(Deterrent):
            rs = by_det.get(d, [])
            if not rs:
                continue
            gone = [r for r in rs if r["gone"]]
            times = [r["seconds_until_gone"] for r in gone if r["seconds_until_gone"] is not None]
            avg = f"{sum(times) / len(times):.1f}" if times else "-"
            lines.append(f"{d:10} {len(rs):4} {len(gone) / len(rs):10.0%} "
                         f"{sum(r['turned_around'] for r in rs):7} {avg:>15}")
        return "\n".join(lines)


if __name__ == "__main__":
    print(EncounterLog().summary())
