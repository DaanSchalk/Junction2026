"""A made-up site, so the whole loop can run with no camera, no model and no minis.

SimWorld plays both the camera and the hardware for run.Pipeline: animals wander
into view, the real decision engine picks a response, and the animals react
according to the hidden rules below. Time is virtual, so two weeks take a second.

    python run.py --simulate 20                       # watch it
    python mock_data/make_mock_encounters.py          # write encounters_mock.jsonl

The hidden rules (so you can check that the engine and your dashboard find them):
  * animals come mostly at dusk, at night and at dawn; deer are the most common, bears the rarest
  * sound works very well on deer at first, but every time they hear it they care a bit less
  * light works much better in the dark than in daylight, and best on bears
  * with no deterrent, some animals still turn around on their own (the baseline to beat)
  * a little mess: unsure detections, animals seen in one frame only, animals lost from sight
Everything here is invented. It says nothing about real animals.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from settings import Settings

LOCAL = timezone(timedelta(hours=2))  # local time of the imaginary site

ANIMAL_WEIGHTS = {"deer": 45, "fox": 30, "wolf": 15, "bear": 10}
# Chance the animal turns around and leaves, before the modifiers below.
LEAVE = {
    "none":  {"deer": 0.22, "fox": 0.20, "wolf": 0.28, "bear": 0.15},
    "sound": {"deer": 0.88, "fox": 0.55, "wolf": 0.70, "bear": 0.45},
    "light": {"deer": 0.62, "fox": 0.35, "wolf": 0.55, "bear": 0.70},
}
SOUND_FATIGUE = {"deer": 0.975}   # each sound a deer hears multiplies its effect by this...
SOUND_FLOOR = 0.35                # ...down to this share of the original
# hour of day (local) -> relative activity: dusk, night, dawn
HOUR_WEIGHTS = [6, 6, 5, 5, 5, 6, 8, 7, 3, 1, 1, 1, 1, 1, 2, 5, 9, 10, 8, 7, 6, 6, 6, 6]
BODY = 0.15                       # body length as a share of the view's width


def is_dark(hour: int) -> bool:
    return hour < 8 or hour >= 16  # November, far north


@dataclass
class Sighting:
    """What the detector would report for one animal in one burst (same fields as detector.Animal)."""
    name: str
    confidence: float
    frames_seen: int
    direction: str
    approach: str
    dx: float
    dy: float
    position: float


class SimWorld:
    model_name = "simulation"

    def __init__(self, settings: Settings, visits: int = 20, seed: int | None = None,
                 start: str = "2026-11-01", days: int = 14):
        self.s = settings
        self.rng = random.Random(seed)
        self.id_rng = random.Random(seed)  # ids come from the seed too, so the same seed gives the same file
        first_day = datetime.fromisoformat(start).replace(tzinfo=LOCAL)
        arrivals = []
        for _ in range(visits):
            hour = self.rng.choices(range(24), HOUR_WEIGHTS)[0]
            arrivals.append(first_day + timedelta(days=self.rng.randrange(days), hours=hour,
                                                  minutes=self.rng.randrange(60), seconds=self.rng.randrange(60)))
        self.arrivals = sorted(arrivals)
        self.days = days
        self.first_day = first_day
        self.clock = first_day
        self.sounds_heard = {a: 0 for a in ANIMAL_WEIGHTS}
        self.v = None  # the current visitor

    # ---- the camera's part ---------------------------------------------

    def now(self) -> float:
        return self.clock.timestamp()

    def timestamp(self) -> str:
        return self.clock.astimezone(timezone.utc).isoformat(timespec="seconds")

    def wait(self, seconds: float) -> None:
        if self.v and not self.v["reacted"]:
            self._react("none")          # nothing was fired: it does what it would have done anyway
        self._advance(seconds)

    def wait_for_trigger(self) -> bool:
        if not self.arrivals:
            return False  # the simulated period is over
        self.clock = max(self.clock + timedelta(seconds=30), self.arrivals.pop(0))
        self._new_visitor()
        return True

    def observe(self, following: bool = False) -> list[Sighting]:
        v, s = self.v, self.s
        duration = (s.burst_frames - 1) * s.burst_gap_s
        x0 = v["x"]
        self._advance(duration)
        if not v["visible"] or not 0 <= v["x"] <= 1:
            return []
        moved = (v["x"] - x0) / BODY                       # in body lengths, + = toward the protected area
        if abs(moved) < 0.25:
            direction = v["idle_direction"]
        else:
            direction = s.critical_side if moved > 0 else s.away_side
        seen_once = self.rng.random() < 0.02
        dx = moved if s.critical_side == "right" else -moved
        position = v["x"] if s.critical_side == "right" else 1 - v["x"]
        return [Sighting(
            name=v["animal"],
            confidence=v["confidence"],
            frames_seen=1 if seen_once else s.burst_frames,
            direction="unknown" if seen_once else direction,
            approach="unknown" if seen_once else v["approach"],
            dx=0.0 if seen_once else round(dx, 2),
            dy=0.0,
            position=round(min(max(position, 0.0), 1.0), 3),
        )]

    def new_id(self) -> str:
        return f"{self.id_rng.getrandbits(48):012x}"

    def save_last(self, encounter_id: str, prefix: str, single: bool = False) -> list[str]:
        return []  # no photos in a simulation

    # ---- the hardware's part -------------------------------------------

    def fire(self, deterrent: str) -> None:
        self._react(deterrent)

    def temperature(self) -> float | None:
        if self.rng.random() < 0.03:
            return None  # sensor hiccup
        progress = (self.clock - self.first_day).days / max(self.days - 1, 1)
        return round(self.rng.gauss(-1.5 - 3 * progress - (2 if is_dark(self.clock.hour) else 0), 2.0), 1)

    # ---- the animals ---------------------------------------------------

    def _new_visitor(self) -> None:
        rng = self.rng
        animal = rng.choices(list(ANIMAL_WEIGHTS), ANIMAL_WEIGHTS.values())[0]
        heading = rng.choices(["toward", "away", "still", "up", "down"], [55, 28, 9, 4, 4])[0]
        speed = rng.uniform(0.5, 1.3) * BODY               # share of the view per second
        confidence = min(0.97, max(0.4, rng.gauss(0.8, 0.09)))
        if rng.random() < 0.03:
            confidence = rng.uniform(0.40, 0.55)           # an unsure detection
        self.v = {
            "animal": animal,
            "confidence": round(confidence, 3),
            "x": rng.uniform(0.12, 0.55) if heading == "toward" else rng.uniform(0.2, 0.7),
            "vx": {"toward": speed, "away": -speed}.get(heading, 0.0),
            "idle_direction": heading if heading in ("up", "down") else "still",
            "approach": rng.choices(["approaching", "same", "leaving"], [45, 40, 15])[0],
            "visible": True,
            "reacted": False,
            "plan": [],   # (seconds from now, new vx, becomes visible?) changes still to come
        }
        if rng.random() < 0.04:                            # already grazing inside the protected area
            self.v.update(x=rng.uniform(1 - self.s.critical_width + 0.02, 0.95), vx=0.0, idle_direction="still")

    def _react(self, deterrent: str) -> None:
        """Decide, once, what this animal does after the response (or the lack of one)."""
        v, rng = self.v, self.rng
        v["reacted"] = True
        animal, hour = v["animal"], self.clock.hour
        p = LEAVE.get(deterrent, LEAVE["sound"])[animal]
        if deterrent == "sound":
            fatigue = SOUND_FATIGUE.get(animal, 1.0) ** self.sounds_heard[animal]
            p *= max(SOUND_FLOOR, fatigue)
            self.sounds_heard[animal] += 1
        if deterrent == "light":
            p *= 1.15 if is_dark(hour) else 0.45           # hardly noticed in daylight
        fast = deterrent != "none"

        if rng.random() < 0.04:                            # stops, then walks behind something: lost from sight
            v["vx"] = 0.0
            v["plan"].append((rng.uniform(1, 5), 0.0, False))
        elif rng.random() < min(p, 0.97):                  # freezes, then turns around and leaves
            hesitation = rng.uniform(0, 2) if fast else rng.uniform(2, 8)
            flee = (rng.uniform(1.2, 2.5) if fast else rng.uniform(0.5, 0.9)) * BODY
            v["vx"] = 0.0
            v["plan"].append((hesitation, -flee, True))
        elif rng.random() < 0.65:                          # carries on into the protected area
            v["vx"] = max(v["vx"], rng.uniform(0.4, 1.0) * BODY)
        else:                                              # hangs around
            v["plan"].append((rng.uniform(0, 3), 0.0, True))
        v["approach"] = "leaving" if v["plan"] and v["plan"][-1][1] < 0 else rng.choice(["approaching", "same", "same"])

    def _advance(self, seconds: float) -> None:
        """Move virtual time forward, moving the animal along and applying planned changes."""
        v = self.v
        left = seconds
        while v and left > 0:
            step = min(left, v["plan"][0][0]) if v["plan"] else left
            v["x"] += v["vx"] * step
            left -= step
            if v["plan"]:
                wait, vx, visible = v["plan"][0]
                if step >= wait:
                    v["plan"].pop(0)
                    v["vx"], v["visible"] = vx, visible
                else:
                    v["plan"][0] = (wait - step, vx, visible)
        self.clock += timedelta(seconds=seconds)
