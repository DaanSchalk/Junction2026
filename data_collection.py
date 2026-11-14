"""One record per animal encounter: what came, which way it went, what we did, what happened.

Records are appended as one JSON line each to encounters.jsonl, which is easy to
write on the UNO Q and survives a crash halfway through a demo.

run.py does all of this for you. By hand it looks like this:

    decision = engine.decide(animal.name, animal.confidence, animal.direction, animal.position)
    ...respond, then keep checking until the animal is gone -> a FollowUp...
    enc = build_encounter(animal, decision, follow_up, images=[...])
    EncounterLog().append(enc)
    engine.learn(enc)

`python data_collection.py` prints a summary of the log: how each response did.
`python data_collection.py encounters_mock.jsonl` does the same for the mock data.

This file does not import the detector on purpose: it only needs objects with
.name, .confidence, .frames_seen, .direction, .approach, .dx and .position
(detector.Animal has those).
"""

from __future__ import annotations

import json
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from settings import OPPOSITE, Settings, load_settings

SETTINGS = load_settings()

Direction = Literal["left", "right", "up", "down", "still", "unknown"]  # as seen in the image
Approach = Literal["approaching", "leaving", "same", "unknown"]       # box growing / shrinking
# What happened after we responded (or deliberately didn't):
#   left          gone from view, away from the protected area   -> the response worked
#   entered       went into the protected area                   -> it didn't
#   stayed        still in view when we stopped checking          -> it didn't
#   unknown       lost sight of it, can't tell where it went      -> not counted
#   not_a_threat  we never followed it (wrong animal for the scenario, unsure, or not heading our way)
Outcome = Literal["left", "entered", "stayed", "unknown", "not_a_threat"]

# Which side of the camera view the protected area is on (settings.yaml). Each record also
# stores the side that was in force when it was made, so moving the camera never rewrites history.
CRITICAL_SIDE: Direction = SETTINGS.critical_side

LOG_FILE = Path(__file__).resolve().parent / "encounters.jsonl"
MOCK_LOG_FILE = Path(__file__).resolve().parent / "encounters_mock.jsonl"  # invented data, never mix the two


@dataclass
class FollowUp:
    """What the repeated checks after the response found."""
    outcome: Outcome = "not_a_threat"
    gone: bool = False                       # no longer in view when we stopped checking
    seconds_until_gone: float | None = None  # from response to the first check without the animal
    checks: int = 0                          # how many times we looked again
    direction_after: Direction = "unknown"   # from the last time we still saw it
    approach_after: Approach = "unknown"
    images: list[str] = field(default_factory=list)


@dataclass
class Encounter:
    animal: str
    confidence: float                  # detector confidence, 0..1
    direction_before: Direction
    approach_before: Approach
    dx_before: float                   # sideways movement in body lengths (+ = right)
    position_before: float | None      # where in the view it was: 0 = left edge, 1 = right edge
    threat: bool                       # did it qualify for a response? Only threats count as trials.
    reason: str                        # why the engine decided what it did
    deterrent: str                     # "none", or one of settings.yaml's deterrents
    outcome: Outcome
    gone: bool                         # no longer in view at the end of the follow-up
    seconds_until_gone: float | None   # None unless it left
    checks: int                        # follow-up checks made
    direction_after: Direction         # "unknown" if it was not seen again
    approach_after: Approach
    scenario: str = SETTINGS.scenario
    critical_side: str = SETTINGS.critical_side  # where the protected area was when this was recorded
    temperature_c: float | None = None # optional: the UNO Q has no sensor on board
    images: list[str] = field(default_factory=list)  # saved frames -> reuse as training data!
    model: str = ""                    # which model made the detection
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))

    @property
    def toward_critical_area(self) -> bool:
        return self.direction_before == self.critical_side

    @property
    def turned_around(self) -> bool:
        """Was heading toward the protected area, and went the other way after the response."""
        return self.toward_critical_area and self.direction_after == OPPOSITE[self.critical_side]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["toward_critical_area"] = self.toward_critical_area  # stored for convenience, derived on read
        d["turned_around"] = self.turned_around
        return d


# ---- helpers -----------------------------------------------------------

def main_animal(animals: list, known=SETTINGS.animals, is_threat=None):
    """The animal to react to. Threats come first (pass is_threat=callable to say what one is),
    then whichever was seen in the most frames, most confidently. None if nothing usable."""
    usable = [a for a in animals if a.name in known]

    def rank(a):
        return (bool(is_threat(a)) if is_threat else False, a.frames_seen, a.confidence)

    return max(usable, key=rank, default=None)


def build_encounter(before, decision, follow_up: FollowUp | None = None, images: list[str] | None = None,
                    temperature_c: float | None = None, model: str = "", settings: Settings = SETTINGS,
                    **overrides) -> Encounter:
    """`before` is one detector.Animal, `decision` a decision_engine.Decision,
    `follow_up` what the checks after the response found (None for animals we didn't follow),
    `settings` the settings the device was running with."""
    f = follow_up or FollowUp()
    left = f.outcome == "left"
    return Encounter(
        animal=before.name,
        confidence=round(before.confidence, 3),
        direction_before=before.direction,
        approach_before=before.approach,
        dx_before=before.dx,
        position_before=None if getattr(before, "position", None) is None else round(before.position, 3),
        threat=decision.threat,
        reason=decision.reason,
        deterrent=decision.action,
        outcome=f.outcome,
        gone=f.gone,
        seconds_until_gone=round(f.seconds_until_gone, 1) if left and f.seconds_until_gone is not None else None,
        checks=f.checks,
        direction_after=f.direction_after,
        approach_after=f.approach_after,
        scenario=settings.scenario,
        critical_side=settings.critical_side,
        temperature_c=temperature_c,
        images=list(images or []) + list(f.images),
        model=model,
        **overrides,
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
        rows = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written line, e.g. the power was cut mid-write: skip it, keep the rest
        return rows

    def summary(self) -> str:
        rows = self.read()
        if not rows:
            return f"No encounters in {self.path} yet."
        threats = [r for r in rows if r.get("threat")]
        by_det = defaultdict(list)
        for r in threats:
            by_det[r["deterrent"]].append(r)
        lines = [f"{len(rows)} encounters in {self.path.name}: {len(threats)} threats, "
                 f"{len(rows) - len(threats)} left alone",
                 f"{'response':10} {'n':>4} {'left':>6} {'entered':>8} {'stayed':>7} {'unclear':>8} {'avg s to leave':>15}"]
        for d in sorted(by_det, key=lambda d: (d != "none", d)):
            rs = by_det[d]
            n = {o: sum(r["outcome"] == o for r in rs) for o in ("left", "entered", "stayed", "unknown")}
            times = [r["seconds_until_gone"] for r in rs if r["seconds_until_gone"] is not None]
            avg = f"{sum(times) / len(times):.1f}" if times else "-"
            lines.append(f"{d:10} {len(rs):4} {n['left'] / len(rs):6.0%} {n['entered']:8} {n['stayed']:7} "
                         f"{n['unknown']:8} {avg:>15}")
        return "\n".join(lines)


class MemoryLog:
    """Same interface as EncounterLog, but keeps everything in memory (simulations, tests)."""

    def __init__(self):
        self.rows: list[dict] = []
        self.path = Path("(memory)")

    def append(self, enc: Encounter) -> None:
        self.rows.append(enc.to_dict())

    def read(self) -> list[dict]:
        return list(self.rows)

    summary = EncounterLog.summary


if __name__ == "__main__":
    print(EncounterLog(sys.argv[1] if len(sys.argv) > 1 else LOG_FILE).summary())
