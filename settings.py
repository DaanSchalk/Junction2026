"""Loads settings.yaml (how the device behaves) plus the animal list from vision/config.yaml.

    from settings import load_settings
    s = load_settings()                      # the active scenario from settings.yaml
    s = load_settings(scenario="livestock")  # or force another one

Everything else in the project takes its numbers from here, so there is one
place to change them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
SETTINGS_FILE = ROOT / "settings.yaml"
VISION_CONFIG = ROOT / "vision" / "config.yaml"

OPPOSITE = {"left": "right", "right": "left"}


@dataclass(frozen=True)
class Settings:
    scenario: str
    description: str
    scenarios: tuple[str, ...]     # every scenario that exists in settings.yaml
    animals: tuple[str, ...]       # everything the model can recognise
    deter: frozenset[str]          # animals that get a response in this scenario
    deterrents: tuple[str, ...]    # e.g. ("sound", "light")
    critical_side: str             # "left" or "right"
    critical_width: float          # share of the view that is the protected area
    min_confidence: float
    control_share: float
    memory: float
    check_every_s: float
    give_up_after_s: float
    empty_looks: int               # looks in a row without the animal before it counts as gone
    follow_confidence: float       # detector cut-off while following an animal we already identified
    burst_frames: int
    burst_gap_s: float

    @property
    def actions(self) -> tuple[str, ...]:
        return ("none", *self.deterrents)

    @property
    def away_side(self) -> str:
        return OPPOSITE[self.critical_side]

    def toward(self, position: float | None) -> float | None:
        """How far along the way to the protected area: 0 = far edge of the view, 1 = the area's edge."""
        if position is None:
            return None
        return position if self.critical_side == "right" else 1 - position

    def in_area(self, position: float | None) -> bool:
        """Is this position (0 = left edge of the view, 1 = right edge) inside the protected area?"""
        if position is None or self.critical_width <= 0:
            return False
        return self.toward(position) >= 1 - self.critical_width


def load_settings(path: Path | str = SETTINGS_FILE, scenario: str | None = None,
                  vision_config: Path | str = VISION_CONFIG) -> Settings:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f)
    with open(vision_config) as f:
        animals = tuple(yaml.safe_load(f)["classes"])

    def fail(msg: str):
        raise SystemExit(f"{path.name}: {msg}")

    scenarios = raw.get("scenarios") or {}
    name = scenario or raw.get("scenario")
    if name not in scenarios:
        fail(f"scenario '{name}' not found. Available: {', '.join(scenarios)}")
    deter = scenarios[name].get("deter") or []
    unknown = [a for a in deter if a not in animals]
    if unknown:
        fail(f"scenario '{name}' lists {unknown}, but vision/config.yaml only knows {list(animals)}")

    area = raw.get("critical_area") or {}
    side = area.get("side", "right")
    if side not in OPPOSITE:
        fail(f"critical_area.side must be 'left' or 'right', not '{side}'")
    width = float(area.get("width", 0))
    if not 0 <= width < 1:
        fail("critical_area.width must be between 0 and 1")

    deterrents = tuple(raw.get("deterrents") or [])
    if not deterrents or "none" in deterrents:
        fail("deterrents needs at least one entry, and 'none' is added automatically")

    learning, follow, burst = raw.get("learning") or {}, raw.get("follow_up") or {}, raw.get("burst") or {}
    return Settings(
        scenario=name,
        description=scenarios[name].get("description", ""),
        scenarios=tuple(scenarios),
        animals=animals,
        deter=frozenset(deter),
        deterrents=deterrents,
        critical_side=side,
        critical_width=width,
        min_confidence=float(raw.get("min_confidence", 0.6)),
        control_share=float(learning.get("control_share", 0.1)),
        memory=float(learning.get("memory", 1.0)),
        check_every_s=float(follow.get("check_every_s", 1)),
        give_up_after_s=float(follow.get("give_up_after_s", 20)),
        empty_looks=max(1, int(follow.get("empty_looks_before_gone", 2))),
        follow_confidence=float(follow.get("min_confidence", 0.25)),
        burst_frames=int(burst.get("frames", 3)),
        burst_gap_s=float(burst.get("gap_s", 0.4)),
    )
