"""The whole loop, without the Arduino:
see -> decide -> respond -> keep checking until the animal is gone -> log -> learn.

    python run.py --simulate 20              # no camera, no model: a made-up world, to watch the loop work
    python run.py --camera 0                 # real camera + your trained model
    python run.py --camera clip.mp4          # a recorded video instead of a camera
    python run.py --camera 0 --model yolo26n.pt --alias person=deer
                                             # no minis yet? Use the stock model and play the deer yourself

    --scenario livestock                     # another scenario from settings.yaml
    --settings site_b.yaml                   # another settings file altogether

Responses are only printed for now. The Hardware class at the top of this file is
the one place to connect the Arduino later: the motion sensor, the speaker and
light, and the temperature sensor.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from data_collection import LOG_FILE, EncounterLog, FollowUp, MemoryLog, build_encounter, main_animal
from decision_engine import DecisionEngine, is_threatening
from settings import SETTINGS_FILE, Settings, load_settings

ROOT = Path(__file__).resolve().parent
ENCOUNTER_IMAGES = ROOT / "vision" / "data" / "encounters"


# ---- the Arduino side (not connected yet) ----------------------------------

class Hardware:
    """Everything the microcontroller will do. For now it only prints."""

    def fire(self, deterrent: str) -> None:
        """Switch a deterrent on: 'sound', 'light', or whatever settings.yaml lists."""
        print(f"    [hardware] would fire: {deterrent}")

    def temperature(self) -> float | None:
        """Outside temperature in °C, or None without a sensor."""
        return None


# ---- the camera side -------------------------------------------------------

class SourceEnded(Exception):
    """The video file is finished, or the camera stopped delivering frames."""


class CameraEyes:
    """Camera (or video file) + detector. Gives the pipeline what it needs:
    wait_for_trigger(), observe(), now(), wait(), save_last()."""

    def __init__(self, source: str, model: str | None, settings: Settings, conf: float = 0.4,
                 aliases: dict[str, str] | None = None, sensitivity: float = 0.01, cooldown_s: float = 1.0,
                 imgsz: int | None = None):
        sys.path.insert(0, str(ROOT / "vision"))
        import cv2
        from capture import MotionDetector
        from detector import AnimalDetector

        self.cv2 = cv2
        self.settings = settings
        self.conf = conf
        self.aliases = aliases or {}
        # Only track what we care about, so other objects in view can't be mistaken for the animal.
        self.detector = AnimalDetector(model, conf=conf, imgsz=imgsz, only=[*settings.animals, *self.aliases])
        self.model_name = Path(model).name if model else "animals_ncnn_model"
        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera '{source}'.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # A video file has its own clock (frame number / fps), so runs on a file are repeatable.
        self.is_file = not source.isdigit() and not source.startswith(("/dev/", "rtsp://", "http://", "https://"))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frames_read = 0
        self.motion = MotionDetector(sensitivity)
        self.cooldown_s = cooldown_s
        self.last_frames: list = []

    def now(self) -> float:
        return self.frames_read / self.fps if self.is_file else time.monotonic()

    def timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            raise SourceEnded
        self.frames_read += 1
        return frame

    def wait(self, seconds: float) -> None:
        """Let time pass while still reading frames, so the next photo is fresh."""
        until = self.now() + seconds
        while self.now() < until:
            self.motion(self.read())  # keeps the motion background up to date too

    def wait_for_trigger(self) -> bool:
        """Block until something moves in view. (Later: the Arduino's motion sensor.)"""
        quiet_until = self.now() + self.cooldown_s
        while True:
            moved = self.motion.triggered(self.read())
            if moved and self.now() >= quiet_until:
                return True

    def observe(self, following: bool = False) -> list:
        """Take a burst of photos and return the animals in it, with direction and position.
        following=True: we are watching an animal we already identified, so weaker sightings count too."""
        s = self.settings
        frames = []
        for i in range(s.burst_frames):
            if i:
                self.wait(s.burst_gap_s)
            frames.append(self.read())
        self.last_frames = frames
        floor = min(self.conf, s.follow_confidence) if following else None
        animals = self.detector.analyze_burst(frames, conf=floor)
        for a in animals:
            a.name = self.aliases.get(a.name, a.name)
        return animals

    def save_last(self, encounter_id: str, prefix: str, single: bool = False) -> list[str]:
        """Save the photos of the last burst (single=True: only the newest one).
        They double as training data later."""
        folder = ENCOUNTER_IMAGES / encounter_id
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        frames = list(enumerate(self.last_frames))
        for i, frame in frames[-1:] if single else frames:
            path = folder / f"{prefix}_f{i}.jpg"
            self.cv2.imwrite(str(path), frame)
            paths.append(str(path.relative_to(ROOT)))
        return paths


# ---- the loop --------------------------------------------------------------

class Pipeline:
    """One encounter at a time: see, decide, respond, follow up, log, learn.

    `eyes` needs wait_for_trigger() (False = nothing more will come), observe(following=False), now(), wait(),
    save_last(id, prefix, single=False), timestamp() and model_name.
    `hardware` needs fire() and temperature(). CameraEyes + Hardware are the real ones;
    simulation.SimWorld plays both parts.
    """

    same_visit_s = 30  # an animal that is just hanging around is logged once, not at every look

    def __init__(self, eyes, hardware, engine: DecisionEngine, log, settings: Settings, quiet: bool = False):
        self.eyes, self.hardware, self.engine, self.log, self.settings = eyes, hardware, engine, log, settings
        self.quiet = quiet
        self.last_logged = (None, float("-inf"))  # (animal, when)

    def say(self, text: str) -> None:
        if not self.quiet:
            print(text)

    def is_threat(self, animal) -> bool:
        return is_threatening(animal.name, animal.confidence, animal.direction,
                              getattr(animal, "position", None), self.settings)

    def run(self, max_encounters: int = 0) -> int:
        """Handle encounters until the source ends (or max_encounters is reached). Returns how many."""
        done = 0
        try:
            while not max_encounters or done < max_encounters:
                if not self.eyes.wait_for_trigger():
                    break
                if self.run_once() is not None:
                    done += 1
        except (SourceEnded, KeyboardInterrupt):
            pass
        return done

    def run_once(self):
        """Something moved: look, and deal with what is there. Returns the Encounter, or None if no animal."""
        animals = self.eyes.observe()
        target = main_animal(animals, known=self.settings.animals, is_threat=self.is_threat)
        if target is None:
            return None

        position = getattr(target, "position", None)
        decision = self.engine.decide(target.name, target.confidence, target.direction, position)
        last_name, last_time = self.last_logged
        if not decision.threat and target.name == last_name and self.eyes.now() - last_time < self.same_visit_s:
            self.last_logged = (target.name, self.eyes.now())
            return None  # same visitor as a moment ago, still no threat: nothing new to record
        encounter_id = self.eyes.new_id() if hasattr(self.eyes, "new_id") else uuid.uuid4().hex[:12]
        images = self.eyes.save_last(encounter_id, "before")
        moving = {"still": "standing still", "unknown": "direction unknown"}.get(target.direction,
                                                                                   f"moving {target.direction}")
        self.say(f"{target.name} {target.confidence:.0%}, {moving} -> {decision.action}. {decision.reason}")

        if decision.action != "none":
            self.hardware.fire(decision.action)
        follow = self.follow_up(target, encounter_id) if decision.threat else FollowUp()

        encounter = build_encounter(
            target, decision, follow, images=images,
            temperature_c=self.hardware.temperature(),
            model=self.eyes.model_name,
            settings=self.settings,
            id=encounter_id,
            timestamp=self.eyes.timestamp(),
        )
        self.log.append(encounter)
        self.engine.learn(encounter)
        self.last_logged = (target.name, self.eyes.now())
        if decision.threat:
            took = f" after {encounter.seconds_until_gone:.0f} s" if encounter.seconds_until_gone is not None else ""
            self.say(f"    outcome: {encounter.outcome}{took} ({encounter.checks} checks)")
        return encounter

    def follow_up(self, before, encounter_id: str) -> FollowUp:
        """Keep looking until the animal is gone, is inside the protected area, or time runs out."""
        s = self.settings
        start = self.eyes.now()
        last = before                    # the last time we saw it: tells us where it was
        moving = None                    # the last look after the response that showed which way it moved
        started_inside = s.in_area(getattr(before, "position", None))
        misses, first_miss_at, checks, images = 0, None, 0, []
        outcome, gone = "stayed", False

        try:
            # Empty looks that started just before the time limit still get their confirming looks.
            while self.eyes.now() - start < s.give_up_after_s or 0 < misses < s.empty_looks:
                if checks:               # the first look is straight away, to catch its reaction
                    self.eyes.wait(s.check_every_s)
                looked_at = self.eyes.now()
                animals = self.eyes.observe(following=True)
                checks += 1
                same = self.same_animal(animals, last)

                if same is None:
                    misses += 1
                    if misses == 1:
                        first_miss_at = looked_at
                    if misses >= s.empty_looks:   # one empty look could be a missed detection
                        gone = True
                        outcome = self.where_did_it_go(last, moving)
                        break
                    continue

                misses, last = 0, same
                if same.direction != "unknown":   # seen in one photo only = no direction; keep the earlier one
                    moving = same
                images += self.eyes.save_last(encounter_id, f"check{checks}", single=True)
                if not started_inside and s.in_area(getattr(same, "position", None)):
                    outcome = "entered"  # no need to keep watching: it got in
                    break
            else:
                # Time is up and it is still in view.
                outcome = "entered" if s.in_area(getattr(last, "position", None)) else "stayed"
        except SourceEnded:
            outcome = "unknown"          # the video ended before we could tell

        return FollowUp(
            outcome=outcome,
            gone=gone,
            seconds_until_gone=(first_miss_at - start) if gone else None,
            checks=checks,
            direction_after=moving.direction if moving else "unknown",
            approach_after=moving.approach if moving else "unknown",
            images=images,
        )

    def same_animal(self, animals: list, last):
        """Find the animal we are following in a new look. Same species first; otherwise something
        close to where it was (the detector sometimes changes its mind about the species)."""
        known = [a for a in animals if a.name in self.settings.animals]
        last_pos = getattr(last, "position", None)

        def distance(a):
            pos = getattr(a, "position", None)
            return abs(pos - last_pos) if pos is not None and last_pos is not None else 0.0

        same_species = [a for a in known if a.name == last.name]
        if same_species:
            return min(same_species, key=distance)
        nearby = [a for a in known if distance(a) <= 0.25]
        return min(nearby, key=distance, default=None)

    def where_did_it_go(self, last, moving=None) -> str:
        """It vanished. Decide whether it left or went into the protected area.
        `last` = the last sighting (where it was); `moving` = the last sighting after the response
        that showed a direction (its reaction), or None if we never saw one."""
        s = self.settings
        position = getattr(last, "position", None)
        if s.in_area(position):
            return "entered"
        direction = moving.direction if moving is not None else None
        if direction == s.critical_side:
            return "entered"
        if direction == s.away_side:
            return "left"
        # No sideways movement to go by (or it vanished before we saw it react): use where it was.
        toward = s.toward(position)
        if toward is None:
            return "unknown"
        if toward < 1 / 3:
            return "left"
        if toward > 2 / 3:
            return "entered"
        return "unknown"


# ---- command line ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--simulate", type=int, nargs="?", const=20, metavar="N",
                      help="play N made-up encounters (default 20); nothing is saved")
    mode.add_argument("--camera", help="camera index, /dev/videoN, video file or stream URL")
    ap.add_argument("--settings", type=Path, default=SETTINGS_FILE,
                    help="settings file for this site (default: settings.yaml)")
    ap.add_argument("--scenario", help="override the scenario named in the settings file")
    ap.add_argument("--model", help="detector weights; default: model_path from vision/config.yaml")
    ap.add_argument("--alias", action="append", default=[], metavar="SEEN=TREAT_AS",
                    help="treat one detected class as another, e.g. person=deer (repeatable)")
    ap.add_argument("--conf", type=float, default=0.4, help="detector confidence cut-off")
    ap.add_argument("--imgsz", type=int, help="detector image size; default: imgsz from vision/config.yaml "
                                              "(the stock yolo26n.pt sees small things better at 640)")
    ap.add_argument("--log", type=Path, default=LOG_FILE, help="where real encounters are written")
    ap.add_argument("--max", type=int, default=0, help="stop after this many encounters (0 = keep going)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    settings = load_settings(args.settings, scenario=args.scenario)
    rng = random.Random(args.seed)
    print(f"Scenario: {settings.scenario} ({settings.description}) Deterring: {', '.join(sorted(settings.deter))}")

    if args.simulate is not None:
        from simulation import SimWorld
        log = MemoryLog()
        world = SimWorld(settings, visits=args.simulate, seed=args.seed)
        Pipeline(world, world, DecisionEngine(log, settings, rng), log, settings).run()
        print("\n" + log.summary())
        return

    try:
        aliases = dict(pair.split("=", 1) for pair in args.alias)
    except ValueError:
        raise SystemExit("--alias needs SEEN=TREAT_AS, e.g. --alias person=deer")
    log = EncounterLog(args.log)
    eyes = CameraEyes(args.camera, args.model, settings, conf=args.conf, aliases=aliases, imgsz=args.imgsz)
    engine = DecisionEngine(log, settings, rng)
    print(f"Watching camera {args.camera}. Logging to {log.path.name}. Ctrl+C to stop.")
    n = Pipeline(eyes, Hardware(), engine, log, settings).run(args.max)
    print(f"\n{n} encounters this run.\n" + log.summary())


if __name__ == "__main__":
    main()
