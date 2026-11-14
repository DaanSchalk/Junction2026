"""Generate FAKE encounter data in the exact format data_collection.py writes,
so you can build the dashboard / analysis before the minis exist.

    python mock_data/make_mock_encounters.py               # 200 encounters over 14 nights
    python mock_data/make_mock_encounters.py --n 500 --days 30 --seed 7

Writes mock_data/encounters_mock.jsonl (same format as encounters.jsonl) and
mock_data/encounters_mock.csv (for a spreadsheet).

Patterns deliberately hidden in the data, so you can check that your analysis finds them:
  * animals mostly come at dusk, at night and at dawn; deer are the most common, bears the rarest
  * sound works well at first, but deer get used to it: its effect on deer halves over the period
  * light works much better in the dark than in daylight, and best on bears
  * with "none", animals sometimes leave anyway (the baseline you compare against)
  * ~5% messy records: seen in only one frame, a missing temperature, low confidence
Everything here is invented. Never mix this file with real encounters.jsonl.
"""

import argparse
import csv
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data_collection import Encounter, EncounterLog  # noqa: E402

LOCAL = timezone(timedelta(hours=2))  # local time of the (imaginary) site

ANIMAL_WEIGHTS = {"deer": 45, "fox": 30, "wolf": 15, "bear": 10}
# chance the animal leaves the area after the deterrent, at the start of the period
LEAVE = {
    "none":  {"deer": 0.25, "fox": 0.20, "wolf": 0.30, "bear": 0.15},
    "sound": {"deer": 0.85, "fox": 0.55, "wolf": 0.70, "bear": 0.45},
    "light": {"deer": 0.65, "fox": 0.35, "wolf": 0.55, "bear": 0.70},
}
# hour of day (local) -> relative activity: dusk, night, dawn
HOUR_WEIGHTS = [6, 6, 5, 5, 5, 6, 8, 7, 3, 1, 1, 1, 1, 1, 2, 5, 9, 10, 8, 7, 6, 6, 6, 6]
OPPOSITE = {"left": "right", "right": "left", "up": "down", "down": "up"}


def is_dark(hour: int) -> bool:
    return hour < 8 or hour >= 16  # November, far north


def make_one(rng: random.Random, start: datetime, days: int, idx: int) -> Encounter:
    day = rng.randrange(days)
    hour = rng.choices(range(24), HOUR_WEIGHTS)[0]
    t_local = (start + timedelta(days=day)).replace(hour=hour, minute=rng.randrange(60), second=rng.randrange(60))
    progress = day / max(days - 1, 1)  # 0 at start, 1 at end of the period

    animal = rng.choices(list(ANIMAL_WEIGHTS), ANIMAL_WEIGHTS.values())[0]
    deterrent = rng.choice(["none", "sound", "light"])  # random, like pick_deterrent()

    direction_before = rng.choices(["right", "left", "still", "up", "down"], [55, 28, 9, 4, 4])[0]
    approach_before = rng.choices(["approaching", "same", "leaving"], [45, 40, 15])[0]
    dx = {"right": 1, "left": -1}.get(direction_before, 0) * round(rng.uniform(0.3, 1.6), 2)
    if direction_before in ("still", "up", "down"):
        dx = round(rng.uniform(-0.2, 0.2), 2)

    p = LEAVE[deterrent][animal]
    if deterrent == "sound" and animal == "deer":
        p *= 1 - 0.5 * progress                      # habituation
    if deterrent == "light":
        p *= 1.15 if is_dark(hour) else 0.45         # light hardly noticed in daylight
    gone = rng.random() < min(p, 0.97)

    if gone:
        direction_after = OPPOSITE.get(direction_before, rng.choice(["left", "right"])) if rng.random() < 0.8 \
            else rng.choice(["unknown", direction_before])
        approach_after = rng.choice(["leaving", "leaving", "same"])
        fast = deterrent != "none"
        seconds = round(rng.uniform(0.6, 2.5) if fast else rng.uniform(2.0, 8.0), 1)
    else:
        direction_after = rng.choices([direction_before, "still", OPPOSITE.get(direction_before, "still")], [60, 30, 10])[0]
        approach_after = rng.choice(["approaching", "same", "same"])
        seconds = None

    confidence = round(min(0.97, max(0.4, rng.gauss(0.8, 0.09))), 3)
    temperature = round(rng.gauss(-1.5 - 3 * progress - (2 if is_dark(hour) else 0), 2.0), 1)

    # a little mess, like real data
    if rng.random() < 0.02:
        direction_before, approach_before, dx = "unknown", "unknown", 0.0
    if rng.random() < 0.03:
        temperature = None
    if rng.random() < 0.03:
        confidence = round(rng.uniform(0.40, 0.5), 3)

    eid = f"{rng.getrandbits(48):012x}"
    images = [f"data/encounters/{eid}/before_f{i}.jpg" for i in range(3)] + \
             [f"data/encounters/{eid}/after_f{i}.jpg" for i in range(4)]
    return Encounter(
        animal=animal, confidence=confidence,
        direction_before=direction_before, approach_before=approach_before, dx_before=dx,
        deterrent=deterrent, direction_after=direction_after, approach_after=approach_after,
        gone=gone, seconds_until_gone=seconds, temperature_c=temperature,
        images=images, model="animals_ncnn_model (mock)",
        id=eid, timestamp=t_local.astimezone(timezone.utc).isoformat(timespec="seconds"),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", default="2026-11-01", help="first day (local date)")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    start = datetime.fromisoformat(args.start).replace(tzinfo=LOCAL)
    encounters = sorted((make_one(rng, start, args.days, i) for i in range(args.n)), key=lambda e: e.timestamp)

    jsonl = HERE / "encounters_mock.jsonl"
    jsonl.unlink(missing_ok=True)
    log = EncounterLog(jsonl)
    for e in encounters:
        log.append(e)

    rows = [e.to_dict() for e in encounters]
    with open(HERE / "encounters_mock.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "images"] + ["n_images"])
        w.writeheader()
        for r in rows:
            r = dict(r); r["n_images"] = len(r.pop("images"))
            w.writerow(r)

    print(log.summary())
    print(f"\nWrote {jsonl.name} and encounters_mock.csv in {HERE}")


if __name__ == "__main__":
    main()
