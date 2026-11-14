"""Generate FAKE encounter data by running the real loop (run.Pipeline + the real
decision engine) against the made-up world in simulation.py. The file has exactly
the format the device writes, so you can build the dashboard / analysis before the
minis exist, and you can watch what the engine learns.

    python mock_data/make_mock_encounters.py                        # 300 encounters over 14 nights
    python mock_data/make_mock_encounters.py --n 600 --days 30 --seed 7
    python mock_data/make_mock_encounters.py --scenario road        # every animal is a threat

Writes encounters_mock.jsonl in the project folder (same format as encounters.jsonl)
and mock_data/encounters_mock.csv (for a spreadsheet).

The patterns hidden in the data are listed at the top of simulation.py.
Everything here is invented. Never mix this file with real encounters.jsonl.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from data_collection import MOCK_LOG_FILE, EncounterLog  # noqa: E402
from decision_engine import DecisionEngine  # noqa: E402
from run import Pipeline  # noqa: E402
from settings import load_settings  # noqa: E402
from simulation import SimWorld  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--start", default="2026-11-01", help="first day (local date)")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--scenario", help="override the scenario from settings.yaml")
    args = ap.parse_args()

    settings = load_settings(scenario=args.scenario)
    MOCK_LOG_FILE.unlink(missing_ok=True)
    log = EncounterLog(MOCK_LOG_FILE)
    world = SimWorld(settings, visits=args.n, seed=args.seed, start=args.start, days=args.days)
    engine = DecisionEngine(log, settings, random.Random(args.seed))
    Pipeline(world, world, engine, log, settings, quiet=True).run()

    rows = log.read()
    with open(HERE / "encounters_mock.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k != "images"])
        w.writeheader()
        for r in rows:
            r.pop("images")
            w.writerow(r)

    print(f"Scenario: {settings.scenario}")
    print(log.summary())
    print(f"\nWrote {MOCK_LOG_FILE.name} and mock_data/encounters_mock.csv")


if __name__ == "__main__":
    main()
