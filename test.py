"""Checks for the decision engine, the follow-up loop and the encounter log.

    python test.py          # prints what the engine has learned so far, then runs every check
    pytest test.py          # the same checks under pytest

No camera, model or Arduino needed.
"""

import dataclasses
import random
import sys
import tempfile
from pathlib import Path

from data_collection import LOG_FILE, MOCK_LOG_FILE, EncounterLog, MemoryLog
from decision_engine import ACTIONS, DecisionEngine, assess_threat, evaluate_outcome, is_threatening
from run import Pipeline
from settings import load_settings
from simulation import Sighting

SETTINGS = load_settings(scenario="crops")


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def seen(name="deer", direction="right", position=0.4, confidence=0.9):
    """One animal as the detector would report it."""
    return Sighting(name=name, confidence=confidence, frames_seen=3, direction=direction,
                    approach="same", dx=0.0, dy=0.0, position=position)


def trial(engine, animal, deterrent, outcome):
    """Feed the engine one finished encounter."""
    engine.learn({"animal": animal, "deterrent": deterrent, "threat": True, "outcome": outcome})


def play(engine, chance, rounds, animal="deer"):
    """Let the engine choose `rounds` times in a world where each action makes the animal
    leave with the given chance (a number, or a function of the round). Returns its choices."""
    rng = random.Random(1)
    picks = []
    for i in range(rounds):
        action = engine.choose_deterrent(animal, 0.9, SETTINGS.critical_side, 0.4)
        p = chance[action](i) if callable(chance[action]) else chance[action]
        trial(engine, animal, action, "left" if rng.random() < p else "entered")
        picks.append(action)
    return picks


class ScriptedEyes:
    """A camera that plays back prepared looks. Each look is the list of animals in view."""
    model_name = "test"

    def __init__(self, looks):
        self.looks, self.t = list(looks), 0.0
        self.following = []  # for each look: were we following an animal?

    def now(self):
        return self.t

    def wait(self, seconds):
        self.t += seconds

    def wait_for_trigger(self):
        return bool(self.looks)

    def observe(self, following=False):
        self.t += 0.8
        self.following.append(following)
        return self.looks.pop(0) if self.looks else []

    def save_last(self, encounter_id, prefix, single=False):
        return []

    def timestamp(self):
        return "2026-11-14T12:00:00+00:00"


class RecordingHardware:
    def __init__(self):
        self.fired = []

    def fire(self, deterrent):
        self.fired.append(deterrent)

    def temperature(self):
        return None


def encounter_from(looks, settings=SETTINGS):
    """Run one encounter through the real pipeline. Returns (encounter, hardware)."""
    hardware, log = RecordingHardware(), MemoryLog()
    engine = DecisionEngine(log, settings, random.Random(0))
    pipeline = Pipeline(ScriptedEyes(looks), hardware, engine, log, settings, quiet=True)
    return pipeline.run_once(), hardware


# --------------------------------------------------
# What counts as a threat
# --------------------------------------------------

def test_crops_scenario_only_deters_deer_heading_for_the_area():
    assert is_threatening("deer", 0.9, "right", 0.4, SETTINGS)
    assert not is_threatening("fox", 0.9, "right", 0.4, SETTINGS), "fox is not on the crops list"
    assert not is_threatening("deer", 0.9, "left", 0.4, SETTINGS), "moving away"
    assert not is_threatening("deer", 0.9, "still", 0.4, SETTINGS), "standing outside the area"
    assert not is_threatening("deer", 0.9, "unknown", 0.4, SETTINGS), "direction unknown"
    assert not is_threatening("deer", 0.4, "right", 0.4, SETTINGS), "detector not sure enough"


def test_animal_inside_the_area_is_a_threat_even_when_standing_still():
    assert is_threatening("deer", 0.9, "still", 0.9, SETTINGS)
    assert "inside" in assess_threat("deer", 0.9, "still", 0.9, SETTINGS)[1]


def test_switching_scenario_changes_who_is_deterred():
    livestock, road = load_settings(scenario="livestock"), load_settings(scenario="road")
    assert is_threatening("wolf", 0.9, "right", 0.4, livestock)
    assert not is_threatening("deer", 0.9, "right", 0.4, livestock)
    assert all(is_threatening(a, 0.9, "right", 0.4, road) for a in road.animals)


def test_area_on_the_left_flips_the_direction():
    left = dataclasses.replace(SETTINGS, critical_side="left")
    assert is_threatening("deer", 0.9, "left", 0.6, left)
    assert not is_threatening("deer", 0.9, "right", 0.6, left)
    assert is_threatening("deer", 0.9, "still", 0.1, left), "inside the area on the left"
    # ...and the whole loop follows: heading left is the threat, leaving to the right is the success
    enc, _ = encounter_from([[seen(direction="left", position=0.6)], [seen(direction="right", position=0.7)],
                             [seen(direction="right", position=0.9)], [], []], settings=left)
    assert enc.threat and enc.outcome == "left" and enc.critical_side == "left"
    assert enc.toward_critical_area and enc.turned_around


def test_unknown_direction_is_not_a_threat_and_says_so():
    threat, why = assess_threat("deer", 0.9, "unknown", 0.4, SETTINGS)
    assert not threat and "can't tell" in why
    assert "59%" in assess_threat("deer", 0.596, "right", 0.4, SETTINGS)[1], "never '60% sure (need 60%)'"


def test_non_threats_never_get_a_deterrent():
    engine = DecisionEngine(None, SETTINGS, random.Random(0))
    for _ in range(300):
        assert engine.choose_deterrent("fox", 0.99, "right", 0.5) == "none"
        assert engine.choose_deterrent("deer", 0.99, "left", 0.5) == "none"
        assert engine.choose_deterrent("deer", 0.30, "right", 0.5) == "none"


def test_settings_reject_an_animal_the_camera_does_not_know():
    text = (Path(__file__).parent / "settings.yaml").read_text().replace("deter: [deer]", "deter: [moose]")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "settings.yaml"
        path.write_text(text)
        try:
            load_settings(path)
        except SystemExit as e:
            assert "moose" in str(e)
        else:
            raise AssertionError("a scenario with an unknown animal should be refused")


# --------------------------------------------------
# Learning
# --------------------------------------------------

def test_outcomes_count_the_right_way():
    assert evaluate_outcome({"outcome": "left"}) == "success"
    assert evaluate_outcome({"outcome": "entered"}) == "failure"
    assert evaluate_outcome({"outcome": "stayed"}) == "failure"
    assert evaluate_outcome({"outcome": "unknown"}) == "inconclusive"
    assert evaluate_outcome({}) == "inconclusive"


def test_only_threats_are_trials():
    engine = DecisionEngine(None, SETTINGS)
    engine.learn({"animal": "fox", "deterrent": "none", "threat": False, "outcome": "not_a_threat"})
    assert engine.model.get_statistics("fox", "none")["trials"] == 0
    trial(engine, "deer", "sound", "left")
    trial(engine, "deer", "sound", "unknown")
    stats = engine.model.get_statistics("deer", "sound")
    assert (stats["trials"], stats["successes"], stats["inconclusive"]) == (2, 1, 1)


def test_learns_which_deterrent_works():
    engine = DecisionEngine(None, SETTINGS, random.Random(0))
    picks = play(engine, {"none": 0.15, "sound": 0.80, "light": 0.40}, rounds=400)
    late = picks[200:]
    assert late.count("sound") > 0.6 * len(late), f"sound should dominate, got {late.count('sound')}/{len(late)}"
    report = engine.get_effectiveness_report("deer")
    assert report["sound"]["effectiveness"] > report["light"]["effectiveness"] > report["none"]["effectiveness"]
    assert report["sound"]["low"] < report["sound"]["effectiveness"] < report["sound"]["high"]


def test_never_gets_stuck_on_an_early_impression():
    # The situation that froze the first version: "none" got lucky once, the deterrents didn't.
    engine = DecisionEngine(None, SETTINGS, random.Random(0))
    trial(engine, "deer", "none", "left")
    for outcome in ("left", "left", "entered", "entered"):
        trial(engine, "deer", "sound", outcome)
    for _ in range(4):
        trial(engine, "deer", "light", "unknown")
    picks = [engine.choose_deterrent("deer", 0.9, "right", 0.4) for _ in range(300)]
    assert picks.count("sound") > 30 and picks.count("light") > 30, "both deterrents must keep getting tried"


def test_keeps_a_control_group():
    engine = DecisionEngine(None, SETTINGS, random.Random(0))
    kinds = [engine.decide("deer", 0.9, "right", 0.4).kind for _ in range(3000)]
    share = kinds.count("control") / len(kinds)
    assert abs(share - SETTINGS.control_share) < 0.03, f"control share was {share:.1%}"


def test_notices_when_a_deterrent_stops_working():
    # Sound works at first, then the animals get used to it. Light stays the same.
    engine = DecisionEngine(None, SETTINGS, random.Random(0))
    sound = lambda i: 0.85 if i < 150 else 0.10
    picks = play(engine, {"none": 0.15, "sound": sound, "light": 0.50}, rounds=450)
    early, late = picks[50:150], picks[350:]
    assert early.count("sound") > early.count("light"), "sound should lead while it works"
    assert late.count("light") > late.count("sound"), "the engine should have moved to light"


def test_learning_survives_a_restart():
    with tempfile.TemporaryDirectory() as tmp:
        log = EncounterLog(Path(tmp) / "encounters.jsonl")
        looks = [[seen()], [seen(direction="left", position=0.3)], [], []]
        hardware = RecordingHardware()
        engine = DecisionEngine(log, SETTINGS, random.Random(0))
        encounter = Pipeline(ScriptedEyes(looks), hardware, engine, log, SETTINGS, quiet=True).run_once()
        restarted = DecisionEngine(log, SETTINGS)
        before = engine.model.get_statistics("deer", encounter.deterrent)
        after = restarted.model.get_statistics("deer", encounter.deterrent)
        assert before == after and after["trials"] == 1


# --------------------------------------------------
# The follow-up: keep checking until the animal is gone
# --------------------------------------------------

def test_animal_that_turns_and_leaves_counts_as_left():
    enc, hardware = encounter_from([[seen()], [seen(direction="left", position=0.3)],
                                    [seen(direction="left", position=0.1)], [], []])
    assert enc.threat and enc.outcome == "left" and enc.gone
    assert enc.direction_after == "left" and enc.turned_around
    assert hardware.fired == ([enc.deterrent] if enc.deterrent != "none" else [])
    # Timed from the response to the START of the first look that no longer found it:
    # look (0.8 s) + pause (1 s) + look (0.8 s) + pause (1 s) = 3.6 s.
    assert enc.seconds_until_gone == 3.6


def test_last_glimpse_in_one_photo_does_not_erase_the_direction():
    # Slipping out of view, the animal is caught in just one photo of the last burst: no direction
    # from that look, so the one before it must be kept.
    glimpse = seen(direction="unknown", position=0.02)
    enc, _ = encounter_from([[seen()], [seen(direction="left", position=0.3)], [glimpse], [], []])
    assert enc.outcome == "left" and enc.direction_after == "left" and enc.turned_around


def test_animal_that_walks_into_the_area_counts_as_entered():
    enc, _ = encounter_from([[seen()], [seen(position=0.6)], [seen(position=0.8)], [seen(position=0.9)]])
    assert enc.outcome == "entered" and enc.checks == 2, "stops watching as soon as it is inside"


def test_animal_that_hangs_around_counts_as_stayed():
    enc, _ = encounter_from([[seen()]] + [[seen(direction="still", position=0.5)]] * 30)
    assert enc.outcome == "stayed" and not enc.gone and enc.seconds_until_gone is None


def test_one_missed_look_is_not_gone():
    looks = [[seen()], [seen(direction="left", position=0.3)], [], [seen(direction="left", position=0.2)], [], []]
    enc, _ = encounter_from(looks)
    assert enc.outcome == "left" and enc.checks == 5
    # ...and the number of empty looks it takes is a setting
    patient = dataclasses.replace(SETTINGS, empty_looks=3)
    enc, _ = encounter_from(looks + [[]], settings=patient)
    assert enc.outcome == "left" and enc.checks == 6


def test_follow_up_looks_are_marked_as_following():
    # The camera uses that mark to accept weaker sightings of an animal it already identified.
    eyes = ScriptedEyes([[seen()], [seen(direction="left", position=0.3)], [], []])
    log = MemoryLog()
    Pipeline(eyes, RecordingHardware(), DecisionEngine(log, SETTINGS, random.Random(0)), log, SETTINGS,
             quiet=True).run_once()
    assert eyes.following == [False, True, True, True]


def test_vanishing_before_we_see_a_reaction():
    far, _ = encounter_from([[seen(position=0.2)], [], []])
    middle, _ = encounter_from([[seen(position=0.5)], [], []])
    assert far.outcome == "left", "it was far from the area, it cannot have slipped in unseen"
    assert middle.outcome == "unknown", "from the middle of the view we cannot tell"


def test_animal_already_inside_that_stays_counts_as_entered():
    enc, _ = encounter_from([[seen(direction="still", position=0.9)]] * 31)
    assert enc.threat and enc.outcome == "entered"


def test_other_species_do_not_confuse_the_follow_up():
    # A fox wanders through the far side of the view while we follow a deer that leaves.
    fox = seen("fox", direction="right", position=0.9)
    enc, _ = encounter_from([[seen()], [seen(direction="left", position=0.2), fox], [fox], [fox]])
    assert enc.animal == "deer" and enc.outcome == "left"


def test_non_threat_is_logged_and_left_alone():
    enc, hardware = encounter_from([[seen("fox")]])
    assert not enc.threat and enc.outcome == "not_a_threat" and enc.deterrent == "none"
    assert hardware.fired == [] and enc.checks == 0


def test_lingering_animal_is_logged_once_but_still_handled_when_it_becomes_a_threat():
    log, hardware = MemoryLog(), RecordingHardware()
    eyes = ScriptedEyes([[seen(direction="still")], [seen(direction="still")], [seen(direction="still")],
                         [seen(direction="right")], [], []])
    pipeline = Pipeline(eyes, hardware, DecisionEngine(log, SETTINGS, random.Random(0)), log, SETTINGS, quiet=True)
    for _ in range(4):
        pipeline.run_once()
        eyes.wait(5)
    assert [r["threat"] for r in log.rows] == [False, True], "one record for the lingering, one for the threat"


def test_threat_is_handled_before_a_bystander():
    enc, _ = encounter_from([[seen("fox", confidence=0.99), seen("deer", confidence=0.7)], [], []])
    assert enc.animal == "deer"


def test_log_survives_a_half_written_line():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "encounters.jsonl"
        path.write_text('{"animal": "deer", "threat": true, "deterrent": "sound", "outcome": "left"}\n'
                        '{"animal": "deer", "threat": tr')  # the power was cut mid-write
        log = EncounterLog(path)
        assert len(log.read()) == 1
        assert DecisionEngine(log, SETTINGS).model.get_statistics("deer", "sound")["successes"] == 1


# --------------------------------------------------
# The mock data
# --------------------------------------------------

def test_mock_data_has_the_current_format():
    rows = EncounterLog(MOCK_LOG_FILE).read()
    assert rows, f"{MOCK_LOG_FILE.name} is missing: run python mock_data/make_mock_encounters.py"
    for key in ("animal", "threat", "reason", "deterrent", "outcome", "checks", "position_before", "scenario",
                "critical_side"):
        assert all(key in r for r in rows), f"mock data lacks '{key}': regenerate it"
    engine = DecisionEngine(EncounterLog(MOCK_LOG_FILE), load_settings(scenario=rows[0]["scenario"]))
    threats = sum(r["threat"] for r in rows)
    learned = sum(engine.model.get_statistics(a, d)["trials"] for a in engine.settings.animals for d in engine.actions)
    assert learned == threats, "the engine should learn from every threat, and only from threats"


# --------------------------------------------------
# Report + runner
# --------------------------------------------------

def print_effectiveness_report(engine, animals):
    print("\n" + "=" * 78)
    print("EFFECTIVENESS REPORT (how often the animal left; range = 90% plausible)")
    print("=" * 78)

    for animal in sorted(animals):
        if not any(engine.model.get_statistics(animal, action)["trials"] for action in ACTIONS):
            continue
        print(f"\nAnimal: {animal}")
        print("-" * 78)

        for action in ACTIONS:
            stats = engine.model.get_statistics(animal, action)

            print(
                f"{action:6} | "
                f"Trials: {stats['trials']:3} | "
                f"Left: {stats['successes']:3} | "
                f"Didn't: {stats['failures']:3} | "
                f"Unclear: {stats['inconclusive']:2} | "
                f"Effectiveness: {stats['effectiveness']:4.0%} ({stats['low']:.0%}-{stats['high']:.0%})"
            )


def main():
    # 1. What has the engine learned so far? Real encounters if there are any, mock data otherwise.
    path = LOG_FILE if LOG_FILE.exists() else MOCK_LOG_FILE
    log = EncounterLog(path)
    records = log.read()
    print(f"Loaded {len(records)} encounters from {path.name}" + ("" if path == LOG_FILE else " (invented data)"))
    if records:
        print(log.summary())
        engine = DecisionEngine(log, load_settings(scenario=records[-1].get("scenario")))
        print_effectiveness_report(engine, {r["animal"] for r in records})

    # 2. Run every check.
    print("\n" + "=" * 78)
    print("CHECKS")
    print("=" * 78)
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS | {name[5:].replace('_', ' ')}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL | {name[5:].replace('_', ' ')}: {e}")
    print(f"\n{len(tests) - failed} of {len(tests)} checks passed.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
