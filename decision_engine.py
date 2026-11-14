"""Decides how to respond to an animal, and learns from what happened next.

    engine = DecisionEngine(EncounterLog())      # learns from every earlier encounter
    decision = engine.decide("deer", confidence=0.9, direction="right", position=0.4)
    decision.action    # "none", "sound" or "light"
    decision.reason    # one line you can show on screen
    ...respond, watch what the animal does, build the Encounter...
    engine.learn(encounter)

How it chooses (all numbers live in settings.yaml):
  1. Is it a threat? The animal must be on the active scenario's deter list, the
     detector must be sure enough, and the animal must be heading for (or already
     inside) the protected area. Anything else gets "none" and is not a trial.
  2. Control. A fixed share of threats gets "none" on purpose, so we always know
     what animals do when we do nothing.
  3. Otherwise: Thompson sampling. For each option the engine draws a plausible
     success rate from what it has seen so far and picks the highest draw. Options
     that work get picked more and more, but the others keep getting a turn, so the
     engine can never get stuck on a wrong early impression.
  Old results slowly count less (learning.memory), so when animals get used to a
  deterrent the engine notices and shifts to another one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from settings import Settings, load_settings

# --------------------------------------------------
# Configuration (from settings.yaml)
# --------------------------------------------------

SETTINGS = load_settings()

ACTIONS = SETTINGS.actions
# Animals that never receive a deterrent in the active scenario.
ALLOWED_ANIMALS = set(SETTINGS.animals) - SETTINGS.deter
CONFIDENCE_THRESHOLD = SETTINGS.min_confidence

# How an encounter's outcome counts for learning.
OUTCOME_RESULT = {"left": "success", "entered": "failure", "stayed": "failure"}


# --------------------------------------------------
# Threat assessment
# --------------------------------------------------

def assess_threat(animal, confidence, direction, position=None, settings: Settings = SETTINGS):
    """Return (is_threat, reason). `direction` is left/right/... as seen in the image,
    `position` is where the animal is in the view (0 = left edge, 1 = right edge)."""

    # Animals outside the scenario are logged, never deterred.
    if animal not in settings.deter:
        return False, f"{animal} is not deterred in the '{settings.scenario}' scenario"

    # Ignore uncertain detections.
    if confidence < settings.min_confidence:
        sure = int(confidence * 100)  # rounded down, so 59.6% never reads as "60% (need 60%)"
        return False, f"only {sure}% sure it is a {animal} (need {settings.min_confidence:.0%})"

    if settings.in_area(position):
        return True, f"{animal} is inside the protected area"

    if direction == settings.critical_side:
        return True, f"{animal} is heading for the protected area"

    if direction == "unknown":
        return False, f"can't tell yet which way the {animal} is going"

    return False, f"{animal} is not heading for the protected area"


def is_threatening(animal, confidence, direction, position=None, settings: Settings = SETTINGS):
    """Determine whether an animal qualifies for intervention."""
    return assess_threat(animal, confidence, direction, position, settings)[0]


# --------------------------------------------------
# Outcome evaluation
# --------------------------------------------------

def evaluate_outcome(encounter):
    """
    Turn what happened after the response into a learning signal.

    Success:       the animal left the view, away from the protected area.
    Failure:       it went into the protected area, or was still around
                   when we stopped checking.
    Inconclusive:  we lost sight of it and can't tell which way it went.
    """

    if hasattr(encounter, "to_dict"):
        encounter = encounter.to_dict()

    return OUTCOME_RESULT.get(encounter.get("outcome"), "inconclusive")


# --------------------------------------------------
# Bayesian effectiveness model
# --------------------------------------------------

class EffectivenessModel:
    """One Beta distribution per (animal, deterrent): how often does the animal leave?"""

    def __init__(self, memory: float = 1.0):
        self.memory = memory
        self.estimates = {}

    def _stats(self, animal, deterrent):
        return self.estimates.setdefault((animal, deterrent), {
            "alpha": 1.0,
            "beta": 1.0,
            "trials": 0,
            "successes": 0,
            "failures": 0,
            "inconclusive": 0,
        })

    def update(self, animal, deterrent, outcome):
        """Update the Beta distribution for an animal-deterrent pair."""

        stats = self._stats(animal, deterrent)

        # Count every eligible trial, including inconclusive ones.
        stats["trials"] += 1

        if outcome not in ("success", "failure"):
            # Inconclusive encounters do not change alpha or beta.
            stats["inconclusive"] += 1
            return

        # Let earlier evidence fade a little, so the estimate can follow a change
        # (for example animals getting used to a sound).
        stats["alpha"] = 1 + (stats["alpha"] - 1) * self.memory
        stats["beta"] = 1 + (stats["beta"] - 1) * self.memory

        if outcome == "success":
            stats["alpha"] += 1
            stats["successes"] += 1
        else:
            stats["beta"] += 1
            stats["failures"] += 1

    def get_effectiveness(self, animal, deterrent):
        """Return the Bayesian posterior mean effectiveness."""
        stats = self._stats(animal, deterrent)
        return stats["alpha"] / (stats["alpha"] + stats["beta"])

    def sample(self, animal, deterrent, rng):
        """Draw one plausible effectiveness, given what we have seen (Thompson sampling)."""
        stats = self._stats(animal, deterrent)
        return rng.betavariate(stats["alpha"], stats["beta"])

    def get_statistics(self, animal, deterrent):
        """Return effectiveness, a 90% range around it, and the observation counts."""
        stats = self._stats(animal, deterrent)
        rng = random.Random(0)  # fixed seed: the same data always gives the same range
        draws = sorted(rng.betavariate(stats["alpha"], stats["beta"]) for _ in range(2000))
        return {
            **stats,
            "effectiveness": self.get_effectiveness(animal, deterrent),
            "low": draws[100],     # 5th percentile
            "high": draws[1899],   # 95th percentile
        }


# --------------------------------------------------
# Decision engine
# --------------------------------------------------

@dataclass
class Decision:
    action: str      # "none" or one of the deterrents
    threat: bool     # did the animal qualify for a response?
    kind: str        # "not_a_threat", "control" or "learned"
    reason: str      # human-readable, for the log and the screen
    estimates: dict = field(default_factory=dict)  # action -> estimated chance the animal leaves


class DecisionEngine:

    def __init__(self, encounter_log=None, settings: Settings = SETTINGS, rng: random.Random | None = None):
        self.settings = settings
        self.actions = settings.actions
        self.rng = rng or random.Random()
        self.model = EffectivenessModel(memory=settings.memory)

        # Rebuild the model from previous encounters.
        # This means learning persists across program restarts.
        if encounter_log is not None:
            for encounter in encounter_log.read():
                self.learn(encounter)

    def learn(self, encounter):
        """
        Learn from a completed encounter.

        Accepts either an Encounter object or a dictionary.
        """

        if hasattr(encounter, "to_dict"):
            encounter = encounter.to_dict()

        # Only threats are trials. Animals that were just passing teach us nothing
        # about deterrents.
        if not encounter.get("threat"):
            return

        deterrent = encounter.get("deterrent")
        if deterrent not in self.actions:
            return

        self.model.update(
            animal=encounter.get("animal"),
            deterrent=deterrent,
            outcome=evaluate_outcome(encounter),
        )

    def decide(self, animal, confidence, direction, position=None) -> Decision:
        """Choose a response and explain why."""

        # 1. Assess whether intervention is necessary.
        threat, why = assess_threat(animal, confidence, direction, position, self.settings)
        if not threat:
            return Decision("none", False, "not_a_threat", why)

        estimates = {a: self.model.get_effectiveness(animal, a) for a in self.actions}
        so_far = "chance it leaves: " + ", ".join(f"{a} {estimates[a]:.0%}" for a in self.actions)

        # 2. Control group: sometimes do nothing on purpose.
        if self.rng.random() < self.settings.control_share:
            return Decision("none", True, "control",
                            f"{why}; control trial: no deterrent, to measure what happens without one", estimates)

        # 3. Thompson sampling over none + every deterrent.
        draws = {a: self.model.sample(animal, a, self.rng) for a in self.actions}
        action = max(self.actions, key=lambda a: draws[a])

        conclusive = sum(self.model._stats(animal, a)["successes"] + self.model._stats(animal, a)["failures"]
                         for a in self.actions)
        best = max(self.actions, key=lambda a: estimates[a])
        if conclusive < 2 * len(self.actions):
            reason = f"still learning what works on {animal}: trying {action}"
        elif action == best:
            reason = (f"{action} works best on {animal} so far" if action != "none"
                      else f"no deterrent beats doing nothing on {animal} so far") + f" ({so_far})"
        else:
            reason = f"re-testing {action} on {animal}, in case things changed ({so_far})"
        return Decision(action, True, "learned", f"{why}; {reason}", estimates)

    def choose_deterrent(self, animal, confidence, direction, position=None):
        """Choose none or a deterrent based on behaviour and historical effectiveness."""
        return self.decide(animal, confidence, direction, position).action

    def get_effectiveness_report(self, animal):
        """
        Get the learned statistics for all deterrents
        for a particular animal.
        """

        return {
            action: self.model.get_statistics(animal, action)
            for action in self.actions
        }
