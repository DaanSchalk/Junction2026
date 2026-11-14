
import random
from collections import defaultdict


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ACTIONS = ("none", "sound", "light")

# Animals that should never receive a deterrent.
# Replace these examples with your actual conservation policy.
ALLOWED_ANIMALS = {"deer"}
CONFIDENCE_THRESHOLD = 0.80

# Number of eligible encounters per action before
# the engine starts choosing based on effectiveness.
MIN_TRIALS_PER_ACTION = 3


# --------------------------------------------------
# Threat assessment
# --------------------------------------------------

def is_threatening(
    animal,
    confidence,
    approach,
):
    """
    Determine whether an animal qualifies for intervention.
    """

    # Never intervene against allowed animals.
    if animal in ALLOWED_ANIMALS:
        return False

    # Ignore uncertain detections.
    if confidence < CONFIDENCE_THRESHOLD:
        return False

    # Only intervene when the animal is approaching.
    if approach != "approaching":
        return False

    return True


# --------------------------------------------------
# Outcome evaluation
# --------------------------------------------------

def evaluate_outcome(encounter):
    """
    Evaluate the animal's behaviour after an encounter.

    Success:
        Animal was approaching before and leaving afterwards.

    Failure:
        Animal was approaching before and continues approaching.

    Inconclusive:
        Any other result.
    """

    if encounter.get("approach_before") != "approaching":
        return "inconclusive"

    approach_after = encounter.get("approach_after")

    if approach_after == "leaving":
        return "success"

    if approach_after == "approaching":
        return "failure"

    return "inconclusive"


# --------------------------------------------------
# Bayesian effectiveness model
# --------------------------------------------------

class EffectivenessModel:

    def __init__(self):
        self.estimates = defaultdict(
            lambda: {
                "alpha": 1,
                "beta": 1,
                "trials": 0,
                "successes": 0,
                "failures": 0,
                "inconclusive": 0,
            }
        )

    def update(self, animal, deterrent, outcome):
        """
        Update the Beta distribution for an animal-deterrent pair.
        """

        key = (animal, deterrent)
        stats = self.estimates[key]

        # Count every eligible trial, including inconclusive ones.
        stats["trials"] += 1

        if outcome == "success":
            stats["alpha"] += 1
            stats["successes"] += 1

        elif outcome == "failure":
            stats["beta"] += 1
            stats["failures"] += 1

        else:
            # Inconclusive encounters do not change alpha or beta.
            stats["inconclusive"] += 1

    def get_effectiveness(self, animal, deterrent):
        """
        Return the Bayesian posterior mean effectiveness.
        """

        stats = self.estimates[(animal, deterrent)]

        alpha = stats["alpha"]
        beta = stats["beta"]

        return alpha / (alpha + beta)

    def get_statistics(self, animal, deterrent):
        """
        Return effectiveness and supporting observation counts.
        """

        stats = self.estimates[(animal, deterrent)]

        return {
            **stats,
            "effectiveness": self.get_effectiveness(
                animal, deterrent
            ),
        }


# --------------------------------------------------
# Decision engine
# --------------------------------------------------

class DecisionEngine:

    def __init__(self, encounter_log):
        self.model = EffectivenessModel()

        # Rebuild the model from previous encounters.
        # This means learning persists across program restarts.
        for encounter in encounter_log.read():
            self.learn(encounter)

    def learn(self, encounter):
        """
        Learn from a completed encounter.

        Accepts either an Encounter object or a dictionary.
        """

        if hasattr(encounter, "to_dict"):
            encounter = encounter.to_dict()

        animal = encounter.get("animal")
        deterrent = encounter.get("deterrent")
        confidence = encounter.get("confidence", 0)

        # Only learn from valid, eligible encounters.
        if animal in ALLOWED_ANIMALS:
            return

        if confidence < CONFIDENCE_THRESHOLD:
            return

        if encounter.get("approach_before") != "approaching":
            return

        if deterrent not in ACTIONS:
            return

        outcome = evaluate_outcome(encounter)

        self.model.update(
            animal=animal,
            deterrent=deterrent,
            outcome=outcome,
        )

    def choose_deterrent(
        self,
        animal,
        confidence,
        approach,
    ):
        """
        Choose none, sound or light based on behaviour
        and historical effectiveness.
        """

        # 1. Assess whether intervention is necessary.
        if not is_threatening(
            animal,
            confidence,
            approach,
        ):
            return "none"

        # 2. Explore actions that have been tested least.
        trial_counts = {
            action: self.model.get_statistics(
                animal, action
            )["trials"]
            for action in ACTIONS
        }

        least_tested = min(trial_counts.values())

        if least_tested < MIN_TRIALS_PER_ACTION:
            candidates = [
                action
                for action in ACTIONS
                if trial_counts[action] == least_tested
            ]

            return random.choice(candidates)

        # 3. Compare each deterrent against doing nothing.
        baseline = self.model.get_effectiveness(
            animal, "none"
        )

        scores = {
            "none": 0.0,
            "sound": (
                self.model.get_effectiveness(animal, "sound")
                - baseline
            ),
            "light": (
                self.model.get_effectiveness(animal, "light")
                - baseline
            ),
        }

        # 4. Choose the best option.
        # Ties favour less intervention: none, then sound, then light.
        best_action = max(
            ACTIONS,
            key=lambda action: (
                scores[action],
                -ACTIONS.index(action),
            ),
        )

        # Do not intervene if neither deterrent beats the baseline.
        if scores[best_action] <= 0:
            return "none"

        return best_action

    def get_effectiveness_report(self, animal):
        """
        Get the learned statistics for all deterrents
        for a particular animal.
        """

        return {
            action: self.model.get_statistics(animal, action)
            for action in ACTIONS
        }
