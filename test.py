
from data_collection import EncounterLog, LOG_FILE
from decision_engine import (
    DecisionEngine,
    ACTIONS,
    ALLOWED_ANIMALS,
)


def print_effectiveness_report(engine, animals):
    print("\n" + "=" * 65)
    print("EFFECTIVENESS REPORT")
    print("=" * 65)

    for animal in sorted(animals):
        print(f"\nAnimal: {animal}")
        print("-" * 45)

        for action in ACTIONS:
            stats = engine.model.get_statistics(animal, action)

            print(
                f"{action:10} | "
                f"Trials: {stats['trials']:3} | "
                f"Successes: {stats['successes']:3} | "
                f"Failures: {stats['failures']:3} | "
                f"Inconclusive: {stats['inconclusive']:3} | "
                f"Effectiveness: {stats['effectiveness']:.2%}"
            )


def test_non_threatening_behaviour(engine):
    print("\n" + "=" * 65)
    print("NON-THREATENING BEHAVIOUR TESTS")
    print("=" * 65)

    test_cases = [
        ("Approaching with low confidence", "fox", 0.40, "approaching"),
        ("Leaving the protected area", "fox", 0.95, "leaving"),
        ("Moving in the same direction", "fox", 0.95, "same"),
        ("Unknown approach", "fox", 0.95, "unknown"),
    ]

    for description, animal, confidence, approach in test_cases:
        action = engine.choose_deterrent(
            animal=animal,
            confidence=confidence,
            approach=approach,
        )

        passed = action == "none"

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{description:35} | Selected: {action}"
        )

        assert passed, f"Unexpected intervention: {description}"


def test_allowed_animals(engine):
    print("\n" + "=" * 65)
    print("ALLOWED ANIMAL TEST")
    print("=" * 65)

    if not ALLOWED_ANIMALS:
        print("SKIPPED: ALLOWED_ANIMALS is empty.")
        print("Configure at least one allowed animal to test this rule.")
        return

    for animal in ALLOWED_ANIMALS:
        action = engine.choose_deterrent(
            animal=animal,
            confidence=0.99,
            approach="approaching",
        )

        passed = action == "none"

        print(
            f"{'PASS' if passed else 'FAIL'} | "
            f"{animal:15} | Selected: {action}"
        )

        assert passed, f"Intervention selected for allowed animal: {animal}"


def test_threatening_behaviour(engine, animals):
    print("\n" + "=" * 65)
    print("THREATENING BEHAVIOUR TEST")
    print("=" * 65)

    eligible_animals = animals - ALLOWED_ANIMALS

    if not eligible_animals:
        print("SKIPPED: No non-allowed animals found in the dataset.")
        return

    for animal in sorted(eligible_animals):
        action = engine.choose_deterrent(
            animal=animal,
            confidence=0.99,
            approach="approaching",
        )

        print(f"{animal:15} | Selected action: {action}")


def main():
    print("=" * 65)
    print("DECISION ENGINE TEST")
    print("=" * 65)

    print(f"\nLoading mock data from: {LOG_FILE}")

    # Load historical encounters.
    encounter_log = EncounterLog()
    records = encounter_log.read()

    if not records:
        print("No encounters found. Check your encounters.jsonl file.")
        return

    print(f"Loaded {len(records)} encounters.")

    # Initialise the engine. It learns from the existing JSONL records.
    engine = DecisionEngine(encounter_log)

    animals = {
        record["animal"]
        for record in records
        if record.get("animal")
    }

    print(f"Animals found: {', '.join(sorted(animals))}")

    # 1. Check historical effectiveness estimates.
    print_effectiveness_report(engine, animals)

    # 2. Check that non-threatening situations produce no intervention.
    test_non_threatening_behaviour(engine)

    # 3. Check that allowed animals never receive an intervention.
    test_allowed_animals(engine)

    # 4. Check decisions for potentially threatening situations.
    test_threatening_behaviour(engine, animals)

    print("\n" + "=" * 65)
    print("TEST SCRIPT COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
