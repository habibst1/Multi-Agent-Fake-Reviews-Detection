"""Run one fake-world scenario through the original or the migrated pipeline.

Usage: python tests/run_scenario.py {legacy|new} <scenario> <output.json> [--sequential]

Writes the pipeline's result and every prompt it sent to the LLM to <output.json>.
"""

import json
import os
import sys
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


def main() -> None:
    impl, scenario_name, output_path = sys.argv[1:4]
    sequential = "--sequential" in sys.argv[4:]

    import fakes
    world = fakes.World(fakes.SCENARIOS[scenario_name])

    if impl == "legacy":
        # The original code reads "Hotel Reviews with claims" from the working directory
        os.chdir(FIXTURES_DIR)
        sys.path.insert(0, str(TESTS_DIR / "legacy"))
    elif impl == "new":
        os.environ["REVIEWS_DIR"] = str(FIXTURES_DIR / "Hotel Reviews with claims")
        sys.path.insert(0, str(PROJECT_ROOT))
    else:
        raise SystemExit(f"Unknown implementation: {impl}")
    fakes.install(world, impl)

    import main as pipeline
    scenario = world.scenario
    if impl == "legacy":
        result = pipeline.hotel_review_fact_checker(scenario["hotel"], scenario["review"])
    else:
        result = pipeline.hotel_review_fact_checker(scenario["hotel"], scenario["review"], sequential=sequential)

    output = {"result": result, "prompts": world.prompts}
    Path(output_path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
