"""Regression checks for task-first routing and anti-fallback verification.

Run from the project root:
    python scripts/regression_routing_tests.py
"""

from __future__ import annotations

import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("MW_USE_LLM", "0")

from backend.agent import MaintenanceWizard  # noqa: E402


TESTS = [
    {
        "name": "error_code_preserved",
        "prompt": "What does error code E-045 mean on the blast furnace blower motor, and what steps should I take immediately?",
        "mode_not": "plant_priority",
        "intent": "error_code_lookup",
        "must_contain": ["E-045", "blast furnace blower motor"],
        "must_not_contain": ["E-054", "GBX-17"],
    },
    {
        "name": "spares_not_plant_priority",
        "prompt": "What spare parts will I need if I replace the ladle car wheel assembly this weekend? What's the procurement lead time?",
        "mode_not": "plant_priority",
        "intent": "spare_procurement_query",
        "must_contain": ["ladle car wheel assembly"],
        "must_not_contain": ["Choose GBX-17"],
    },
    {
        "name": "emergency_stoppage_not_plant_priority",
        "prompt": "Our conveyor belt on line 3 just stopped. Walk me through the first checks I should do right now.",
        "mode_not": "plant_priority",
        "intent": "emergency_troubleshooting",
        "must_contain": ["conveyor belt on line 3"],
        "must_not_contain": ["Choose GBX-17"],
    },
    {
        "name": "logbook_no_asset_invention",
        "prompt": "Generate a digital logbook entry for today's planned maintenance on the EAF transformer cooling system. Technician: R. Kumar. Work done: oil top-up and fan belt inspection.",
        "mode": "logbook_template",
        "must_contain": ["EAF transformer cooling system", "R. Kumar", "oil top-up and fan belt inspection", "Asset ID: not provided"],
        "must_not_contain": ["Asset ID: GBX-17"],
    },
    {
        "name": "trend_rul_not_plant_priority",
        "prompt": "Lube oil temperature on the BOF tilting drive is trending: 52 C one week ago to 58 C yesterday to 63 C today. Predict remaining useful life and recommend when to intervene.",
        "mode_not": "plant_priority",
        "intent": "trend_rul_analysis",
        "must_contain": ["BOF tilting drive"],
        "must_not_contain": ["Choose GBX-17"],
    },
    {
        "name": "explicit_one_asset_keeps_plant_priority",
        "prompt": "If I can maintain only one asset today, which one should I choose and why?",
        "mode": "plant_priority",
        "must_contain": ["Choose"],
    },
]


def check_case(wizard: MaintenanceWizard, case: dict) -> tuple[bool, str]:
    result = wizard.chat(case["prompt"])
    text = (result.get("answer") or "") + "\n" + str(result.get("decision_packet") or {})
    mode = str(result.get("mode"))
    intent = str(result.get("intent"))
    failures: list[str] = []
    if case.get("mode") and mode != case["mode"]:
        failures.append(f"mode {mode!r} != {case['mode']!r}")
    if case.get("mode_not") and mode == case["mode_not"]:
        failures.append(f"mode unexpectedly {mode!r}")
    if case.get("intent") and intent != case["intent"]:
        failures.append(f"intent {intent!r} != {case['intent']!r}")
    for needle in case.get("must_contain", []):
        if needle.lower() not in text.lower():
            failures.append(f"missing {needle!r}")
    for needle in case.get("must_not_contain", []):
        if needle.lower() in text.lower():
            failures.append(f"forbidden {needle!r}")
    ok = not failures
    detail = "PASS" if ok else "FAIL: " + "; ".join(failures)
    return ok, detail


def main() -> int:
    wizard = MaintenanceWizard()
    passed = 0
    for case in TESTS:
        ok, detail = check_case(wizard, case)
        passed += int(ok)
        print(f"{case['name']}: {detail}")
    print(f"\nPassed {passed}/{len(TESTS)} routing/verifier regression checks.")
    return 0 if passed == len(TESTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
