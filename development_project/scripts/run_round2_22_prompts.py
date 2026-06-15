from __future__ import annotations

import json
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path


API_URL = "http://127.0.0.1:8600/api/chat"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"
RAW_PATH = REPORT_DIR / "round2_22_prompts_raw.json"
SUMMARY_PATH = REPORT_DIR / "round2_22_prompts_summary.json"
PROGRESS_PATH = REPORT_DIR / "round2_22_prompts_progress.log"


PROMPTS = [
    {
        "id": "easy_01_error_code",
        "difficulty": "easy",
        "prompt": "What does error code E-045 mean on the blast furnace blower motor, and what steps should I take immediately?",
        "expect": ["diagnosis", "steps"],
    },
    {
        "id": "easy_02_hydraulic_pump_seal_sop",
        "difficulty": "easy",
        "prompt": "Show me the standard SOP for replacing a hydraulic pump seal on the rolling mill.",
        "expect": ["sop", "steps"],
    },
    {
        "id": "easy_03_walking_beam_furnace_records",
        "difficulty": "easy",
        "prompt": "Pull up all maintenance records for the Walking Beam Furnace from the last 90 days and give me a summary.",
        "expect": ["history", "summary"],
    },
    {
        "id": "easy_04_ladle_car_spares",
        "difficulty": "easy",
        "prompt": "What spare parts will I need if I replace the ladle car wheel assembly this weekend? What's the procurement lead time?",
        "expect": ["spares", "lead_time"],
    },
    {
        "id": "easy_05_conveyor_first_checks",
        "difficulty": "easy",
        "prompt": "Our conveyor belt on line 3 just stopped. Walk me through the first checks I should do right now.",
        "expect": ["checks", "diagnosis"],
    },
    {
        "id": "easy_06_logbook_entry",
        "difficulty": "easy",
        "prompt": "Generate a digital logbook entry for today's planned maintenance on the EAF transformer cooling system. Technician: R. Kumar. Work done: oil top-up and fan belt inspection.",
        "expect": ["logbook"],
    },
    {
        "id": "medium_07_caster_vibration",
        "difficulty": "medium",
        "prompt": "The vibration sensor on the continuous caster mold oscillation unit is reading 12 mm/s RMS - double the normal 6 mm/s. Classify the risk level and tell me what to do.",
        "expect": ["risk", "action"],
    },
    {
        "id": "medium_08_bof_gas_alert",
        "difficulty": "medium",
        "prompt": "Generate an abnormal alert report for the BOF gas cleaning system - dust collector differential pressure has spiked 3 times this shift above 1.8 kPa.",
        "expect": ["alert", "risk"],
    },
    {
        "id": "medium_09_hot_strip_pattern",
        "difficulty": "medium",
        "prompt": "We've had 4 unplanned stoppages on the hot strip mill work roll chock bearings in 6 weeks. Analyze the incident records and identify any common pattern.",
        "expect": ["pattern", "root_cause"],
    },
    {
        "id": "medium_10_bof_tilting_rul",
        "difficulty": "medium",
        "prompt": "Lube oil temperature on the BOF tilting drive is trending: 52 C (1 week ago) -> 58 C (yesterday) -> 63 C (today). Predict remaining useful life and recommend when to intervene.",
        "expect": ["rul", "intervention"],
    },
    {
        "id": "medium_11_three_jobs_two_crews",
        "difficulty": "medium",
        "prompt": "We have 3 critical maintenance jobs and only 2 crews available this weekend. Jobs: (A) caster roll alignment - 8hr, high criticality; (B) reheat furnace burner - 4hr, medium criticality; (C) descaler pump seal - 2hr, spares in stock. Prioritize.",
        "expect": ["prioritize", "constraints"],
    },
    {
        "id": "medium_12_supervisor_summary",
        "difficulty": "medium",
        "prompt": "Prepare a maintenance decision summary for the plant supervisor covering this week's top 3 equipment risks, actions taken, and pending items.",
        "expect": ["summary", "top_3"],
    },
    {
        "id": "medium_13_slab_pitting",
        "difficulty": "medium",
        "prompt": "I want to understand what's causing slab surface pitting on caster #2. Walk me through which equipment conditions could be contributing to this defect.",
        "expect": ["defect", "root_cause"],
    },
    {
        "id": "hard_14_fume_extraction_repeated_failures",
        "difficulty": "hard",
        "prompt": "The fume extraction fan has had 3 impeller failures in 18 months, each time following the SOP correctly. What systematic issue could explain repeated failures despite correct procedure, and what should change?",
        "expect": ["systemic", "sop_change"],
    },
    {
        "id": "hard_15_full_risk_procurement",
        "difficulty": "hard",
        "prompt": "Bearing temp is 87 C (limit 90), vibration 9.5 mm/s (limit 10), oil pressure 2.1 bar (normal 2.5). Replacement part has a 14-day procurement lead time and we run 24/7. Give me a full risk assessment and maintenance plan.",
        "expect": ["risk", "rul", "procurement"],
    },
    {
        "id": "hard_16_missed_prediction_postmortem",
        "difficulty": "hard",
        "prompt": "Two weeks ago you rated torpedo ladle car bearings as low risk. Yesterday one failed catastrophically, causing a 6-hour production loss. Conduct a post-mortem on the missed prediction.",
        "expect": ["postmortem", "feedback"],
    },
    {
        "id": "hard_17_spindle_deferral_scenario",
        "difficulty": "hard",
        "prompt": "Run a scenario for me: if we defer the roughing mill spindle coupling replacement by 2 more weeks, what is the estimated probability of failure and what's the worst-case production impact?",
        "expect": ["scenario", "probability", "impact"],
    },
    {
        "id": "hard_18_multiturn_pickle_line_motor",
        "difficulty": "hard",
        "turns": [
            "Our pickle line tension reel motor is making an unusual noise. What could it be?",
            "The noise is intermittent, worse at startup, and we noticed slight overheating last week. Does that change your diagnosis?",
        ],
        "expect": ["multiturn", "diagnosis_change"],
    },
    {
        "id": "expert_19_skid_pipe_rul_schedule",
        "difficulty": "expert",
        "prompt": "Based on 6 months of sensor history on the reheat furnace skid pipes, predict the remaining useful life per pipe section and build a proactive replacement schedule that avoids production curtailment.",
        "expect": ["rul", "schedule"],
    },
    {
        "id": "expert_20_cbm_framework",
        "difficulty": "expert",
        "prompt": "Design a condition-based monitoring and early warning framework for our 8 rolling mill drives - including sensor thresholds, escalation rules, and what notifications go to field engineers vs the plant supervisor.",
        "expect": ["framework", "thresholds", "notifications"],
    },
    {
        "id": "expert_21_shutdown_schedule",
        "difficulty": "expert",
        "prompt": "We're planning our 48-hour annual shutdown. We have 47 open maintenance jobs, 6 crews, 3 crane slots, 12 items on critical path, and 5 jobs with long-lead spares not yet arrived. Build an optimised shutdown schedule with risk flags and contingency recommendations.",
        "expect": ["schedule", "risk_flags", "contingency"],
    },
    {
        "id": "expert_22_cbm_roadmap",
        "difficulty": "expert",
        "prompt": "We want to shift 12 assets from time-based to condition-based maintenance. Using their failure history and sensor profiles, identify which assets are ready for the transition and propose a phased CBM roadmap.",
        "expect": ["cbm", "roadmap"],
    },
]


def post_prompt(prompt: str, user_id: str) -> dict:
    payload = json.dumps({"prompt": prompt, "user_id": user_id}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=260) as response:
        return json.loads(response.read().decode("utf-8"))


def append_progress(message: str) -> None:
    print(message, flush=True)
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def contains_any(text: str, words: list[str]) -> bool:
    lower = text.lower()
    return any(word.replace("_", " ") in lower or word in lower for word in words)


def assess_case(case: dict, outputs: list[dict]) -> dict:
    combined_answer = "\n\n".join(str(item.get("answer", "")) for item in outputs)
    combined_lower = combined_answer.lower()
    modes = [str(item.get("mode", "")) for item in outputs]
    verifier_checks = []
    for item in outputs:
        verifier_checks.extend(item.get("verifier_checks") or [])
    has_fail_check = any(str(check.get("status", "")).lower() == "fail" for check in verifier_checks if isinstance(check, dict))
    no_bad_tokens = not any(token in combined_lower for token in ["nan", "traceback", "exception", "unexpected token"])
    nonempty = len(combined_answer.strip()) >= 120
    llm_used = any(bool(item.get("llm_used")) for item in outputs)
    evidence_or_tools = any((item.get("tool_calls") or item.get("retrieved_docs") or item.get("decision_packet")) for item in outputs)
    expect_hit = contains_any(combined_lower, case.get("expect", []))
    quality_hits = sum([nonempty, no_bad_tokens, not has_fail_check, llm_used, evidence_or_tools, expect_hit])
    passed = quality_hits >= 5
    return {
        "passed": passed,
        "quality_hits": quality_hits,
        "nonempty": nonempty,
        "no_bad_tokens": no_bad_tokens,
        "has_fail_check": has_fail_check,
        "llm_used": llm_used,
        "evidence_or_tools": evidence_or_tools,
        "expect_hit": expect_hit,
        "modes": modes,
        "answer_preview": combined_answer[:900],
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_PATH.write_text("", encoding="utf-8")
    append_progress(f"Starting 22-prompt run at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    raw_results = []
    start = time.time()
    for index, case in enumerate(PROMPTS, 1):
        case_start = time.time()
        outputs = []
        append_progress(f"[{index:02d}/22] RUN {case['id']}")
        try:
            turns = case.get("turns") or [case["prompt"]]
            user_id = f"round2_22_{case['id']}"
            for turn_index, prompt in enumerate(turns, 1):
                turn_start = time.time()
                result = post_prompt(prompt, user_id=user_id)
                result["_turn_index"] = turn_index
                result["_latency_sec"] = round(time.time() - turn_start, 2)
                outputs.append(result)
            assessment = assess_case(case, outputs)
            status = "PASS" if assessment["passed"] else "REVIEW"
            append_progress(
                f"[{index:02d}/22] {status} {case['id']} "
                f"mode={','.join(assessment['modes'])} latency={round(time.time() - case_start, 2)}s"
            )
            raw_results.append(
                {
                    "index": index,
                    "id": case["id"],
                    "difficulty": case["difficulty"],
                    "prompts": case.get("turns") or [case["prompt"]],
                    "assessment": assessment,
                    "outputs": outputs,
                    "latency_sec": round(time.time() - case_start, 2),
                    "error": "",
                }
            )
        except Exception as exc:
            append_progress(f"[{index:02d}/22] ERROR {case['id']}: {exc}")
            raw_results.append(
                {
                    "index": index,
                    "id": case["id"],
                    "difficulty": case["difficulty"],
                    "prompts": case.get("turns") or [case.get("prompt", "")],
                    "assessment": {"passed": False, "quality_hits": 0, "modes": [], "answer_preview": ""},
                    "outputs": [],
                    "latency_sec": round(time.time() - case_start, 2),
                    "error": traceback.format_exc(),
                }
            )
        RAW_PATH.write_text(json.dumps(raw_results, indent=2), encoding="utf-8")

    passed = sum(1 for row in raw_results if row["assessment"].get("passed"))
    failed = len(raw_results) - passed
    summary = {
        "prompt_count": len(raw_results),
        "pass_count": passed,
        "review_or_fail_count": failed,
        "pass_rate_pct": round(100 * passed / max(len(raw_results), 1), 2),
        "duration_sec": round(time.time() - start, 2),
        "avg_latency_sec": round(sum(row["latency_sec"] for row in raw_results) / max(len(raw_results), 1), 2),
        "llm_used_count": sum(1 for row in raw_results if row["assessment"].get("llm_used")),
        "modes": {},
        "review_cases": [
            {
                "id": row["id"],
                "difficulty": row["difficulty"],
                "modes": row["assessment"].get("modes", []),
                "quality_hits": row["assessment"].get("quality_hits"),
                "error": row.get("error", ""),
                "preview": row["assessment"].get("answer_preview", "")[:500],
            }
            for row in raw_results
            if not row["assessment"].get("passed")
        ],
    }
    for row in raw_results:
        for mode in row["assessment"].get("modes", []):
            summary["modes"][mode] = summary["modes"].get(mode, 0) + 1
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    append_progress(f"Completed: {passed}/{len(raw_results)} passed, {failed} review/fail")
    append_progress(f"Raw: {RAW_PATH}")
    append_progress(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
