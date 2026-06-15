"""Run multiple prompt styles against the Maintenance Wizard agent."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard  # noqa: E402


PROMPTS = [
    {
        "case_id": "asset_diagnosis_gbx17",
        "prompt": "GBX-17 has abnormal vibration and temperature rise. Act as an autonomous maintenance agent: diagnose root cause, prioritize risk, plan spares, and generate an alert.",
    },
    {
        "case_id": "asset_diagnosis_mtr204",
        "prompt": "MTR-204 motor current and temperature are high. Give me the agent plan, risk score reasoning, RUL, immediate work order, and spare strategy.",
    },
    {
        "case_id": "plant_supervisor",
        "prompt": "Compare all equipment and tell the supervisor what to prioritize today. Include your tool calls and verifier checks.",
    },
    {
        "case_id": "spare_followup",
        "prompt": "For the same asset, what spare should I arrange next and why?",
    },
    {
        "case_id": "public_data_governance",
        "prompt": "What public dataset was used? Explain leakage control and how it is separated from steel app decisions.",
    },
]


def main() -> None:
    wizard = MaintenanceWizard()
    wizard.initialize(load_llm=False)

    rows = []
    raw_outputs = {}

    for item in PROMPTS:
        started = time.time()
        result = wizard.chat(item["prompt"], user_id="prompt_lab")
        latency = round(time.time() - started, 2)
        packet = result.get("decision_packet", {})
        rows.append(
            {
                "case_id": item["case_id"],
                "asset_id": result.get("asset_id"),
                "priority": result.get("priority"),
                "mode": packet.get("mode", result.get("mode", "")),
                "risk_level": packet.get("risk_level", result.get("risk_priority", {}).get("risk_level")),
                "tool_calls": len(result.get("tool_calls", [])),
                "verifier_checks": len(result.get("verifier_checks", [])),
                "agent_plan_steps": len(result.get("agent_plan", [])),
                "latency_sec": latency,
                "answer_preview": result.get("answer", "")[:500],
            }
        )
        raw_outputs[item["case_id"]] = result

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(exist_ok=True)
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(report_dir / "prompt_lab_summary.csv", index=False)
    (report_dir / "prompt_lab_outputs.json").write_text(json.dumps(raw_outputs, indent=2, default=str), encoding="utf-8")

    print("Prompt lab complete.")
    print(summary_df.to_string(index=False))
    print(f"\nSaved: {report_dir / 'prompt_lab_summary.csv'}")
    print(f"Saved: {report_dir / 'prompt_lab_outputs.json'}")


if __name__ == "__main__":
    main()
