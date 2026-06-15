"""Build Qwen instruction-tuning data from steel maintenance evidence.

The goal is not to fine-tune on raw CSV rows. The model should learn how a
maintenance agent writes grounded decisions from locked facts: risk, RUL,
evidence confidence, SOP/history/spares context, active memory, and missing
evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DATA_DIR, DOC_DIR, REPORT_DIR
from .decision_intelligence import build_decision_intelligence_table, top_original_vs_dynamic


SYSTEM_PROMPT = (
    "You are a serious agentic AI maintenance copilot for steel manufacturing. "
    "Use only the provided evidence. Do not invent history, SOPs, spares, sensor values, or root causes. "
    "When evidence is missing, say what is missing and give a safe next action."
)


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _clean_json_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_json_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json_value(v) for v in obj]
    if isinstance(obj, tuple):
        return [_clean_json_value(v) for v in obj]
    try:
        if pd.isna(obj):
            return "not_provided"
    except Exception:
        pass
    if hasattr(obj, "item"):
        try:
            return _clean_json_value(obj.item())
        except Exception:
            pass
    return obj


def _json(obj: Any) -> str:
    return json.dumps(_clean_json_value(obj), ensure_ascii=False, default=str, allow_nan=False)


def _asset_context(asset_id: str, decision_table: pd.DataFrame) -> dict[str, Any]:
    rows = decision_table[decision_table["asset_id"].astype(str) == str(asset_id)]
    if rows.empty:
        return {"asset_id": asset_id, "evidence": "No decision-intelligence row available."}
    row = rows.iloc[0].to_dict()

    history = _read_csv(DATA_DIR / "maintenance_history.csv")
    failures = _read_csv(DATA_DIR / "failure_reports.csv")
    spares = _read_csv(DATA_DIR / "spares_inventory.csv")

    history_rows = history[history.get("asset_id", pd.Series(dtype=str)).astype(str) == str(asset_id)].head(3).to_dict("records") if not history.empty else []
    failure_rows = failures[failures.get("asset_id", pd.Series(dtype=str)).astype(str) == str(asset_id)].head(3).to_dict("records") if not failures.empty else []
    spare_rows = spares[spares.get("asset_id", pd.Series(dtype=str)).astype(str) == str(asset_id)].head(5).to_dict("records") if not spares.empty else []
    docs = []
    for p in list(DOC_DIR.glob(f"*{asset_id.replace('-', '_')}*.txt")) + list(DOC_DIR.glob(f"*{asset_id}*.txt")):
        docs.append({"source": p.name, "excerpt": p.read_text(encoding="utf-8", errors="ignore")[:900]})

    return {
        "decision": row,
        "maintenance_history": history_rows,
        "failure_reports": failure_rows,
        "spares": spare_rows,
        "sop_excerpts": docs,
    }


def _assistant_asset_report(asset_id: str, context: dict[str, Any]) -> str:
    d = context.get("decision", {})
    missing = str(d.get("missing_evidence", "")).strip() or "none"
    return (
        f"{asset_id} should be treated as {d.get('priority', 'REVIEW')}/{d.get('risk_band', 'REVIEW')} based on the available maintenance evidence.\n\n"
        f"The current decision score is {d.get('decision_score')} with failure risk {d.get('failure_risk')} and estimated RUL "
        f"{d.get('estimated_rul_days')} day(s). The main drivers are alarm count {d.get('alarm_count')}, vibration {d.get('vibration')}, "
        f"temperature {d.get('temperature')}, delay impact INR {d.get('delay_cost_impact_inr')}, and procurement risk {d.get('procurement_risk')}.\n\n"
        f"Evidence confidence is {d.get('evidence_confidence')}. Missing evidence: {missing}. "
        f"Do not invent unavailable records; verify missing items before intrusive work.\n\n"
        f"Recommended next action: {d.get('next_system_action')}\n\n"
        f"Spare plan: {d.get('spare_plan')}"
    )


def _messages(user: str, assistant: str, system: str = SYSTEM_PROMPT) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def build_qwen_sft_examples(output_jsonl: str | Path | None = None, min_examples: int = 120) -> list[dict[str, Any]]:
    """Create instruction examples and optionally save JSONL."""

    decision_table = build_decision_intelligence_table()
    examples: list[dict[str, Any]] = []

    for _, row in decision_table.iterrows():
        asset_id = str(row["asset_id"])
        context = _asset_context(asset_id, decision_table)
        ctx_text = _json(context)
        examples.extend(
            [
                _messages(
                    f"Create a maintenance decision report for {asset_id}. Use this evidence packet and do not invent missing data:\n{ctx_text}",
                    _assistant_asset_report(asset_id, context),
                ),
                _messages(
                    f"Estimate risk, RUL, priority, procurement concern, and first action for {asset_id}. Evidence:\n{ctx_text}",
                    _assistant_asset_report(asset_id, context),
                ),
                _messages(
                    f"An engineer asks: should {asset_id} be maintained today? Answer with evidence confidence and missing evidence. Evidence:\n{ctx_text}",
                    _assistant_asset_report(asset_id, context),
                ),
            ]
        )

    comparison = top_original_vs_dynamic()
    examples.append(
        _messages(
            "Compare the highest-risk original demo asset with the highest-risk dynamic asset. Pick one for immediate maintenance. Evidence packet:\n"
            + _json(comparison),
            _comparison_answer(comparison),
        )
    )

    examples.extend(_memory_examples())
    examples.extend(_general_steel_examples())

    # Repeat varied examples when the local demo has few assets. This is small-data SFT,
    # so we keep repeats explicit and deterministic rather than pretending to have more data.
    base = list(examples)
    idx = 0
    while len(examples) < min_examples and base:
        examples.append(base[idx % len(base)])
        idx += 1

    if output_jsonl is None:
        output_jsonl = DATA_DIR / "qwen3_8b_maintenance_sft.jsonl"
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "qwen_sft_dataset_report.json").write_text(
        json.dumps({"examples": len(examples), "output_jsonl": str(output_path)}, indent=2),
        encoding="utf-8",
    )
    return examples


def _comparison_answer(packet: dict[str, Any]) -> str:
    original = packet.get("top_original") or {}
    dynamic = packet.get("top_dynamic") or {}
    winner = packet.get("winner") or original or dynamic
    second = dynamic if winner == original else original
    return (
        f"Choose {winner.get('asset_id', 'the highest-scored asset')} first for immediate maintenance.\n\n"
        f"Top original asset: {original.get('asset_id', 'none')} with score {original.get('decision_score')}, "
        f"priority {original.get('priority')}, RUL {original.get('estimated_rul_days')} day(s), "
        f"evidence confidence {original.get('evidence_confidence')}.\n\n"
        f"Top dynamic asset: {dynamic.get('asset_id', 'none')} with score {dynamic.get('decision_score')}, "
        f"priority {dynamic.get('priority')}, RUL {dynamic.get('estimated_rul_days')} day(s), "
        f"evidence confidence {dynamic.get('evidence_confidence')}.\n\n"
        f"The winner beats the second-ranked asset because its combined decision score, RUL urgency, production delay/procurement risk, "
        f"and rule-adjusted condition create the stronger immediate maintenance case. Missing evidence should remain visible and must not be invented."
    )


def _memory_examples() -> list[dict[str, Any]]:
    return [
        _messages(
            "Add a new asset BFB-21, blast furnace blower, vibration 7.4 mm/s, temperature 86 C, current 91 A, pressure 8.1 bar, alarm count 3. What should the agent do?",
            "Store BFB-21 in dynamic asset memory, preserve the source query, score it as an active dynamic asset, and include it in future active ranking. Because it is a blast furnace blower with high vibration and temperature, it should be reviewed for P1/CRITICAL treatment if remembered safety rules match. Missing items such as historical work orders, SOP, and spares must be reported instead of invented.",
        ),
        _messages(
            "Update DES-08: vibration reduced to 4.2 mm/s, pressure improved to 6.8 bar, alarm count reduced to 1, but operator reports loud cavitation noise. Did risk actually reduce?",
            "Risk reduces numerically because vibration, pressure, and alarms improved, but it should not be closed automatically. The operator cavitation report is fresh qualitative evidence, so keep DES-08 elevated for inspection until suction strainer, tank level, air ingress, seal leakage, and impeller condition are checked. Record both old and new readings in dynamic asset history.",
        ),
        _messages(
            "Rank active dynamic assets only. Do not include original demo assets.",
            "Rank only records from dynamic memory where active is true. Exclude GBX-17, MTR-204, PMP-09, and HPP-12 even if their scores are higher. Show priority, score, RUL, applied rules, evidence confidence, and missing evidence for each dynamic asset.",
        ),
        _messages(
            "Remember this safety rule: any blast furnace blower with vibration above 6.5 and temperature above 80 must be P1. How should duplicates be handled?",
            "Normalize the rule into a stable rule key using equipment scope, area scope, thresholds, and override. If an equivalent active rule already exists, report that it is already remembered and do not create a duplicate. Apply the rule only when both vibration and temperature conditions match the asset state.",
        ),
    ]


def _general_steel_examples() -> list[dict[str, Any]]:
    return [
        _messages(
            "What does error code E-045 mean on a blast furnace blower motor and what steps should I take immediately?",
            "I do not have a site-specific E-045 code table in the provided evidence, so I cannot claim a definitive meaning. Treat this as a blower motor fault requiring safe triage: check HMI alarm detail, motor protection relay, vibration, bearing temperature, current imbalance, damper position, cooling airflow, and standby blower status. If vibration or temperature is above safety limits, notify the furnace control room, prepare a P1 inspection, and verify lockout requirements before intrusive work.",
        ),
        _messages(
            "Our conveyor belt on line 3 just stopped. Walk me through the first checks right now.",
            "First make the area safe: stop upstream feed, barricade the belt, and confirm no trapped personnel or material slide risk. Then check E-stop/pull-cord status, motor overload or VFD trip, belt tracking, blocked chute, pulley/idler seizure, gearbox oil/temperature, coupling, brake release, and local sensor status. Do not restart until guards are in place, jammed material is cleared, and the operator confirms the line is safe.",
        ),
        _messages(
            "Generate a digital logbook entry for planned maintenance on the EAF transformer cooling system. Technician R. Kumar. Work done: oil top-up and fan belt inspection.",
            "Create a draft logbook entry for EAF transformer cooling system: technician R. Kumar performed oil top-up and fan belt inspection. Mark status as pending verification until post-maintenance oil level, fan belt condition, cooling fan run check, temperature trend, leakage check, and supervisor acceptance are captured. Do not mark the job closed without closure evidence.",
        ),
        _messages(
            "Lube oil temperature on the BOF tilting drive is trending 52 C to 58 C to 63 C. Predict RUL and recommend when to intervene.",
            "The rising lube oil temperature trend is an early warning, but exact RUL cannot be calculated without baseline limits, vibration, oil analysis, load, and alarm history. Treat it as a short-horizon risk: inspect within the next shift, check oil level, cooler condition, filter differential pressure, gearbox/bearing temperature, vibration, and contamination. If temperature continues rising or vibration/alarms appear, escalate to P1 and prepare controlled shutdown.",
        ),
    ]


def load_sft_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
