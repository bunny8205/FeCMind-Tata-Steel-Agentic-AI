"""General steel-domain agent layer.

This module handles broad steel plant prompts that are not tied to one of the
demo asset IDs. It gives the app a Codex-like "agent workspace" for steel
maintenance, process reliability, safety, procurement, and operations requests.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Iterable


STEEL_TERMS = {
    "steel",
    "plant",
    "mill",
    "blast furnace",
    "bf",
    "bof",
    "eaf",
    "caster",
    "casting",
    "continuous caster",
    "slab",
    "surface pitting",
    "pitting",
    "tundish",
    "ladle",
    "mold",
    "mould",
    "rolling",
    "hot strip",
    "cold rolling",
    "plate mill",
    "finishing mill",
    "sinter",
    "pellet",
    "coke oven",
    "reheating furnace",
    "reheat furnace",
    "skid pipe",
    "skid pipes",
    "crane",
    "conveyor",
    "descaler",
    "hydraulic",
    "gearbox",
    "motor",
    "drive",
    "drives",
    "pump",
    "bearing",
    "impeller",
    "spindle",
    "coupling",
    "tension reel",
    "work roll",
    "chock",
    "lubrication",
    "cooling water",
    "vibration",
    "temperature",
    "pressure",
    "current",
    "scada",
    "plc",
    "agent",
    "agentic",
    "architecture",
    "workflow",
    "system design",
    "condition-based",
    "condition based",
    "cbm",
    "roadmap",
    "monitoring framework",
    "early warning",
    "threshold",
    "thresholds",
    "escalation",
    "notifications",
    "sop",
    "rca",
    "root cause",
    "maintenance",
    "equipment",
    "asset",
    "operator",
    "supervisor",
    "shift",
    "downtime",
    "reliability",
    "availability",
    "mtbf",
    "mttr",
    "breakdown",
    "shutdown",
    "spares",
    "procurement",
    "work order",
    "rul",
    "failure",
    "defect",
    "quality",
    "refractory",
    "tuyere",
    "stave",
    "safety",
    "loto",
}


INTENT_RULES = [
    (
        "incident_report",
        ["incident", "shift report", "report", "summary", "handover", "explain to supervisor"],
    ),
    (
        "root_cause_analysis",
        ["rca", "root cause", "why", "cause", "fishbone", "5 why", "failure analysis"],
    ),
    (
        "sop_generation",
        ["sop", "procedure", "checklist", "standard operating", "step by step", "inspection checklist"],
    ),
    (
        "work_order_planning",
        ["work order", "maintenance plan", "repair plan", "shutdown plan", "plan the job", "action plan", "schedule"],
    ),
    (
        "spares_procurement",
        ["spare", "procurement", "lead time", "inventory", "stock", "purchase", "reserve"],
    ),
    (
        "risk_prioritization",
        ["prioritize", "priority", "risk", "critical", "bottleneck", "what first", "urgent"],
    ),
    (
        "failure_prediction",
        ["predict", "rul", "remaining useful", "early warning", "anomaly", "alarm", "alarms", "breakout", "degradation", "forecast"],
    ),
    (
        "safety_control",
        ["safety", "loto", "permit", "isolation", "hazard", "unsafe", "confined", "hot work"],
    ),
    (
        "process_quality",
        ["defect", "quality", "surface", "crack", "scale", "camber", "thickness", "flatness"],
    ),
    (
        "data_agent_design",
        ["architecture", "agent", "workflow", "design", "system", "data flow", "model"],
    ),
]


SUBJECT_PATTERNS = [
    (r"annual\s+shutdown|shutdown\s+schedule|48[-\s]?hour\s+shutdown|critical\s+path", "Annual Shutdown Planning"),
    (r"blast\s+furnace|bf\b", "Blast Furnace"),
    (r"\bbof\b|basic\s+oxygen", "BOF Converter"),
    (r"\beaf\b|electric\s+arc", "Electric Arc Furnace"),
    (r"slab.*pitting|surface\s+pitting|pitting", "Caster Slab Surface Quality"),
    (r"continuous\s+caster|caster|casting|mould|mold|tundish", "Continuous Caster"),
    (r"rolling\s+mill\s+drives?|mill\s+drives?", "Rolling Mill Drives"),
    (r"rolling|hot\s+strip|cold\s+rolling|plate\s+mill|finishing\s+mill", "Rolling Mill"),
    (r"gearbox|gear\s+box|gear", "Gearbox"),
    (r"motor|drive", "Motor or Drive"),
    (r"pump|cavitation", "Pump System"),
    (r"hydraulic|power\s+pack|actuator", "Hydraulic System"),
    (r"crane|hoist", "Crane and Hoist"),
    (r"conveyor|belt", "Conveyor System"),
    (r"skid\s+pipes?", "Reheat Furnace Skid Pipes"),
    (r"reheating\s+furnace|reheat\s+furnace|furnace", "Reheating Furnace"),
    (r"sinter", "Sinter Plant"),
    (r"coke\s+oven", "Coke Oven"),
    (r"ladle", "Ladle Handling System"),
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def is_steel_domain_query(query: str) -> bool:
    q = _norm(query)
    return any(term in q for term in STEEL_TERMS)


def classify_steel_intent(query: str) -> str:
    q = _norm(query)
    if any(term in q for term in ["agentic workflow", "agent workflow", "workflow design", "system architecture", "data flow"]):
        return "predictive_maintenance_workflow_design"
    if "predictive maintenance" in q and any(term in q for term in ["design", "workflow", "agent", "architecture", "logs", "sops", "sensor", "feedback"]):
        return "predictive_maintenance_workflow_design"
    if re.search(r"\b(?:error|fault)\s+code\b", q) or re.search(r"\be[- ]?\d{2,4}\b", q):
        return "error_code_lookup"
    if "logbook" in q and any(term in q for term in ["entry", "work done", "technician", "planned maintenance"]):
        return "logbook_entry"
    if any(term in q for term in ["spare", "spares", "procurement", "lead time", "stock", "inventory"]):
        return "spare_procurement_query"
    if any(term in q for term in ["sop", "standard operating procedure", "procedure for", "replace"]) and any(term in q for term in ["seal", "pump", "hydraulic", "bearing", "assembly", "motor", "gearbox"]):
        return "sop_request"
    if any(term in q for term in ["just stopped", "stopped", "tripped", "first checks", "right now", "walk me through"]):
        return "emergency_troubleshooting"
    if any(term in q for term in ["trending", "trend", "remaining useful life", "rul", "predict remaining", "intervene"]):
        return "trend_rul_analysis"
    if any(term in q for term in ["threshold", "differential pressure", "alert report", "create an alert", "alarm"]):
        return "abnormal_alert_report"
    if any(term in q for term in ["last 90 days", "incidents", "incident pattern", "maintenance records", "failure history"]):
        return "incident_pattern_analysis"
    if any(term in q for term in ["crew", "technician", "schedule", "weekend", "shift plan"]):
        return "crew_job_scheduling"
    if any(term in q for term in ["weekly summary", "supervisor summary", "supervisor update"]):
        return "supervisor_weekly_summary"
    if any(term in q for term in ["annual shutdown", "shutdown schedule", "48-hour shutdown", "48 hour shutdown", "critical path"]):
        return "work_order_planning"
    if any(term in q for term in ["condition-based", "condition based", "cbm", "early warning framework", "monitoring framework", "cbm roadmap"]):
        return "cbm_framework_design"
    if any(term in q for term in ["surface pitting", "slab pitting", "process defect", "quality defect"]):
        return "process_quality_analysis"
    if any(term in q for term in ["repeated failure", "repeat failure", "keeps failing", "recurring failure", "recurrence"]):
        return "repeated_failure_rca"
    if any(term in q for term in ["which one", "choose one", "only one", "what should i choose", "first priority"]):
        return "risk_prioritization"
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_RULES:
        scores[intent] = sum(1 for keyword in keywords if keyword in q)
    best_intent, best_score = max(scores.items(), key=lambda item: item[1])
    return best_intent if best_score > 0 else "general_steel_copilot"


def infer_steel_subject(query: str) -> str:
    q = _norm(query)
    for pattern, label in SUBJECT_PATTERNS:
        if re.search(pattern, q):
            return label
    return "Steel Plant"


def _source_lines(docs: list[dict], limit: int = 5) -> str:
    if not docs:
        return "- No indexed document matched; response uses steel maintenance first principles and asks for confirmation data."
    lines = []
    seen = set()
    for doc in docs:
        label = f"{doc.get('source')} ({doc.get('equipment_type')}/{doc.get('issue_type')})"
        if label in seen:
            continue
        seen.add(label)
        text = " ".join(str(doc.get("text", "")).split())[:260]
        lines.append(f"- {label} - {text}")
        if len(lines) >= limit:
            break
    return "\n".join(lines)


def _health_lines(health_rows: list[dict], limit: int = 4) -> str:
    if not health_rows:
        return "- No live health table available."
    lines = []
    for row in health_rows[:limit]:
        lines.append(
            "- {asset_id}: {asset_type}, {risk_band}, hybrid risk {risk}, RUL {rul} days, area {area}".format(
                asset_id=row.get("asset_id", "asset"),
                asset_type=row.get("asset_type", "equipment"),
                risk_band=row.get("risk_band", row.get("risk_level", "unknown risk")),
                risk=round(float(row.get("hybrid_failure_risk", row.get("failure_risk", 0)) or 0), 3),
                rul=round(float(row.get("estimated_rul_days", 0) or 0), 1),
                area=row.get("area", "plant"),
            )
        )
    return "\n".join(lines)


def _intent_actions(intent: str, subject: str) -> list[str]:
    libraries = {
        "predictive_maintenance_workflow_design": [
            "Design the agent loop: perceive, retrieve, reason, act, verify, log, and learn.",
            "Use logs, SOPs, sensor alerts, failure reports, spares, production delay, and feedback as separate tool inputs.",
            "Keep ML risk, RUL, priority, safety gates, and selected target as locked deterministic fields.",
            "Use the LLM for planning, explanation, multi-turn interaction, report writing, and operator/supervisor adaptation.",
        ],
        "root_cause_analysis": [
            f"Freeze the failure statement for {subject}: symptom, asset boundary, time window, operating mode.",
            "Separate symptoms from causes; compare sensor trends, operator logs, maintenance history, and process changes.",
            "Run 5-Why/FMEA style ranking and mark each cause as confirmed, probable, or needs evidence.",
            "Define containment, permanent corrective action, verification test, and recurrence-prevention owner.",
        ],
        "sop_generation": [
            f"Create a field-safe SOP for {subject} with prerequisites, isolation, inspection order, acceptance limits, and escalation.",
            "Add hold points where an engineer must approve continuation.",
            "Map each step to evidence to capture: readings, photos, vibration spectrum, oil sample, or PLC alarm export.",
            "End with restart checks and logbook entries.",
        ],
        "sop_request": [
            f"Create a field-safe SOP/checklist for {subject} and clearly separate verified site evidence from generic safe practice.",
            "Confirm model, isolation points, permit needs, spare kit, and acceptance limits before hands-on work.",
            "Include LOTO, depressurization/zero-energy checks, inspection hold points, restart checks, and logbook evidence.",
        ],
        "error_code_lookup": [
            f"Preserve the exact requested error code for {subject} and check whether a verified code table exists.",
            "If no OEM/manual match is available, do not invent the meaning; give safe immediate checks and required evidence.",
            "Capture MCC/VFD/PLC alarm export, timestamp, load state, and standby-equipment availability.",
        ],
        "spare_procurement_query": [
            f"Build a spare and procurement plan for {subject} using exact equipment boundary, parts master, stock, and lead time.",
            "Flag missing part numbers or stock records instead of inventing availability.",
            "Reserve critical items and raise procurement/escalation for long-lead or zero-stock spares.",
        ],
        "emergency_troubleshooting": [
            f"Give safe first checks for {subject} before diagnosing root cause.",
            "Start with personnel safety, trips/interlocks, local alarms, obstruction, overload, and upstream/downstream permissives.",
            "Require LOTO before physical inspection and escalate on smoke, heat, abnormal noise, repeated trips, or stored-energy hazards.",
        ],
        "trend_rul_analysis": [
            f"Use the user-supplied trend values for {subject} as provisional observations and estimate RUL as a confidence band.",
            "Explain trend slope, threshold crossing, missing baseline, and intervention timing.",
            "Request load, alarm, oil/vibration, inspection, and history data needed to improve confidence.",
        ],
        "abnormal_alert_report": [
            f"Create a scoped abnormal-condition alert for {subject}, using supplied thresholds as provisional evidence.",
            "Verify instrument health, process condition, repeated alarm count, and escalation owner.",
            "Do not convert the alert into plant-wide ranking unless the user explicitly asks for ranking.",
        ],
        "incident_pattern_analysis": [
            f"Review incident or maintenance-history patterns for {subject} without inventing records.",
            "Group repeated symptoms by component, operating mode, time window, and corrective action effectiveness.",
            "Recommend RCA evidence and recurrence-prevention actions.",
        ],
        "crew_job_scheduling": [
            f"Create a practical crew/job schedule for {subject} using permits, isolation, tools, spares, and critical-path constraints.",
            "Separate immediate safety work from planned-window tasks and keep missing manpower/spare data visible.",
        ],
        "supervisor_weekly_summary": [
            f"Write a supervisor-ready weekly maintenance summary for {subject}.",
            "Highlight critical risks, completed work, open actions, spares blockers, and decisions needed.",
        ],
        "work_order_planning": [
            f"Convert the request into an optimized maintenance schedule for {subject}, including risk flags, crew/crane/resource constraints, critical path, long-lead spares, and contingency recommendations.",
            "Split into immediate containment, planned shutdown work, manpower/tools, spares, permits, and restart checks.",
            "Sequence tasks to reduce downtime, protect the critical path, and avoid repeat isolation.",
            "Create acceptance criteria, fallback windows, and escalation triggers before releasing the asset back to operation.",
        ],
        "spares_procurement": [
            f"Build a spare strategy for {subject} by criticality, lead time, consumption, and failure consequence.",
            "Reserve available critical spares; raise procurement for zero-stock or long-lead items.",
            "Define substitutes only with engineering approval and OEM compatibility.",
            "Link procurement priority to production bottleneck and safety exposure.",
        ],
        "risk_prioritization": [
            f"Rank {subject} risk by safety, production bottleneck, RUL, criticality, spares readiness, and delay history.",
            "Classify risk as P1/P2/P3/P4 and explain why.",
            "Select the first intervention and the condition that would trigger escalation.",
            "Create a supervisor-level decision summary.",
        ],
        "failure_prediction": [
            f"Estimate RUL bands for {subject}, build an intervention/replacement schedule, and define early-warning signals: trend slope, anomaly count, threshold crossing, and historical failure pattern.",
            "Estimate RUL as a band, not a false exact value, unless a trained model supplies it.",
            "List the additional signals needed to improve prediction confidence.",
            "Trigger inspection before catastrophic failure indicators become irreversible.",
        ],
        "safety_control": [
            f"Treat {subject} as a safety-critical maintenance activity until hazards are cleared.",
            "Define energy sources, LOTO points, permits, exclusion zones, and PPE.",
            "Add stop-work triggers for temperature, pressure, suspended load, stored energy, or gas risk.",
            "Require supervisor sign-off before restart.",
        ],
        "process_quality": [
            f"Connect {subject} equipment condition to process and quality defect mechanisms.",
            "Separate mechanical, thermal, hydraulic, and control-loop causes.",
            "Link defect evidence to process parameters and maintenance checks.",
            "Recommend containment, sampling plan, and permanent corrective action.",
        ],
        "process_quality_analysis": [
            f"Connect {subject} condition to the quality symptom and isolate mechanical, cooling, thermal, and control-loop causes.",
            "Use caster/process evidence first and avoid unrelated equipment sources.",
            "Recommend containment, inspection, sampling, and permanent corrective action.",
        ],
        "repeated_failure_rca": [
            f"Run repeated-failure RCA for {subject}, separating recurrence pattern, failed corrective actions, and missing evidence.",
            "Use history/failure records if available and clearly state gaps if they are absent.",
            "Recommend containment, permanent corrective action, verification, and recurrence-prevention owner.",
        ],
        "cbm_framework_design": [
            f"Design a condition-based monitoring framework for {subject} with asset classes, sensors, thresholds, escalation rules, field-engineer notifications, and plant-supervisor notifications.",
            "Define field-engineer notifications for early warnings, supervisor notifications for P1/P2 risk, and auto-escalation for repeated alarms or falling RUL.",
            "Create a phased CBM roadmap: pilot assets, data-quality checks, model validation, SOP integration, spares alignment, and feedback learning.",
            "Use failure history, sensor profiles, RUL bands, procurement lead time, and production criticality to decide which assets are ready for transition.",
        ],
        "data_agent_design": [
            "Design the agent loop: perceive, retrieve, reason, act, verify, log, learn.",
            "Keep deterministic safety/ML fields locked and use the LLM for synthesis and interaction.",
            "Use RAG over manuals, SOPs, history, incident records, spares, and sensor summaries.",
            "Add feedback learning and audit trails so every decision is traceable.",
        ],
    }
    return libraries.get(
        intent,
        [
            f"Understand the steel-plant objective around {subject}.",
            "Collect asset, process, safety, production, and spare constraints.",
            "Generate a traceable decision with immediate actions and follow-up evidence.",
            "Log the outcome and learn from engineer feedback.",
        ],
    )


def build_general_plan(query: str, intent: str, subject: str) -> list[dict]:
    objective = {
        "predictive_maintenance_workflow_design": "Design an agentic predictive-maintenance workflow and demonstrate it on live plant context",
        "root_cause_analysis": "Diagnose probable root cause and corrective actions",
        "sop_generation": "Generate a safe executable SOP",
        "sop_request": "Generate a safe executable SOP/checklist",
        "error_code_lookup": "Resolve fault-code meaning safely without inventing OEM evidence",
        "spare_procurement_query": "Plan spares, inventory checks, and procurement lead-time actions",
        "emergency_troubleshooting": "Guide immediate safe first checks for an abnormal stoppage",
        "trend_rul_analysis": "Assess trend evidence and estimate RUL/intervention timing",
        "abnormal_alert_report": "Create a scoped abnormal-condition alert",
        "incident_pattern_analysis": "Analyze incident patterns and repeated symptoms",
        "crew_job_scheduling": "Plan crew, permit, spare, and schedule constraints",
        "supervisor_weekly_summary": "Summarize maintenance status for supervisors",
        "work_order_planning": "Create a maintenance execution plan",
        "spares_procurement": "Plan spares and procurement",
        "risk_prioritization": "Prioritize risk and intervention",
        "failure_prediction": "Predict failure risk and early warnings",
        "safety_control": "Create safety controls and stop-work logic",
        "process_quality": "Connect equipment health to process quality",
        "process_quality_analysis": "Connect equipment health to process quality",
        "repeated_failure_rca": "Diagnose repeated failures and recurrence controls",
        "cbm_framework_design": "Design a condition-based monitoring framework and CBM rollout roadmap",
        "data_agent_design": "Design or explain an agentic AI workflow",
    }.get(intent, "Solve the steel operations request")
    tasks = [
        ("Supervisor Agent", objective),
        ("Triage Agent", f"Classify intent as {intent} and subject as {subject}"),
        ("Retrieval Agent", "Pull manuals, SOPs, history, failures, spares, policy, and live health evidence"),
        ("Reasoning Agent", "Convert evidence into hypotheses, risks, actions, assumptions, and missing data"),
        ("Planning Agent", "Create ordered actions, owner handoffs, spare plan, and escalation logic"),
        ("Verifier Agent", "Check traceability, safety, decision completeness, and uncertainty"),
        ("Memory Agent", "Prepare logbook and feedback hooks for continuous learning"),
        ("Reporter Agent", "Write a concise engineer-ready answer"),
    ]
    return [
        {"step": idx, "agent": agent, "task": task, "target": subject, "status": "complete"}
        for idx, (agent, task) in enumerate(tasks, 1)
    ]


def build_general_tool_calls(
    query: str,
    intent: str,
    subject: str,
    docs: list[dict],
    health_rows: list[dict],
    feedback_rows: int,
) -> list[dict]:
    task_specific = intent in {
        "sop_request",
        "error_code_lookup",
        "spare_procurement_query",
        "emergency_troubleshooting",
        "logbook_entry",
        "trend_rul_analysis",
        "abnormal_alert_report",
    }
    calls = [
        {
            "tool": "intent_classifier",
            "agent": "Triage Agent",
            "input": query,
            "output": f"intent={intent}, subject={subject}",
            "status": "success",
        },
        {
            "tool": "rag_retriever",
            "agent": "Retrieval Agent",
            "input": f"{subject}, {intent}",
            "output": f"{len(docs)} evidence chunks retrieved",
            "status": "success" if docs else "review",
        },
        {
            "tool": "safety_gate",
            "agent": "Verifier Agent",
            "input": subject,
            "output": "LOTO/permit/escalation considered for maintenance-facing answer",
            "status": "success",
        },
        {
            "tool": "feedback_memory",
            "agent": "Memory Agent",
            "input": "feedback_log.csv",
            "output": f"{feedback_rows} feedback rows available for future learning",
            "status": "success",
        },
        {
            "tool": "decision_packet_builder",
            "agent": "Reporter Agent",
            "input": "plan + evidence + verifier checks",
            "output": "structured steel-agent response generated",
            "status": "success",
        },
    ]
    if not task_specific:
        calls.insert(
            1,
            {
                "tool": "plant_health_snapshot",
                "agent": "Sensor Agent",
                "input": "asset_health_summary.csv",
                "output": f"{len(health_rows)} live asset health rows reviewed",
                "status": "success" if health_rows else "review",
            },
        )
    return calls


def build_general_verifier_checks(intent: str, docs: list[dict], health_rows: list[dict]) -> list[dict]:
    task_specific = intent in {
        "sop_request",
        "error_code_lookup",
        "spare_procurement_query",
        "emergency_troubleshooting",
        "logbook_entry",
        "trend_rul_analysis",
        "abnormal_alert_report",
    }
    checks = [
        ("Steel-domain intent classified", bool(intent)),
        ("Evidence retrieved or uncertainty declared", bool(docs)),
        ("Live plant health considered when needed", bool(health_rows) or task_specific),
        ("Action plan included", True),
        ("Safety/LOTO gate included", True),
        ("Traceability section included", True),
        ("Feedback/logbook path included", True),
    ]
    return [
        {"check": name, "status": "pass" if ok else "review", "detail": "verified" if ok else "needs more site data"}
        for name, ok in checks
    ]


def build_general_decision_packet(
    query: str,
    intent: str,
    subject: str,
    docs: list[dict],
    health_rows: list[dict],
) -> dict:
    first_action = _intent_actions(intent, subject)[0]
    top_sources = [doc.get("source") for doc in docs[:5]]
    applied_target = health_rows[0].get("asset_id") if health_rows else None
    live_risks = [
        {
            "asset_id": row.get("asset_id"),
            "risk_band": row.get("risk_band"),
            "rul_days": row.get("estimated_rul_days"),
        }
        for row in health_rows[:3]
    ]
    workflow_mode = intent == "predictive_maintenance_workflow_design"
    return {
        "mode": "agentic_workflow_design" if workflow_mode else "general_steel_agent",
        "intent": intent,
        "objective": query,
        "selected_asset": "Steel Plant Workflow" if workflow_mode else subject,
        "subject": subject,
        "applied_demo_target": applied_target if workflow_mode else None,
        "risk_level": "Contextual",
        "priority": "Agent-assessed",
        "urgency": "Depends on live measurements; safety-critical symptoms escalate immediately",
        "recommended_first_action": first_action,
        "top_sources": top_sources,
        "live_risk_snapshot": live_risks,
        "next_system_action": "collect_missing_field_data_then_create_or_update_work_order",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def build_general_answer(
    query: str,
    intent: str,
    subject: str,
    docs: list[dict],
    health_rows: list[dict],
    agent_plan: list[dict],
    tool_calls: list[dict],
    verifier_checks: list[dict],
    decision_packet: dict,
) -> str:
    actions = _intent_actions(intent, subject)
    evidence = _source_lines(docs)
    health = _health_lines(health_rows)
    plan_lines = "\n".join(f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan)
    tool_lines = "\n".join(
        f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}"
        for t in tool_calls
    )
    check_lines = "\n".join(f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks)
    action_lines = "\n".join(f"- {action}" for action in actions)
    applied_target = decision_packet.get("applied_demo_target")
    applied_row = next((row for row in health_rows if row.get("asset_id") == applied_target), {}) if applied_target else {}

    if intent == "error_code_lookup":
        code_match = re.search(r"\b([A-Z]{1,4}[- ]?\d{2,4})\b", query, flags=re.IGNORECASE)
        code = code_match.group(1).upper().replace(" ", "-") if code_match else "the requested code"
        exact_docs = [
            doc for doc in docs
            if code.lower() in " ".join(str(doc.get(key, "")) for key in ["source", "issue_type", "text"]).lower()
        ]
        exact_status = "verified" if exact_docs else "not verified"
        source_status = (
            f"I found an exact supporting document for {code}; use the listed source and equipment model before acting."
            if exact_docs
            else f"I could not verify the exact meaning of {code} from the available OEM, VFD, MCC, PLC, relay, or blower-motor manual evidence."
        )
        general_guidance = _source_lines(
            [
                doc for doc in docs
                if any(term in " ".join(str(doc.get(key, "")) for key in ["source", "equipment_type", "issue_type", "text"]).lower()
                       for term in ["safety", "loto", "blast furnace", "blower", "motor"])
            ],
            limit=3,
        )
        return f"""
**Code Meaning**
{source_status}

**Confidence / Source Status**
- Exact code preserved: `{code}`
- Code-definition confidence: {exact_status.upper()}
- Do not assume the meaning, repeatedly reset, or restart blindly until the exact manufacturer/manual entry is checked.

**Immediate Actions**
1. Confirm whether the blast-furnace blower motor is running, tripped, or on standby.
2. Record the alarm source: VFD, MCC, motor protection relay, PLC, local panel, or DCS.
3. Capture the full alarm text, timestamp, preceding alarms, drive/relay model, firmware, and a photo/export of the alarm screen.
4. Check motor current, winding or bearing temperature, vibration, cooling airflow, lubrication indication, discharge pressure/flow, and standby blower availability.
5. If the blower has abnormal noise, smoke, overheating, excessive vibration, repeated trips, or unstable furnace impact, stop/escalate to the area supervisor and control room.
6. Apply LOTO before opening guards, panels, coupling covers, motor terminals, or blower housings.
7. Look up `{code}` in the exact OEM/VFD/MCC/relay manual for that installed equipment model before clearing or resetting the alarm.

**Information Required**
- Manufacturer, controller/VFD/protection-relay model, firmware/version, panel name, full alarm text, timestamp, operating state, and recent maintenance/inspection notes.

**Evidence**
- Exact `{code}` definition: not found in the loaded evidence unless a matching OEM/manual entry is provided.
- General safety guidance used:
{general_guidance}
""".strip()

    if intent == "sop_request":
        return f"""
**Generic Safe SOP: Hydraulic Pump Seal Replacement - {subject}**

Use this as a field-safe procedure and verify it against the exact pump OEM manual, site-approved SOP, isolation drawing, and permit requirements before execution. Where torque values, clearances, seal dimensions, pressure limits, or acceptance limits are not available in the loaded evidence, use the OEM/site manual and do not guess.

**Procedure**
1. Obtain the work permit, line-break permit if required, and job safety briefing. Confirm the rolling-mill equipment boundary, affected hydraulic power pack, pump tag, isolation points, and responsible operator.
2. Verify the replacement seal kit by part number, material, size, rotation/orientation requirement, and compatibility with the hydraulic oil. Stage clean tools, lint-free cloths, drip trays, blanking caps, spill kit, and approved lubricant.
3. Stop the hydraulic unit through the approved operating sequence and inform the mill pulpit/control room. Place the equipment in a safe maintenance state.
4. Isolate all energy sources: electrical supply to the pump motor, hydraulic pressure, accumulators, pneumatic controls, mechanical stored energy, and any gravity-loaded actuator. Apply lockout/tagout and try-start verification.
5. Release stored hydraulic energy and depressurize the circuit. Confirm zero pressure at the correct gauge/test point before loosening any fitting. Treat residual pressure as a stop-work condition.
6. Clean the pump and surrounding area before opening the system. Drain oil below the pump/seal level if required and cap or plug opened lines immediately to prevent contamination.
7. Mark coupling position, hose orientation, pump mounting position, and alignment references. Disconnect guards, coupling, instruments, and hydraulic lines only after zero-pressure verification.
8. Remove the pump or seal housing as per the OEM method. Support the pump safely and avoid side-loading the shaft.
9. Remove the old seal carefully. Inspect the shaft sleeve, seal seat, keyway, faces, bearings, housing bore, and drain path for scoring, corrosion, wear, runout signs, or contamination. Do not install a new seal onto a damaged sleeve or seat.
10. Clean the housing and shaft with approved solvent and lint-free material. Install the new seal in the correct orientation using the approved lubricant/installation sleeve. Do not hammer the seal face. Tighten fasteners only to OEM/site torque values.
11. Reassemble the pump, reconnect lines, replace disturbed O-rings/gaskets, restore guards, and check coupling alignment. Correct soft foot or misalignment before restart.
12. Refill with filtered approved oil if oil was drained. Bleed trapped air from the circuit using the site procedure.
13. Restore hydraulic pressure gradually while checking for leaks at the seal, fittings, drain, and housing. Keep personnel clear of pressurized leak paths.
14. Run a controlled restart at low load first. Verify pressure stability, oil level, temperature, vibration, motor current, abnormal noise, and actuator response.
15. Accept the job only when there is no seal leakage, system pressure is stable, reservoir oil level is correct, oil temperature, vibration, motor current, and noise are normal for the unit, guards are restored, interlocks are healthy, and the operator/supervisor signs off.
16. Reinspect for leakage after 10-15 minutes of controlled operation and again after the first operating cycle.
17. Record the seal kit used, oil added, pressure/temperature/vibration/current readings, leak-test result, root cause observations, photos if available, and any follow-up action in the digital logbook.

**Hold Points**
- Do not loosen hydraulic fittings until zero pressure is verified.
- Do not reuse a scored shaft sleeve, damaged seal seat, or contaminated seal.
- Do not invent torque values or acceptance limits; use the OEM/site manual.
- Stop and escalate if pressure returns unexpectedly, oil sprays, the pump runs hot/noisy, or leakage remains after restart.

This procedure supports safe execution, but it must be verified against site rules, the approved isolation plan, and the exact pump OEM manual.

**Evidence Used**
{evidence}
""".strip()

    if intent == "predictive_maintenance_workflow_design":
        return f"""
**Agentic Predictive Maintenance Workflow Design**

**Mode**
- agentic_workflow_design

**Intent**
- predictive_maintenance_workflow_design

**Applied Live Demo Target**
- {applied_target or "No live target available"}

**Framing**
- The workflow design is the main answer.
- To demonstrate the workflow on live plant data, the agent applied it to the current asset-health table and selected {applied_target or "the highest-risk asset"} as the first maintenance target.

**1. Agentic Workflow Design**
{action_lines}

**2. Live Execution Example**
- Current highest-risk live target: {applied_target or "not available"}
- Risk band: {applied_row.get("risk_band", "not available")}
- Hybrid risk: {round(float(applied_row.get("hybrid_failure_risk", applied_row.get("failure_risk", 0)) or 0), 3)}
- RUL: {applied_row.get("estimated_rul_days", "not available")} days

**3. Why The Demo Target Was Selected**
- The agent ranks live assets by hybrid health score, failure risk, RUL, criticality, delay impact, and evidence availability.
- The selected target is used only as a demonstration of the workflow, not as a replacement for the architecture design.

**4. Autonomous Execution Plan**
{plan_lines}

**5. Tool Calls Executed**
{tool_lines}

**6. Verifier Checks**
{check_lines}

**7. Evidence**
{evidence}

**Decision Packet**
- Mode: {decision_packet["mode"]}
- Intent: {decision_packet["intent"]}
- Applied live demo target: {decision_packet.get("applied_demo_target")}
- Next system action: {decision_packet["next_system_action"]}
""".strip()

    return f"""
**Steel Plant Agent Response**

**Objective**
- {query}

**Interpreted Intent**
- Intent: {intent.replace("_", " ")}
- Subject: {subject}
- Agent stance: I will treat this as a steel-plant decision task, not a generic chatbot answer.

**Autonomous Execution Plan**
{plan_lines}

**Tool Calls Executed**
{tool_lines}

**Verifier Checks**
{check_lines}

**Live Plant Context Considered**
{health}

**Working Diagnosis / Approach**
- For {subject}, the agent should first separate immediate safety risk, production bottleneck risk, equipment health risk, and evidence uncertainty.
- If the prompt is about a known asset ID, the system should switch to the live ML/RAG asset path. If it is a broader plant question, this general steel agent produces the operating plan and evidence checklist.
- The answer is intentionally operational: what to inspect, what to verify, what to procure, what to log, and when to escalate.

**Action Plan**
{action_lines}

**Evidence To Collect Next**
- Latest sensor trend: temperature, vibration, current, pressure, speed, alarm count, and trend slope.
- PLC/SCADA fault chronology with timestamps and operating mode.
- Recent maintenance work orders, lubrication/oil/filter records, and operator observations.
- Spare stock, lead time, substitute policy, and shutdown window constraints.
- Safety permits, isolation points, and restart acceptance readings.

**Risk And Priority Logic**
- P1/Critical if there is personnel risk, catastrophic failure risk, very low RUL, repeated critical alarms, or a direct production bottleneck.
- P2/High if degradation is clear but controlled intervention within 24 hours is feasible.
- P3/Medium if the issue can be planned inside a maintenance window.
- P4/Low if only monitoring and confirmation data are required.

**Traceability / Sources**
{evidence}

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Intent: {decision_packet["intent"]}
- Subject: {decision_packet["subject"]}
- Recommended first action: {decision_packet["recommended_first_action"]}
- Next system action: {decision_packet["next_system_action"]}

**Feedback And Learning**
- Save engineer corrections, accepted actions, actual root cause, downtime, and restart readings to the feedback log.
- Use the next confirmed outcome to adjust future root-cause ranking and action order.
""".strip()


def summarize_health_rows(rows: Iterable[dict]) -> list[dict]:
    def score(row: dict) -> float:
        return float(row.get("hybrid_health_score", row.get("hybrid_failure_risk", row.get("failure_risk", 0))) or 0)

    return sorted([dict(row) for row in rows], key=score, reverse=True)
