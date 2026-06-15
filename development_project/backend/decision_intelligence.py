"""Decision-intelligence utilities for the steel maintenance agent.

This layer does not replace the chat agent. It creates a stronger factual
packet for the LLM: normalized multi-source asset state, evidence confidence,
delay cost, procurement risk, active/inactive dynamic assets, and remembered
rule effects.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, DOC_DIR, REPORT_DIR
from .dynamic_assets import load_dynamic_assets, score_dynamic_assets


AREA_DELAY_COST_INR_PER_HOUR = {
    "Blast Furnace": 1_500_000,
    "Basic Oxygen Furnace": 1_250_000,
    "Caster": 950_000,
    "Caster Utility": 850_000,
    "Finishing Mill": 850_000,
    "Hot Strip Mill": 700_000,
    "Roughing Mill": 650_000,
    "Plate Mill": 500_000,
    "Sinter Plant": 450_000,
    "Utilities": 350_000,
}


def _read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def _num(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _priority_from_score(score: float) -> tuple[str, str]:
    if score >= 82:
        return "P1", "CRITICAL"
    if score >= 62:
        return "P2", "HIGH"
    if score >= 38:
        return "P3", "MEDIUM"
    return "P4", "LOW"


def _safe_latest(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "asset_id" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    if "timestamp" in out.columns:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", format="mixed")
        out = out.sort_values(["asset_id", "timestamp"])
    return out.groupby("asset_id", as_index=False).tail(1).reset_index(drop=True)


def _evidence_for_asset(asset_id: str, source: str) -> dict[str, Any]:
    history = _read_csv(DATA_DIR / "maintenance_history.csv")
    failures = _read_csv(DATA_DIR / "failure_reports.csv")
    spares = _read_csv(DATA_DIR / "spares_inventory.csv")
    docs = list(DOC_DIR.glob(f"*{asset_id.replace('-', '_')}*.txt")) + list(DOC_DIR.glob(f"*{asset_id}*.txt"))

    history_count = 0 if history.empty else int((history.get("asset_id", pd.Series(dtype=str)).astype(str) == asset_id).sum())
    failure_count = 0 if failures.empty else int((failures.get("asset_id", pd.Series(dtype=str)).astype(str) == asset_id).sum())
    spare_count = 0 if spares.empty else int((spares.get("asset_id", pd.Series(dtype=str)).astype(str) == asset_id).sum())
    sop_count = len(docs)

    if source == "steel_demo_app":
        confidence = "HIGH" if history_count or failure_count or spare_count or sop_count else "MEDIUM"
    elif history_count or failure_count or spare_count or sop_count:
        confidence = "MEDIUM"
    else:
        confidence = "REVIEW"

    missing = []
    if history_count == 0:
        missing.append("maintenance history")
    if failure_count == 0:
        missing.append("failure reports")
    if spare_count == 0:
        missing.append("spares inventory")
    if sop_count == 0:
        missing.append("asset-specific SOP")

    return {
        "history_count": history_count,
        "failure_count": failure_count,
        "spare_count": spare_count,
        "sop_count": sop_count,
        "evidence_confidence": confidence,
        "missing_evidence": missing,
    }


def _spares_packet(asset_id: str, spare_lead_time_days: float) -> dict[str, Any]:
    spares = _read_csv(DATA_DIR / "spares_inventory.csv")
    if spares.empty or "asset_id" not in spares.columns:
        lead = _num(spare_lead_time_days, 0.0)
        return {
            "spare_stock_qty": np.nan,
            "max_spare_lead_time_days": lead,
            "procurement_risk": "UNKNOWN" if lead <= 0 else "HIGH",
            "spare_plan": "No spares record available; verify inventory before maintenance release.",
        }

    rows = spares[spares["asset_id"].astype(str) == str(asset_id)].copy()
    if rows.empty:
        lead = _num(spare_lead_time_days, 0.0)
        return {
            "spare_stock_qty": np.nan,
            "max_spare_lead_time_days": lead,
            "procurement_risk": "UNKNOWN" if lead <= 0 else "HIGH",
            "spare_plan": "No asset-specific spares record; raise inventory check and procurement confirmation.",
        }

    stock = float(pd.to_numeric(rows["stock_qty"], errors="coerce").fillna(0).sum())
    lead = float(pd.to_numeric(rows["lead_time_days"], errors="coerce").fillna(0).max())
    critical = rows["spare_criticality"].astype(str).str.upper().str.contains("CRITICAL").any()
    if stock <= 0 and (critical or lead >= 7):
        risk = "CRITICAL"
    elif stock <= 1 or lead >= 7:
        risk = "HIGH"
    elif lead >= 3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    parts = ", ".join(rows["spare_part"].astype(str).head(4).tolist())
    return {
        "spare_stock_qty": stock,
        "max_spare_lead_time_days": lead,
        "procurement_risk": risk,
        "spare_plan": f"Reserve/verify: {parts}. Max lead time {lead:.0f} day(s).",
    }


def _score_base_row(row: dict[str, Any]) -> dict[str, Any]:
    score = _num(row.get("hybrid_health_score"), np.nan)
    failure_risk = _num(row.get("failure_risk"), np.nan)
    rul = _num(row.get("estimated_rul_days"), np.nan)
    alarms = _num(row.get("alarm_count"), 0.0)
    criticality_score = _num(row.get("criticality_score"), 2.0)

    if math.isnan(score):
        temperature = _num(row.get("temperature"), 45.0)
        vibration = _num(row.get("vibration"), 2.0)
        current = _num(row.get("current"), 35.0)
        pressure = _num(row.get("pressure"), 9.0)
        temp_risk = _clip01((temperature - 55.0) / 35.0)
        vib_risk = _clip01((vibration - 3.0) / 7.0)
        current_risk = _clip01((current - 45.0) / 45.0)
        pressure_risk = _clip01((7.0 - pressure) / 4.0)
        alarm_risk = _clip01(alarms / 4.0)
        criticality_risk = _clip01(criticality_score / 4.0)
        failure_risk = _clip01(0.25 * temp_risk + 0.30 * vib_risk + 0.15 * current_risk + 0.15 * pressure_risk + 0.10 * alarm_risk + 0.05 * criticality_risk)
        score = round(failure_risk * 100.0, 2)

    if math.isnan(rul) or rul <= 0:
        rul = round(max(1.0, 35.0 * (1.0 - _clip01(score / 100.0))), 1)

    priority, risk_band = _priority_from_score(score)
    return {
        "hybrid_health_score": round(score, 2),
        "failure_risk": round(_clip01(failure_risk if not math.isnan(failure_risk) else score / 100.0), 4),
        "estimated_rul_days": round(rul, 1),
        "priority": row.get("priority") or priority,
        "risk_band": row.get("risk_band") or risk_band,
    }


def build_decision_intelligence_table(force: bool = True) -> pd.DataFrame:
    """Build and save a richer asset-ranking table for agent prompts and judging."""

    health = _read_csv(DATA_DIR / "asset_health_summary.csv")
    steel_latest = _safe_latest(health if not health.empty else _read_csv(DATA_DIR / "steel_sensor_logs_scored.csv"))
    if steel_latest.empty:
        steel_latest = _safe_latest(_read_csv(DATA_DIR / "steel_sensor_logs.csv"))

    rows: list[dict[str, Any]] = []
    for _, sr in steel_latest.iterrows():
        row = sr.to_dict()
        row["data_origin"] = row.get("source", "steel_demo_app")
        row["is_dynamic"] = 0
        rows.append(row)

    dynamic_raw = load_dynamic_assets()
    active_dynamic_count = 0
    inactive_dynamic_count = 0
    if not dynamic_raw.empty:
        active_dynamic_count = int(dynamic_raw.get("active", True).astype(bool).sum()) if "active" in dynamic_raw.columns else len(dynamic_raw)
        inactive_dynamic_count = int(len(dynamic_raw) - active_dynamic_count)
        scored_dynamic = score_dynamic_assets(dynamic_raw, active_only=False)
        if not scored_dynamic.empty:
            for _, dr in scored_dynamic.iterrows():
                row = dr.to_dict()
                row["data_origin"] = "dynamic_user_memory"
                row["is_dynamic"] = 1
                rows.append(row)

    enriched: list[dict[str, Any]] = []
    for row in rows:
        asset_id = str(row.get("asset_id", "")).strip()
        if not asset_id:
            continue
        active = bool(row.get("active", True))
        score_packet = _score_base_row(row)
        row.update(score_packet)

        area = str(row.get("area", "Utilities"))
        delay_cost_per_hour = AREA_DELAY_COST_INR_PER_HOUR.get(area, 350_000)
        delay_hours = _num(row.get("delay_hours"), 0.0)
        delay_cost_impact = delay_hours * delay_cost_per_hour
        spare_packet = _spares_packet(asset_id, row.get("spare_lead_time_days", 0.0))
        evidence = _evidence_for_asset(asset_id, str(row.get("data_origin", "")))

        procurement_multiplier = {"LOW": 0.05, "MEDIUM": 0.12, "HIGH": 0.22, "CRITICAL": 0.30, "UNKNOWN": 0.18}.get(spare_packet["procurement_risk"], 0.18)
        evidence_penalty = {"HIGH": 0.0, "MEDIUM": 2.5, "REVIEW": 6.0, "LOW": 10.0}.get(evidence["evidence_confidence"], 6.0)
        delay_component = min(10.0, delay_cost_impact / 1_000_000.0)
        procurement_component = procurement_multiplier * 15.0
        rule_bonus = 6.0 if _num(row.get("applied_rule_count"), 0) > 0 else 0.0
        decision_score = round(min(100.0, row["hybrid_health_score"] + delay_component + procurement_component + rule_bonus - evidence_penalty), 2)
        decision_priority, decision_risk = _priority_from_score(decision_score)

        applied_rules = row.get("applied_rules", [])
        if isinstance(applied_rules, str):
            try:
                applied_rules = json.loads(applied_rules)
            except Exception:
                applied_rules = [applied_rules] if applied_rules else []

        enriched.append(
            {
                "asset_id": asset_id,
                "asset_type": row.get("asset_type", "Unknown"),
                "area": area,
                "criticality": row.get("criticality", "Medium"),
                "active": active,
                "is_dynamic": int(row.get("is_dynamic", 0)),
                "data_origin": row.get("data_origin", ""),
                "temperature": row.get("temperature"),
                "vibration": row.get("vibration"),
                "current": row.get("current"),
                "pressure": row.get("pressure"),
                "rpm": row.get("rpm"),
                "alarm_count": row.get("alarm_count"),
                "raw_health_score": row["hybrid_health_score"],
                "rule_adjusted_score": row.get("hybrid_health_score"),
                "decision_score": decision_score,
                "failure_risk": row["failure_risk"],
                "estimated_rul_days": row["estimated_rul_days"],
                "priority": decision_priority,
                "risk_band": decision_risk,
                "delay_hours": delay_hours,
                "delay_cost_per_hour_inr": delay_cost_per_hour,
                "delay_cost_impact_inr": round(delay_cost_impact, 2),
                "procurement_risk": spare_packet["procurement_risk"],
                "spare_stock_qty": spare_packet["spare_stock_qty"],
                "max_spare_lead_time_days": spare_packet["max_spare_lead_time_days"],
                "spare_plan": spare_packet["spare_plan"],
                "evidence_confidence": evidence["evidence_confidence"],
                "missing_evidence": "; ".join(evidence["missing_evidence"]),
                "history_count": evidence["history_count"],
                "failure_report_count": evidence["failure_count"],
                "sop_count": evidence["sop_count"],
                "applied_rule_count": int(_num(row.get("applied_rule_count"), 0)),
                "applied_rules": json.dumps(applied_rules, ensure_ascii=True),
                "next_system_action": _next_action(row, decision_priority, spare_packet["procurement_risk"]),
            }
        )

    out = pd.DataFrame(enriched)
    if not out.empty:
        out = out.sort_values(["active", "decision_score", "estimated_rul_days"], ascending=[False, False, True]).reset_index(drop=True)
    out.to_csv(DATA_DIR / "asset_decision_intelligence.csv", index=False)

    summary = {
        "rows": int(len(out)),
        "original_demo_assets": int((out.get("is_dynamic", pd.Series(dtype=int)) == 0).sum()) if not out.empty else 0,
        "active_dynamic_assets": active_dynamic_count,
        "inactive_dynamic_assets": inactive_dynamic_count,
        "top_asset": "" if out.empty else str(out.iloc[0]["asset_id"]),
        "top_priority": "" if out.empty else str(out.iloc[0]["priority"]),
        "output": str(DATA_DIR / "asset_decision_intelligence.csv"),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "decision_intelligence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out


def _next_action(row: dict[str, Any], priority: str, procurement_risk: str) -> str:
    asset_type = str(row.get("asset_type", "")).lower()
    if priority == "P1":
        prefix = "Create immediate P1 work order, notify supervisor, and reserve critical spares."
    elif priority == "P2":
        prefix = "Schedule inspection within 24 hours and verify spare readiness."
    else:
        prefix = "Monitor trend and plan inspection during next maintenance window."

    if "gearbox" in asset_type:
        action = "Run vibration spectrum, oil sample, bearing temperature check, backlash and alignment inspection."
    elif "motor" in asset_type:
        action = "Check bearing lubrication, cooling path, current imbalance, coupling alignment, and insulation temperature."
    elif "pump" in asset_type:
        action = "Inspect suction strainer, suction head, seal leakage, impeller condition, and pressure recovery."
    elif "hydraulic" in asset_type:
        action = "Inspect filter differential pressure, relief valve, oil level, leakage points, and pump noise."
    elif "blower" in asset_type or "fan" in asset_type:
        action = "Check vibration spectrum, bearing temperature, impeller fouling, damper position, motor current, and standby availability."
    else:
        action = "Create equipment-specific inspection checklist from SOP/manual evidence."

    if procurement_risk in {"CRITICAL", "HIGH", "UNKNOWN"}:
        action += " Procurement review required before releasing intrusive work."
    return f"{prefix} {action}"


def top_original_vs_dynamic() -> dict[str, Any]:
    table = build_decision_intelligence_table()
    active = table[table["active"].astype(bool)].copy() if not table.empty else table
    original = active[active["is_dynamic"] == 0].head(1)
    dynamic = active[active["is_dynamic"] == 1].head(1)
    winner = active.head(1)
    return {
        "top_original": original.to_dict("records")[0] if not original.empty else None,
        "top_dynamic": dynamic.to_dict("records")[0] if not dynamic.empty else None,
        "winner": winner.to_dict("records")[0] if not winner.empty else None,
    }

