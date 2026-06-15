"""Data generation and public benchmark ingestion for Maintenance Wizard."""

from __future__ import annotations

import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DOC_DIR, PUBLIC_DIR


ASSETS = [
    {
        "asset_id": "MTR-204",
        "asset_type": "Induction Motor",
        "area": "Hot Strip Mill",
        "criticality": "High",
        "criticality_score": 3,
        "main_issue": "overheating and high current",
    },
    {
        "asset_id": "GBX-17",
        "asset_type": "Gearbox",
        "area": "Finishing Mill",
        "criticality": "Critical",
        "criticality_score": 4,
        "main_issue": "abnormal vibration and bearing wear",
    },
    {
        "asset_id": "PMP-09",
        "asset_type": "Cooling Water Pump",
        "area": "Caster Utility",
        "criticality": "High",
        "criticality_score": 3,
        "main_issue": "cavitation and low suction pressure",
    },
    {
        "asset_id": "HPP-12",
        "asset_type": "Hydraulic Power Pack",
        "area": "Plate Mill",
        "criticality": "Medium",
        "criticality_score": 2,
        "main_issue": "low hydraulic pressure and leakage risk",
    },
]


COMMON_COLUMNS = [
    "source",
    "timestamp",
    "asset_id",
    "asset_type",
    "area",
    "criticality",
    "criticality_score",
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "alarm_count",
    "delay_hours",
    "spare_lead_time_days",
    "failure_label",
    "failure_mode",
]


PUBLIC_AI4I_URLS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv",
    "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip",
]


def _ramp(periods: int, start: int, end_value: float = 1.0) -> np.ndarray:
    values = np.zeros(periods)
    if start < periods:
        values[start:] = np.linspace(0, end_value, periods - start)
    return values


def _write_doc(filename: str, text: str) -> None:
    (DOC_DIR / filename).write_text(text.strip() + "\n", encoding="utf-8")


def generate_steel_demo_data(force: bool = False) -> pd.DataFrame:
    """Generate deterministic steel maintenance demo tables and SOP docs."""

    target = DATA_DIR / "steel_sensor_logs.csv"
    if target.exists() and not force:
        return pd.read_csv(target)

    rng = np.random.default_rng(42)
    asset_master = pd.DataFrame(ASSETS)
    asset_master.to_csv(DATA_DIR / "asset_master.csv", index=False)

    periods = 720
    end_ts = pd.Timestamp.now().floor("h")
    timestamps = pd.date_range(end=end_ts, periods=periods, freq="h")
    rows = []

    for asset in ASSETS:
        aid = asset["asset_id"]
        stage = _ramp(periods, 470, 1.0)

        for i, ts in enumerate(timestamps):
            s = stage[i]

            if aid == "MTR-204":
                temperature = 58 + 30 * s + rng.normal(0, 1.6)
                vibration = 2.2 + 3.6 * s + rng.normal(0, 0.25)
                current = 48 + 34 * s + rng.normal(0, 2.0)
                pressure = 9.5 + rng.normal(0, 0.25)
                rpm = 1480 - 55 * s + rng.normal(0, 8)
                alarm_count = int(temperature > 78) + int(current > 75) + int(vibration > 5.0)
                failure_label = int(temperature > 84 and current > 78)
            elif aid == "GBX-17":
                temperature = 52 + 18 * s + rng.normal(0, 1.4)
                vibration = 3.0 + 8.2 * s + rng.normal(0, 0.35)
                current = 42 + 14 * s + rng.normal(0, 1.8)
                pressure = 10.0 + rng.normal(0, 0.2)
                rpm = 980 - 35 * s + rng.normal(0, 6)
                alarm_count = int(vibration > 7.0) + int(vibration > 9.0) + int(temperature > 65)
                failure_label = int(vibration > 9.3)
            elif aid == "PMP-09":
                temperature = 49 + 13 * s + rng.normal(0, 1.2)
                vibration = 2.5 + 4.2 * s + rng.normal(0, 0.3)
                current = 38 + 23 * s + rng.normal(0, 2.0)
                pressure = 9.2 - 4.8 * s + rng.normal(0, 0.25)
                rpm = 1440 - 70 * s + rng.normal(0, 10)
                alarm_count = int(pressure < 6.2) + int(vibration > 5.3) + int(current > 58)
                failure_label = int(pressure < 5.7 and vibration > 5.1)
            else:
                temperature = 46 + 22 * s + rng.normal(0, 1.5)
                vibration = 1.8 + 2.5 * s + rng.normal(0, 0.25)
                current = 34 + 14 * s + rng.normal(0, 1.7)
                pressure = 10.5 - 5.4 * s + rng.normal(0, 0.3)
                rpm = 1200 + rng.normal(0, 5)
                alarm_count = int(pressure < 6.4) + int(temperature > 62) + int(pressure < 5.6)
                failure_label = int(pressure < 5.5)

            rows.append(
                {
                    "source": "steel_demo_app",
                    "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "asset_id": aid,
                    "asset_type": asset["asset_type"],
                    "area": asset["area"],
                    "criticality": asset["criticality"],
                    "criticality_score": asset["criticality_score"],
                    "temperature": round(float(temperature), 3),
                    "vibration": round(float(vibration), 3),
                    "current": round(float(current), 3),
                    "pressure": round(float(pressure), 3),
                    "rpm": round(float(rpm), 3),
                    "alarm_count": int(alarm_count),
                    "delay_hours": round(float(max(0, alarm_count * 1.5 + s * 4)), 2),
                    "spare_lead_time_days": int(2 + asset["criticality_score"] + alarm_count),
                    "failure_label": int(failure_label),
                    "failure_mode": asset["main_issue"] if failure_label else "normal",
                }
            )

    steel_df = pd.DataFrame(rows)
    steel_df.to_csv(target, index=False)

    pd.DataFrame(
        [
            ["WO-1001", "MTR-204", "2026-05-02 09:30:00", "Motor temperature rising", "Cleaned cooling fins and checked bearing grease", "Temporary improvement", 2.0],
            ["WO-1002", "GBX-17", "2026-05-08 14:00:00", "Gearbox vibration above warning", "Oil sample collected, alignment checked", "Bearing wear suspected", 4.5],
            ["WO-1003", "PMP-09", "2026-05-14 11:15:00", "Pump cavitation noise", "Checked suction strainer and impeller clearance", "Suction restriction found", 3.0],
            ["WO-1004", "HPP-12", "2026-05-20 16:20:00", "Hydraulic pressure fluctuation", "Inspected relief valve and filter", "Filter choking observed", 2.5],
        ],
        columns=["work_order_id", "asset_id", "timestamp", "issue", "action_taken", "result", "downtime_hours"],
    ).to_csv(DATA_DIR / "maintenance_history.csv", index=False)

    pd.DataFrame(
        [
            ["FR-201", "MTR-204", "2026-04-18 07:45:00", "Overheating", "Bearing lubrication degradation", "Regrease bearing, inspect load current, clean cooling path", "Line speed reduction for 3 hours"],
            ["FR-202", "GBX-17", "2026-04-26 18:10:00", "High vibration", "Bearing pitting and gear mesh misalignment", "Replace bearing set, verify coupling alignment", "Finishing mill delay risk"],
            ["FR-203", "PMP-09", "2026-05-04 10:40:00", "Cavitation", "Low suction head due to clogged strainer", "Clean strainer, inspect impeller erosion", "Cooling instability risk"],
            ["FR-204", "HPP-12", "2026-05-11 13:05:00", "Low pressure", "Hydraulic oil leakage and filter choking", "Replace filter, inspect seals, top up oil", "Actuator slow response"],
        ],
        columns=["report_id", "asset_id", "timestamp", "failure_mode", "root_cause", "corrective_action", "business_impact"],
    ).to_csv(DATA_DIR / "failure_reports.csv", index=False)

    pd.DataFrame(
        [
            ["SP-001", "MTR-204", "DE bearing 6312", 2, 5, "Critical", 18000],
            ["SP-002", "MTR-204", "Cooling fan kit", 1, 7, "High", 12000],
            ["SP-003", "GBX-17", "Gearbox bearing set", 1, 12, "Critical", 85000],
            ["SP-004", "GBX-17", "Synthetic gear oil drum", 4, 2, "High", 15000],
            ["SP-005", "PMP-09", "Mechanical seal kit", 3, 4, "High", 22000],
            ["SP-006", "PMP-09", "Impeller assembly", 1, 10, "Critical", 65000],
            ["SP-007", "HPP-12", "Hydraulic filter element", 6, 3, "High", 8000],
            ["SP-008", "HPP-12", "Relief valve cartridge", 1, 9, "Critical", 32000],
        ],
        columns=["spare_id", "asset_id", "spare_part", "stock_qty", "lead_time_days", "spare_criticality", "unit_cost_inr"],
    ).to_csv(DATA_DIR / "spares_inventory.csv", index=False)

    pd.DataFrame(
        [
            ["DL-301", "MTR-204", "Hot Strip Mill", 3.5, "Reduced motor load due to overheating"],
            ["DL-302", "GBX-17", "Finishing Mill", 8.0, "Gearbox vibration may stop finishing stand"],
            ["DL-303", "PMP-09", "Caster Utility", 5.0, "Cooling water instability can affect caster"],
            ["DL-304", "HPP-12", "Plate Mill", 4.0, "Hydraulic low pressure slows actuator cycle"],
        ],
        columns=["delay_id", "asset_id", "area", "delay_hours", "delay_reason"],
    ).to_csv(DATA_DIR / "delay_logs.csv", index=False)

    pd.DataFrame(
        [
            ["INC-401", "MTR-204", "Temperature crossed high alarm during peak load", "High", "Inspect bearing and cooling path"],
            ["INC-402", "GBX-17", "Repeated vibration spikes during finishing load", "Critical", "Plan immediate vibration inspection"],
            ["INC-403", "PMP-09", "Cavitation noise reported by operator", "High", "Check suction strainer and pump inlet"],
            ["INC-404", "HPP-12", "Pressure recovery slow after valve actuation", "Medium", "Inspect filter and relief valve"],
        ],
        columns=["incident_id", "asset_id", "incident_summary", "severity", "recommended_response"],
    ).to_csv(DATA_DIR / "incident_records.csv", index=False)

    if force or not (DATA_DIR / "digital_logbook.csv").exists():
        pd.DataFrame(
            [
                ["2026-06-01 08:00:00", "system", "MTR-204", "Initial condition review created", "Open"],
                ["2026-06-01 08:05:00", "system", "GBX-17", "Vibration watchlist created", "Open"],
                ["2026-06-01 08:10:00", "system", "PMP-09", "Cavitation watchlist created", "Open"],
                ["2026-06-01 08:15:00", "system", "HPP-12", "Pressure watchlist created", "Open"],
            ],
            columns=["timestamp", "user", "asset_id", "log_entry", "status"],
        ).to_csv(DATA_DIR / "digital_logbook.csv", index=False)

    if force or not (DATA_DIR / "feedback_log.csv").exists():
        pd.DataFrame(
            columns=["timestamp", "user_id", "asset_id", "query", "feedback_type", "feedback_text", "corrected_action", "outcome"]
        ).to_csv(DATA_DIR / "feedback_log.csv", index=False)

    _write_doc(
        "SOP_MTR_204_motor_overheating.txt",
        """
        Asset MTR-204 induction motor SOP.
        Symptoms: high winding temperature, rising current, vibration increase, repeated thermal alarms.
        Likely root causes: bearing lubrication loss, blocked cooling fins, overloaded drive, rotor imbalance.
        Inspection order: bearing temperature and grease condition, cooling fan, air path, current imbalance, coupling alignment.
        Recommended action: reduce load if temperature exceeds 80 C, inspect lubrication, clean cooling path, plan bearing replacement if vibration exceeds 5 mm/s.
        Required spares: DE bearing 6312, cooling fan kit, grease, current clamp meter.
        """,
    )
    _write_doc(
        "SOP_GBX_17_gearbox_vibration.txt",
        """
        Asset GBX-17 gearbox SOP.
        Symptoms: vibration above 7 mm/s warning or above 9 mm/s critical, temperature rise, abnormal noise.
        Likely root causes: bearing pitting, gear mesh wear, coupling misalignment, oil contamination.
        Inspection order: vibration spectrum, oil sample, bearing temperature, coupling alignment, gear backlash.
        Recommended action: if vibration exceeds 9 mm/s create P1 alert, prepare bearing set, inspect oil, schedule controlled shutdown.
        Required spares: gearbox bearing set, synthetic gear oil, coupling shim pack.
        """,
    )
    _write_doc(
        "SOP_PMP_09_cavitation.txt",
        """
        Asset PMP-09 cooling water pump SOP.
        Symptoms: low suction pressure, vibration increase, rattling sound, fluctuating discharge pressure.
        Likely root causes: clogged suction strainer, low tank level, air ingress, impeller erosion.
        Inspection order: suction strainer, tank level, inlet valve position, seal leakage, impeller condition.
        Recommended action: clean strainer, verify suction head, inspect impeller and mechanical seal, monitor pressure recovery.
        Required spares: mechanical seal kit, impeller assembly, suction gasket.
        """,
    )
    _write_doc(
        "SOP_HPP_12_hydraulic_pressure.txt",
        """
        Asset HPP-12 hydraulic power pack SOP.
        Symptoms: low system pressure, slow actuator response, high oil temperature, pressure alarm.
        Likely root causes: filter choking, relief valve leakage, pump wear, hydraulic oil leakage.
        Inspection order: filter differential pressure, relief valve setting, oil level, pump noise, hose leakage.
        Recommended action: replace filter, inspect relief valve cartridge, top up oil, plan pump inspection if pressure remains below limit.
        Required spares: hydraulic filter element, relief valve cartridge, hydraulic oil.
        """,
    )
    _write_doc(
        "maintenance_prioritization_policy.txt",
        """
        Maintenance priority policy.
        Prioritize by safety risk, asset criticality, failure risk, estimated RUL, production delay, spare availability, and repair lead time.
        P1 Critical: high failure risk, low RUL, critical asset, production bottleneck, or safety risk.
        P2 High: elevated failure risk or repeated alerts with meaningful production impact.
        P3 Medium: warning condition that can be planned.
        P4 Low: monitor only.
        Always provide diagnosis, root cause, risk, RUL estimate, recommended repair plan, spares strategy, evidence sources, and alerting action.
        """,
    )
    _write_doc(
        "feedback_learning_policy.txt",
        """
        Feedback loop policy.
        Engineer feedback must be stored in feedback_log.csv.
        Accepted recommendations increase confidence for similar future cases.
        Rejected recommendations must be used to revise action order and root cause ranking.
        Digital maintenance logbook entries must be created for every agent recommendation and alert.
        """,
    )
    _write_doc(
        "steel_agent_operating_model.txt",
        """
        Steel Plant Agentic AI operating model.
        The agent loop is perceive, retrieve, reason, act, verify, log, and learn.
        Perceive natural language query, asset identifiers, symptoms, sensor summaries, fault messages, process area, and user role.
        Retrieve SOPs, manuals, maintenance history, failure reports, incident records, spares, production delay logs, and current asset health.
        Reason by classifying intent, identifying equipment boundary, separating symptoms from causes, ranking hypotheses, estimating safety and production risk, and selecting the best next action.
        Act by creating maintenance recommendations, alert reports, work order plans, spare strategies, supervisor summaries, SOPs, or RCA reports.
        Verify traceability, risk classification, safety controls, missing data, RUL uncertainty, spare feasibility, and restart acceptance criteria.
        Log each recommendation in the digital maintenance logbook.
        Learn from engineer feedback, actual root cause, outcome, downtime, and corrected action.
        """,
    )
    _write_doc(
        "continuous_caster_breakout_prevention.txt",
        """
        Continuous caster abnormality and breakout prevention guide.
        Critical symptoms include mold temperature rise, uneven heat flux, friction increase, mold level instability, sudden casting speed change, abnormal oscillator vibration, breakout prediction alarm, water flow imbalance, and stopper rod instability.
        Likely causes include sticker formation, poor lubrication, SEN clogging, mold powder mismatch, copper plate wear, improper taper, cooling water blockage, unstable superheat, and control-loop disturbance.
        Immediate action: reduce casting speed under procedure, check mold level and heat flux trend, verify mold cooling water flow and delta temperature, inspect powder feed and lubrication, alert caster pulpit and mechanical maintenance, and prepare controlled stop if breakout risk remains high.
        Evidence: heat flux map, mold thermocouple trend, caster speed trend, water flow readings, oscillation marks, shell thickness estimate, powder batch, SEN condition, and recent tundish or ladle events.
        """,
    )
    _write_doc(
        "blast_furnace_maintenance_sop.txt",
        """
        Blast furnace maintenance decision guide.
        Common concerns include tuyere leakage, stave cooling abnormality, hot blast valve issues, charging equipment faults, burden distribution problems, gas cleaning equipment degradation, and skip or bell-less top failures.
        Risk factors include water ingress, high shell temperature, abnormal tuyere camera observation, cooling water delta temperature rise, high top pressure fluctuation, gas leakage, refractory wear, and unstable burden descent.
        Immediate response: isolate unsafe zones, notify furnace control room, verify cooling circuit pressure and flow, check gas detection and CO exposure controls, reduce operating stress if approved, and involve refractory, mechanical, and process experts.
        Root cause analysis should separate process causes, cooling system causes, refractory wear, instrumentation error, and mechanical failure.
        """,
    )
    _write_doc(
        "rolling_mill_vibration_quality_guide.txt",
        """
        Rolling mill vibration and quality guide.
        Symptoms include high stand vibration, chatter marks, roll force variation, strip thickness deviation, camber, flatness defects, bearing temperature rise, gearbox noise, and motor current oscillation.
        Mechanical causes include roll bearing wear, chock clearance, gearbox gear mesh defect, coupling misalignment, foundation looseness, lubrication starvation, and roll imbalance.
        Process or control causes include incorrect reduction schedule, unstable tension, roll bite instability, cooling mismatch, AGC control issue, and speed resonance.
        Inspection order: trend vibration by stand, compare drive and operator side readings, check roll force and motor current oscillation, verify lubrication flow, inspect roll chocks and bearings, take oil sample, check coupling alignment, and compare recent roll changes.
        """,
    )
    _write_doc(
        "industrial_safety_loto_policy.txt",
        """
        Industrial maintenance safety and LOTO policy.
        Before intrusive maintenance, identify electrical, hydraulic, pneumatic, mechanical, gravitational, thermal, chemical, and stored process energy.
        Minimum controls include lockout-tagout, zero-energy verification, permit to work, line break permit, hot work permit, confined space permit where required, barricading, gas testing, PPE, and control room communication.
        Stop-work triggers include unexpected movement, uncontrolled pressure release, high temperature exposure, CO or gas alarm, unstable suspended load, missing isolation confirmation, water ingress into hot process equipment, and repeated critical alarm without clear cause.
        Restart requires tool clearance, guard restoration, interlock verification, no leakage, normal vibration or pressure reading, operator acceptance, and logbook closure.
        """,
    )
    _write_doc(
        "spares_procurement_strategy.txt",
        """
        Steel maintenance spares and procurement strategy.
        Classify spares by criticality, lead time, consumption rate, failure consequence, and substitute availability.
        Critical long-lead spares must be reserved before planned shutdown. Zero-stock critical spares require immediate purchase requisition and management visibility.
        For each maintenance recommendation, state available stock, lead time, repairability, substitute rules, and whether procurement should be immediate, planned, or monitored.
        Procurement priority increases when an asset is a bottleneck, RUL is low, safety risk is high, previous delay hours are high, or no substitute is approved.
        """,
    )

    return steel_df


def _download_ai4i_dataset() -> tuple[pd.DataFrame, str]:
    last_error = None
    for url in PUBLIC_AI4I_URLS:
        try:
            if url.endswith(".csv"):
                return pd.read_csv(url), url

            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
            urllib.request.urlretrieve(url, tmp_path)
            with zipfile.ZipFile(tmp_path, "r") as zf:
                csv_names = [name for name in zf.namelist() if name.lower().endswith(".csv")]
                if not csv_names:
                    raise ValueError("No CSV found inside AI4I zip")
                with zf.open(csv_names[0]) as fh:
                    return pd.read_csv(fh), url
        except Exception as exc:  # pragma: no cover - network fallback path
            last_error = str(exc)
    raise RuntimeError(f"Could not download AI4I dataset. Last error: {last_error}")


def convert_ai4i_to_common_schema(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert AI4I to app schema without using target leakage as features."""

    df = raw_df.copy()
    required = [
        "UDI",
        "Type",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Machine failure",
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"AI4I missing required columns: {missing}")

    udi = pd.to_numeric(df["UDI"], errors="coerce")
    udi_fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index)
    udi = udi.fillna(udi_fallback).astype(int)

    process_c = pd.to_numeric(df["Process temperature [K]"], errors="coerce") - 273.15
    rpm = pd.to_numeric(df["Rotational speed [rpm]"], errors="coerce")
    torque = pd.to_numeric(df["Torque [Nm]"], errors="coerce")
    wear = pd.to_numeric(df["Tool wear [min]"], errors="coerce")
    machine_type = df["Type"].astype(str)

    torque_delta = (torque - torque.median()).abs()
    rpm_low_signal = (rpm.quantile(0.90) - rpm).clip(lower=0)

    alarm_count = (
        (wear > wear.quantile(0.80)).astype(int)
        + (torque > torque.quantile(0.90)).astype(int)
        + (rpm < rpm.quantile(0.10)).astype(int)
        + (process_c > process_c.quantile(0.90)).astype(int)
    )

    criticality_map = {"H": ("High", 3), "M": ("Medium", 2), "L": ("Low", 1)}
    timestamps = pd.date_range(start=pd.Timestamp("2025-01-01 00:00:00"), periods=len(df), freq="min")

    out = pd.DataFrame(
        {
            "source": "public_ai4i_benchmark",
            "timestamp": timestamps.strftime("%Y-%m-%d %H:%M:%S"),
            "asset_id": ["AI4I-" + str(value) for value in udi],
            "asset_type": "Public AI4I Machine",
            "area": "Public Benchmark",
            "criticality": machine_type.map(lambda x: criticality_map.get(x, ("Medium", 2))[0]),
            "criticality_score": machine_type.map(lambda x: criticality_map.get(x, ("Medium", 2))[1]).astype(int),
            "temperature": process_c.round(3),
            "vibration": (2.0 + (torque_delta / 12.0) + (wear / 160.0) + (rpm_low_signal / 900.0)).round(3),
            "current": (35.0 + (torque * 0.75) + (wear / 12.0)).round(3),
            "pressure": (9.5 - (torque_delta / 18.0) - (wear / 260.0)).round(3),
            "rpm": rpm.round(3),
            "alarm_count": alarm_count.astype(int),
            "delay_hours": 0.0,
            "spare_lead_time_days": 0,
            "failure_label": pd.to_numeric(df["Machine failure"], errors="coerce").fillna(0).astype(int),
            "failure_mode": "public_ai4i_target_label_only",
        }
    )
    return out[COMMON_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna(
        subset=["temperature", "vibration", "current", "pressure", "rpm", "alarm_count", "failure_label"]
    )


def ingest_public_ai4i(force: bool = False) -> pd.DataFrame:
    """Download and convert AI4I. Returns an empty frame when offline."""

    out_path = DATA_DIR / "public_ai4i_common_schema.csv"
    if out_path.exists() and not force:
        return pd.read_csv(out_path)

    try:
        raw, source_url = _download_ai4i_dataset()
        raw.to_csv(PUBLIC_DIR / "public_ai4i_raw.csv", index=False)
        raw.to_csv(DATA_DIR / "public_ai4i_raw.csv", index=False)
        public_df = convert_ai4i_to_common_schema(raw)
        available = True
        error = ""
    except Exception as exc:  # pragma: no cover - depends on internet availability
        source_url = ""
        public_df = pd.DataFrame(columns=COMMON_COLUMNS)
        available = False
        error = str(exc)

    public_df.to_csv(out_path, index=False)
    report = {
        "public_ai4i_available": available,
        "source_url": source_url,
        "error": error,
        "public_rows": int(len(public_df)),
        "leakage_control": {
            "machine_failure_used_as_feature": False,
            "twf_hdf_pwf_osf_rnf_used_as_features": False,
            "machine_failure_used_only_as_target": True,
            "steel_app_decision_model_separate_from_public_benchmark": True,
        },
    }
    (DATA_DIR / "public_ai4i_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return public_df


def create_compatibility_sensor_log() -> pd.DataFrame:
    steel_df = pd.read_csv(DATA_DIR / "steel_sensor_logs.csv")
    steel_df["source_public"] = 0
    steel_df["is_demo_asset"] = 1

    public_path = DATA_DIR / "public_ai4i_common_schema.csv"
    if public_path.exists():
        public_df = pd.read_csv(public_path)
    else:
        public_df = pd.DataFrame(columns=COMMON_COLUMNS)
    public_df["source_public"] = 1
    public_df["is_demo_asset"] = 0

    combined = pd.concat([steel_df, public_df], ignore_index=True, sort=False)
    combined.to_csv(DATA_DIR / "sensor_logs.csv", index=False)
    combined.to_csv(DATA_DIR / "combined_training_logs.csv", index=False)
    return combined


def write_data_sources_doc() -> None:
    (DATA_DIR.parent / "DATA_SOURCES.md").write_text(
        """# Data Sources

## Steel demo data
Synthetic steel maintenance data is generated for four industrial assets:
MTR-204, GBX-17, PMP-09, and HPP-12.

It includes sensor logs, maintenance history, failure reports, delay logs,
incident records, spare inventory, feedback logs, and a digital logbook.

## Public AI4I benchmark
The AI4I 2020 Predictive Maintenance dataset is used only as an external
public benchmark to demonstrate ML validation.

Leakage control:
- `Machine failure` is used only as the supervised target label.
- Failure subtype labels such as TWF, HDF, PWF, OSF, and RNF are not used as model features.
- Sensor proxy fields are engineered only from non-target process variables.
- The steel app decision layer uses a separate steel demo model plus operational rules.
""",
        encoding="utf-8",
    )


def prepare_data(force: bool = False) -> dict:
    steel_df = generate_steel_demo_data(force=force)
    public_df = ingest_public_ai4i(force=force)
    combined = create_compatibility_sensor_log()
    write_data_sources_doc()
    return {
        "steel_rows": int(len(steel_df)),
        "public_rows": int(len(public_df)),
        "combined_rows": int(len(combined)),
        "public_available": bool(len(public_df) > 0),
    }
