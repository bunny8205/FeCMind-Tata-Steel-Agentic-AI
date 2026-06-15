"""Dynamic asset memory, updates, and scoring for user-added plant equipment."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR


DYNAMIC_ASSET_PATH = DATA_DIR / "dynamic_assets.csv"
DYNAMIC_ASSET_HISTORY_PATH = DATA_DIR / "dynamic_asset_history.csv"
DYNAMIC_RULE_PATH = DATA_DIR / "dynamic_rules.csv"
DYNAMIC_RULE_APPLICATION_PATH = DATA_DIR / "dynamic_rule_applications.csv"

DYNAMIC_ASSET_COLUMNS = [
    "asset_id",
    "asset_type",
    "area",
    "criticality",
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "alarm_count",
    "delay_hours",
    "spare_lead_time_days",
    "operator_notes",
    "active",
    "created_at",
    "updated_at",
    "source_query",
]

DYNAMIC_HISTORY_COLUMNS = [
    "event_id",
    "event_type",
    "asset_id",
    "changed_at",
    "changed_fields",
    "previous_score",
    "new_score",
    "previous_priority",
    "new_priority",
    "previous_risk_band",
    "new_risk_band",
    "source_query",
    "previous_record",
    "new_record",
]

DYNAMIC_RULE_COLUMNS = [
    "rule_id",
    "timestamp",
    "rule_type",
    "equipment_pattern",
    "area_pattern",
    "condition_text",
    "priority_override",
    "risk_override",
    "source_text",
    "normalized_rule_text",
    "rule_key",
    "active",
]

DYNAMIC_RULE_APPLICATION_COLUMNS = [
    "event_id",
    "timestamp",
    "rule_id",
    "asset_id",
    "base_priority",
    "final_priority",
    "base_risk_band",
    "final_risk_band",
    "base_score",
    "final_score",
    "source",
]

NUMERIC_FIELDS = [
    "temperature",
    "vibration",
    "current",
    "pressure",
    "rpm",
    "alarm_count",
    "delay_hours",
    "spare_lead_time_days",
]

READING_LIMITS = {
    "temperature": (-50.0, 250.0, "temperature must be physically plausible for industrial equipment"),
    "vibration": (0.0, 100.0, "vibration cannot be negative or beyond practical sensor range"),
    "current": (0.0, 5000.0, "current cannot be negative or beyond configured plant sensor range"),
    "pressure": (0.0, 500.0, "pressure cannot be negative or beyond configured plant sensor range"),
    "rpm": (0.0, 20000.0, "rpm cannot be negative or beyond configured rotating-equipment range"),
    "alarm_count": (0.0, 1000.0, "alarm count cannot be negative"),
    "delay_hours": (0.0, 10000.0, "delay hours cannot be negative"),
    "spare_lead_time_days": (0.0, 3650.0, "lead time cannot be negative"),
}

READING_CONNECTOR = (
    r"(?:\s+(?:is|was|has|have|now|changed|increased|decreased|reduced|improved|"
    r"returned|stabilized|normalised|normalized|dropped|rose|fell|went|became|set|updated|of))*\s*(?:=|:|to)?\s*"
)


def validate_dynamic_asset_readings(asset: dict | pd.Series) -> list[dict]:
    """Return invalid physical readings without mutating memory."""

    record = dict(asset)
    issues: list[dict] = []
    for field, (low, high, reason) in READING_LIMITS.items():
        if field not in record or _is_missing(record.get(field)):
            continue
        value = _num(record.get(field), np.nan)
        if np.isnan(value):
            continue
        if value < low or value > high:
            issues.append(
                {
                    "field": field,
                    "value": value,
                    "allowed_min": low,
                    "allowed_max": high,
                    "reason": reason,
                }
            )
    return issues


def _clean_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, float) and np.isnan(value):
        return default
    text = str(value).strip()
    return text if text else default


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _num(value: Any, default: float = np.nan) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def _extract_text(patterns: list[str], text: str, default: str = "") -> str:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" :,-.;")
            if value:
                return value
    return default


def _extract_number(patterns: list[str], text: str, default: float | None = None) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    return default


def _reading_pattern(label: str) -> str:
    return rf"\b{label}{READING_CONNECTOR}(-?\d+(?:\.\d+)?)"


def _extract_operator_notes(text: str) -> str:
    notes: list[str] = []
    patterns = [
        r"\b(?:operator|technician|engineer|shift team|maintenance team)\s+(?:reports?|reported|observes?|observed|mentions?|mentioned|hears?|heard|sees?|saw)\s+([^.;\n]+)",
        r"\b(?:symptom|observation|operator note|field note|note)\s*(?:is|=|:)?\s*([^.;\n]+)",
        r"\b(?:but|and)\s+(?:there is|there are|there was|there were|it has|asset has|shows?|showing)\s+([^.;\n]*(?:noise|cavitation|leak|smoke|smell|surge|sparking|overheat|overheating|vibration|chatter|rubbing)[^.;\n]*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            note = re.sub(r"\s+", " ", match.group(1)).strip(" ,:-")
            if note and note.lower() not in {n.lower() for n in notes}:
                notes.append(note)
    lowered = text.lower()
    symptom_phrases = [
        "high temperature",
        "high oil temperature",
        "rising oil temperature",
        "high current",
        "abnormal vibration",
        "repeated trips",
        "metallic particles",
        "metal particles",
        "no spare bearing available",
        "no spare available",
        "loud cavitation noise",
        "cavitation noise",
        "abnormal noise",
        "bearing noise",
        "burning smell",
        "oil leak",
        "hydraulic leak",
        "smoke",
        "sparking",
        "surging",
    ]
    for phrase in symptom_phrases:
        existing = {n.lower() for n in notes}
        if phrase in lowered and phrase not in existing and not any(phrase in note or note in phrase for note in existing):
            notes.append(phrase)
    return "; ".join(notes)


def _json_clean(record: dict) -> dict:
    clean: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (np.integer,)):
            clean[key] = int(value)
        elif isinstance(value, (np.floating,)):
            clean[key] = None if np.isnan(value) else float(value)
        elif isinstance(value, float) and np.isnan(value):
            clean[key] = None
        elif pd.isna(value) if not isinstance(value, (list, dict, tuple, set)) else False:
            clean[key] = None
        else:
            clean[key] = value
    return clean


def load_dynamic_assets() -> pd.DataFrame:
    if DYNAMIC_ASSET_PATH.exists():
        df = pd.read_csv(DYNAMIC_ASSET_PATH)
        for col in DYNAMIC_ASSET_COLUMNS:
            if col == "active" and col not in df.columns:
                df[col] = True
            elif col not in df.columns:
                df[col] = np.nan
        for col in NUMERIC_FIELDS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["active"] = df["active"].apply(lambda value: str(value).strip().lower() not in {"false", "0", "no", "inactive", ""})
        for col in ["asset_id", "asset_type", "area", "criticality", "operator_notes", "created_at", "updated_at", "source_query"]:
            df[col] = df[col].astype("object")
        df["asset_id"] = df["asset_id"].astype(str).str.upper()
        return df[DYNAMIC_ASSET_COLUMNS].copy()
    return pd.DataFrame(columns=DYNAMIC_ASSET_COLUMNS)


def save_dynamic_assets(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for col in DYNAMIC_ASSET_COLUMNS:
        if col == "active" and col not in df.columns:
            df[col] = True
        elif col not in df.columns:
            df[col] = np.nan
    out = df[DYNAMIC_ASSET_COLUMNS].copy()
    out["active"] = out["active"].apply(lambda value: str(value).strip().lower() not in {"false", "0", "no", "inactive", ""})
    for col in ["asset_id", "asset_type", "area", "criticality", "operator_notes", "created_at", "updated_at", "source_query"]:
        out[col] = out[col].astype("object")
    out["asset_id"] = out["asset_id"].astype(str).str.upper()
    out.to_csv(DYNAMIC_ASSET_PATH, index=False)


def load_dynamic_asset_history() -> pd.DataFrame:
    if DYNAMIC_ASSET_HISTORY_PATH.exists():
        df = pd.read_csv(DYNAMIC_ASSET_HISTORY_PATH)
        for col in DYNAMIC_HISTORY_COLUMNS:
            if col not in df.columns:
                df[col] = ""
        return df[DYNAMIC_HISTORY_COLUMNS].copy()
    return pd.DataFrame(columns=DYNAMIC_HISTORY_COLUMNS)


def append_dynamic_asset_history(rows: list[dict]) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current = load_dynamic_asset_history()
    if not rows:
        return current
    incoming = pd.DataFrame(rows)
    for col in DYNAMIC_HISTORY_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = ""
    merged = pd.concat([current, incoming[DYNAMIC_HISTORY_COLUMNS]], ignore_index=True, sort=False)
    merged.to_csv(DYNAMIC_ASSET_HISTORY_PATH, index=False)
    return merged


def load_dynamic_rules() -> pd.DataFrame:
    if DYNAMIC_RULE_PATH.exists():
        df = pd.read_csv(DYNAMIC_RULE_PATH)
        for col in DYNAMIC_RULE_COLUMNS:
            if col not in df.columns:
                df[col] = "" if col != "active" else True
        if "active" in df.columns:
            df["active"] = df["active"].apply(lambda value: str(value).strip().lower() not in {"false", "0", "no", ""})
        for idx, row in df.iterrows():
            normalized = normalize_rule_text(row.get("condition_text") or row.get("source_text") or "")
            df.at[idx, "normalized_rule_text"] = normalized
            df.at[idx, "rule_key"] = dynamic_rule_key(row.to_dict())
        return df[DYNAMIC_RULE_COLUMNS].copy()
    return pd.DataFrame(columns=DYNAMIC_RULE_COLUMNS)


def save_dynamic_rules(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for col in DYNAMIC_RULE_COLUMNS:
        if col not in df.columns:
            df[col] = "" if col != "active" else True
    out = df[DYNAMIC_RULE_COLUMNS].copy()
    for idx, row in out.iterrows():
        normalized = normalize_rule_text(row.get("condition_text") or row.get("source_text") or "")
        out.at[idx, "normalized_rule_text"] = normalized
        out.at[idx, "rule_key"] = dynamic_rule_key(row.to_dict())
    out["active"] = out["active"].apply(lambda value: str(value).strip().lower() not in {"false", "0", "no", ""})
    out.to_csv(DYNAMIC_RULE_PATH, index=False)


def append_dynamic_rule_application(rows: list[dict]) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DYNAMIC_RULE_APPLICATION_PATH.exists():
        current = pd.read_csv(DYNAMIC_RULE_APPLICATION_PATH)
    else:
        current = pd.DataFrame(columns=DYNAMIC_RULE_APPLICATION_COLUMNS)
    if not rows:
        return current
    incoming = pd.DataFrame(rows)
    for col in DYNAMIC_RULE_APPLICATION_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = ""
        if col not in current.columns:
            current[col] = ""
    merged = pd.concat([current, incoming[DYNAMIC_RULE_APPLICATION_COLUMNS]], ignore_index=True, sort=False)
    merged.to_csv(DYNAMIC_RULE_APPLICATION_PATH, index=False)
    return merged


def latest_dynamic_asset_change(asset_id: str) -> dict | None:
    history = load_dynamic_asset_history()
    if history.empty:
        return None
    rows = history[
        (history["asset_id"].astype(str).str.upper() == str(asset_id).upper())
        & (history["event_type"].astype(str).str.lower() == "update")
    ].copy()
    if rows.empty:
        return None
    rows["changed_at_sort"] = pd.to_datetime(rows["changed_at"], errors="coerce", format="mixed")
    row = rows.sort_values("changed_at_sort").iloc[-1].to_dict()
    for key in ["previous_record", "new_record"]:
        try:
            row[key] = json.loads(row.get(key) or "{}")
        except json.JSONDecodeError:
            row[key] = {}
    try:
        row["changed_fields"] = json.loads(row.get("changed_fields") or "[]")
    except json.JSONDecodeError:
        row["changed_fields"] = []
    return row


def dynamic_asset_ids(active_only: bool = True) -> list[str]:
    df = load_dynamic_assets()
    if df.empty:
        return []
    if active_only and "active" in df.columns:
        df = df[df["active"].astype(bool)].copy()
    return sorted(df["asset_id"].dropna().astype(str).str.upper().unique().tolist())


def is_asset_ingestion_query(query: str) -> bool:
    q = str(query).lower()
    has_asset_id = bool(re.search(r"\b[A-Z]{2,10}-\d{1,5}\b", str(query), flags=re.IGNORECASE))
    if has_asset_id and re.search(r"\b(register|add|remember|create|ingest)\b.{0,80}\b(named|called)\b", q, flags=re.IGNORECASE):
        return True
    add_terms = [
        "add a new asset",
        "add new asset",
        "add asset",
        "register asset",
        "register a new",
        "register new",
        "register a new asset",
        "register new asset",
        "register new assets",
        "register three new assets",
        "remember this asset",
        "remember these assets",
        "remember all",
        "ingest asset",
        "create asset",
        "new asset:",
    ]
    return has_asset_id and any(term in q for term in add_terms)


def is_asset_update_query(query: str) -> bool:
    if is_asset_ingestion_query(query):
        return False
    if is_rule_ingestion_query(query):
        return False
    q = str(query).lower()
    explicit_update = q.strip().startswith(("update ", "change ", "set "))
    if not explicit_update and any(
        term in q
        for term in [
            "predict",
            "remaining useful",
            "rul",
            "risk assessment",
            "maintenance plan",
            "decision summary",
            "supervisor",
            "scenario",
            "schedule",
            "prioritize",
            "priority",
            "procurement lead time",
        ]
    ):
        return False
    update_terms = [
        "update",
        "change",
        "changed",
        "reduced",
        "increased",
        "improved",
        "dropped",
        " set ",
        "new reading",
        "new readings",
        "latest reading",
        "latest readings",
        "reading changed",
        "readings changed",
        "increased to",
        "decreased to",
    ]
    has_update_language = any(term in q for term in update_terms)
    has_reference_language = any(
        term in q
        for term in [
            "update it",
            "update that",
            "update the same asset",
            "now update it",
            "that asset",
            "same asset",
        ]
    )
    if not extract_asset_ids(query) and not has_reference_language and not explicit_update:
        return False
    return (has_update_language or has_reference_language) and (
        bool(extract_reading_fields(query))
        or bool(extract_asset_ids(query))
        or has_reference_language
    )


def is_rule_ingestion_query(query: str) -> bool:
    q = str(query).lower()
    if any(term in q for term in ["self-test", "self test", "verifier summary", "tools used", "agent objective"]):
        return False
    explicit_rule_phrases = [
        "remember the safety rule",
        "remember the sop rule",
        "remember the rule",
        "remember that any",
        "remember any",
        "remember this safety rule",
        "remember this sop rule",
        "remember this policy",
        "remember this rule",
        "save this safety rule",
        "save this sop rule",
        "save this rule",
        "store this safety rule",
        "store this sop rule",
        "store this rule",
        "learn this safety rule",
        "learn this sop rule",
        "learn this rule",
        "add safety rule",
        "add sop rule",
        "add rule",
    ]
    if not any(phrase in q for phrase in explicit_rule_phrases):
        if is_rule_apply_query(query):
            return False
        return False
    memory_patterns = [
        r"\bremember\b",
        r"\bsave\b",
        r"\bstore\b",
        r"\blearn\b",
        r"\bkeep this\b",
    ]
    rule_terms = ["rule", "safety rule", "sop rule", "policy", "override", "must be", "should trigger", "must be inspected", "requires supervisor"]
    return any(re.search(pattern, q) for pattern in memory_patterns) and any(term in q for term in rule_terms)


def is_rule_apply_query(query: str) -> bool:
    q = str(query).lower()
    if is_asset_reactivation_query(query) or is_asset_deactivation_query(query):
        return False
    plant_decision_terms = [
        "choose exactly one",
        "only one maintenance crew",
        "one maintenance crew",
        "plant ranking",
        "rest of the plant",
        "rank original",
        "rank active",
        "original demo assets",
        "active dynamic assets together",
        "plant-wide",
        "immediate maintenance",
        "compare gbx",
        "compare ",
        "prioritize",
        "which asset",
    ]
    if any(term in q for term in plant_decision_terms):
        return False

    explicit_apply_to_asset = bool(
        re.search(
            r"\bapply\b.{0,80}\b(rule|rules|remembered rule|remembered safety|safety rule|sop rule)\b.{0,50}\bto\b\s+[A-Z]{2,10}-\d{1,5}\b",
            str(query),
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    explicit_apply_to_context = any(
        term in q
        for term in [
            "apply that rule",
            "apply this rule",
            "apply the rule",
            "apply remembered rules to this asset",
            "apply remembered safety rules to this asset",
            "apply remembered rules to that asset",
            "apply remembered safety rules to that asset",
            "apply remembered rules to it",
            "apply remembered safety rules to it",
        ]
    )
    return explicit_apply_to_asset or explicit_apply_to_context


def is_asset_deactivation_query(query: str) -> bool:
    q = str(query).lower()
    terms = [
        "mark asset as inspected",
        "mark as inspected",
        "inspected and cleared",
        "cleared",
        "remove from active ranking",
        "remove it from active ranking",
        "deactivate",
        "make inactive",
        "no longer active",
        "stand down",
    ]
    return any(term in q for term in terms)


def is_asset_reactivation_query(query: str) -> bool:
    q = str(query).lower()
    terms = [
        "reactivate",
        "make active",
        "back to active",
        "return to active ranking",
        "put back in active",
        "reopen",
    ]
    return any(term in q for term in terms)


def is_priority_change_query(query: str) -> bool:
    q = str(query).lower()
    return any(
        term in q
        for term in [
            "priority change",
            "priority changed",
            "did priority change",
            "what changed after",
            "after the update",
            "after update",
            "changed after update",
        ]
    )


def query_mentions_new_asset_reference(query: str) -> bool:
    q = str(query).lower()
    return any(
        term in q
        for term in [
            "same new asset",
            "this new asset",
            "that new asset",
            "the new asset",
            "this new equipment",
            "that new equipment",
            "newly added asset",
            "newly added assets",
            "added asset",
            "added assets",
        ]
    )


def extract_reading_fields(text: str) -> dict[str, Any]:
    segment = str(text)
    fields: dict[str, Any] = {}
    patterns = {
        "temperature": [_reading_pattern(r"temp(?:erature)?")],
        "vibration": [_reading_pattern(r"vib(?:ration)?")],
        "current": [_reading_pattern(r"current")],
        "pressure": [_reading_pattern(r"pressure")],
        "rpm": [_reading_pattern(r"rpm")],
        "alarm_count": [
            _reading_pattern(r"alarm\s*count"),
            _reading_pattern(r"alarms?"),
        ],
    }
    for field, pats in patterns.items():
        value = _extract_number(pats, segment, None)
        if value is not None:
            fields[field] = value
    note = _extract_operator_notes(segment)
    if note:
        fields["operator_notes"] = note
    return fields


def extract_asset_ids(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in re.finditer(r"\b[A-Z]{2,10}-\d{1,5}\b", str(text), flags=re.IGNORECASE):
        asset_id = match.group(0).upper()
        if asset_id not in seen:
            seen.add(asset_id)
            out.append(asset_id)
    return out


def _segment_by_asset_id(query: str) -> list[tuple[str, str]]:
    ids = list(re.finditer(r"\b[A-Z]{2,10}-\d{1,5}\b", query, flags=re.IGNORECASE))
    segments: list[tuple[str, str]] = []
    for idx, match in enumerate(ids):
        start = match.start()
        end = ids[idx + 1].start() if idx + 1 < len(ids) else len(query)
        segment = query[start:end]
        segments.append((match.group(0).upper(), segment))
    return segments


def _infer_asset_type(asset_id: str, segment: str, explicit: str) -> str:
    if explicit and explicit.lower() not in {"unspecified equipment", "unknown", "not provided"}:
        return explicit
    tail = re.sub(r"^\s*(?:id\s*)?" + re.escape(asset_id) + r"\b", "", segment, flags=re.IGNORECASE).strip(" ,:-")
    tail = re.split(
        r"\b(?:area|criticality|temperature|temp|vibration|vib|current|pressure|rpm|alarm|alarms|available readings|readings|spares|spare|with|has|showing|after)\b",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    candidate = re.sub(r"\b(?:named|called|with|has|showing)\b", " ", tail, flags=re.IGNORECASE)
    candidate = re.sub(r"\s+", " ", candidate).strip(" ,:-.;")
    if len(candidate) >= 3 and not re.search(r"\d", candidate):
        return candidate.title()

    prefix = asset_id.split("-", 1)[0]
    prefix_map = {
        "BF": "Blast Furnace Equipment",
        "GBX": "Gearbox",
        "MTR": "Motor",
        "PMP": "Pump",
        "HPP": "Hydraulic Power Pack",
        "RMF": "Roughing Mill Fan",
        "DES": "Descaler Pump",
        "CCM": "Caster Mold Oscillator",
        "SNT": "Sinter Plant Exhaust Fan",
    }
    return prefix_map.get(prefix, "Unspecified Equipment")


def _infer_area(asset_id: str, asset_type: str, explicit: str) -> str:
    if explicit and explicit.lower() not in {"unspecified area", "unknown", "not provided"}:
        return explicit.title()
    text = f"{asset_id} {asset_type}".lower()
    if "sinter" in text:
        return "Sinter Plant"
    if "roughing mill" in text or asset_id.startswith("RMF-"):
        return "Roughing Mill"
    if "descaler" in text or asset_id.startswith("DES-"):
        return "Descaling"
    if "caster" in text or asset_id.startswith("CCM-"):
        return "Caster"
    if "blast furnace" in text or asset_id.startswith("BF-"):
        return "Blast Furnace"
    if "finishing mill" in text or "gearbox" in text:
        return "Finishing Mill"
    if "hydraulic" in text:
        return "Hydraulic Station"
    if "pump" in text:
        return "Utility"
    return "Unspecified Area"


def _criticality_from_segment(segment: str) -> str:
    explicit = _extract_text(
        [
            r"\bcriticality\s*(?:is|=|:)?\s*(critical|high|medium|low)",
            r"\b(critical|high|medium|low)\s+criticality\b",
        ],
        segment,
        "",
    )
    if explicit:
        return explicit.title()
    return "Medium"


def parse_dynamic_assets(query: str) -> list[dict]:
    """Extract one or more user-specified assets from natural language."""

    assets: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")
    for asset_id, segment in _segment_by_asset_id(str(query)):
        explicit_type = _extract_text(
            [
                r"(?:asset\s*type|equipment\s*type|type)\s*(?:is|=|:)?\s*([^,;\n.]+)",
                r"(?:asset|equipment)\s+" + re.escape(asset_id) + r"\s*(?:is|:)?\s*([^,;\n.]+)",
            ],
            segment,
            "",
        )
        asset_type = _infer_asset_type(asset_id, segment, explicit_type)
        explicit_area = _extract_text([r"\barea\s*(?:is|=|:)?\s*([^,;\n.]+)"], segment, "")
        area = _infer_area(asset_id, asset_type, explicit_area)

        asset = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "area": area,
            "criticality": _criticality_from_segment(segment),
            "temperature": _extract_number([_reading_pattern(r"temp(?:erature)?")], segment, np.nan),
            "vibration": _extract_number([_reading_pattern(r"vib(?:ration)?")], segment, np.nan),
            "current": _extract_number([_reading_pattern(r"current")], segment, np.nan),
            "pressure": _extract_number([_reading_pattern(r"pressure")], segment, np.nan),
            "rpm": _extract_number([_reading_pattern(r"rpm")], segment, 1480.0),
            "alarm_count": _extract_number(
                [
                    _reading_pattern(r"alarm\s*count"),
                    _reading_pattern(r"alarms?"),
                ],
                segment,
                0.0,
            ),
            "delay_hours": _extract_number([_reading_pattern(r"delay(?:\s*hours)?")], segment, 0.0),
            "spare_lead_time_days": _extract_number([_reading_pattern(r"lead\s*time(?:\s*days)?")], segment, 0.0),
            "operator_notes": _extract_operator_notes(segment),
            "created_at": now,
            "updated_at": now,
            "source_query": str(query),
        }
        assets.append(asset)
    return assets


def parse_dynamic_asset_updates(query: str, fallback_asset_id: str | None = None) -> list[dict]:
    updates: list[dict] = []
    now = datetime.now().isoformat(timespec="seconds")
    segments = _segment_by_asset_id(str(query))
    if not segments and fallback_asset_id:
        segments = [(str(fallback_asset_id).upper(), str(query))]
    for asset_id, segment in segments:
        fields = extract_reading_fields(segment)
        if fields:
            updates.append({"asset_id": asset_id, "fields": fields, "updated_at": now, "source_query": str(query)})
    return updates


def upsert_dynamic_assets(assets: list[dict]) -> pd.DataFrame:
    current = load_dynamic_assets()
    if not assets:
        return current

    incoming = pd.DataFrame(assets)
    for col in DYNAMIC_ASSET_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = np.nan
    incoming["asset_id"] = incoming["asset_id"].astype(str).str.upper()

    rows: list[dict] = []
    for _, new_row in incoming[DYNAMIC_ASSET_COLUMNS].iterrows():
        asset_id = str(new_row["asset_id"]).upper()
        previous = current[current["asset_id"].astype(str).str.upper() == asset_id]
        row = new_row.to_dict()
        row["active"] = True
        if not previous.empty:
            old = previous.iloc[0].to_dict()
            row["created_at"] = old.get("created_at") or row.get("created_at")
            row["active"] = old.get("active", True)
        rows.append(row)

    keep = current[~current["asset_id"].astype(str).str.upper().isin({row["asset_id"] for row in rows})].copy()
    merged = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True, sort=False)
    save_dynamic_assets(merged)
    return merged


def list_active_dynamic_assets() -> pd.DataFrame:
    df = load_dynamic_assets()
    if df.empty:
        return df
    return df[df["active"].astype(bool)].copy()


def list_inactive_dynamic_assets() -> pd.DataFrame:
    df = load_dynamic_assets()
    if df.empty:
        return df
    return df[~df["active"].astype(bool)].copy()


def _asset_state_history_row(event_type: str, asset_id: str, previous: dict, new: dict, query: str, reason: str = "") -> dict:
    previous_scored = score_dynamic_asset(previous) if previous else {}
    new_scored = score_dynamic_asset(new) if new else {}
    changed_fields = ["active"]
    if reason:
        changed_fields.append("reason")
    return {
        "event_id": str(uuid.uuid4()),
        "event_type": event_type,
        "asset_id": asset_id,
        "changed_at": datetime.now().isoformat(timespec="seconds"),
        "changed_fields": json.dumps(changed_fields),
        "previous_score": previous_scored.get("hybrid_health_score"),
        "new_score": new_scored.get("hybrid_health_score"),
        "previous_priority": previous_scored.get("priority"),
        "new_priority": new_scored.get("priority"),
        "previous_risk_band": previous_scored.get("risk_band"),
        "new_risk_band": new_scored.get("risk_band"),
        "source_query": query or reason,
        "previous_record": json.dumps(_json_clean(previous_scored)),
        "new_record": json.dumps(_json_clean(new_scored)),
    }


def mark_dynamic_asset_inactive(asset_id: str, reason: str = "", query: str = "") -> dict:
    current = load_dynamic_assets()
    target = str(asset_id).upper()
    idx = current.index[current["asset_id"].astype(str).str.upper() == target].tolist()
    if not idx:
        return {"asset_id": target, "status": "missing", "reason": reason, "history": None}
    row_idx = idx[0]
    previous = current.loc[row_idx].to_dict()
    current.at[row_idx, "active"] = False
    current.at[row_idx, "updated_at"] = datetime.now().isoformat(timespec="seconds")
    current.at[row_idx, "source_query"] = query or reason or "deactivated"
    new = current.loc[row_idx].to_dict()
    save_dynamic_assets(current)
    history = _asset_state_history_row("deactivate", target, previous, new, query, reason)
    append_dynamic_asset_history([history])
    return {"asset_id": target, "status": "inactive", "reason": reason, "history": history, "record": new}


def reactivate_dynamic_asset(asset_id: str, updates: dict | None = None, query: str = "") -> dict:
    current = load_dynamic_assets()
    target = str(asset_id).upper()
    idx = current.index[current["asset_id"].astype(str).str.upper() == target].tolist()
    if not idx:
        return {"asset_id": target, "status": "missing", "history": None}
    row_idx = idx[0]
    previous = current.loc[row_idx].to_dict()
    current.at[row_idx, "active"] = True
    if updates:
        for field, value in updates.items():
            if field in DYNAMIC_ASSET_COLUMNS:
                current.at[row_idx, field] = value
    current.at[row_idx, "updated_at"] = datetime.now().isoformat(timespec="seconds")
    current.at[row_idx, "source_query"] = query or "reactivated"
    new = current.loc[row_idx].to_dict()
    save_dynamic_assets(current)
    history = _asset_state_history_row("reactivate", target, previous, new, query)
    append_dynamic_asset_history([history])
    return {"asset_id": target, "status": "active", "history": history, "record": new}


def update_dynamic_assets_from_query(query: str, fallback_asset_id: str | None = None) -> dict:
    updates = parse_dynamic_asset_updates(query, fallback_asset_id=fallback_asset_id)
    current = load_dynamic_assets()
    if not updates:
        return {"updated": [], "missing": [], "history": [], "invalid": []}

    now = datetime.now().isoformat(timespec="seconds")
    updated_rows: list[dict] = []
    missing: list[str] = []
    invalid: list[dict] = []
    history_rows: list[dict] = []

    for update in updates:
        asset_id = str(update["asset_id"]).upper()
        idx = current.index[current["asset_id"].astype(str).str.upper() == asset_id].tolist()
        if not idx:
            missing.append(asset_id)
            continue

        row_idx = idx[0]
        previous_raw = current.loc[row_idx].to_dict()
        previous_scored = score_dynamic_asset(previous_raw)
        changed_fields: list[str] = []
        for field, value in update["fields"].items():
            if field in DYNAMIC_ASSET_COLUMNS:
                if field == "operator_notes":
                    old_note = _clean_text(current.at[row_idx, field], "")
                    new_note = _clean_text(value, "")
                    if old_note and new_note and new_note.lower() not in old_note.lower():
                        value = f"{old_note}; {new_note}"
                    elif old_note and not new_note:
                        value = old_note
                current.at[row_idx, field] = value
                changed_fields.append(field)
        current.at[row_idx, "updated_at"] = now
        current.at[row_idx, "source_query"] = str(query)
        new_raw = current.loc[row_idx].to_dict()
        invalid_fields = validate_dynamic_asset_readings(new_raw)
        if invalid_fields:
            for issue in invalid_fields:
                issue["asset_id"] = asset_id
            invalid.extend(invalid_fields)
            for field in changed_fields:
                current.at[row_idx, field] = previous_raw.get(field)
            current.at[row_idx, "updated_at"] = previous_raw.get("updated_at")
            current.at[row_idx, "source_query"] = previous_raw.get("source_query")
            continue
        new_scored = score_dynamic_asset(new_raw)
        updated_rows.append(new_scored)
        history_rows.append(
            {
                "event_id": str(uuid.uuid4()),
                "event_type": "update",
                "asset_id": asset_id,
                "changed_at": now,
                "changed_fields": json.dumps(changed_fields),
                "previous_score": previous_scored.get("hybrid_health_score"),
                "new_score": new_scored.get("hybrid_health_score"),
                "previous_priority": previous_scored.get("priority"),
                "new_priority": new_scored.get("priority"),
                "previous_risk_band": previous_scored.get("risk_band"),
                "new_risk_band": new_scored.get("risk_band"),
                "source_query": str(query),
                "previous_record": json.dumps(_json_clean(previous_scored)),
                "new_record": json.dumps(_json_clean(new_scored)),
            }
        )

    if updated_rows:
        save_dynamic_assets(current)
        append_dynamic_asset_history(history_rows)
    return {"updated": updated_rows, "missing": missing, "history": history_rows, "invalid": invalid}


def criticality_score(criticality: Any) -> int:
    c = str(criticality).strip().lower()
    if c == "critical":
        return 3
    if c == "high":
        return 2
    if c == "medium":
        return 1
    return 0


def _make_rule_id() -> str:
    return "RULE-" + uuid.uuid4().hex[:8].upper()


def normalize_rule_text(text: Any) -> str:
    normalized = str(text or "").lower()
    normalized = re.sub(
        r"^\s*(remember|save|store|learn|keep)\s+(this\s+)?(safety\s+|sop\s+|policy\s+)?rule\s*[:\-]?\s*",
        "",
        normalized,
    )
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, number in number_words.items():
        normalized = re.sub(rf"\b{word}\b", number, normalized)

    replacements = {
        "alarm_count": "alarm count",
        "alarm counts": "alarm count",
        "alarms": "alarm count",
        "greater than": "above",
        "more than": "above",
        "over": "above",
        ">=": "above",
        ">": "above",
        "less than": "below",
        "under": "below",
        "<=": "below",
        "<": "below",
        "must be": "trigger",
        "should be": "trigger",
        "must receive immediate": "trigger",
        "must receive": "trigger",
        "should receive": "trigger",
        "make it": "trigger",
        "priority p1": "trigger p1",
        "priority p2": "trigger p2",
        "priority p3": "trigger p3",
        "immediate p1 priority": "trigger p1",
        "mm/s": "",
        "deg c": "",
        " c ": " ",
    }
    for old, new in replacements.items():
        normalized = normalized.replace(old, new)
    normalized = re.sub(r"\bbecause\b.*$", "", normalized)
    normalized = re.sub(r"[^a-z0-9\s><=.-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _rule_condition_signature(text: Any) -> str:
    normalized = normalize_rule_text(text)
    checks: list[str] = []
    aliases = {
        "vibration": r"vib(?:ration)?",
        "temperature": r"temp(?:erature)?",
        "current": r"current",
        "pressure": r"pressure",
        "alarm count": r"alarm\s*count",
    }
    for label, pattern in aliases.items():
        for match in re.finditer(rf"\b(?:{pattern})\b\s*(above|below|at least)\s*(-?\d+(?:\.\d+)?)", normalized):
            op = "above" if match.group(1) == "at least" else match.group(1)
            value = float(match.group(2))
            checks.append(f"{label}:{op}:{value:g}")
        for match in re.finditer(rf"\b(above|below|at least)\s*(-?\d+(?:\.\d+)?)\s*(?:{pattern})\b", normalized):
            op = "above" if match.group(1) == "at least" else match.group(1)
            value = float(match.group(2))
            checks.append(f"{label}:{op}:{value:g}")
    if checks:
        return "&".join(sorted(set(checks)))
    return normalized


def dynamic_rule_key(rule: dict | pd.Series) -> str:
    r = dict(rule)
    normalized = _rule_condition_signature(r.get("condition_text") or r.get("source_text") or "")
    parts = [
        normalized,
        _clean_text(r.get("equipment_pattern"), ".*"),
        _clean_text(r.get("area_pattern"), ".*"),
        _clean_text(r.get("priority_override"), ""),
        _clean_text(r.get("risk_override"), ""),
    ]
    return "||".join(parts)


def deactivate_duplicate_rules() -> pd.DataFrame:
    rules = load_dynamic_rules()
    if rules.empty:
        return rules
    seen: set[str] = set()
    for idx, row in rules.iterrows():
        key = dynamic_rule_key(row.to_dict())
        rules.at[idx, "normalized_rule_text"] = normalize_rule_text(row.get("condition_text") or row.get("source_text") or "")
        rules.at[idx, "rule_key"] = key
        if key in seen and bool(row.get("active", True)):
            rules.at[idx, "active"] = False
        elif bool(row.get("active", True)):
            seen.add(key)
    save_dynamic_rules(rules)
    return load_dynamic_rules()


def list_dynamic_rules_grouped() -> pd.DataFrame:
    rules = load_dynamic_rules()
    if rules.empty:
        return rules
    grouped = (
        rules.groupby(["rule_key", "equipment_pattern", "area_pattern", "priority_override", "risk_override"], dropna=False)
        .agg(
            rule_count=("rule_id", "count"),
            active_count=("active", "sum"),
            latest_timestamp=("timestamp", "max"),
            sample_condition=("condition_text", "first"),
        )
        .reset_index()
    )
    return grouped


def active_unique_dynamic_rules() -> pd.DataFrame:
    rules = deactivate_duplicate_rules()
    if rules.empty:
        return rules
    active = rules[rules["active"].astype(bool)].copy()
    if "rule_key" not in active.columns:
        active["rule_key"] = active.apply(lambda row: dynamic_rule_key(row.to_dict()), axis=1)
    return active.drop_duplicates(subset=["rule_key"], keep="last").copy()


def _infer_rule_equipment_pattern(text: str) -> str:
    q = str(text).lower()
    if "bof" in q and "gearbox" in q:
        return r"bof.*gearbox|basic oxygen furnace.*gearbox|tilting gearbox"
    if "blast furnace" in q and any(term in q for term in ["blower", "fan"]):
        return r"blast furnace.*(blower|fan)|(blower|fan).*blast furnace"
    if "descaler" in q and "pump" in q:
        return r"descaler.*pump|descaling.*pump"
    if "pump" in q:
        return r"pump"
    if "caster" in q or "mold oscillator" in q or "mould oscillator" in q:
        return r"caster|mold oscillator|mould oscillator"
    if "motor" in q:
        return r"motor"
    if "gearbox" in q:
        return r"gearbox"
    if any(term in q for term in ["blower", "fan", "compressor"]):
        return r"blower|fan|compressor"
    if "hydraulic" in q:
        return r"hydraulic"
    return r".*"


def _infer_rule_area_pattern(text: str) -> str:
    q = str(text).lower()
    if "bof" in q or "basic oxygen furnace" in q:
        return r"bof|basic oxygen furnace"
    if "blast furnace" in q:
        return r"blast furnace"
    if "caster" in q:
        return r"caster"
    if "roughing mill" in q:
        return r"roughing mill"
    if "sinter" in q:
        return r"sinter"
    if "descal" in q:
        return r"descal"
    return r".*"


def parse_dynamic_rule(text: str) -> dict:
    q = str(text).lower()
    priority_override = ""
    normalized = normalize_rule_text(text)
    if (
        any(term in q for term in ["must be p1", "should be p1", "trigger p1", "priority p1", "make it p1", "must receive immediate p1", "immediate p1 priority"])
        or "trigger p1" in normalized
        or "within four hours" in q
        or "within 4 hours" in q
    ):
        priority_override = "P1"
    elif any(term in q for term in ["must be p2", "should be p2", "trigger p2", "priority p2"]) or "trigger p2" in normalized:
        priority_override = "P2"
    elif any(term in q for term in ["must be p3", "should be p3", "trigger p3", "priority p3"]) or "trigger p3" in normalized:
        priority_override = "P3"

    risk_override = ""
    if priority_override == "P1":
        risk_override = "CRITICAL"
    elif priority_override == "P2":
        risk_override = "HIGH"
    elif priority_override == "P3":
        risk_override = "MEDIUM"

    rule = {
        "rule_id": _make_rule_id(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rule_type": "safety_override" if priority_override else "policy_rule",
        "equipment_pattern": _infer_rule_equipment_pattern(text),
        "area_pattern": _infer_rule_area_pattern(text),
        "condition_text": str(text),
        "priority_override": priority_override,
        "risk_override": risk_override,
        "source_text": str(text),
        "normalized_rule_text": normalize_rule_text(text),
        "rule_key": "",
        "active": True,
    }
    rule["rule_key"] = dynamic_rule_key(rule)
    return rule


def remember_dynamic_rule(text: str) -> dict:
    rule = parse_dynamic_rule(text)
    rules = load_dynamic_rules()
    if not rules.empty:
        rules["rule_key"] = rules.apply(lambda row: dynamic_rule_key(row.to_dict()), axis=1)
        match = rules[(rules["rule_key"] == rule["rule_key"]) & (rules["active"].astype(bool))]
        if not match.empty:
            existing = match.iloc[-1].to_dict()
            existing["duplicate"] = True
            existing["message"] = "Rule already remembered."
            return existing
    rule["duplicate"] = False
    rule["message"] = "Rule remembered."
    rules = pd.concat([rules, pd.DataFrame([rule])], ignore_index=True, sort=False)
    save_dynamic_rules(rules)
    deactivate_duplicate_rules()
    return rule


def _regex_match(pattern: Any, text: str) -> bool:
    pattern_text = _clean_text(pattern, ".*")
    if not pattern_text or pattern_text == ".*":
        return True
    try:
        return bool(re.search(pattern_text, text, flags=re.IGNORECASE))
    except re.error:
        return pattern_text.lower() in text.lower()


def rule_matches_asset(rule: dict | pd.Series, asset_state: dict) -> bool:
    rule_dict = dict(rule)
    combined = " ".join(
        _clean_text(asset_state.get(field), "")
        for field in ["asset_id", "asset_type", "area", "criticality"]
    )
    equipment_ok = _regex_match(rule_dict.get("equipment_pattern"), combined)
    area_ok = _regex_match(rule_dict.get("area_pattern"), combined)
    return equipment_ok and area_ok


def _rule_threshold_check(text: str, label_regex: str, value: float) -> list[bool]:
    checks: list[bool] = []
    patterns = [
        rf"\b(?:{label_regex})\b\s*(?:is|are|was|were)?\s*(above|over|greater than|>=|>|at least|below|under|less than|<=|<)\s*(-?\d+(?:\.\d+)?)",
        rf"\b(above|over|greater than|>=|>|at least|below|under|less than|<=|<)\s*(-?\d+(?:\.\d+)?)\s*(?:{label_regex})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            operator = match.group(1).lower()
            threshold = float(match.group(2))
            if pd.isna(value):
                checks.append(False)
                continue
            if operator in {"above", "over", "greater than", ">"}:
                checks.append(value > threshold)
            elif operator in {">=", "at least"}:
                checks.append(value >= threshold)
            elif operator in {"below", "under", "less than", "<"}:
                checks.append(value < threshold)
            elif operator == "<=":
                checks.append(value <= threshold)
    return checks


def rule_condition_met(rule: dict | pd.Series, asset_state: dict) -> bool:
    text = _clean_text(dict(rule).get("condition_text"), "").lower()
    temp = _num(asset_state.get("temperature"), np.nan)
    vib = _num(asset_state.get("vibration"), np.nan)
    current = _num(asset_state.get("current"), np.nan)
    pressure = _num(asset_state.get("pressure"), np.nan)
    alarms = _num(asset_state.get("alarm_count"), np.nan)
    risk_band = _clean_text(asset_state.get("risk_band") or asset_state.get("risk_level"), "").upper()
    rul = _num(asset_state.get("estimated_rul_days") or asset_state.get("rul_days"), np.nan)

    checks: list[bool] = []
    checks.extend(_rule_threshold_check(text, r"vib(?:ration)?", vib))
    checks.extend(_rule_threshold_check(text, r"temp(?:erature)?", temp))
    checks.extend(_rule_threshold_check(text, r"current", current))
    checks.extend(_rule_threshold_check(text, r"pressure", pressure))
    checks.extend(_rule_threshold_check(text, r"alarm(?:\s*count)?|alarms?", alarms))
    if "critical risk" in text or "risk critical" in text or "risk band critical" in text:
        checks.append(risk_band == "CRITICAL")
    word_numbers = {
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
    }
    for match in re.finditer(r"\b(?:rul|remaining useful life)\b.{0,20}\b(below|under|less than|<=|<)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?", text):
        token = match.group(2)
        threshold = float(token) if re.match(r"\d", token) else word_numbers.get(token, np.nan)
        if pd.isna(rul) or pd.isna(threshold):
            checks.append(False)
        elif match.group(1) in {"<=", "less than", "below", "under"}:
            checks.append(rul < threshold if match.group(1) != "<=" else rul <= threshold)
    if not checks:
        return bool(_clean_text(dict(rule).get("priority_override"), ""))
    return all(checks)


def apply_dynamic_rules(asset_state: dict, scored: dict) -> dict:
    final = dict(scored)
    final["base_priority"] = scored.get("priority")
    final["base_risk_band"] = scored.get("risk_band")
    final["base_hybrid_health_score"] = scored.get("hybrid_health_score")
    applied: list[dict] = []
    rules = active_unique_dynamic_rules()
    if rules.empty:
        final["applied_rules"] = []
        final["applied_rule_count"] = 0
        return final

    for _, rule_row in rules.iterrows():
        rule = rule_row.to_dict()
        if not bool(rule.get("active", True)):
            continue
        if not rule_matches_asset(rule, asset_state):
            continue
        if not rule_condition_met(rule, asset_state):
            continue

        rule_summary = {
            "rule_id": rule.get("rule_id"),
            "rule_type": rule.get("rule_type"),
            "condition_text": rule.get("condition_text"),
            "priority_override": rule.get("priority_override"),
            "risk_override": rule.get("risk_override"),
        }
        applied.append(rule_summary)

        if rule.get("priority_override") == "P1":
            final["priority"] = "P1"
            final["risk_band"] = "CRITICAL"
            final["urgency"] = "Immediate action"
            final["hybrid_health_score"] = round(max(_num(final.get("hybrid_health_score"), 0.0), 85.0), 2)
            final["operational_rule_score"] = round(max(_num(final.get("operational_rule_score"), 0.0), 85.0), 2)
            final["hybrid_failure_risk"] = round(max(_num(final.get("hybrid_failure_risk"), 0.0), 0.85), 4)
            final["failure_risk"] = round(max(_num(final.get("failure_risk"), 0.0), 0.85), 4)
            final["ml_failure_pred"] = 1
            final["is_anomaly"] = 1
            final["estimated_rul_days"] = min(_num(final.get("estimated_rul_days"), 50.0), 7.5)
            final["anomaly_events_24h"] = max(int(_num(final.get("anomaly_events_24h"), 0)), 12)
        elif rule.get("priority_override") == "P2" and final.get("priority") not in {"P1", "P2"}:
            final["priority"] = "P2"
            final["risk_band"] = "HIGH"
            final["urgency"] = "Action within 24 hours"
            final["hybrid_health_score"] = round(max(_num(final.get("hybrid_health_score"), 0.0), 58.0), 2)
            final["operational_rule_score"] = round(max(_num(final.get("operational_rule_score"), 0.0), 58.0), 2)
            final["hybrid_failure_risk"] = round(max(_num(final.get("hybrid_failure_risk"), 0.0), 0.58), 4)

    final["applied_rules"] = applied
    final["applied_rule_count"] = len(applied)
    final["dynamic_rule_note"] = (
        f"{len(applied)} remembered safety/SOP rule(s) applied to this asset."
        if applied
        else ""
    )
    return final


def score_dynamic_asset(row: pd.Series | dict) -> dict:
    r = dict(row)
    asset_type = _clean_text(r.get("asset_type"), "Unspecified Equipment")
    typ = asset_type.lower()
    area = _clean_text(r.get("area"), "Unspecified Area")
    criticality = _clean_text(r.get("criticality"), "Medium").title()

    raw_temp = _num(r.get("temperature"), np.nan)
    raw_vib = _num(r.get("vibration"), np.nan)
    raw_current = _num(r.get("current"), np.nan)
    raw_pressure = _num(r.get("pressure"), np.nan)
    raw_rpm = _num(r.get("rpm"), 1480.0)
    raw_alarms = _num(r.get("alarm_count"), 0.0)
    operator_notes = _clean_text(r.get("operator_notes"), "")
    notes = operator_notes.lower()

    temp = 0.0 if np.isnan(raw_temp) else raw_temp
    vib = 0.0 if np.isnan(raw_vib) else raw_vib
    current = 0.0 if np.isnan(raw_current) else raw_current
    pressure = 8.0 if np.isnan(raw_pressure) else raw_pressure
    alarms = 0.0 if np.isnan(raw_alarms) else raw_alarms
    rpm = 1480.0 if np.isnan(raw_rpm) else raw_rpm

    missing = [
        field
        for field, value in [
            ("temperature", raw_temp),
            ("vibration", raw_vib),
            ("current", raw_current),
            ("pressure", raw_pressure),
        ]
        if np.isnan(value)
    ]

    score = 0.0
    if temp >= 70:
        score += 8
    if temp >= 80:
        score += 20
    if temp >= 90:
        score += 15
    if vib >= 5:
        score += 10
    if vib >= 6:
        score += 20
    if vib >= 8:
        score += 20
    if current >= 75:
        score += 8
    if current >= 85:
        score += 15
    if pressure <= 6:
        score += 18
    if alarms >= 2:
        score += 10
    if alarms >= 3:
        score += 10

    crit = criticality.lower()
    if crit == "critical":
        score += 15
    elif crit == "high":
        score += 10
    elif crit == "medium":
        score += 5

    if "blast furnace" in (typ + " " + str(area).lower()) and vib >= 6.5 and temp >= 80:
        score += 20
    if any(word in typ for word in ["blower", "fan", "compressor"]) and vib >= 6 and current >= 80:
        score += 12
    if "hydraulic" in typ and pressure <= 6:
        score += 10
    if "pump" in typ and pressure <= 6:
        score += 10
    if "caster" in (typ + " " + str(area).lower()) and vib >= 6:
        score += 8
    if "descaler" in typ and pressure <= 6:
        score += 10
    if any(term in notes for term in ["cavitation", "suction noise"]) and any(term in typ for term in ["pump", "descaler"]):
        score += 25
        score = max(score, 58)
    if any(term in notes for term in ["high temperature", "high oil temperature", "rising oil temperature"]):
        score += 18
        score = max(score, 45)
    if "high current" in notes:
        score += 12
        score = max(score, 40)
    if "abnormal vibration" in notes:
        score += 20
        score = max(score, 50)
    if "repeated trips" in notes:
        score += 18
        score = max(score, 55)
    if any(term in notes for term in ["metallic particles", "metal particles"]):
        score += 28
        score = max(score, 68)
    if any(term in notes for term in ["no spare bearing available", "no spare available"]):
        score += 8
    if any(term in notes for term in ["loud noise", "abnormal noise", "bearing noise", "rubbing", "chatter"]):
        score += 10
        score = max(score, 45)
    if any(term in notes for term in ["leak", "oil leak", "hydraulic leak"]):
        score += 15
    if any(term in notes for term in ["smoke", "sparking", "burning smell"]):
        score += 30
        score = max(score, 75)

    score = float(np.clip(score, 0, 100))
    if score >= 75:
        risk = "CRITICAL"
        priority = "P1"
        urgency = "Immediate action"
    elif score >= 55:
        risk = "HIGH"
        priority = "P2"
        urgency = "Action within 24 hours"
    elif score >= 35:
        risk = "MEDIUM"
        priority = "P3"
        urgency = "Plan in maintenance window"
    else:
        risk = "LOW"
        priority = "P4"
        urgency = "Monitor only"

    rul = max(1.0, round(50 * (1 - score / 100), 1))
    ml_risk_proxy = round(float(np.clip(0.20 + 0.0065 * score, 0.01, 0.92)), 4)

    scored = {
        **r,
        "asset_id": str(r.get("asset_id", "")).upper(),
        "asset_type": asset_type,
        "area": area,
        "criticality": criticality,
        "criticality_score": criticality_score(criticality),
        "temperature": raw_temp,
        "vibration": raw_vib,
        "current": raw_current,
        "pressure": raw_pressure,
        "rpm": rpm,
        "alarm_count": int(alarms),
        "operator_notes": operator_notes,
        "temperature_for_scoring": temp,
        "vibration_for_scoring": vib,
        "current_for_scoring": current,
        "pressure_for_scoring": pressure,
        "missing_readings": ", ".join(missing),
        "provisional_scoring_note": "Missing readings held as unknown; neutral defaults used only for provisional risk scoring." if missing else "",
        "qualitative_risk_note": "Operator/field symptom used as risk override evidence." if operator_notes else "",
        "anomaly_score": round(score / 100, 4),
        "is_anomaly": int(score >= 55),
        "failure_risk": round(score / 100, 4),
        "ml_failure_risk": ml_risk_proxy,
        "ml_failure_pred": int(score >= 55),
        "operational_rule_score": round(score, 2),
        "hybrid_health_score": round(score, 2),
        "hybrid_failure_risk": round(score / 100, 4),
        "risk_band": risk,
        "priority": priority,
        "urgency": urgency,
        "estimated_rul_days": rul,
        "anomaly_events_24h": int(max(alarms * 4, 1 if score >= 55 else 0)),
        "temperature_slope_24h": 0.0,
        "vibration_slope_24h": 0.0,
        "pressure_slope_24h": 0.0,
        "data_origin": "dynamic_user_memory",
        "is_dynamic": 1,
    }
    return apply_dynamic_rules({**r, **scored}, scored)


def score_dynamic_assets(df: pd.DataFrame | None = None, active_only: bool = True) -> pd.DataFrame:
    source = load_dynamic_assets() if df is None else df.copy()
    if source.empty:
        return pd.DataFrame()
    if "active" not in source.columns:
        source["active"] = True
    if active_only:
        source = source[source["active"].astype(bool)].copy()
    if source.empty:
        return pd.DataFrame()
    return pd.DataFrame([score_dynamic_asset(row) for _, row in source.iterrows()])


def dynamic_root_cause(asset_type: str) -> str:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        return "bearing wear, impeller imbalance, fouling, duct restriction, motor overload, or coupling misalignment"
    if "bearing" in typ:
        return "bearing wear, lubrication breakdown, misalignment, overload, contamination, or inadequate cooling"
    if "pump" in typ:
        return "low suction head, clogged strainer, seal wear, bearing wear, impeller erosion, or air ingress"
    if "compressor" in typ:
        return "bearing wear, surge condition, filter restriction, lubrication issue, or coupling misalignment"
    if "conveyor" in typ:
        return "belt mistracking, pulley bearing wear, material buildup, gearbox issue, or idler seizure"
    if "furnace" in typ:
        return "cooling restriction, airflow instability, refractory hot spot, bearing overload, or process imbalance"
    if "caster" in typ:
        return "oscillator drive wear, mold friction instability, lubrication issue, alignment drift, or servo fault"
    return "abnormal operating condition inferred from current readings, alarm state, criticality, and equipment class"


def dynamic_actions(asset_type: str) -> list[str]:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        return [
            "Verify blower/fan vibration spectrum, bearing temperature, motor current balance, damper position, and foundation bolts.",
            "Check impeller fouling, duct restriction, lubrication condition, coupling alignment, and standby equipment availability.",
        ]
    if "bearing" in typ:
        return [
            "Check bearing temperature, lubrication condition, vibration spectrum, alignment, loading, contamination, and cooling path.",
            "Prepare controlled inspection with spare bearing set, seals, lifting plan, and restart acceptance readings.",
        ]
    if "pump" in typ:
        return [
            "Inspect suction strainer, suction head, inlet valve position, seal leakage, bearing condition, and impeller erosion.",
        ]
    if "compressor" in typ:
        return [
            "Check suction filter, discharge pressure trend, bearing temperature, lubrication, surge alarms, and coupling alignment.",
        ]
    if "conveyor" in typ:
        return [
            "Inspect belt tracking, pulley bearings, idlers, take-up tension, gearbox oil, and emergency pull-cord status.",
        ]
    if "caster" in typ:
        return [
            "Inspect mold oscillator drive, servo response, lubrication condition, alignment, vibration spectrum, and breakout-prevention interlocks.",
        ]
    if "furnace" in typ:
        return [
            "Check cooling flow, hot-spot trend, airflow stability, refractory condition, interlock alarms, and safe isolation plan.",
        ]
    return ["Inspect the asset boundary, confirm sensor readings, isolate energy sources if required, and create a maintenance work order."]


def dynamic_spares(asset_id: str, asset_type: str) -> list[dict]:
    typ = str(asset_type).lower()
    if "blower" in typ or "fan" in typ:
        parts = [
            ("Blower/fan bearing set", 1, 10, "Critical", 95000),
            ("Coupling insert/alignment kit", 2, 3, "High", 18000),
        ]
    elif "pump" in typ:
        parts = [
            ("Pump seal kit", 1, 5, "High", 28000),
            ("Pump bearing set", 1, 7, "High", 42000),
        ]
    elif "caster" in typ:
        parts = [
            ("Mold oscillator bearing kit", 1, 14, "Critical", 135000),
            ("Servo feedback sensor", 1, 9, "High", 52000),
        ]
    elif "compressor" in typ:
        parts = [
            ("Compressor bearing kit", 1, 14, "Critical", 120000),
            ("Suction filter element", 3, 4, "High", 12000),
        ]
    elif "conveyor" in typ:
        parts = [
            ("Pulley bearing set", 2, 7, "High", 32000),
            ("Idler roller set", 8, 2, "Medium", 5000),
        ]
    else:
        parts = [
            ("Inspection consumables and isolation kit", 1, 1, "Medium", 5000),
            ("Equipment-specific critical spare to verify", 0, 7, "Critical", 0),
        ]
    return [
        {
            "asset_id": asset_id,
            "spare_part": spare,
            "stock_qty": qty,
            "lead_time_days": lead,
            "spare_criticality": crit,
            "unit_cost_inr": cost,
            "source": "dynamic_spares_inference",
        }
        for spare, qty, lead, crit, cost in parts
    ]
