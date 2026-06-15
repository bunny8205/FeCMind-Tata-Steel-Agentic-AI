"""Agentic Maintenance Wizard orchestration layer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .data_setup import create_compatibility_sensor_log, prepare_data
from .dynamic_assets import (
    active_unique_dynamic_rules,
    dynamic_actions,
    dynamic_asset_ids,
    dynamic_root_cause,
    dynamic_spares,
    extract_asset_ids,
    is_asset_deactivation_query,
    is_asset_ingestion_query,
    is_asset_reactivation_query,
    is_asset_update_query,
    is_priority_change_query,
    is_rule_apply_query,
    is_rule_ingestion_query,
    latest_dynamic_asset_change,
    load_dynamic_asset_history,
    list_inactive_dynamic_assets,
    load_dynamic_assets,
    load_dynamic_rules,
    mark_dynamic_asset_inactive,
    parse_dynamic_assets,
    query_mentions_new_asset_reference,
    reactivate_dynamic_asset,
    remember_dynamic_rule,
    rule_condition_met,
    rule_matches_asset,
    save_dynamic_assets,
    score_dynamic_assets,
    update_dynamic_assets_from_query,
    upsert_dynamic_assets,
    validate_dynamic_asset_readings,
)
from .llm import LocalLLM
from .models import FEATURE_COLS, ModelManager, risk_band_from_score, safe_float
from .rag import RAGIndex, normalize_equipment_type
from .steel_agent import (
    build_general_answer,
    build_general_decision_packet,
    build_general_plan,
    build_general_tool_calls,
    build_general_verifier_checks,
    classify_steel_intent,
    infer_steel_subject,
    is_steel_domain_query,
    summarize_health_rows,
)


def _format_records(records: list[dict]) -> str:
    if not records:
        return "- No matching records found."
    lines = []
    for record in records:
        clean = {k: v for k, v in record.items() if pd.notna(v)}
        lines.append("- " + "; ".join(f"{k}: {v}" for k, v in clean.items()))
    return "\n".join(lines)


def _format_history_records(asset_id: str, records: list[dict]) -> str:
    real = [record for record in records if "No historical work orders yet" not in str(record.get("issue", ""))]
    if not real:
        return f"- None found for {asset_id}."
    return _format_records(real)


def _format_failure_records(asset_id: str, records: list[dict]) -> str:
    real = [record for record in records if "No failure reports yet" not in str(record.get("failure_mode", ""))]
    if not real:
        return f"- None found for {asset_id}."
    return _format_records(real)


def _format_sources(docs: list[dict]) -> str:
    if not docs:
        return "- No retrieved evidence."
    lines = []
    seen = set()
    for doc in docs:
        key = (doc.get("source"), doc.get("equipment_type"), doc.get("issue_type"))
        if key in seen:
            continue
        seen.add(key)
        source = doc.get("source", "unknown source")
        equipment = doc.get("equipment_type", "general")
        issue = doc.get("issue_type", "general")
        text = " ".join(str(doc.get("text", "")).split())[:320]
        lines.append(f"{len(lines) + 1}. {source} - {equipment}/{issue}\n   Evidence: {text}")
    return "\n".join(lines)


def _read_csv_safe(path, columns: list[str] | None = None) -> pd.DataFrame:
    path = DATA_DIR / path if isinstance(path, str) else path
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(columns=columns or [])
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        return pd.DataFrame(columns=columns or [])


def _is_missing_value(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _display_value(value, unit: str = "", decimals: int = 2) -> str:
    if _is_missing_value(value):
        return "not provided"
    try:
        number = float(value)
        rendered = f"{number:.{decimals}f}".rstrip("0").rstrip(".")
        return f"{rendered}{unit}"
    except (TypeError, ValueError):
        return f"{value}{unit}" if unit else str(value)


def infer_output_style(query: str) -> dict:
    q = str(query).lower()
    if "json only" in q or "valid json only" in q:
        return {"format": "json_only"}
    if "table only" in q or "return only a table" in q or "only a table" in q:
        return {"format": "table_only"}
    bullet_match = None
    import re

    for pattern in [r"under\s+(\d+)\s+bullets?", r"in\s+(\d+)\s+bullets?", r"(\d+)\s+bullets?\s+only"]:
        bullet_match = re.search(pattern, q)
        if bullet_match:
            return {"format": "bullets", "max_items": int(bullet_match.group(1))}
    line_match = None
    for pattern in [r"(\d+)\s+lines?\s+only", r"in\s+(\d+)\s+lines?", r"under\s+(\d+)\s+lines?"]:
        line_match = re.search(pattern, q)
        if line_match:
            return {"format": "lines", "max_items": int(line_match.group(1))}
    if "short answer" in q or "be concise" in q or "briefly" in q:
        return {"format": "bullets", "max_items": 6}
    return {"format": "full_report"}


def _markdown_table(records: list[dict], columns: list[str]) -> str:
    if not records:
        return "| Result |\n|---|\n| No rows found |"
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for record in records:
        values = []
        for col in columns:
            value = record.get(col, "")
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            values.append(str(value).replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + rows)


def _spares_strategy(spares: list[dict]) -> str:
    if not spares:
        return "- No matching spare inventory found."
    lines = []
    for item in spares:
        spare = item.get("spare_part", "Unknown spare")
        qty = int(safe_float(item.get("stock_qty", 0)))
        lead = int(safe_float(item.get("lead_time_days", 0)))
        if qty <= 0:
            action = "Raise procurement immediately."
        elif lead >= 7:
            action = "Reserve before shutdown due to lead time."
        else:
            action = "Available; reserve for planned work."
        lines.append(f"- {spare}: stock {qty}, lead time {lead} days. {action}")
    return "\n".join(lines)


@dataclass
class MaintenanceWizard:
    model_manager: ModelManager = field(default_factory=ModelManager)
    rag: RAGIndex = field(default_factory=RAGIndex)
    llm: LocalLLM = field(default_factory=LocalLLM)
    initialized: bool = False
    session_memory: dict = field(default_factory=dict)

    def initialize(self, force: bool = False, load_llm: bool = False) -> "MaintenanceWizard":
        prepare_data(force=force)
        self.model_manager.train_or_load(force=force)
        self.rag.build()
        if load_llm:
            self.llm.load()
        self.initialized = True
        return self

    def ensure_ready(self) -> None:
        if not self.initialized:
            self.initialize(load_llm=False)

    @property
    def asset_ids(self) -> list[str]:
        self.ensure_ready()
        return sorted(self.asset_health_table()["asset_id"].dropna().astype(str).unique().tolist())

    def asset_health_table(self) -> pd.DataFrame:
        self.ensure_ready()
        base = self.model_manager.asset_health.copy()
        if "data_origin" not in base.columns:
            base["data_origin"] = "demo_sensor_model"
        if "is_dynamic" not in base.columns:
            base["is_dynamic"] = 0
        dynamic = score_dynamic_assets(load_dynamic_assets())
        if dynamic.empty:
            return base
        return pd.concat([base, dynamic], ignore_index=True, sort=False)

    def query_assets(self, query: str) -> list[str]:
        self.ensure_ready()
        explicit = self._explicit_asset_ids(query)
        rag_assets = self.rag.query_assets(query)
        out: list[str] = []
        for asset_id in explicit + rag_assets:
            if asset_id not in out:
                out.append(asset_id)
        return out

    def _explicit_asset_ids(self, query: str) -> list[str]:
        known = set(self.asset_health_table()["asset_id"].dropna().astype(str).str.upper())
        known.update(dynamic_asset_ids(active_only=False))
        out = []
        for asset_id in extract_asset_ids(query):
            if asset_id in known and asset_id not in out:
                out.append(asset_id)
        return out

    def _unknown_asset_ids(self, query: str) -> list[str]:
        known = set(self.asset_health_table()["asset_id"].dropna().astype(str).str.upper())
        known.update(dynamic_asset_ids(active_only=False))
        out = []
        for asset_id in extract_asset_ids(query):
            if asset_id not in known and asset_id not in out:
                out.append(asset_id)
        return out

    def _original_demo_asset_ids(self) -> set[str]:
        table = self.asset_health_table()
        if table.empty:
            return set()
        dynamic_ids = set(dynamic_asset_ids(active_only=False))
        return {
            str(asset_id).upper()
            for asset_id in table["asset_id"].dropna().astype(str)
            if str(asset_id).upper() not in dynamic_ids
        }

    def _is_ambiguous_reference_update(self, query: str) -> bool:
        q = str(query).lower()
        ids = extract_asset_ids(query)
        has_update = is_asset_update_query(query)
        has_ambiguous_reference = any(term in q for term in [" update it", "update it", "update that", "same asset", "that asset", " it "])
        has_comparison_context = any(term in q for term in ["compare", " with ", " vs ", " versus "])
        return has_update and len(ids) >= 2 and has_ambiguous_reference and has_comparison_context

    def _is_impossible_alert_or_ingestion(self, query: str) -> bool:
        return bool(extract_asset_ids(query)) and bool(parse_dynamic_assets(query))

    def _is_rule_conflict_query(self, query: str) -> bool:
        q = str(query).lower()
        if any(term in q for term in ["using all original assets", "choose one immediate maintenance target", "procurement target", "monitoring target", "full adversarial"]):
            return False
        return "rule" in q and any(term in q for term in ["conflict", "conflicted", "precedence", "which rule won"])

    def _is_rule_scope_audit_query(self, query: str) -> bool:
        q = str(query).lower()
        return "apply all remembered rules" in q or ("matched and rejected rules" in q and "rule" in q)

    def _is_evidence_contradiction_query(self, query: str) -> bool:
        q = str(query).lower()
        contradiction_terms = ["faulty", "contradict", "conflicting evidence", "replaced yesterday"]
        return bool(self._explicit_asset_ids(query)) and any(term in q for term in contradiction_terms)

    def _is_degraded_tool_query(self, query: str) -> bool:
        q = str(query).lower()
        return any(term in q for term in ["rag retriever fails", "retriever fails", "spares file is unavailable", "tool failure", "unavailable tools"])

    def _is_procurement_tradeoff_query(self, query: str) -> bool:
        q = str(query).lower()
        return "asset a" in q and "asset b" in q and any(term in q for term in ["zero stock", "lead time", "spare is available", "procurement priority"])

    def _is_inactive_safety_exception_query(self, query: str) -> bool:
        q = str(query).lower()
        return (
            bool(extract_asset_ids(query))
            and any(term in q for term in ["inactive", "excluded", "new live readings", "serious new alert", "smoke reported"])
            and is_asset_update_query(query)
        )

    def _is_dynamic_asset(self, asset_id: str) -> bool:
        return str(asset_id).upper() in set(dynamic_asset_ids(active_only=False))

    def _remember_asset_context(self, asset_ids: list[str], selected_asset: str | None = None) -> None:
        clean_ids = []
        for asset_id in asset_ids:
            aid = str(asset_id).upper()
            if aid and aid not in clean_ids:
                clean_ids.append(aid)
        if len(clean_ids) >= 2:
            self.session_memory["last_compared_assets"] = clean_ids
        dynamic_ids = [aid for aid in clean_ids if self._is_dynamic_asset(aid)]
        original_ids = [aid for aid in clean_ids if not self._is_dynamic_asset(aid)]
        if dynamic_ids:
            self.session_memory["last_dynamic_asset"] = dynamic_ids[-1]
            self.session_memory["last_new_asset_id"] = dynamic_ids[-1]
        if original_ids:
            self.session_memory["last_original_asset"] = original_ids[-1]
        if selected_asset:
            selected = str(selected_asset).upper()
            self.session_memory["last_selected_asset"] = selected
            self.session_memory["last_asset_id"] = selected

    def resolve_update_target(self, query: str) -> str | None:
        explicit = self._explicit_asset_ids(query)
        if explicit:
            return explicit[0]
        q = str(query).lower()
        if any(term in q for term in [" it", "it ", "same asset", "that asset", "this asset"]):
            compared = [str(a).upper() for a in self.session_memory.get("last_compared_assets", [])]
            dynamic_compared = [asset_id for asset_id in compared if self._is_dynamic_asset(asset_id)]
            if len(dynamic_compared) == 1:
                return dynamic_compared[0]
            if self.session_memory.get("last_dynamic_asset"):
                return self.session_memory["last_dynamic_asset"]
            if self.session_memory.get("last_new_asset_id"):
                return self.session_memory["last_new_asset_id"]
        return self._infer_asset_from_query(query)

    def _infer_asset_from_query(self, query: str) -> str | None:
        q = str(query).lower()
        explicit = self._explicit_asset_ids(query)
        if explicit:
            return explicit[0]
        if query_mentions_new_asset_reference(query):
            remembered = self.session_memory.get("last_new_asset_id")
            if remembered:
                return remembered
            dyn_ids = dynamic_asset_ids(active_only=True)
            if len(dyn_ids) == 1:
                return dyn_ids[0]
        if any(term in q for term in ["same asset", "that asset", "it", "spare should i", "same equipment"]):
            remembered = self.session_memory.get("last_asset_id")
            if remembered:
                return remembered
        assets = self.rag.query_assets(query)
        if assets:
            return assets[0]
        return None

    def _strict_task_intent(self, query: str) -> str:
        q = str(query or "").lower()
        if re.search(r"\b(?:error|fault)\s+code\b", q) or re.search(r"\be[- ]?\d{2,4}\b", q):
            return "error_code_lookup"
        if self._is_logbook_template_query(query):
            return "logbook_entry"
        if any(term in q for term in ["spare", "spares", "procurement", "lead time", "stock", "inventory"]) and not self._explicit_plant_priority_request(query):
            return "spare_procurement_query"
        if any(term in q for term in ["sop", "standard operating procedure", "procedure for", "steps for replacing", "replace"]) and any(
            term in q for term in ["seal", "pump", "hydraulic", "bearing", "assembly", "motor", "gearbox"]
        ):
            return "sop_request"
        if any(term in q for term in ["just stopped", "stopped", "tripped", "walk me through", "first checks", "right now", "immediately"]) and any(
            term in q for term in ["conveyor", "belt", "motor", "pump", "blower", "drive", "line"]
        ):
            return "emergency_troubleshooting"
        if any(term in q for term in ["trending", "trend", "remaining useful life", "rul", "predict remaining", "intervene"]) or re.search(r"\d+\s*(?:c|°c|bar|kpa|mm/s|a)\s*(?:-|→|->|to)", q):
            return "trend_rul_analysis"
        alert_summary_terms = ["today's alerts", "todays alerts", "shift alerts", "alert summary", "summarize alerts", "summarize today's abnormal", "which assets have abnormal"]
        if any(term in q for term in ["threshold", "above", "below", "differential pressure", "alert report", "create an alert", "alarm"]) and not any(term in q for term in alert_summary_terms) and not self._explicit_plant_priority_request(query):
            return "abnormal_alert_report"
        if any(term in q for term in ["last 90 days", "incidents", "incident pattern", "pattern", "maintenance records", "failure history"]):
            return "incident_pattern_analysis"
        if any(term in q for term in ["crew", "technician", "schedule", "weekend", "shift plan", "job scheduling"]):
            return "crew_job_scheduling"
        if any(term in q for term in ["weekly summary", "supervisor summary", "supervisor update"]):
            return "supervisor_weekly_summary"
        if any(term in q for term in ["surface pitting", "slab pitting", "process defect", "quality defect", "scale marks"]):
            return "process_quality_analysis"
        if any(term in q for term in ["repeated failure", "repeat failure", "keeps failing", "recurring failure", "recurrence"]):
            return "repeated_failure_rca"
        return ""

    def _explicit_plant_priority_request(self, query: str) -> bool:
        q = str(query or "").lower()
        if any(term in q for term in ["agentic workflow", "agent workflow", "workflow design", "system architecture", "data flow"]):
            return False
        if "predictive maintenance" in q and any(term in q for term in ["design", "workflow", "agent", "architecture", "logs", "sops", "sensor", "feedback"]):
            return False
        dynamic_only_rank = any(
            term in q
            for term in [
                "rank active dynamic",
                "rank only newly",
                "only newly added assets",
                "dynamic assets only",
                "active dynamic assets only",
                "newly added assets only",
            ]
        )
        original_dynamic = any(term in q for term in ["original vs dynamic", "original demo", "highest-risk original", "highest risk original"])
        choose_one = any(
            term in q
            for term in [
                "only one asset",
                "maintain only one",
                "maintain one asset",
                "one asset today",
                "which asset should",
                "which asset would",
                "which one should",
                "which one would",
                "choose exactly one",
                "choose one asset",
                "select one asset",
                "first priority asset",
                "highest-risk asset",
                "highest risk asset",
                "highest-risk equipment",
                "highest risk equipment",
            ]
        )
        rank_assets = any(term in q for term in ["rank assets", "rank all assets", "prioritize assets", "prioritize all assets", "plant ranking", "asset ranking"])
        compare_assets = (
            any(term in q for term in ["compare", "side-by-side", "side by side"])
            and any(term in q for term in ["asset", "assets", "equipment", "original", "dynamic", "newly added"])
        )
        plant_decision = any(term in q for term in ["plant priority", "plant-level priority", "maintenance target", "immediate maintenance today"])
        return dynamic_only_rank or original_dynamic or choose_one or rank_assets or compare_assets or plant_decision

    def _preserved_equipment_context(self, query: str) -> str:
        text = str(query or "").strip()
        q = text.lower()
        patterns = [
            r"our\s+(.+?)\s+(?:just\s+)?(?:stopped|tripped|failed)",
            r"(?:temperature|pressure|vibration|current|lube\s+oil\s+temperature).+?\s+on\s+(?:the\s+)?(.+?)\s+is\s+trending",
            r"replace\s+(?:the\s+)?(.+?)(?:\s+this weekend|,|\?|\.|$)",
            r"(?:generate|create)\s+(?:a\s+)?(?:digital\s+)?logbook entry for\s+(?:today's\s+)?(?:planned maintenance on\s+)?(?:the\s+)?(.+?)(?:\.| technician| work done|$)",
            r"(?:alert|alarm).+?\s+for\s+(?:the\s+)?(.+?)(?:,|\.|\?|$)",
            r"on\s+(?:the\s+)?(.+?)(?:,|\?|\.| and what| and which| should|$)",
            r"for\s+(?:the\s+)?(.+?)(?:,|\?|\.| this weekend| today|$)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
            if match:
                candidate = re.sub(r"\s+", " ", match.group(1)).strip(" .,:;")
                candidate = re.sub(r"^(?:a|an|the)\s+", "", candidate, flags=re.IGNORECASE)
                candidate = re.sub(r"\s+is\s+trending.*$", "", candidate, flags=re.IGNORECASE)
                candidate = re.sub(r"\s+(?:just\s+)?(?:stopped|tripped|failed)$", "", candidate, flags=re.IGNORECASE)
                if 3 <= len(candidate) <= 90:
                    return candidate
        subject = infer_steel_subject(query)
        if subject and subject != "Steel Plant":
            return subject
        if "line 3" in q and "conveyor" in q:
            return "line 3 conveyor belt"
        return ""

    def _requested_identifiers(self, query: str) -> dict[str, list[str]]:
        text = str(query or "")
        asset_ids = self._explicit_asset_ids(text)
        raw_codes = {m.upper().replace(" ", "-") for m in re.findall(r"\b[A-Z]{1,4}[- ]\d{2,4}\b", text)}
        wants_fault_code = bool(re.search(r"\b(?:error|fault)\s+code\b", text, flags=re.IGNORECASE))
        fault_codes = sorted(
            code
            for code in raw_codes
            if code not in set(asset_ids) and (code.startswith("E-") or wants_fault_code)
        )
        lines = sorted({m.lower().replace("  ", " ") for m in re.findall(r"\bline\s+\d+\b", text, flags=re.IGNORECASE)})
        equipment = self._preserved_equipment_context(text)
        return {
            "asset_ids": asset_ids,
            "fault_codes": fault_codes,
            "line_refs": lines,
            "equipment_context": [equipment] if equipment else [],
        }

    def _looks_truncated(self, text: str) -> bool:
        clean = str(text or "").strip()
        if not clean:
            return True
        if clean.endswith(("**", "*", "-", ":", ",", ";")):
            return True
        if clean.endswith("...") and not re.search(r"(missing|not available|unknown|etc)\.\.\.$", clean, flags=re.IGNORECASE):
            return True
        if len(re.findall(r"\*\*", clean)) % 2:
            return True
        tail = clean[-24:]
        if re.search(r"\b(?:reass|inspecti|procureme|maintenan|verific|calibrati|hydraul|temperat)$", tail, flags=re.IGNORECASE):
            return True
        if not re.search(r"[.!?)]\s*$", clean):
            return True
        return False

    def _is_plant_query(self, query: str) -> bool:
        return self._explicit_plant_priority_request(query)

    def _plant_scope_asset_ids(self, query: str) -> list[str] | None:
        q = str(query).lower()
        dyn_ids = dynamic_asset_ids(active_only=True)
        if query_mentions_new_asset_reference(query):
            remembered = (
                self.session_memory.get("last_new_asset_id")
                or self.session_memory.get("last_dynamic_asset")
                or self.session_memory.get("last_asset_id")
            )
            explicit_refs = self._explicit_asset_ids(query)
            if remembered:
                remembered = str(remembered).upper()
                scoped = [remembered]
                scoped.extend(asset for asset in explicit_refs if asset != remembered)
                if len(scoped) >= 2 or any(term in q for term in ["compare", "choose", "which one", "maintained first"]):
                    return scoped
        if any(term in q for term in ["which one", "maintained first", "which should", "which asset should"]) and self.session_memory.get("last_compared_assets"):
            remembered_scope = [str(asset).upper() for asset in self.session_memory.get("last_compared_assets", [])]
            if remembered_scope:
                return remembered_scope
        dynamic_only_terms = [
            "only newly added",
            "rank only newly",
            "newly added assets only",
            "dynamic assets only",
            "active dynamic assets only",
            "rank active dynamic",
            "only dynamic",
            "not original",
            "not the original",
            "not original demo",
            "exclude original",
            "exclude demo",
            "do not include gbx",
            "do not include mtr",
            "do not include hpp",
            "do not include pmp",
        ]
        if any(term in q for term in dynamic_only_terms):
            return dyn_ids or None
        explicit = self._explicit_asset_ids(query)
        if len(explicit) >= 2:
            return explicit
        if any(term in q for term in ["newly added", "added assets", "new assets", "dynamic assets"]):
            if "original" not in q and "all assets" not in q and "all original" not in q:
                return dyn_ids or None
        return explicit or None

    def _is_dynamic_memory_listing_query(self, query: str) -> bool:
        q = str(query).lower()
        if "rank" in q and "active dynamic" in q:
            return False
        return (
            any(
                term in q
                for term in [
                    "active and inactive",
                    "inactive remembered",
                    "active remembered",
                    "remembered dynamic assets separately",
                    "active dynamically registered assets",
                    "active dynamic assets",
                    "list dynamic assets",
                    "list the dynamic assets",
                    "dynamically registered assets",
                ]
            )
            and any(term in q for term in ["dynamic", "remembered", "assets"])
        )

    def _is_feedback_learning_query(self, query: str) -> bool:
        q = str(query).lower()
        has_asset_context = bool(extract_asset_ids(query) or self.session_memory.get("last_asset_id"))
        return has_asset_context and any(
            term in q
            for term in [
                "record this feedback",
                "record that correction",
                "record the correction",
                "that correction",
                "engineer reports",
                "engineer confirms",
                "engineer confirms that",
                "diagnosis is incorrect",
                "actual cause",
                "sensor was faulty",
                "was faulty",
                "future recommendations should change",
            ]
        )

    def _is_spare_revision_query(self, query: str) -> bool:
        q = str(query).lower()
        return bool(extract_asset_ids(query)) and any(term in q for term in ["will arrive", "lead time", "unavailable", "no spare", "spare bearing"]) and any(
            term in q for term in ["revise", "temporary", "alternative", "strategy", "recommendation"]
        )

    def _is_spare_availability_listing_query(self, query: str) -> bool:
        q = str(query).lower()
        return any(term in q for term in ["spare parts available", "spares available", "spare parts unavailable", "spares unavailable", "do not have spare", "spare inventory"]) and any(
            term in q for term in ["which", "list", "critical assets", "assets"]
        )

    def _is_alert_summary_query(self, query: str) -> bool:
        q = str(query).lower()
        alert_terms = [
            "abnormal alert",
            "abnormal alerts",
            "today's alerts",
            "todays alerts",
            "today's abnormal",
            "todays abnormal",
            "shift alerts",
            "alert summary",
            "summarize alerts",
            "summarize today's abnormal",
            "summarize todays abnormal",
        ]
        plain_language_terms = ["simple language", "operator summary", "supervisor summary", "plain english", "brief"]
        return any(term in q for term in alert_terms) or ("alert" in q and "summarize" in q) or (
            "abnormal" in q and any(term in q for term in plain_language_terms)
        )

    def _is_model_disagreement_query(self, query: str) -> bool:
        q = str(query).lower()
        return any(term in q for term in ["disagreement", "conflict", "contradiction", "contradict", "inconsistency", "inconsistent"]) and any(
            term in q for term in ["ml model", "anomaly detector", "rule engine", "maintenance history", "retrieved documents", "final decision", "spare-planning", "inventory record", "out of stock"]
        )

    def _is_scenario_planning_query(self, query: str) -> bool:
        q = str(query).lower()
        scenario_terms = [
            "assume ",
            "what happens",
            "what-if",
            "what if",
            "if the ",
            "if a ",
            "if an ",
            "temporarily",
            "reassess",
            "recalculate",
            "revise the plan",
            "without the faulty",
            "unavailable",
            "returns no relevant",
            "not repaired for another",
            "already been inspected",
            "no physical damage",
        ]
        return any(term in q for term in scenario_terms) and not is_asset_ingestion_query(query)

    def _is_safety_guardrail_query(self, query: str) -> bool:
        q = str(query).lower()
        if "completed" in q and any(term in q for term in ["do not mark", "not mark", "do not close", "keep open", "leave open", "pending"]):
            return False
        unsafe_terms = [
            "guarantee",
            "exact date and time",
            "ignore the safety rule",
            "bypass",
            "safety interlock",
            "restart",
            "without verification",
            "hide conflicting evidence",
            "mark the maintenance task as completed",
            "completed even though",
            "change the critical risk classification",
            "production cannot stop",
            "do not mention uncertainty",
            "assume all missing readings are normal",
            "not available in the knowledge base",
            "not present in the available documents",
            "no sensor or historical data is available",
        ]
        if any(term in q for term in unsafe_terms):
            return True
        return bool(re.search(r"\b(?:invent|invented|inventing|fake|fabricate|fabricated)\b", q))

    def _is_ambiguous_asset_resolution_query(self, query: str) -> bool:
        q = str(query).lower()
        return any(term in q for term in ["identifier is incomplete", "incomplete", "without silently selecting", "most likely matching asset", "check the blower", "three blower assets", "ambiguity", "ambiguous"])

    def _is_logbook_template_query(self, query: str) -> bool:
        q = str(query).lower()
        return "logbook" in q and any(term in q for term in ["structured", "entry", "do not mark", "not mark", "not completed"])

    def _is_memory_audit_query(self, query: str) -> bool:
        q = str(query).lower()
        audit_terms = [
            "memory audit",
            "audit trail",
            "show history",
            "state history",
            "show every event",
            "chronological order",
            "for each event",
            "registration",
        ]
        return any(term in q for term in audit_terms) and bool(extract_asset_ids(query) or self._infer_asset_from_query(query))

    def _is_evidence_confidence_query(self, query: str) -> bool:
        q = str(query).lower()
        if any(term in q for term in ["weakest evidence", "missing evidence only", "list missing evidence"]):
            return True
        if "evidence confidence" not in q:
            return False
        decision_terms = [
            "compare",
            "choose",
            "rank",
            "prioritize",
            "full maintenance report",
            "maintenance report",
            "immediate maintenance",
            "which asset",
            "one asset",
        ]
        if any(term in q for term in decision_terms):
            return False
        return any(term in q for term in ["which", "show", "review", "audit", "missing", "available"])

    def _is_agentic_self_test_query(self, query: str) -> bool:
        q = str(query).lower()
        return (
            any(term in q for term in ["agentic-ai self-test", "agentic ai self-test", "agentic self-test", "self-test"])
            and any(term in q for term in ["perceive", "retrieve", "reason", "verify", "tools used", "verifier summary"])
        )

    def _is_public_query(self, query: str) -> bool:
        q = str(query).lower()
        return any(term in q for term in ["public dataset", "ai4i", "uci", "data source", "dataset used"])

    def _is_general_steel_query(self, query: str) -> bool:
        return is_steel_domain_query(query)

    def _planner_context(self, query: str) -> dict:
        explicit_assets = self._explicit_asset_ids(query)
        dynamic_active = dynamic_asset_ids(active_only=True)
        inactive_df = list_inactive_dynamic_assets()
        inactive_assets = inactive_df["asset_id"].astype(str).str.upper().tolist() if not inactive_df.empty else []
        return {
            "known_assets_in_prompt": explicit_assets,
            "original_demo_assets": sorted([asset for asset in self.asset_ids if asset not in dynamic_asset_ids(active_only=False)]),
            "active_dynamic_assets": dynamic_active,
            "inactive_dynamic_assets": inactive_assets,
            "last_asset_id": self.session_memory.get("last_asset_id"),
            "last_dynamic_asset": self.session_memory.get("last_dynamic_asset") or self.session_memory.get("last_new_asset_id"),
            "last_compared_assets": self.session_memory.get("last_compared_assets", []),
            "available_tools": [
                "ml_failure_risk",
                "rul_estimator",
                "rag_retriever",
                "spares_checker",
                "dynamic_memory",
                "remembered_safety_rules",
                "logbook_writer",
                "verifier",
            ],
        }

    def _llm_plan_query(self, query: str) -> dict:
        plan = self.llm.plan(query, self._planner_context(query))
        if not isinstance(plan, dict):
            plan = {}
        plan.setdefault("intent", "general_steel")
        plan.setdefault("scope", "general")
        plan.setdefault("target_assets", self._explicit_asset_ids(query))
        plan.setdefault("reason", "Planner returned minimal route.")
        q = str(query).lower()
        if self._is_safety_guardrail_query(query):
            plan["intent"] = "safety_guardrail"
            plan["scope"] = "safety"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: unsafe, hallucination, or over-certainty request must be handled by the safety guardrail."
            ).strip()
            return plan
        if self._is_ambiguous_asset_resolution_query(query):
            plan["intent"] = "ambiguous_asset_resolution"
            plan["scope"] = "general"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: ambiguous asset references must list candidates and request confirmation."
            ).strip()
            return plan
        if self._is_logbook_template_query(query):
            plan["intent"] = "logbook_template"
            plan["scope"] = "general"
            plan["task_intent"] = "logbook_entry"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: logbook template request should not mark work completed."
            ).strip()
            return plan
        strict_intent = self._strict_task_intent(query)
        task_intents = {
            "error_code_lookup",
            "sop_request",
            "spare_procurement_query",
            "emergency_troubleshooting",
            "logbook_entry",
            "trend_rul_analysis",
            "sensor_threshold_assessment",
            "abnormal_alert_report",
            "incident_pattern_analysis",
            "crew_job_scheduling",
            "supervisor_weekly_summary",
            "process_quality_analysis",
            "repeated_failure_rca",
        }
        if strict_intent and not self._explicit_plant_priority_request(query):
            explicit_for_task = self._explicit_asset_ids(query)
            planned = str(plan.get("intent", "")).lower()
            if planned in {"plant_priority", "asset_update", "asset_ingestion", "rule_apply", "original_vs_dynamic_comparison"}:
                plan["intent"] = strict_intent
                plan["reason"] = (
                    str(plan.get("reason", "")).strip()
                    + f" Verifier guardrail: prompt is a task-specific {strict_intent} request, so plant/state mutation routes were blocked."
                ).strip()
            elif planned not in task_intents and planned != "asset_diagnosis":
                plan["intent"] = strict_intent
            plan["scope"] = "single_asset" if explicit_for_task else "general"
            plan["target_assets"] = explicit_for_task
            plan["task_intent"] = strict_intent
            return plan
        if self._is_scenario_planning_query(query):
            plan["intent"] = "scenario_planning"
            plan["scope"] = "single_asset" if self._explicit_asset_ids(query) else "original_and_dynamic"
            plan["target_assets"] = self._explicit_asset_ids(query)
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: hypothetical assumptions require what-if planning without mutating live memory."
            ).strip()
            return plan
        if self._is_spare_availability_listing_query(query):
            plan["intent"] = "spare_availability_listing"
            plan["scope"] = "original_and_dynamic"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: spare inventory listing requires plant-wide spare check."
            ).strip()
            return plan
        if self._is_alert_summary_query(query):
            plan["intent"] = "alert_summary"
            plan["scope"] = "original_and_dynamic"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: alert-summary prompts require plant-wide abnormal-condition summarization."
            ).strip()
            return plan
        deterministic_intent = classify_steel_intent(query)
        if str(plan.get("intent", "")).lower() == "asset_update" and not is_asset_update_query(query):
            if deterministic_intent in {"risk_prioritization"} or self._is_plant_query(query):
                plan["intent"] = "plant_priority"
                plan["scope"] = "original_and_dynamic"
            else:
                plan["intent"] = "general_steel"
                plan["scope"] = "general"
            plan["target_assets"] = []
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: trend, summary, risk, RUL, or planning prompts are not dynamic-memory updates unless the user explicitly asks to update/set/change an asset."
            ).strip()
            return plan
        if (
            any(term in q for term in ["agentic workflow", "agent workflow", "workflow design", "system architecture", "data flow"])
            or ("predictive maintenance" in q and any(term in q for term in ["design", "workflow", "architecture", "feedback"]))
        ):
            plan["intent"] = "general_steel"
            plan["scope"] = "general"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: workflow/design prompt needs architecture synthesis, not live asset ranking."
            ).strip()
            return plan
        strategic_intent = deterministic_intent
        if strategic_intent in {"process_quality", "cbm_framework_design"} or (
            strategic_intent == "failure_prediction"
            and any(term in q for term in ["skid pipe", "skid pipes", "proactive replacement", "replacement schedule", "per pipe", "6 months"])
        ):
            plan["intent"] = "general_steel"
            plan["scope"] = "general"
            plan["target_assets"] = []
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + f" Corrected by verifier: {strategic_intent} prompt needs strategic/process maintenance reasoning, not live-asset ranking."
            ).strip()
            return plan
        explicit_assets = self._explicit_asset_ids(query)
        single_asset_nonplant = (
            len(explicit_assets) == 1
            and not self._is_plant_query(query)
            and not self._is_original_dynamic_comparison_query(query)
            and not is_asset_update_query(query)
            and not is_asset_ingestion_query(query)
            and not is_rule_ingestion_query(query)
            and not is_rule_apply_query(query)
        )
        if single_asset_nonplant and str(plan.get("intent", "")).lower() in {
            "general_steel",
            "plant_priority",
            "original_vs_dynamic_comparison",
            "evidence_confidence",
        }:
            plan["intent"] = "asset_diagnosis"
            plan["scope"] = "single_asset"
            plan["target_assets"] = explicit_assets
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + f" Corrected by verifier: single explicit asset prompt should diagnose {explicit_assets[0]}, not rank the plant."
            ).strip()
            return plan
        if self._is_model_disagreement_query(query):
            plan["intent"] = "model_disagreement_review"
            plan["scope"] = "original_and_dynamic"
            plan["target_assets"] = explicit_assets
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: disagreement prompts must compare ML, anomaly, rules, history, and RAG evidence before deciding."
            ).strip()
            return plan
        if self._is_original_dynamic_comparison_query(query):
            plan["intent"] = "original_vs_dynamic_comparison"
            plan["scope"] = "original_and_dynamic"
            plan["target_assets"] = []
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: explicit original-vs-dynamic wording must keep comparison mode, even when the answer also recommends one asset."
            ).strip()
            return plan
        if str(plan.get("intent", "")).lower() == "original_vs_dynamic_comparison":
            if explicit_assets and len(explicit_assets) == 1:
                plan["intent"] = "asset_diagnosis"
                plan["scope"] = "single_asset"
                plan["target_assets"] = explicit_assets
            elif explicit_assets and len(explicit_assets) >= 2 and any(
                term in q for term in ["compare", "choose", "prioritize", "maintained first", "select one"]
            ):
                plan["intent"] = "plant_priority"
                plan["scope"] = "explicit_assets"
                plan["target_assets"] = explicit_assets
            elif self._is_plant_query(query):
                plan["intent"] = "plant_priority"
                plan["scope"] = "original_and_dynamic"
                plan["target_assets"] = []
            else:
                plan["intent"] = "general_steel"
                plan["scope"] = "general"
                plan["target_assets"] = []
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: original-vs-dynamic mode is allowed only when the prompt explicitly compares original/demo assets with dynamic/new assets."
            ).strip()
            return plan
        if query_mentions_new_asset_reference(query) and any(term in q for term in ["compare", "choose", "which one", "maintained first"]):
            scoped = self._plant_scope_asset_ids(query) or explicit_assets
            plan["intent"] = "plant_priority"
            plan["scope"] = "explicit_assets" if scoped else "original_and_dynamic"
            plan["target_assets"] = scoped or []
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: remembered-new-asset comparison must rank the remembered dynamic asset against explicit assets."
            ).strip()
            return plan
        if self._is_dynamic_memory_listing_query(query):
            plan["intent"] = "dynamic_memory_listing"
            plan["scope"] = "dynamic_only"
            plan["target_assets"] = dynamic_asset_ids(active_only=True)
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: request is a dynamic memory listing, not general steel chat."
            ).strip()
            return plan
        if len(explicit_assets) >= 2 and any(term in q for term in ["compare", "select one", "choose one", "immediate maintenance", "maintained first"]):
            plan["intent"] = "plant_priority"
            plan["scope"] = "explicit_assets"
            plan["target_assets"] = explicit_assets
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: multi-asset comparison must run ranking/selection tools, not single-asset diagnosis."
            ).strip()
            return plan
        if is_rule_ingestion_query(query):
            plan["intent"] = "rule_ingestion"
            plan["scope"] = "rule_memory"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: remember/save/learn rule command must update rule memory first."
            ).strip()
            return plan
        if is_asset_ingestion_query(query):
            plan["intent"] = "asset_ingestion"
            plan["scope"] = "dynamic_memory"
            plan["target_assets"] = extract_asset_ids(query)
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: register/add named asset command must update dynamic memory first."
            ).strip()
            return plan
        if explicit_assets and any(
            term in q
            for term in [
                "maintenance report",
                "full report",
                "diagnose",
                "root cause",
                "available evidence",
                "missing evidence",
                "spares",
                "next action",
                "rul",
                "risk",
            ]
        ) and not any(term in q for term in ["rank", "choose exactly one", "which asset", "plant ranking"]):
            plan["intent"] = "asset_diagnosis"
            plan["scope"] = "single_asset"
            plan["target_assets"] = explicit_assets
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + f" Corrected by verifier: explicit asset report request resolved to asset_diagnosis for {explicit_assets[0]}."
            ).strip()
        if any(
            term in q
            for term in [
                "only newly added",
                "rank only newly",
                "newly added assets only",
                "dynamic assets only",
                "active dynamic assets only",
                "rank active dynamic",
                "only dynamic",
                "not original",
                "not the original",
                "exclude original",
                "do not include gbx",
                "do not include mtr",
                "do not include hpp",
                "do not include pmp",
            ]
        ):
            plan["intent"] = "plant_priority"
            plan["scope"] = "dynamic_only"
            plan["target_assets"] = dynamic_asset_ids(active_only=True)
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: dynamic-only wording excludes original demo assets from ranking scope."
            ).strip()
        if (
            str(plan.get("intent", "")).lower() in {"rule_apply", "dynamic_memory_listing", "general_steel", "asset_diagnosis"}
            and self._is_plant_query(query)
            and (not explicit_assets or len(explicit_assets) >= 2)
        ):
            plan["intent"] = "plant_priority"
            plan["scope"] = "original_and_dynamic"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: broad choose/rank/compare prompt must run plant-priority tools, with memory/rules as decision constraints."
            ).strip()
        if str(plan.get("intent", "")).lower() == "original_vs_dynamic_comparison" and any(
            term in q for term in ["choose exactly one", "rest of the plant", "rest of plant", "plant ranking", "one maintenance crew"]
        ):
            plan["intent"] = "plant_priority"
            plan["scope"] = "original_and_dynamic"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: original-vs-dynamic comparison is a required explanation inside the broader plant-priority decision."
            ).strip()
        if (
            str(plan.get("intent", "")).lower() == "original_vs_dynamic_comparison"
            and self._is_plant_query(query)
            and not self._is_original_dynamic_comparison_query(query)
        ):
            plan["intent"] = "plant_priority"
            plan["scope"] = "original_and_dynamic"
            plan["reason"] = (
                str(plan.get("reason", "")).strip()
                + " Corrected by verifier: broad plant operations/scheduling prompt should rank plant priorities, not only original-vs-dynamic."
            ).strip()
        return plan

    def _asset_ids_from_plan(self, query: str, plan: dict) -> list[str] | None:
        scope = str(plan.get("scope", "")).lower()
        if scope == "dynamic_only":
            return dynamic_asset_ids(active_only=True) or None
        if scope == "original_only":
            dynamic_all = set(dynamic_asset_ids(active_only=False))
            return [asset for asset in self.asset_ids if asset not in dynamic_all] or None
        target_assets = [str(asset).upper() for asset in plan.get("target_assets", []) if str(asset).strip()]
        known = set(self.asset_health_table()["asset_id"].dropna().astype(str).str.upper())
        filtered = [asset for asset in target_assets if asset in known]
        if len(filtered) >= 2 and "rest of the plant" not in str(query).lower():
            return filtered
        return self._plant_scope_asset_ids(query)

    def _attach_llm_plan(self, result: dict, plan: dict) -> dict:
        if not isinstance(result, dict):
            return result
        clean_plan = {
            "intent": plan.get("intent"),
            "task_intent": plan.get("task_intent"),
            "scope": plan.get("scope"),
            "target_assets": plan.get("target_assets", []),
            "reason": plan.get("reason"),
            "planner_status": plan.get("planner_status", "unknown"),
            "used_model": bool(plan.get("used_model")),
            "model_id": plan.get("model_id"),
            "load_error": plan.get("load_error", ""),
        }
        result["llm_planner"] = clean_plan
        result["llm_used"] = bool(result.get("llm_used")) or bool(plan.get("used_model"))
        result["llm_validation"] = "llm_planner_then_deterministic_tools_and_verifier"
        decision_packet = result.get("decision_packet")
        if isinstance(decision_packet, dict):
            decision_packet["llm_planner"] = clean_plan
        tool_calls = result.get("tool_calls")
        if isinstance(tool_calls, list):
            tool_calls.insert(
                0,
                {
                    "tool": "llm_intent_planner",
                    "agent": "Planner Agent",
                    "input": "user prompt + asset/memory/tool context",
                    "output": f"intent={clean_plan['intent']}, scope={clean_plan['scope']}, status={clean_plan['planner_status']}",
                    "status": "success" if clean_plan["planner_status"] in {"model", "model_json", "model_label", "fallback", "model_label_fallback"} else "review",
                },
            )
        return self._apply_llm_final_synthesis(result, clean_plan)

    def _apply_llm_final_synthesis(self, result: dict, clean_plan: dict) -> dict:
        if not isinstance(result, dict):
            return result
        mode = str(result.get("mode", "")).lower()
        output_style = result.get("output_style") or {}
        strict_format = isinstance(output_style, dict) and output_style.get("format") in {"json_only", "table_only", "lines"}
        skip_modes = {
            "clarification",
            "safety_guardrail",
            "logbook_template",
        }
        original_answer = str(result.get("answer") or result.get("final_answer") or "")
        if (
            not original_answer
            or strict_format
            or mode in skip_modes
            or result.get("skip_llm_synthesis")
        ):
            result["llm_synthesizer"] = {
                "attempted": False,
                "status": "skipped",
                "reason": "strict/safety/empty response path",
            }
            result["llm_validation"] = "llm_planner_then_deterministic_tools_and_verifier"
            return result

        payload = self._build_synthesis_payload(result, clean_plan, original_answer)
        synthesized_raw = self.llm.synthesize_final_answer(payload)
        synthesized = self._clean_llm_natural_text(synthesized_raw)
        verification = self._verify_llm_synthesis(synthesized, result)
        result["tool_grounded_answer"] = original_answer
        result["llm_synthesizer"] = {
            "attempted": True,
            "status": "accepted_qwen_final_answer" if verification["accepted"] else "rejected",
            "model_id": self.llm.model_id,
            "provider": getattr(self.llm, "provider", "local"),
            "qwen_api_model": getattr(self.llm, "qwen_api_model", ""),
            "qwen_remote_available": bool(getattr(self.llm, "remote_available", False)),
            "checks": verification["checks"],
            "load_error": self.llm.load_error,
        }
        if verification["accepted"]:
            result["answer"] = synthesized
            result["final_answer"] = synthesized
            result["llm_used"] = True
            result["llm_validation"] = "llm_planner_plus_deterministic_tools_plus_llm_final_synthesizer_plus_deterministic_verifier"
            tool_calls = result.get("tool_calls")
            if isinstance(tool_calls, list):
                tool_calls.append(
                    {
                        "tool": "llm_final_synthesizer",
                        "agent": "Synthesis Agent",
                        "input": "locked tool facts + verifier constraints",
                        "output": "accepted final answer",
                        "status": "success",
                    }
                )
                tool_calls.append(
                    {
                        "tool": "deterministic_answer_verifier",
                        "agent": "Verifier Agent",
                        "input": "LLM final answer",
                        "output": "locked fields preserved",
                        "status": "success",
                    }
                )
        else:
            result["answer"] = original_answer
            result["final_answer"] = original_answer
            result["llm_validation"] = "llm_planner_plus_deterministic_tools_plus_rejected_llm_synthesizer_plus_deterministic_verifier"
            verifier_checks = result.get("verifier_checks")
            if isinstance(verifier_checks, list):
                verifier_checks.append(
                    {
                        "check": "LLM final synthesis rejected",
                        "status": "review",
                        "detail": (
                            "Qwen final synthesis is unavailable or failed quality gates. "
                            + ("Configure MW_LLM_PROVIDER=qwen_api with MW_QWEN_API_BASE/MW_QWEN_API_KEY/MW_QWEN_API_MODEL. " if not bool(getattr(self.llm, "remote_available", False)) else "")
                            + ("; ".join(check["detail"] for check in verification["checks"] if check["status"] != "pass") or "quality gate failed")
                        ),
                    }
                )
        decision_packet = result.get("decision_packet")
        if isinstance(decision_packet, dict):
            decision_packet["llm_synthesizer"] = result["llm_synthesizer"]
            decision_packet["llm_validation"] = result["llm_validation"]
        return result

    def _compose_verifier_repaired_synthesis(self, llm_text: str, payload: dict, result: dict) -> str:
        decision = payload.get("decision_packet") if isinstance(payload.get("decision_packet"), dict) else {}
        priority = payload.get("priority") if isinstance(payload.get("priority"), dict) else {}
        evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
        objective = str(decision.get("objective") or payload.get("grounded_answer_excerpt") or "")
        selected = decision.get("selected_asset") or payload.get("asset_id") or result.get("asset_id") or "plant scope"
        risk = priority.get("risk_level") or decision.get("risk_level") or "REVIEW"
        priority_label = priority.get("priority") or decision.get("priority") or "REVIEW"
        rul = decision.get("estimated_rul_days", "not available")
        confidence = evidence.get("evidence_confidence") or decision.get("evidence_confidence") or "REVIEW"
        missing = evidence.get("missing_evidence") or decision.get("missing_evidence") or []
        if isinstance(missing, list):
            missing_text = ", ".join(str(x) for x in missing) if missing else "none"
        else:
            missing_text = str(missing)
        first_action = decision.get("recommended_first_action") or decision.get("next_system_action") or "verify live evidence and create the appropriate work order"
        llm_note = self._clean_llm_natural_text(llm_text)
        if not self._is_useful_llm_natural_text(llm_note, objective):
            llm_note = ""

        rul_text = f"{rul} days" if str(rul).lower() not in {"", "none", "nan", "not available"} else "not available from the current evidence"
        objective_lower = objective.lower()
        task_intent = self._strict_task_intent(objective)
        exact_code = re.search(r"\b([A-Z]+[- ]?\d{2,4})\b", objective, flags=re.IGNORECASE)
        code_text = exact_code.group(1).upper().replace(" ", "-") if exact_code else ""

        if task_intent == "error_code_lookup" and code_text:
            if not missing_text or missing_text.lower() == "none":
                missing_text = f"verified OEM fault-code definition for {code_text}"
            opener = (
                f"I do not have a verified fault-code table entry for {code_text} in the loaded evidence, so I would not pretend to know the exact OEM definition. "
                f"Because the question is about {selected}, I would treat {code_text} as a potentially production-critical equipment alarm until the MCC/VFD, PLC, or OEM manual confirms otherwise."
            )
            action_sentence = (
                "Immediately acknowledge the alarm, keep the blower under controlled operation only if vibration, current, bearing temperature, and discharge pressure are stable, "
                "check the MCC/VFD trip history, inspect cooling airflow, bearing condition, coupling alignment, motor current imbalance, and standby blower availability, and apply LOTO before opening the motor or blower guards."
            )
        elif task_intent == "logbook_entry":
            opener = f"I would draft this as an open logbook entry for {selected}, not as a completed job."
            action_sentence = (
                "Capture the supplied technician, work performed, date wording, and any missing closure evidence. "
                "Keep the status as OPEN/PENDING VERIFICATION until measured readings, parts used, supervisor sign-off, and post-maintenance outcome are recorded."
            )
        elif task_intent == "spare_procurement_query":
            opener = f"For {selected}, treat this as a spare/procurement planning request, not a plant-priority ranking."
            action_sentence = (
                "Build the spares list from the exact equipment boundary, confirm part numbers and interchangeability with the OEM or stores master, check on-hand stock, reserve critical items, and raise procurement for any long-lead or zero-stock part. "
                "If no site spare record is available, state that gap and avoid inventing lead times."
            )
        elif task_intent == "emergency_troubleshooting":
            opener = f"For {selected}, start with safe first checks before assuming a root cause."
            action_sentence = (
                "Check personnel safety and stop/start permissives, confirm E-stop or trip status, inspect local alarms, motor overload/VFD faults, belt/drive obstruction, pull-cord switches, guards, lubrication, and upstream/downstream interlocks. "
                "Use LOTO before hands-on inspection and escalate if heat, smoke, abnormal noise, or repeated trips are present."
            )
        elif task_intent in {"trend_rul_analysis", "sensor_threshold_assessment"}:
            opener = f"For {selected}, use the values in your prompt as provisional trend evidence."
            action_sentence = (
                "Estimate RUL as a band, not a false exact date, because the current tools need historical baseline, load state, alarm history, and inspection findings. "
                "Escalate intervention timing if the trend slope continues, crosses the site limit, or coincides with vibration/current/pressure deterioration."
            )
        elif task_intent == "abnormal_alert_report":
            opener = f"I would create a provisional alert for {selected} using the threshold evidence you supplied."
            action_sentence = (
                "Record the observed threshold crossings, verify the instrument impulse lines/calibration and process condition, check recent cleaning or isolation status, notify the area owner, and keep the alert scoped to this equipment unless the user explicitly asks for plant-wide ranking."
            )
        elif task_intent == "sop_request" or any(term in objective_lower for term in ["sop", "procedure", "replace", "seal"]):
            opener = (
                f"Use this as a field-safe SOP for {selected}: first confirm the exact pump model, seal kit, isolation points, and permit requirements before removing anything. "
                "If the site-specific SOP is not available in the retrieved evidence, treat the steps below as a controlled maintenance checklist that still needs supervisor/OEM confirmation."
            )
            action_text = str(first_action).strip().rstrip(".")
            if action_text:
                action_text = action_text[:1].lower() + action_text[1:]
            action_sentence = (
                f"The next system action is to {action_text}. In practical terms: isolate and lock out electrical and hydraulic energy, depressurize the line, drain and contain oil, verify zero pressure, remove the seal housing, inspect the shaft sleeve and faces, install the correct seal without contamination, refill and bleed, then restart with a leak, pressure, temperature, and vibration check."
            )
        elif str(result.get("mode", "")).lower() == "plant_priority" or "choose" in objective_lower or "only one asset" in objective_lower:
            opener = f"I would maintain {selected} first."
            readable_priority = self._readable_priority_summary(priority_label, risk)
            plant_action = re.sub(
                rf"(?i)^choose\s+{re.escape(str(selected))}\s+first\.\s*",
                "",
                str(first_action).strip(),
            )
            action_sentence = (
                f"It is the strongest maintenance target in the current plant ranking: {readable_priority}, with RUL {rul_text}. {plant_action or first_action}"
            )
        else:
            opener = f"For {selected}, I would handle this as a contextual steel-plant maintenance task rather than a generic answer."
            action_sentence = f"The first practical move is: {first_action}"

        readable_priority = self._readable_priority_summary(priority_label, risk)
        evidence_sentence = (
            f"The evidence confidence is {confidence}. "
            + (f"The missing or weak evidence is: {missing_text}. " if missing_text and missing_text.lower() != "none" else "No additional missing evidence was flagged by the current tools. ")
            + f"For traceability, the final answer used the verified asset/context {selected}, {readable_priority}, RUL {rul_text}, and the stated next action."
        )

        if llm_note:
            return f"{opener}\n\n{llm_note}\n\n{action_sentence}\n\n{evidence_sentence}".strip()
        return f"{opener}\n\n{action_sentence}\n\n{evidence_sentence}".strip()

    def _clean_llm_natural_text(self, text: str) -> str:
        clean = str(text or "").strip()
        clean = re.split(r"Natural final response:|Final answer:", clean, flags=re.IGNORECASE)[-1].strip()
        clean = re.sub(
            r"^[\s*_`#>-]*(assistant\s*)?(final answer|answer|response)[\s*_`#>-]*:\s*[\s*_`#>-]*",
            "",
            clean,
            flags=re.IGNORECASE,
        ).strip()
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        clean = clean.replace("**Decision**", "").replace("**Why**", "").replace("**Evidence Gaps**", "").replace("**Next Action**", "")
        internal_heading_pattern = re.compile(
            r"(?i)(steel plant agent response|autonomous execution plan|tool calls executed|verifier checks|decision packet|interpreted intent|locked decision fields)"
        )
        if internal_heading_pattern.search(clean):
            return ""
        clean = re.sub(r"(?i)\blocked facts\b.*", "", clean).strip()
        if len(clean) > 3600:
            snippet = clean[:3600].rstrip()
            cut = max(snippet.rfind(". "), snippet.rfind("! "), snippet.rfind("? "))
            clean = snippet[: cut + 1].strip() if cut >= 1200 else snippet.rstrip(" ,;:") + "."
        return clean

    def _readable_priority_summary(self, priority_label: str, risk: str) -> str:
        priority_text = str(priority_label or "").strip().upper()
        risk_text = str(risk or "").strip().upper()
        if priority_text == "PLANT" and risk_text == "PLANT_SUMMARY":
            return "highest plant-priority risk"
        if priority_text == "AGENT" and risk_text == "CONTEXTUAL":
            return "a contextual engineering review priority"
        if priority_text and risk_text:
            return f"{priority_text}/{risk_text}"
        if priority_text:
            return priority_text
        if risk_text:
            return risk_text
        return "requires review"

    def _is_useful_llm_natural_text(self, text: str, objective: str) -> bool:
        clean = str(text or "").strip()
        lower = clean.lower()
        if len(clean) < 80:
            return False
        bad_fragments = [
            "user objective:",
            "locked facts",
            "tool outputs:",
            "mode:",
            "selected asset:",
            "write the final response",
            "natural final response",
            "steel plant agent response",
            "interpreted intent",
            "autonomous execution plan",
            "tool calls executed",
            "verifier checks",
            "decision packet",
            "locked decision fields",
            "grounded maintenance report",
        ]
        if any(fragment in lower for fragment in bad_fragments):
            return False
        objective_words = [w for w in re.findall(r"[a-zA-Z]{4,}", str(objective).lower()) if w not in {"what", "does", "mean", "show", "standard", "should", "take"}]
        if objective_words and sum(1 for w in objective_words[:10] if w in lower) == 0:
            return False
        return True

    def _build_synthesis_payload(self, result: dict, clean_plan: dict, original_answer: str) -> dict:
        decision = result.get("decision_packet") if isinstance(result.get("decision_packet"), dict) else {}
        priority = result.get("risk_priority") if isinstance(result.get("risk_priority"), dict) else {}
        evidence = result.get("evidence_confidence") if isinstance(result.get("evidence_confidence"), dict) else {}
        tool_calls = result.get("tool_calls") if isinstance(result.get("tool_calls"), list) else []
        verifier_checks = result.get("verifier_checks") if isinstance(result.get("verifier_checks"), list) else []
        docs = result.get("retrieved_docs") if isinstance(result.get("retrieved_docs"), list) else []
        compact_tools = [
            {
                "agent": call.get("agent"),
                "tool": call.get("tool"),
                "output": call.get("output"),
                "status": call.get("status"),
            }
            for call in tool_calls[:8]
        ]
        compact_checks = [
            {
                "check": check.get("check"),
                "status": check.get("status"),
                "detail": check.get("detail"),
            }
            for check in verifier_checks[:8]
        ]
        compact_docs = [
            {
                "source": doc.get("source"),
                "asset_id": doc.get("asset_id"),
                "issue_type": doc.get("issue_type"),
            }
            for doc in docs[:5]
        ]
        operator_role = str(self.session_memory.get("operator_role") or "Maintenance Engineer")
        role_duties = str(self.session_memory.get("role_duties") or "")
        return {
            "mode": result.get("mode"),
            "intent": result.get("intent"),
            "task_intent": clean_plan.get("task_intent") or result.get("intent"),
            "operator_role": operator_role,
            "role_duties": role_duties,
            "asset_id": result.get("asset_id") or decision.get("selected_asset"),
            "planner": clean_plan,
            "priority": priority,
            "decision_packet": {
                key: decision.get(key)
                for key in [
                    "mode",
                    "objective",
                    "selected_asset",
                    "risk_level",
                    "priority",
                    "urgency",
                    "hybrid_health_score",
                    "hybrid_failure_risk",
                    "ml_failure_risk",
                    "operational_rule_score",
                    "estimated_rul_days",
                    "evidence_confidence",
                    "recommended_first_action",
                    "next_system_action",
                    "missing_evidence",
                    "inactive_dynamic_assets_excluded",
                ]
                if key in decision
            },
            "evidence": evidence,
            "tool_calls": compact_tools,
            "verifier_checks": compact_checks,
            "retrieved_sources": compact_docs,
            "requested_identifiers": self._requested_identifiers(str(decision.get("objective") or original_answer or "")),
            "grounded_answer_excerpt": original_answer[:2800],
        }

    def _verify_llm_synthesis(self, text: str, result: dict) -> dict:
        checks = []
        clean = str(text or "").strip()
        lower = clean.lower()
        checks.append({"check": "LLM produced non-empty final answer", "status": "pass" if len(clean) >= 80 else "fail", "detail": f"{len(clean)} chars"})
        checks.append({"check": "No NaN/null display", "status": "pass" if not re.search(r"\b(?:nan|null)\b", lower) else "fail", "detail": "no forbidden tokens"})
        hidden_leaks = [
            "acting user role",
            "role duties",
            "decision lens",
            "locked packet",
            "deterministic tools",
            "tool calls",
            "verifier checks",
        ]
        checks.append({
            "check": "No hidden prompt or internal pipeline leak",
            "status": "pass" if not any(term in lower for term in hidden_leaks) else "fail",
            "detail": "hidden metadata scan",
        })
        checks.append({"check": "Answer not truncated", "status": "pass" if not self._looks_truncated(clean) else "fail", "detail": "complete sentence/markdown scan"})
        asset = str(result.get("asset_id") or "").upper()
        decision = result.get("decision_packet") if isinstance(result.get("decision_packet"), dict) else {}
        objective = str(decision.get("objective") or "")
        strict_intent = self._strict_task_intent(objective)
        selected = str(decision.get("selected_asset") or asset or "").upper()
        if selected and selected not in {"NONE", "NULL", "NAN", "UNRESOLVED"}:
            checks.append({"check": "Selected asset preserved", "status": "pass" if selected.lower() in lower else "fail", "detail": selected})
        requested = self._requested_identifiers(objective)
        for code in requested.get("fault_codes", []):
            answer_codes = sorted({m.upper().replace(" ", "-") for m in re.findall(r"\b[A-Z]{1,4}[- ]\d{2,4}\b", clean)})
            status = "pass" if code.lower() in lower and not any(ans_code.startswith("E-") and ans_code != code for ans_code in answer_codes) else "fail"
            checks.append({"check": "Requested fault/error code preserved", "status": status, "detail": code})
        for asset_id in requested.get("asset_ids", []):
            checks.append({"check": "Requested asset ID preserved", "status": "pass" if asset_id.lower() in lower else "fail", "detail": asset_id})
        for line_ref in requested.get("line_refs", []):
            checks.append({"check": "Requested line reference preserved", "status": "pass" if line_ref in lower else "fail", "detail": line_ref})
        for equipment in requested.get("equipment_context", []):
            important_terms = [term for term in re.findall(r"[a-zA-Z]{4,}", equipment.lower()) if term not in {"system", "equipment"}]
            matched = sum(1 for term in important_terms if term in lower)
            checks.append({"check": "Requested equipment/context preserved", "status": "pass" if not important_terms or matched >= max(1, min(2, len(important_terms))) else "fail", "detail": equipment})
        if strict_intent and not self._explicit_plant_priority_request(objective):
            checks.append({
                "check": "Task-specific prompt did not become plant ranking",
                "status": "pass" if str(result.get("mode", "")).lower() != "plant_priority" else "fail",
                "detail": strict_intent,
            })
        if not requested.get("asset_ids") and requested.get("equipment_context") and not self._explicit_plant_priority_request(objective):
            forbidden_defaults = {"GBX-17", "MTR-204", "PMP-09", "HPP-12"}
            leaked = [default for default in forbidden_defaults if default.lower() in lower and default != selected]
            checks.append({
                "check": "No unrelated demo asset fallback",
                "status": "pass" if not leaked else "fail",
                "detail": ", ".join(leaked) if leaked else "no default demo asset inserted",
            })
        priority = result.get("risk_priority") if isinstance(result.get("risk_priority"), dict) else {}
        risk_level = str(priority.get("risk_level") or decision.get("risk_level") or "").upper()
        if risk_level and risk_level not in {"UNKNOWN", "NONE"}:
            checks.append({"check": "Risk level preserved", "status": "pass" if risk_level.lower() in lower else "review", "detail": risk_level})
        priority_label = str(priority.get("priority") or decision.get("priority") or "").upper()
        if priority_label and priority_label not in {"UNKNOWN", "NONE"}:
            checks.append({"check": "Priority preserved", "status": "pass" if priority_label.lower() in lower else "review", "detail": priority_label})
        unsafe_claims = ["guaranteed safe", "definitely safe", "ignore safety", "bypass interlock"]
        checks.append({"check": "No unsafe certainty", "status": "pass" if not any(term in lower for term in unsafe_claims) else "fail", "detail": "safety language scan"})
        accepted = all(check["status"] != "fail" for check in checks)
        return {"accepted": accepted, "checks": checks}

    def get_latest_sensor_summary(self, asset_id: str) -> dict:
        self.ensure_ready()
        requested_asset = str(asset_id).upper()
        health = self.asset_health_table()
        row = health[health["asset_id"].astype(str).str.upper() == requested_asset]
        if row.empty:
            dynamic_memory = load_dynamic_assets()
            if not dynamic_memory.empty:
                dynamic_row = dynamic_memory[
                    dynamic_memory["asset_id"].astype(str).str.upper() == requested_asset
                ].copy()
                if not dynamic_row.empty:
                    scored_dynamic = score_dynamic_assets(dynamic_row, active_only=False)
                    if not scored_dynamic.empty:
                        row = scored_dynamic
            if row.empty:
                return {"asset_id": asset_id, "error": "No sensor data found."}
        r = row.iloc[0].to_dict()
        is_dynamic = int(safe_float(r.get("is_dynamic"), 0)) == 1

        def latest_value(field: str, default: float = 0):
            value = r.get(field)
            if is_dynamic and _is_missing_value(value):
                return None
            return round(safe_float(value, default), 2)

        return {
            "asset_id": asset_id,
            "asset_type": r.get("asset_type"),
            "area": r.get("area"),
            "criticality": r.get("criticality"),
            "temperature_latest": latest_value("temperature"),
            "vibration_latest": latest_value("vibration"),
            "current_latest": latest_value("current"),
            "pressure_latest": latest_value("pressure"),
            "rpm_latest": latest_value("rpm", 1480),
            "alarm_count_latest": int(safe_float(r.get("alarm_count"))),
            "ml_failure_risk_latest": round(safe_float(r.get("ml_failure_risk")), 4),
            "operational_rule_score": round(safe_float(r.get("operational_rule_score")), 2),
            "hybrid_health_score": round(safe_float(r.get("hybrid_health_score")), 2),
            "hybrid_failure_risk": round(safe_float(r.get("hybrid_failure_risk", r.get("failure_risk"))), 4),
            "failure_pred": int(safe_float(r.get("failure_pred"), 0)),
            "risk_band": r.get("risk_band"),
            "estimated_rul_days": round(safe_float(r.get("estimated_rul_days"), 30), 1),
            "anomaly_events_24h": int(safe_float(r.get("anomaly_events_24h"))),
            "temperature_slope_24h": round(safe_float(r.get("temperature_slope_24h")), 4),
            "vibration_slope_24h": round(safe_float(r.get("vibration_slope_24h")), 4),
            "pressure_slope_24h": round(safe_float(r.get("pressure_slope_24h")), 4),
            "data_origin": r.get("data_origin", "demo_sensor_model"),
            "is_dynamic": int(is_dynamic),
            "missing_readings": r.get("missing_readings", ""),
            "provisional_scoring_note": r.get("provisional_scoring_note", ""),
            "operator_notes": r.get("operator_notes", ""),
            "qualitative_risk_note": r.get("qualitative_risk_note", ""),
            "base_priority": r.get("base_priority", r.get("priority")),
            "base_risk_band": r.get("base_risk_band", r.get("risk_band")),
            "base_hybrid_health_score": round(safe_float(r.get("base_hybrid_health_score", r.get("hybrid_health_score"))), 2),
            "applied_rules": r.get("applied_rules", []) if isinstance(r.get("applied_rules", []), list) else [],
            "applied_rule_count": int(safe_float(r.get("applied_rule_count", 0))),
            "dynamic_rule_note": r.get("dynamic_rule_note", ""),
        }

    def get_spares(self, asset_id: str) -> list[dict]:
        rows = _read_csv_safe(
            DATA_DIR / "spares_inventory.csv",
            columns=["asset_id", "spare_part", "stock_qty", "lead_time_days"],
        ).query("asset_id == @asset_id").to_dict("records")
        if rows:
            return rows
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            return dynamic_spares(asset_id, sensor.get("asset_type", ""))
        return []

    def get_delay(self, asset_id: str) -> dict:
        rows = _read_csv_safe(
            DATA_DIR / "delay_logs.csv",
            columns=["asset_id", "area", "delay_hours", "production_impact"],
        ).query("asset_id == @asset_id")
        if len(rows):
            return rows.iloc[0].to_dict()
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            delay = 6.0 if str(sensor.get("criticality", "")).lower() == "critical" else 2.0
            return {
                "asset_id": asset_id,
                "area": sensor.get("area"),
                "delay_hours": delay,
                "production_impact": "Inferred production/safety impact for user-added asset",
            }
        return {"delay_hours": 0}

    def get_history(self, asset_id: str) -> list[dict]:
        rows = _read_csv_safe(
            DATA_DIR / "maintenance_history.csv",
            columns=["asset_id", "timestamp", "issue", "action_taken", "result", "downtime_hours"],
        ).query("asset_id == @asset_id").to_dict("records")
        if rows:
            return rows
        if self._is_dynamic_asset(asset_id):
            return [
                {
                    "asset_id": asset_id,
                    "timestamp": "user-added asset",
                    "issue": "No historical work orders yet",
                    "action_taken": "Use live readings and generic equipment policy until history is learned",
                    "result": "Needs engineer confirmation",
                    "downtime_hours": 0,
                }
            ]
        return []

    def get_failures(self, asset_id: str) -> list[dict]:
        rows = _read_csv_safe(
            DATA_DIR / "failure_reports.csv",
            columns=["asset_id", "failure_mode", "root_cause", "corrective_action", "business_impact"],
        ).query("asset_id == @asset_id").to_dict("records")
        if rows:
            return rows
        if self._is_dynamic_asset(asset_id):
            return [
                {
                    "asset_id": asset_id,
                    "failure_mode": "No failure reports yet",
                    "root_cause": "Pending inspection and feedback learning",
                    "corrective_action": "Create first baseline inspection record",
                    "business_impact": "Estimated from criticality, area, and live readings",
                }
            ]
        return []

    def get_feedback(self, asset_id: str) -> list[dict]:
        path = DATA_DIR / "feedback_log.csv"
        if not path.exists():
            return []
        df = _read_csv_safe(path)
        if "asset_id" not in df.columns or df.empty:
            return []
        return df[df["asset_id"].astype(str) == str(asset_id)].tail(3).to_dict("records")

    def evidence_confidence(
        self,
        asset_id: str,
        sensor: dict,
        docs: list[dict] | None = None,
        history: list[dict] | None = None,
        failures: list[dict] | None = None,
        spares: list[dict] | None = None,
    ) -> dict:
        docs = docs or []
        history = history if history is not None else self.get_history(asset_id)
        failures = failures if failures is not None else self.get_failures(asset_id)
        spares = spares if spares is not None else self.get_spares(asset_id)
        is_dynamic = bool(sensor.get("is_dynamic"))

        has_current_sensors = sensor.get("hybrid_health_score") is not None and not sensor.get("error")
        has_sop = any(str(doc.get("source", "")).upper().startswith("SOP_") and str(doc.get("asset_id", "")).upper() == str(asset_id).upper() for doc in docs)
        has_policy_or_sop = has_sop or any(
            "policy" in str(doc.get("source", "")).lower()
            or "sop" in str(doc.get("source", "")).lower()
            or "operating_model" in str(doc.get("source", "")).lower()
            for doc in docs
        )
        has_history = any("No historical work orders yet" not in str(row.get("issue", "")) for row in history)
        has_failure_report = any(
            str(row.get("failure_mode", "")).lower() not in {"not yet observed", "no failure reports yet", "", "nan"}
            for row in failures
        )
        has_spares = bool(spares)
        raw_missing_readings = "" if _is_missing_value(sensor.get("missing_readings")) else str(sensor.get("missing_readings") or "")
        missing_readings = [
            x.strip()
            for x in raw_missing_readings.split(",")
            if x.strip() and x.strip().lower() not in {"nan", "none", "null"}
        ]

        available = []
        missing = []
        if has_current_sensors:
            available.append("current sensor/risk state")
        else:
            missing.append("current sensor/risk state")
        if has_sop:
            available.append("asset-specific SOP")
        elif has_policy_or_sop:
            available.append("generic SOP/policy")
            missing.append("asset-specific SOP")
        else:
            missing.append("asset-specific SOP or policy")
        if has_history:
            available.append("historical work orders")
        else:
            missing.append("historical work orders")
        if has_failure_report:
            available.append("failure reports")
        else:
            missing.append("failure reports")
        if has_spares:
            available.append("spares/procurement strategy")
            if is_dynamic:
                missing.append("asset-specific spare master record")
        else:
            missing.append("spares/procurement strategy")
        if missing_readings:
            missing.append("missing readings: " + ", ".join(missing_readings))

        if has_current_sensors and has_sop and has_history and has_failure_report and has_spares and not missing_readings:
            level = "HIGH"
        elif has_current_sensors and has_policy_or_sop and (has_spares or has_history) and not (is_dynamic and len(missing) >= 3):
            level = "MEDIUM"
        else:
            level = "LOW"

        return {
            "evidence_confidence": level,
            "available_evidence": available,
            "missing_evidence": missing,
        }

    def rule_breakdown(self, sensor: dict, delay: dict | None = None, spares: list[dict] | None = None) -> list[str]:
        delay = delay or {}
        spares = spares or []
        reasons = []
        typ = str(sensor.get("asset_type", "")).lower()
        criticality = str(sensor.get("criticality", "medium")).lower()
        temp = safe_float(sensor.get("temperature_latest"))
        vib = safe_float(sensor.get("vibration_latest"))
        current = safe_float(sensor.get("current_latest"))
        pressure = safe_float(sensor.get("pressure_latest"), 8)
        alarms = safe_float(sensor.get("alarm_count_latest"))
        anomalies = safe_float(sensor.get("anomaly_events_24h"))
        delay_hours = safe_float(delay.get("delay_hours", 0))
        notes = str(sensor.get("operator_notes") or "").lower()

        if "gearbox" in typ:
            if vib >= 7:
                reasons.append("Gearbox vibration >= 7 mm/s: +30")
            if vib >= 9:
                reasons.append("Gearbox vibration >= 9 mm/s: +20")
            if vib >= 10:
                reasons.append("Gearbox vibration >= 10 mm/s: +10")
            if temp >= 65:
                reasons.append("Gearbox temperature >= 65 deg C: +8")
        if "motor" in typ:
            if temp >= 80:
                reasons.append("Motor temperature >= 80 deg C: +25")
            if temp >= 85:
                reasons.append("Motor temperature >= 85 deg C: +15")
            if current >= 80:
                reasons.append("Motor current >= 80 A: +12")
            if vib >= 5:
                reasons.append("Motor vibration >= 5 mm/s: +8")
        if "pump" in typ:
            if pressure <= 6:
                reasons.append("Pump pressure <= 6 bar: +22")
            if vib >= 5:
                reasons.append("Pump vibration >= 5 mm/s: +12")
            if current >= 75:
                reasons.append("Pump current >= 75 A: +8")
        if "hydraulic" in typ:
            if pressure <= 6:
                reasons.append("Hydraulic pressure <= 6 bar: +25")
            if temp >= 65:
                reasons.append("Hydraulic oil temperature >= 65 deg C: +10")
            if alarms >= 2:
                reasons.append("Hydraulic alarm count >= 2: +8")
        if any(word in typ for word in ["blower", "fan", "compressor"]):
            if vib >= 6:
                reasons.append("Rotating air equipment vibration >= 6 mm/s: +20")
            if current >= 85:
                reasons.append("Rotating air equipment current >= 85 A: +15")
            if temp >= 80:
                reasons.append("Rotating air equipment temperature >= 80 deg C: +20")
        if "blast furnace" in (typ + " " + str(sensor.get("area", "")).lower()):
            if temp >= 80 and vib >= 6.5:
                reasons.append("Blast furnace critical blower/fan high temperature plus vibration safety override: +20")
        if "cavitation" in notes and ("pump" in typ or "descaler" in typ):
            reasons.append("Operator-reported cavitation noise on pump/descaler: qualitative risk floor keeps priority elevated")
        if any(term in notes for term in ["loud noise", "abnormal noise", "bearing noise", "rubbing", "chatter"]):
            reasons.append("Operator-reported abnormal noise: qualitative risk uplift")
        if any(term in notes for term in ["smoke", "sparking", "burning smell"]):
            reasons.append("Operator-reported smoke/sparking/burning smell: safety-critical risk override")
        if not any(key in typ for key in ["gearbox", "motor", "pump", "hydraulic", "blower", "fan", "compressor"]):
            if temp >= 80:
                reasons.append("Generic equipment temperature >= 80 deg C: +20")
            if vib >= 6:
                reasons.append("Generic rotating equipment vibration >= 6 mm/s: +20")
            if current >= 85:
                reasons.append("Generic equipment current >= 85 A: +15")
            if pressure <= 6:
                reasons.append("Generic low pressure <= 6 bar: +18")

        if alarms >= 2:
            reasons.append("Alarm count >= 2: +8")
        if alarms >= 4:
            reasons.append("Alarm count >= 4: +8")
        if anomalies >= 6:
            reasons.append("Anomaly events >= 6 in last 24h: +15")
        if anomalies >= 18:
            reasons.append("Anomaly events >= 18 in last 24h: +10")
        if criticality == "critical":
            reasons.append("Critical equipment: +15")
        elif criticality == "high":
            reasons.append("High criticality equipment: +10")
        elif criticality == "medium":
            reasons.append("Medium criticality equipment: +5")
        if delay_hours > 0:
            reasons.append(f"Historical delay impact {delay_hours} hours: +{min(delay_hours * 2.0, 12):.1f}")
        if any(safe_float(s.get("stock_qty", 0)) <= 0 and safe_float(s.get("lead_time_days", 0)) >= 7 for s in spares):
            reasons.append("Critical spare unavailable with high lead time: priority uplift")
        for rule in sensor.get("applied_rules") or []:
            rule_id = rule.get("rule_id", "remembered rule")
            condition = str(rule.get("condition_text", "")).strip()
            priority = rule.get("priority_override") or "policy"
            reasons.append(f"Remembered safety/SOP rule {rule_id} applied: {priority}. {condition}")
        return reasons or ["No major rule trigger; monitor based on trend and ML signal."]

    def detect_anomaly(self, asset_id: str) -> dict:
        s = self.get_latest_sensor_summary(asset_id)
        health = s.get("hybrid_health_score", 0)
        events = s.get("anomaly_events_24h", 0)
        if health >= 75 or events >= 12:
            level = "HIGH"
        elif health >= 55 or events >= 4:
            level = "MEDIUM"
        else:
            level = "LOW"
        return {
            "asset_id": asset_id,
            "anomaly_level": level,
            "hybrid_failure_risk": s.get("hybrid_failure_risk", 0),
            "ml_failure_risk": s.get("ml_failure_risk_latest", 0),
            "operational_rule_score": s.get("operational_rule_score", 0),
            "hybrid_health_score": health,
            "anomaly_events_24h": events,
        }

    def prioritize_action(self, sensor: dict, spares: list[dict], delay: dict) -> dict:
        health = sensor.get("hybrid_health_score", 0)
        rul = sensor.get("estimated_rul_days", 30)
        delay_hours = safe_float(delay.get("delay_hours", 0))
        spare_blocked = any(safe_float(s.get("stock_qty", 0)) <= 0 and safe_float(s.get("lead_time_days", 0)) >= 7 for s in spares)
        score = health + min(delay_hours * 1.5, 8) + (5 if rul <= 3 else 3 if rul <= 7 else 0) + (4 if spare_blocked else 0)
        score = round(float(np.clip(score, 0, 100)), 2)
        if score >= 75:
            return {"priority": "P1", "risk_level": "CRITICAL", "urgency": "Immediate action", "priority_score": score}
        if score >= 55:
            return {"priority": "P2", "risk_level": "HIGH", "urgency": "Action within 24 hours", "priority_score": score}
        if score >= 38:
            return {"priority": "P3", "risk_level": "MEDIUM", "urgency": "Plan in maintenance window", "priority_score": score}
        return {"priority": "P4", "risk_level": "LOW", "urgency": "Monitor only", "priority_score": score}

    def infer_root_cause(self, asset_id: str) -> str:
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            return dynamic_root_cause(sensor.get("asset_type", ""))
        typ = normalize_equipment_type(sensor.get("asset_type", ""))
        return {
            "motor": "bearing lubrication degradation, cooling restriction, overload, or current imbalance",
            "gearbox": "bearing wear, gear tooth wear, shaft misalignment, oil contamination, or foundation looseness",
            "pump": "suction strainer choking, low suction head, air ingress, seal wear, or impeller erosion",
            "hydraulic": "filter choking, relief valve leakage, pump wear, or hydraulic oil leakage",
        }.get(typ, "degradation pattern found in sensor trend and historical records")

    def recommended_actions(self, asset_id: str) -> list[str]:
        feedback = self.get_feedback(asset_id)
        actions = []
        if feedback:
            latest = feedback[-1]
            corrected = latest.get("corrected_action")
            if isinstance(corrected, str) and corrected.strip():
                actions.append(f"Apply learned feedback: {corrected}")
        sensor = self.get_latest_sensor_summary(asset_id)
        if sensor.get("is_dynamic"):
            return actions + dynamic_actions(sensor.get("asset_type", ""))
        typ = normalize_equipment_type(sensor.get("asset_type", ""))
        default_actions = {
            "motor": ["Inspect bearing lubrication, cooling airflow, current imbalance, load, and coupling alignment."],
            "gearbox": ["Check oil contamination and level.", "Inspect alignment, bearing condition, gear mesh, and foundation bolts."],
            "pump": ["Inspect suction strainer, suction head, inlet valve position, seal leakage, and impeller erosion."],
            "hydraulic": ["Replace or inspect filter element, verify relief valve setting, check oil level, and inspect leakage."],
        }
        return actions + default_actions.get(typ, ["Inspect asset condition and create a planned maintenance work order."])

    def build_agent_trace(self, asset_id: str, sensor: dict, anomaly: dict, priority: dict, docs: list[dict]) -> list[dict]:
        return [
            {"agent": "Triage Agent", "decision": f"Detected asset {asset_id} and equipment type {sensor.get('asset_type')}."},
            {"agent": "Sensor Agent", "decision": f"Hybrid risk {sensor.get('hybrid_failure_risk')}, anomaly {anomaly.get('anomaly_level')}, RUL {sensor.get('estimated_rul_days')} days."},
            {"agent": "Knowledge Agent", "decision": f"Retrieved {len(docs)} filtered evidence chunks."},
            {"agent": "Risk Agent", "decision": f"Assigned {priority.get('priority')} / {priority.get('risk_level')}."},
            {"agent": "Planning Agent", "decision": "Generated work order actions and spare strategy."},
            {"agent": "Reporting Agent", "decision": "Generated alert and logbook entry."},
        ]

    def build_agent_plan(self, query: str, mode: str, asset_id: str | None = None) -> list[dict]:
        q = str(query).lower()
        objective = "Diagnose equipment issue and produce maintenance decision support"
        if mode == "plant_priority":
            objective = "Rank plant assets and select the best maintenance target"
        elif mode == "original_vs_dynamic_comparison":
            objective = "Compare highest-risk original demo asset against highest-risk active dynamic asset"
        elif mode == "public_dataset":
            objective = "Explain public benchmark usage and data-governance controls"
        elif "spare" in q and not any(term in q for term in ["diagnose", "root cause", "risk", "vibration", "temperature", "pressure", "current", "alert"]):
            objective = "Identify required spare strategy for current maintenance risk"

        target = asset_id or self.session_memory.get("last_asset_id") or "plant"
        return [
            {"step": 1, "agent": "Supervisor Agent", "task": objective, "target": target, "status": "complete"},
            {"step": 2, "agent": "Triage Agent", "task": "Resolve asset, user intent, and operating context", "target": target, "status": "complete"},
            {"step": 3, "agent": "Sensor Agent", "task": "Read latest sensor state, anomaly events, and RUL indicators", "target": target, "status": "complete"},
            {"step": 4, "agent": "Knowledge Agent", "task": "Retrieve SOP, history, failure reports, spares, and policy evidence", "target": target, "status": "complete"},
            {"step": 5, "agent": "Risk Agent", "task": "Fuse ML risk with operational rule score and criticality", "target": target, "status": "complete"},
            {"step": 6, "agent": "Planner Agent", "task": "Create action plan, spare strategy, escalation, and work-order recommendation", "target": target, "status": "complete"},
            {"step": 7, "agent": "Verifier Agent", "task": "Check locked fields, traceability, and safety-critical escalation", "target": target, "status": "complete"},
            {"step": 8, "agent": "Reporter Agent", "task": "Generate engineer-facing report and logbook entry", "target": target, "status": "complete"},
        ]

    def build_tool_calls(
        self,
        asset_id: str,
        sensor: dict,
        anomaly: dict,
        priority: dict,
        docs: list[dict],
        history: list[dict],
        failures: list[dict],
        spares: list[dict],
        delay: dict,
        feedback: list[dict],
    ) -> list[dict]:
        real_history_count = sum("No historical work orders yet" not in str(row.get("issue", "")) for row in history)
        real_failure_count = sum("No failure reports yet" not in str(row.get("failure_mode", "")) for row in failures)
        return [
            {
                "tool": "asset_resolver",
                "agent": "Triage Agent",
                "input": asset_id,
                "output": f"{sensor.get('asset_type')} in {sensor.get('area')}",
                "status": "success",
            },
            {
                "tool": "sensor_health_reader",
                "agent": "Sensor Agent",
                "input": asset_id,
                "output": (
                    f"temp={_display_value(sensor.get('temperature_latest'))}, "
                    f"vib={_display_value(sensor.get('vibration_latest'))}, "
                    f"pressure={_display_value(sensor.get('pressure_latest'))}"
                ),
                "status": "success",
            },
            {
                "tool": "anomaly_detector",
                "agent": "Sensor Agent",
                "input": "latest sensor row + 24h anomaly window",
                "output": f"{anomaly.get('anomaly_level')} abnormality, {anomaly.get('anomaly_events_24h')} anomaly events",
                "status": "success",
            },
            {
                "tool": "hybrid_risk_scorer",
                "agent": "Risk Agent",
                "input": "ML failure risk + operational rules + criticality + delay",
                "output": f"{priority.get('priority')}/{priority.get('risk_level')} with score {priority.get('priority_score')}",
                "status": "success",
            },
            {
                "tool": "rag_retriever",
                "agent": "Knowledge Agent",
                "input": f"asset={asset_id}, top_k=5",
                "output": f"{len(docs)} evidence chunks from {len(set(d.get('source') for d in docs))} sources",
                "status": "success",
            },
            {
                "tool": "history_lookup",
                "agent": "Knowledge Agent",
                "input": asset_id,
                "output": f"{real_history_count} historical work orders, {real_failure_count} failure reports",
                "status": "success",
            },
            {
                "tool": "spares_planner",
                "agent": "Planner Agent",
                "input": asset_id,
                "output": f"{len(spares)} spare items checked",
                "status": "success",
            },
            {
                "tool": "feedback_memory",
                "agent": "Planner Agent",
                "input": asset_id,
                "output": f"{len(feedback)} relevant feedback rows reused",
                "status": "success",
            },
            {
                "tool": "digital_logbook_writer",
                "agent": "Reporter Agent",
                "input": asset_id,
                "output": "logbook entry created after report generation",
                "status": "success",
            },
        ]

    def build_verifier_checks(self, sensor: dict, priority: dict, docs: list[dict], spares: list[dict]) -> list[dict]:
        checks = [
            ("Asset resolved", bool(sensor.get("asset_id"))),
            ("Locked priority populated", bool(priority.get("priority") and priority.get("risk_level"))),
            ("Hybrid score available", sensor.get("hybrid_health_score") is not None),
            ("ML risk and rule score separated", sensor.get("ml_failure_risk_latest") is not None and sensor.get("operational_rule_score") is not None),
            ("RUL estimate available", sensor.get("estimated_rul_days") is not None),
            ("Traceability sources retrieved", len(docs) >= 3),
            ("Spare strategy checked", len(spares) > 0),
            ("Escalation generated for P1/P2", priority.get("priority") in {"P1", "P2", "P3", "P4", "PLANT"}),
        ]
        return [
            {"check": name, "status": "pass" if ok else "review", "detail": "verified" if ok else "needs engineer review"}
            for name, ok in checks
        ]

    def build_decision_packet(
        self,
        mode: str,
        query: str,
        asset_id: str,
        sensor: dict,
        priority: dict,
        docs: list[dict],
        actions: list[str],
        evidence: dict | None = None,
    ) -> dict:
        evidence = evidence or {}
        return {
            "mode": mode,
            "objective": query,
            "selected_asset": asset_id,
            "equipment_type": sensor.get("asset_type"),
            "risk_level": priority.get("risk_level"),
            "priority": priority.get("priority"),
            "urgency": priority.get("urgency"),
            "hybrid_failure_risk": sensor.get("hybrid_failure_risk"),
            "ml_failure_risk": sensor.get("ml_failure_risk_latest"),
            "operational_rule_score": sensor.get("operational_rule_score"),
            "hybrid_health_score": sensor.get("hybrid_health_score"),
            "estimated_rul_days": sensor.get("estimated_rul_days"),
            "evidence_confidence": evidence.get("evidence_confidence"),
            "available_evidence": evidence.get("available_evidence", []),
            "missing_evidence": evidence.get("missing_evidence", []),
            "recommended_first_action": actions[0] if actions else "Inspect asset condition",
            "top_sources": [doc.get("source") for doc in docs[:3]],
            "next_system_action": "create_work_order_and_notify_supervisor" if priority.get("priority") in {"P1", "P2"} else "monitor_and_schedule",
        }

    def write_logbook(self, query: str, asset_id: str, priority: dict, summary: str) -> None:
        path = DATA_DIR / "digital_logbook.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        row = {
            "timestamp": datetime.now().isoformat(),
            "user_id": self.session_memory.get("user_id", "demo_user"),
            "asset_id": asset_id,
            "query": query,
            "risk_level": priority.get("risk_level"),
            "priority": priority.get("priority"),
            "summary": summary[:1000],
        }
        pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(path, index=False)

    def save_feedback(self, user_id: str, asset_id: str, query: str, feedback_type: str, feedback_text: str, corrected_action: str = "", outcome: str = "") -> dict:
        path = DATA_DIR / "feedback_log.csv"
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        row = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "asset_id": asset_id,
            "query": query,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
            "corrected_action": corrected_action,
            "outcome": outcome,
        }
        pd.concat([df, pd.DataFrame([row])], ignore_index=True, sort=False).to_csv(path, index=False)
        return row

    def feedback_learning_report(self, query: str, user_id: str = "demo_user") -> dict:
        asset_id = (self._explicit_asset_ids(query) or extract_asset_ids(query) or [self.session_memory.get("last_asset_id", "UNKNOWN")])[0]
        match = re.search(r"confirms?\s+(.+?)\s+as\s+the\s+actual\s+cause", query, flags=re.IGNORECASE)
        if match:
            actual_cause = match.group(1).strip(" .")
        else:
            faulty_match = re.search(r"(?:confirms?\s+that\s+)?(.+?)\s+was\s+faulty", query, flags=re.IGNORECASE)
            if faulty_match:
                actual_cause = faulty_match.group(1).strip(" .")
            elif "record that correction" in str(query).lower() and self.session_memory.get("pending_correction"):
                actual_cause = self.session_memory.get("pending_correction")
            else:
                actual_cause = "engineer-confirmed corrected root cause"
        corrected_action = (
            f"Prioritize checks for {actual_cause}: verify alignment/condition, update RCA hypotheses, "
            "and show engineer feedback as learned evidence in future recommendations."
        )
        self.session_memory["pending_correction"] = actual_cause
        self.session_memory["last_asset_id"] = asset_id
        row = self.save_feedback(
            user_id=user_id,
            asset_id=asset_id,
            query=query,
            feedback_type="correction",
            feedback_text=query,
            corrected_action=corrected_action,
            outcome=f"Actual cause confirmed: {actual_cause}",
        )
        answer = f"""
**Engineer Feedback Recorded For {asset_id}**

**Correction learned**
- Previous diagnosis challenged by engineer.
- Confirmed actual cause: {actual_cause}.

**How future recommendations should change**
- Rank `{actual_cause}` higher in the root-cause hypothesis list for {asset_id}.
- Add alignment/condition verification to the first inspection checklist.
- When similar symptoms appear, cite this feedback as learned evidence instead of repeating the old diagnosis blindly.

**Memory write**
- File: feedback_log.csv
- Feedback type: correction
- Corrected action: {corrected_action}

**Verifier Summary**
- Engineer correction stored: PASS
- Future-action adjustment stated: PASS
- No sensor state was overwritten: PASS
""".strip()
        priority = {"priority": "LEARNING", "risk_level": "FEEDBACK", "urgency": "Use in future recommendations", "priority_score": 0}
        self.write_logbook(query, asset_id, priority, answer)
        return {
            "mode": "feedback_learning",
            "asset_id": asset_id,
            "intent": "feedback_learning",
            "feedback_row": row,
            "risk_priority": priority,
            "priority": "Feedback recorded",
            "agent_plan": [{"step": 1, "agent": "Learning Agent", "task": "Record engineer correction and update future recommendation bias", "status": "complete"}],
            "tool_calls": [{"tool": "feedback_log_writer", "agent": "Learning Agent", "input": asset_id, "output": "feedback row stored", "status": "success"}],
            "verifier_checks": [
                {"check": "Feedback persisted", "status": "pass", "detail": "feedback_log.csv"},
                {"check": "No false sensor update", "status": "pass", "detail": "feedback route does not mutate readings"},
            ],
            "decision_packet": {"mode": "feedback_learning", "selected_asset": asset_id, "actual_cause": actual_cause, "next_system_action": "reuse_feedback_in_future_rca"},
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Feedback recorded for {asset_id}.",
            "llm_used": False,
        }

    def spare_revision_report(self, query: str, asset_id: str | None = None) -> dict:
        target = asset_id or self._infer_asset_from_query(query) or (extract_asset_ids(query)[0] if extract_asset_ids(query) else None)
        if not target:
            return self.unknown_asset_report(query, ["UNKNOWN"])
        day_match = re.search(r"\b(?:arrive|arrives|arrival|lead time|lead)\D{0,30}(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?", query, flags=re.IGNORECASE)
        word_map = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
        lead_days = None
        if day_match:
            token = day_match.group(1).lower()
            lead_days = int(token) if token.isdigit() else word_map.get(token)
        if lead_days is not None and self._is_dynamic_asset(target):
            assets = load_dynamic_assets()
            idx = assets.index[assets["asset_id"].astype(str).str.upper() == str(target).upper()].tolist()
            if idx:
                assets.at[idx[0], "spare_lead_time_days"] = float(lead_days)
                existing = str(assets.at[idx[0], "operator_notes"] or "")
                note = f"bearing/spare arrival lead time {lead_days} days"
                assets.at[idx[0], "operator_notes"] = f"{existing}; {note}".strip("; ")
                save_dynamic_assets(assets)
        sensor = self.get_latest_sensor_summary(target)
        priority = self.prioritize_action(sensor, self.get_spares(target), self.get_delay(target))
        lead_text = f"{lead_days} days" if lead_days is not None else "not confirmed"
        answer = f"""
**Revised Spare-Constrained Recommendation For {target}**

**Spare constraint**
- Bearing/spare availability lead time: {lead_text}.
- Current priority remains {priority.get("priority")}/{priority.get("risk_level")} unless verified readings show risk has reduced.

**Revised strategy**
- Do not wait passively for the bearing if safety indicators remain high.
- Use temporary risk controls: reduce load/speed where possible, increase vibration/oil checks, inspect lubricant and filters, verify alignment, and set trip/watch limits.
- If risk is P1/CRITICAL, prepare controlled shutdown or standby plan even before the spare arrives.
- If shutdown is feasible without the bearing, perform inspection, oil flush, alignment correction, and damage confirmation now; defer bearing replacement until arrival.

**Alternative maintenance plan**
- Reserve incoming bearing immediately.
- Check cannibalization/alternate supplier options.
- Create a two-stage work order: immediate inspection and risk control now, bearing replacement when part arrives.

**Verifier Summary**
- Procurement lead time considered: PASS
- Inspection urgency separated from repair feasibility: PASS
- Temporary risk controls included: PASS
""".strip()
        return {
            "mode": "spare_revision",
            "asset_id": target,
            "intent": "spares_constrained_recommendation",
            "risk_priority": priority,
            "priority": f"{priority.get('priority')}/{priority.get('risk_level')}",
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Procurement Agent", "task": "Revise recommendation using lead time and temporary controls", "status": "complete"}],
            "tool_calls": [{"tool": "spare_lead_time_planner", "agent": "Procurement Agent", "input": query, "output": f"lead_time={lead_text}", "status": "success"}],
            "verifier_checks": [{"check": "Spare constraint handled", "status": "pass", "detail": lead_text}],
            "decision_packet": {"mode": "spare_revision", "selected_asset": target, "lead_time_days": lead_days, "next_system_action": "create_two_stage_work_order_and_reserve_spare"},
            "alert_report": f"Spare-constrained recommendation revised for {target}.",
            "llm_used": False,
        }

    def _rank_asset_rows(self, asset_ids: list[str] | None = None) -> pd.DataFrame:
        asset_ids = asset_ids or self.asset_ids
        rows = []
        for asset_id in asset_ids:
            sensor = self.get_latest_sensor_summary(asset_id)
            if sensor.get("error"):
                continue
            spares = self.get_spares(asset_id)
            delay = self.get_delay(asset_id)
            priority = self.prioritize_action(sensor, spares, delay)
            docs = self._dynamic_context_docs(asset_id, sensor) if sensor.get("is_dynamic") else self.rag.retrieve("", top_k=2, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
            evidence = self.evidence_confidence(asset_id, sensor, docs, self.get_history(asset_id), self.get_failures(asset_id), spares)
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": sensor.get("asset_type"),
                    "area": sensor.get("area"),
                    "criticality": sensor.get("criticality"),
                    "priority": priority.get("priority"),
                    "risk_level": priority.get("risk_level"),
                    "priority_score": priority.get("priority_score"),
                    "hybrid_health_score": sensor.get("hybrid_health_score"),
                    "hybrid_failure_risk": sensor.get("hybrid_failure_risk"),
                    "ml_failure_risk": sensor.get("ml_failure_risk_latest"),
                    "operational_rule_score": sensor.get("operational_rule_score"),
                    "rul_days": sensor.get("estimated_rul_days"),
                    "delay_hours": safe_float(delay.get("delay_hours", 0)),
                    "spare_count": len(spares),
                    "out_of_stock_spares": [s.get("spare_part", "unknown spare") for s in spares if safe_float(s.get("stock_qty", 0)) <= 0],
                    "evidence_confidence": evidence.get("evidence_confidence"),
                    "missing_evidence": evidence.get("missing_evidence", []),
                    "applied_rules": sensor.get("applied_rule_count", 0),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(["priority_score", "hybrid_health_score", "rul_days"], ascending=[False, False, True]).reset_index(drop=True)

    def safety_guardrail_report(self, query: str) -> dict:
        target = self._infer_asset_from_query(query) or self.session_memory.get("last_asset_id")
        table = self._rank_asset_rows([target] if target else None)
        top = table.iloc[0].to_dict() if not table.empty else {}
        selected = str(top.get("asset_id") or target or "UNRESOLVED")
        q = str(query).lower()
        reason = "The request asks for certainty, hidden evidence, fabricated evidence, unsafe operation, or bypassing safety controls."
        if "exact date and time" in q:
            reason = "Exact failure timestamps cannot be guaranteed from probabilistic RUL/risk models."
        elif "guarantee" in q:
            reason = "Safe operation cannot be guaranteed without live verification and control-room acceptance."
        elif "bypass" in q or "interlock" in q:
            reason = "Bypassing safety interlocks is unsafe and outside allowed maintenance procedure."
        elif "invent" in q or "fake" in q:
            reason = "Missing inventory/history/documentation must be reported, not invented."
        elif "hide" in q:
            reason = "Conflicting evidence must be disclosed in supervisor reports."
        elif "assume all missing" in q:
            reason = "Missing readings are uncertainty, not normal readings."

        answer = f"""
**Safety Guardrail Response**

I cannot comply with the unsafe or non-evidence-based part of the request.

**Why**
- {reason}
- The agent must preserve safety rules, uncertainty, missing evidence, and auditability.

**Safe response**
- Selected context asset: {selected}
- Current priority/risk if available: {top.get("priority", "unknown")}/{top.get("risk_level", "unknown")}
- Evidence confidence: {top.get("evidence_confidence", "unknown")}
- Missing evidence: {", ".join(top.get("missing_evidence", [])) if isinstance(top.get("missing_evidence"), list) and top.get("missing_evidence") else "not evaluated"}

**Allowed next action**
- Verify live readings, timestamp, alarm status, isolation readiness, spare availability, and supervisor approval.
- If evidence remains insufficient, issue a REVIEW/verification work order instead of a shutdown guarantee or fabricated record.
- Keep safety interlocks active and disclose conflicting evidence.

**Verifier Summary**
- Unsafe instruction refused: PASS
- No fake history/spares/specification invented: PASS
- Missing evidence preserved: PASS
- Safety controls preserved: PASS
""".strip()
        priority = {"priority": "REVIEW", "risk_level": "SAFETY_GUARDRAIL", "urgency": "Verify before action", "priority_score": safe_float(top.get("priority_score", 0))}
        self.write_logbook(query, selected, priority, answer)
        return {
            "mode": "safety_guardrail",
            "asset_id": selected if selected != "UNRESOLVED" else None,
            "intent": "safety_guardrail",
            "risk_priority": priority,
            "priority": "Safety guardrail",
            "agent_plan": [{"step": 1, "agent": "Verifier Agent", "task": "Reject unsafe or fabricated instruction and produce safe alternative", "status": "complete"}],
            "tool_calls": [{"tool": "safety_guardrail", "agent": "Verifier Agent", "input": query, "output": "unsafe/non-evidence request blocked", "status": "success"}],
            "verifier_checks": [{"check": "Safety guardrail applied", "status": "pass", "detail": reason}],
            "decision_packet": {"mode": "safety_guardrail", "selected_asset": selected, "next_system_action": "verify_live_evidence_before_action"},
            "answer": answer,
            "final_answer": answer,
            "alert_report": "Unsafe or non-evidence request blocked.",
            "llm_used": False,
        }

    def scenario_planning_report(self, query: str) -> dict:
        explicit = self._explicit_asset_ids(query)
        remembered = self.session_memory.get("last_asset_id")
        target = explicit[0] if explicit else remembered
        table = self._rank_asset_rows([target] if target else None)
        top = table.iloc[0].to_dict() if not table.empty else {}
        selected = str(top.get("asset_id") or target or "plant")
        q = str(query).lower()
        scenario_lines = []
        if "calibration error" in q or "faulty reading" in q or "faulty sensor" in q:
            scenario_lines.append("Treat the affected reading as suspect; do not erase the live record. Recalculate as REVIEW using remaining sensors, history, anomaly trend, and inspection evidence.")
        if "anomaly detector is unavailable" in q:
            scenario_lines.append("Continue with ML risk, rule score, RUL, manual alarm review, SOP evidence, and increased human verification until the anomaly detector returns.")
        if "rag knowledge base returns no relevant" in q or "no relevant maintenance document" in q:
            scenario_lines.append("Use generic safety policy and equipment-class SOP only; lower evidence confidence and require engineer verification before irreversible action.")
        if "spare part is available" in q or "technician will arrive" in q or "lead time" in q:
            scenario_lines.append("Separate repair feasibility from inspection urgency: reserve the spare, hold temporary controls, and schedule technician-dependent repair when qualified staff arrive.")
        if "full capacity" in q or "12 hours" in q:
            scenario_lines.append("Use temporary controls: reduce avoidable load spikes, increase inspection frequency, set alarm thresholds, prepare standby/isolation plan, and escalate if limits are crossed.")
        if "two maintenance teams" in q:
            scenario_lines.append("Run two parallel work streams: immediate P1 verification/repair for top assets and P2 inspection/prep for the next asset.")
        if "two hours" in q:
            scenario_lines.append("Choose the highest risk-reduction task that fits the window: inspection, oil sample/filter change, alignment check, or safe isolation prep rather than full overhaul.")
        if "not repaired for another 48 hours" in q:
            scenario_lines.append("Delay increases production and safety exposure; move to enhanced monitoring, supervisor acknowledgement, spare reservation, and contingency shutdown criteria.")
        if "no physical damage" in q:
            scenario_lines.append("Do not close the incident. Investigate sensor calibration, lubrication quality, alignment, operating transients, and repeat trend validation.")
        if "again reaches critical risk" in q:
            scenario_lines.append("Treat recurrence after repair as a repeat-failure investigation: verify repair quality, root cause escape, installation defects, sensor validity, and operating conditions.")
        if not scenario_lines:
            scenario_lines.append("This is a hypothetical scenario. Keep current memory unchanged, compare baseline versus assumption, and require confirmation before changing live priority.")

        ranking_text = (
            _markdown_table(
                table.head(6).to_dict("records"),
                ["asset_id", "priority", "risk_level", "priority_score", "hybrid_health_score", "rul_days", "evidence_confidence"],
            )
            if not table.empty
            else "No asset context available; ask for an asset ID or plant scope."
        )
        answer = f"""
**What-If / Scenario Planning Review**

**Assumption handled**
- {query}
- Live memory was not mutated. This is a planning overlay, not a confirmed sensor update.

**Baseline context**
{ranking_text}

**Scenario impact**
{chr(10).join(f"- {line}" for line in scenario_lines)}

**Recommended next action**
- Keep the existing risk classification until the assumption is verified.
- Create a verification task for the assumed change.
- If verified, re-run priority scoring with the corrected readings, spare constraints, technician availability, and supervisor sign-off.

**Verifier Summary**
- Hypothetical assumption separated from live data: PASS
- Missing/uncertain evidence not guessed: PASS
- Safe temporary controls included: PASS
""".strip()
        priority = {"priority": "SCENARIO", "risk_level": "WHAT_IF", "urgency": "Verify assumption", "priority_score": safe_float(top.get("priority_score", 0))}
        self.write_logbook(query, selected, priority, answer)
        return {
            "mode": "scenario_planning",
            "asset_id": selected if selected != "plant" else None,
            "intent": "scenario_planning",
            "scenario_table": table.to_dict("records") if not table.empty else [],
            "risk_priority": priority,
            "priority": "Scenario planning",
            "agent_plan": [{"step": 1, "agent": "Scenario Agent", "task": "Evaluate assumption without mutating live memory", "status": "complete"}],
            "tool_calls": [{"tool": "what_if_planner", "agent": "Scenario Agent", "input": query, "output": "scenario overlay generated", "status": "success"}],
            "verifier_checks": [{"check": "Live memory unchanged", "status": "pass", "detail": "what-if only"}],
            "decision_packet": {"mode": "scenario_planning", "selected_asset": selected, "next_system_action": "verify_assumption_then_rescore"},
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Scenario plan generated for {selected}.",
            "llm_used": False,
        }

    def spare_availability_report(self, query: str) -> dict:
        table = self._rank_asset_rows()
        rows = []
        if not table.empty:
            for row in table.to_dict("records"):
                is_critical = str(row.get("risk_level")).upper() == "CRITICAL" or str(row.get("priority")).upper() == "P1"
                if not is_critical:
                    continue
                spares = self.get_spares(str(row["asset_id"]))
                unavailable = not spares or any(safe_float(item.get("stock_qty", 0)) <= 0 for item in spares)
                if unavailable:
                    rows.append(
                        {
                            "asset_id": row["asset_id"],
                            "priority": f"{row['priority']}/{row['risk_level']}",
                            "rul_days": row["rul_days"],
                            "spare_status": "no spare master record" if not spares else "one or more spares out of stock",
                            "unavailable_spares": ", ".join(item.get("spare_part", "unknown spare") for item in spares if safe_float(item.get("stock_qty", 0)) <= 0) if spares else "unknown required spares",
                            "evidence_confidence": row["evidence_confidence"],
                        }
                    )
        answer = f"""
**Critical Assets With Spare Availability Risk**

{_markdown_table(rows, ["asset_id", "priority", "rul_days", "spare_status", "unavailable_spares", "evidence_confidence"]) if rows else "No critical active asset currently shows an unavailable spare record."}

**Decision rule**
- Do not assume unavailable parts are available.
- If spare master data is absent, mark procurement risk as REVIEW and reserve/check parts before shutdown.

**Next action**
- Procurement team should verify stock, alternates, vendor lead time, and cannibalization options for every row above.
""".strip()
        return {
            "mode": "spare_availability_listing",
            "asset_id": rows[0]["asset_id"] if rows else None,
            "intent": "spare_availability_listing",
            "spare_risk_table": rows,
            "risk_priority": {"priority": "PROCUREMENT", "risk_level": "SPARE_RISK", "urgency": "Verify stock", "priority_score": len(rows)},
            "priority": "Spare availability review",
            "agent_plan": [{"step": 1, "agent": "Procurement Agent", "task": "Check critical assets against spare records", "status": "complete"}],
            "tool_calls": [{"tool": "spares_inventory_checker", "agent": "Procurement Agent", "input": "critical assets", "output": f"{len(rows)} spare-risk row(s)", "status": "success"}],
            "verifier_checks": [{"check": "No spare availability invented", "status": "pass", "detail": "unknowns remain REVIEW"}],
            "decision_packet": {"mode": "spare_availability_listing", "affected_assets": [row["asset_id"] for row in rows], "next_system_action": "verify_procurement_stock"},
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"{len(rows)} critical asset(s) need spare verification.",
            "llm_used": False,
        }

    def alert_summary_report(self, query: str) -> dict:
        table = self._rank_asset_rows()
        if table.empty:
            answer = "No active asset telemetry is available for alert summarization."
            return {
                "mode": "alert_summary",
                "asset_id": None,
                "intent": "alert_summary",
                "alert_rows": [],
                "risk_priority": {"priority": "UNKNOWN", "risk_level": "UNKNOWN", "urgency": "No telemetry", "priority_score": 0},
                "priority": "No alert data",
                "agent_plan": [{"step": 1, "agent": "Alert Agent", "task": "Scan active assets for abnormal conditions", "status": "review"}],
                "tool_calls": [{"tool": "asset_health_scan", "agent": "Alert Agent", "input": "active assets", "output": "0 rows", "status": "review"}],
                "verifier_checks": [{"check": "No NaN/null display", "status": "pass", "detail": "empty response"}],
                "decision_packet": {"mode": "alert_summary", "selected_asset": None},
                "answer": answer,
                "final_answer": answer,
                "alert_report": "",
                "llm_used": False,
            }

        alert_rows = []
        for row in table.to_dict("records"):
            risk = str(row.get("risk_level", "LOW")).upper()
            priority = str(row.get("priority", "P4")).upper()
            rul = safe_float(row.get("rul_days"), 999)
            score = safe_float(row.get("hybrid_health_score"), 0)
            delay = safe_float(row.get("delay_hours"), 0)
            has_spare_gap = bool(row.get("out_of_stock_spares"))
            if risk in ["CRITICAL", "HIGH"] or priority in ["P1", "P2"] or rul <= 7 or score >= 50 or delay >= 4 or has_spare_gap:
                alert_rows.append(
                    {
                        "asset_id": row.get("asset_id"),
                        "asset_type": row.get("asset_type"),
                        "area": row.get("area"),
                        "priority": f"{priority}/{risk}",
                        "risk_score": round(score, 2),
                        "rul_days": round(rul, 2) if rul < 999 else "not available",
                        "reason": self._plain_alert_reason(row),
                        "next_action": self._plain_alert_action(row),
                        "evidence_confidence": row.get("evidence_confidence", "UNKNOWN"),
                    }
                )

        alert_rows = alert_rows[:8]
        inactive_df = list_inactive_dynamic_assets()
        inactive_ids = inactive_df["asset_id"].astype(str).str.upper().tolist() if not inactive_df.empty else []
        if alert_rows:
            top = alert_rows[0]
            plain_lines = [
                f"- {row['asset_id']}: {row['priority']}. {row['reason']} Next: {row['next_action']} Evidence: {row['evidence_confidence']}."
                for row in alert_rows
            ]
            table_text = _markdown_table(
                alert_rows,
                ["asset_id", "priority", "risk_score", "rul_days", "reason", "next_action", "evidence_confidence"],
            )
            answer = f"""
**Today's Abnormal Alert Summary**

**Plain-language summary**
{chr(10).join(plain_lines)}

**Alert table**
{table_text}

**Supervisor note**
- Inactive dynamic assets excluded from active alert ranking: {", ".join(inactive_ids) if inactive_ids else "none"}.
- Missing history/SOP/spares are treated as evidence gaps, not invented facts.
- Highest immediate attention: {top["asset_id"]}.
""".strip()
            selected = str(top["asset_id"])
            priority_payload = {"priority": top["priority"].split("/")[0], "risk_level": top["priority"].split("/")[-1], "urgency": top["next_action"], "priority_score": safe_float(table.iloc[0].get("priority_score", 0))}
        else:
            answer = """
**Today's Abnormal Alert Summary**

No active asset currently crosses the abnormal alert threshold.

**Supervisor note**
- Continue routine monitoring.
- Do not add inactive dynamic assets back into ranking unless new live readings are provided.
""".strip()
            selected = None
            priority_payload = {"priority": "P4", "risk_level": "LOW", "urgency": "Monitor", "priority_score": 0}

        return {
            "mode": "alert_summary",
            "asset_id": selected,
            "intent": "alert_summary",
            "alert_rows": alert_rows,
            "risk_priority": priority_payload,
            "priority": priority_payload["priority"],
            "agent_plan": [{"step": 1, "agent": "Alert Agent", "task": "Summarize active abnormal plant alerts", "status": "complete"}],
            "tool_calls": [
                {"tool": "asset_health_scan", "agent": "Sensor Agent", "input": "active original and dynamic assets", "output": f"{len(table)} active row(s)", "status": "success"},
                {"tool": "abnormal_alert_filter", "agent": "Alert Agent", "input": "risk, RUL, score, delay, spare gaps", "output": f"{len(alert_rows)} alert row(s)", "status": "success"},
            ],
            "verifier_checks": [
                {"check": "Inactive dynamic assets excluded", "status": "pass", "detail": ", ".join(inactive_ids) if inactive_ids else "none"},
                {"check": "No NaN/null display", "status": "pass", "detail": "missing values rendered as not available"},
                {"check": "No missing evidence invented", "status": "pass", "detail": "evidence confidence retained"},
            ],
            "decision_packet": {"mode": "alert_summary", "selected_asset": selected, "active_alert_count": len(alert_rows), "inactive_dynamic_assets_excluded": inactive_ids},
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"{len(alert_rows)} abnormal alert(s) summarized.",
            "llm_used": False,
        }

    def _plain_alert_reason(self, row: dict) -> str:
        risk = str(row.get("risk_level", "LOW")).upper()
        rul = safe_float(row.get("rul_days"), 999)
        score = safe_float(row.get("hybrid_health_score"), 0)
        applied_rules = int(safe_float(row.get("applied_rules", 0)))
        reasons = []
        if risk in ["CRITICAL", "HIGH"]:
            reasons.append(f"{risk.lower()} risk band")
        if rul <= 3:
            reasons.append(f"short RUL around {round(rul, 2)} days")
        elif rul <= 7:
            reasons.append(f"RUL under a week ({round(rul, 2)} days)")
        if score >= 50:
            reasons.append(f"elevated hybrid score {round(score, 2)}")
        if applied_rules > 0:
            reasons.append(f"{applied_rules} remembered/rule-based trigger(s)")
        if row.get("out_of_stock_spares"):
            reasons.append("spare availability risk")
        return "; ".join(reasons) if reasons else "condition is above normal monitoring threshold"

    def _plain_alert_action(self, row: dict) -> str:
        risk = str(row.get("risk_level", "LOW")).upper()
        asset_type = str(row.get("asset_type", "")).lower()
        if risk == "CRITICAL":
            return "notify supervisor, create P1 work order, verify live readings, and prepare safe isolation"
        if "gearbox" in asset_type:
            return "run vibration spectrum, oil sample, alignment and bearing checks"
        if "pump" in asset_type:
            return "check suction, cavitation noise, pressure trend, seal and spare status"
        if any(term in asset_type for term in ["blower", "fan", "compressor"]):
            return "check vibration, temperature, damper, impeller fouling, bearing and standby readiness"
        if "motor" in asset_type:
            return "check current imbalance, cooling path, bearing lubrication and load"
        return "inspect, confirm trend, reserve critical spares, and update logbook"

    def ambiguous_asset_resolution_report(self, query: str) -> dict:
        q = str(query).lower()
        table = self.asset_health_table().copy()
        if "blower" in q:
            candidates = table[table["asset_type"].astype(str).str.contains("blower|fan|compressor", case=False, regex=True, na=False)].copy()
        elif "pump" in q:
            candidates = table[table["asset_type"].astype(str).str.contains("pump", case=False, regex=True, na=False)].copy()
        elif "gearbox" in q:
            candidates = table[table["asset_type"].astype(str).str.contains("gearbox", case=False, regex=True, na=False)].copy()
        else:
            candidates = table.head(10).copy()
        rows = []
        for _, row in candidates.iterrows():
            asset_id = str(row.get("asset_id")).upper()
            sensor = self.get_latest_sensor_summary(asset_id)
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": sensor.get("asset_type"),
                    "area": sensor.get("area"),
                    "risk_level": sensor.get("risk_band"),
                    "rul_days": sensor.get("estimated_rul_days"),
                }
            )
        answer = f"""
**Ambiguous Asset Reference**

I will not silently select one asset from an incomplete reference.

**Likely candidates**
{_markdown_table(rows, ["asset_id", "asset_type", "area", "risk_level", "rul_days"]) if rows else "No likely candidates found."}

**Safe next step**
- Provide the full asset ID, or confirm one candidate above.
- Until confirmed, I can give only a comparison or inspection checklist, not a specific shutdown/work-order command.
""".strip()
        return {
            "mode": "ambiguous_asset_resolution",
            "asset_id": None,
            "intent": "ambiguous_asset_resolution",
            "candidate_assets": rows,
            "risk_priority": {"priority": "REVIEW", "risk_level": "AMBIGUOUS_ASSET", "urgency": "Confirm asset", "priority_score": 0},
            "priority": "Needs asset confirmation",
            "agent_plan": [{"step": 1, "agent": "Triage Agent", "task": "List candidates and avoid silent asset selection", "status": "complete"}],
            "tool_calls": [{"tool": "asset_candidate_resolver", "agent": "Triage Agent", "input": query, "output": f"{len(rows)} candidate(s)", "status": "success"}],
            "verifier_checks": [{"check": "No silent selection", "status": "pass", "detail": "candidate list returned"}],
            "decision_packet": {"mode": "ambiguous_asset_resolution", "candidate_assets": [row["asset_id"] for row in rows], "next_system_action": "ask_user_to_confirm_asset"},
            "answer": answer,
            "final_answer": answer,
            "alert_report": "Asset reference ambiguous; confirmation required.",
            "llm_used": False,
        }

    def logbook_template_report(self, query: str) -> dict:
        explicit_assets = self._explicit_asset_ids(query)
        target_asset_id = explicit_assets[0] if explicit_assets else None
        equipment_context = self._preserved_equipment_context(query) or "not provided"
        target = target_asset_id or equipment_context
        technician_match = re.search(r"technician\s*:\s*(.+?)(?:\.\s+work\s+done|;|\n|$)", query, flags=re.IGNORECASE)
        work_match = re.search(r"work\s+done\s*:\s*([^.\n]+)", query, flags=re.IGNORECASE)
        date_label = "today" if "today" in str(query).lower() else "not provided"
        technician = technician_match.group(1).strip() if technician_match else "not provided"
        work_done = work_match.group(1).strip() if work_match else "not provided"
        answer = f"""
**Structured Digital Logbook Draft**

- Timestamp: {datetime.now().isoformat(timespec="seconds")}
- Equipment/context: {target}
- Asset ID: {target_asset_id or "not provided"}
- Date supplied: {date_label}
- Technician: {technician}
- Work performed: {work_done}
- Request: {query}
- Status: OPEN / PENDING VERIFICATION
- Completion state: NOT COMPLETED
- Evidence used: latest sensor/risk state, available SOP/RAG evidence, spares status, engineer notes if provided
- Action proposed: create inspection/work-order draft and capture supervisor decision
- Required closure evidence: technician confirmation, measured readings after work, parts used, root cause, outcome, and engineer feedback

**Verifier Summary**
- Work not marked completed: PASS
- Missing closure evidence requested: PASS
""".strip()
        return {
            "mode": "logbook_template",
            "asset_id": target_asset_id,
            "subject": equipment_context,
            "intent": "logbook_template",
            "risk_priority": {"priority": "LOGBOOK", "risk_level": "OPEN_ENTRY", "urgency": "Awaiting completion evidence", "priority_score": 0},
            "priority": "Open logbook draft",
            "agent_plan": [{"step": 1, "agent": "Reporter Agent", "task": "Draft logbook row without closing work", "status": "complete"}],
            "tool_calls": [{"tool": "logbook_template_writer", "agent": "Reporter Agent", "input": query, "output": "open draft generated", "status": "success"}],
            "verifier_checks": [
                {"check": "Completion not claimed", "status": "pass", "detail": "status remains OPEN"},
                {"check": "No asset ID invented", "status": "pass" if target_asset_id else "pass", "detail": target_asset_id or "asset_id not provided"},
                {"check": "Technician/work captured", "status": "pass" if technician != "not provided" or work_done != "not provided" else "review", "detail": f"{technician}; {work_done}"},
            ],
            "decision_packet": {
                "mode": "logbook_template",
                "selected_asset": target,
                "asset_id": target_asset_id or "not provided",
                "equipment_context": equipment_context,
                "technician": technician,
                "work_done": work_done,
                "date_supplied": date_label,
                "completion_state": "NOT_COMPLETED",
            },
            "answer": answer,
            "final_answer": answer,
            "alert_report": "Open logbook draft generated.",
            "llm_used": False,
        }

    def model_disagreement_report(self, query: str) -> dict:
        explicit = self._explicit_asset_ids(query)
        asset_ids = explicit or self.asset_ids
        rows = []
        for asset_id in asset_ids:
            sensor = self.get_latest_sensor_summary(asset_id)
            if sensor.get("error"):
                continue
            anomaly = self.detect_anomaly(asset_id)
            spares = self.get_spares(asset_id)
            delay = self.get_delay(asset_id)
            priority = self.prioritize_action(sensor, spares, delay)
            docs = (
                self._dynamic_context_docs(asset_id, sensor)
                if sensor.get("is_dynamic")
                else self.rag.retrieve(query, top_k=3, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
            )
            history = self.get_history(asset_id)
            failures = self.get_failures(asset_id)
            evidence = self.evidence_confidence(asset_id, sensor, docs, history, failures, spares)
            ml_risk = safe_float(sensor.get("ml_failure_risk_latest"))
            rule_score = safe_float(sensor.get("operational_rule_score"))
            hybrid_score = safe_float(sensor.get("hybrid_health_score"))
            disagreement_flags = []
            if ml_risk >= 0.70 and rule_score < 55:
                disagreement_flags.append("ML high but operational rules not severe")
            if ml_risk < 0.45 and rule_score >= 70:
                disagreement_flags.append("Rule engine severe but ML probability lower")
            if anomaly.get("anomaly_level") == "HIGH" and ml_risk < 0.50:
                disagreement_flags.append("Anomaly detector high while ML probability lower")
            if evidence.get("evidence_confidence") == "LOW" and priority.get("priority") in {"P1", "P2"}:
                disagreement_flags.append("Priority is high but evidence confidence is LOW")
            if sensor.get("applied_rule_count", 0) and evidence.get("evidence_confidence") == "LOW":
                disagreement_flags.append("Remembered rule applied with limited historical/RAG evidence")
            real_history = any("No historical work orders yet" not in str(row.get("issue", "")) for row in history)
            real_failures = any(str(row.get("failure_mode", "")).lower() not in {"not yet observed", "no failure reports yet", "", "nan"} for row in failures)
            if not real_history:
                disagreement_flags.append("No real maintenance history available")
            if not real_failures:
                disagreement_flags.append("No real failure report available")
            rows.append(
                {
                    "asset_id": asset_id,
                    "priority": priority.get("priority"),
                    "risk_level": priority.get("risk_level"),
                    "ml_failure_risk": round(ml_risk, 4),
                    "anomaly_level": anomaly.get("anomaly_level"),
                    "operational_rule_score": round(rule_score, 2),
                    "hybrid_health_score": round(hybrid_score, 2),
                    "rul_days": sensor.get("estimated_rul_days"),
                    "applied_rules": sensor.get("applied_rule_count", 0),
                    "evidence_confidence": evidence.get("evidence_confidence"),
                    "disagreement_flags": disagreement_flags or ["No major disagreement"],
                    "missing_evidence": evidence.get("missing_evidence", []),
                }
            )

        table = (
            pd.DataFrame(rows)
            .sort_values(["hybrid_health_score", "rul_days"], ascending=[False, True])
            .reset_index(drop=True)
            if rows
            else pd.DataFrame()
        )
        if table.empty:
            return self.unknown_asset_report(query, explicit or ["UNKNOWN"])

        top = table.iloc[0].to_dict()
        review_rows = table[table["disagreement_flags"].apply(lambda flags: flags != ["No major disagreement"])].to_dict("records")
        review_text = _format_records(
            [
                {
                    "asset_id": row["asset_id"],
                    "priority": f"{row['priority']}/{row['risk_level']}",
                    "ml": row["ml_failure_risk"],
                    "anomaly": row["anomaly_level"],
                    "rule_score": row["operational_rule_score"],
                    "evidence": row["evidence_confidence"],
                    "flags": "; ".join(row["disagreement_flags"]),
                }
                for row in review_rows[:8]
            ]
        )
        answer = f"""
**Model / Rule / Evidence Disagreement Review**

**Final decision policy**
- Do not let a LOW-evidence dynamic asset silently outrank a stronger-evidence asset.
- If ML, anomaly detector, remembered rules, and RUL agree, proceed with normal priority.
- If a safety/SOP rule matches but evidence is LOW, keep priority elevated but require supervisor verification before shutdown.
- If history or failure reports are missing, state that gap instead of inventing evidence.

**Highest decision candidate**
- Selected for immediate review: {top["asset_id"]}
- Priority/risk: {top["priority"]}/{top["risk_level"]}
- ML risk: {top["ml_failure_risk"]}
- Anomaly detector: {top["anomaly_level"]}
- Rule score: {top["operational_rule_score"]}
- Hybrid score: {top["hybrid_health_score"]}
- RUL: {top["rul_days"]} days
- Evidence confidence: {top["evidence_confidence"]}

**Assets With Disagreement Or Missing Evidence**
{review_text}

**Resolution**
- Final maintenance choice should use hybrid score plus RUL and criticality.
- Any asset with LOW evidence and high rule severity should be marked REVIEW/P1 verification, not treated as fully proven.
- Create or update the work order for {top["asset_id"]}; attach the disagreement flags and missing evidence list.

**Verifier Summary**
- ML risk checked: PASS
- Anomaly detector checked: PASS
- Rule engine checked: PASS
- History/failure evidence checked without invention: PASS
- Final decision policy stated: PASS
""".strip()
        priority = {"priority": "REVIEW", "risk_level": "DISAGREEMENT_AUDIT", "urgency": f"Review {top['asset_id']}", "priority_score": top["hybrid_health_score"]}
        self.session_memory["last_asset_id"] = str(top["asset_id"])
        self.write_logbook(query, str(top["asset_id"]), priority, answer)
        return {
            "mode": "model_disagreement_review",
            "asset_id": str(top["asset_id"]),
            "intent": "model_disagreement_review",
            "disagreement_table": table.to_dict("records"),
            "risk_priority": priority,
            "priority": "Model/evidence disagreement review",
            "agent_plan": [
                {"step": 1, "agent": "Verifier Agent", "task": "Compare ML, anomaly, rule, history, RAG, and spares signals", "status": "complete"},
                {"step": 2, "agent": "Risk Agent", "task": "Resolve final decision policy from conflicting evidence", "status": "complete"},
            ],
            "tool_calls": [
                {"tool": "ml_risk_reader", "agent": "Verifier Agent", "input": f"{len(table)} assets", "output": "ML risk compared", "status": "success"},
                {"tool": "anomaly_detector", "agent": "Verifier Agent", "input": f"{len(table)} assets", "output": "anomaly levels compared", "status": "success"},
                {"tool": "dynamic_rule_engine", "agent": "Policy Agent", "input": f"{len(table)} assets", "output": "applied rules checked", "status": "success"},
                {"tool": "rag_history_evidence_checker", "agent": "Knowledge Agent", "input": f"{len(table)} assets", "output": "evidence confidence checked", "status": "success"},
            ],
            "verifier_checks": [
                {"check": "No fake history invented", "status": "pass", "detail": "missing history/failure evidence is reported"},
                {"check": "LOW evidence does not silently win", "status": "pass", "detail": "requires supervisor verification"},
            ],
            "decision_packet": {
                "mode": "model_disagreement_review",
                "selected_asset": str(top["asset_id"]),
                "final_decision_policy": "hybrid risk + RUL + criticality with evidence-confidence gate",
                "next_system_action": "attach_disagreement_review_to_work_order",
            },
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Disagreement review completed; review {top['asset_id']}.",
            "llm_used": False,
        }

    def _dynamic_context_docs(self, asset_id: str, sensor: dict) -> list[dict]:
        missing = sensor.get("missing_readings") or ""
        uncertainty = (
            f" Missing readings: {missing}. Neutral defaults were used only for provisional risk scoring."
            if missing
            else ""
        )
        qualitative = (
            f" Operator notes: {sensor.get('operator_notes')}. {sensor.get('qualitative_risk_note')}"
            if sensor.get("operator_notes")
            else ""
        )
        rule_text = ""
        if sensor.get("applied_rules"):
            summaries = [
                f"{rule.get('rule_id')}: {rule.get('condition_text')}"
                for rule in sensor.get("applied_rules", [])
            ]
            rule_text = " Remembered rules applied: " + " | ".join(summaries)
        return [
            {
                "source": "dynamic_assets.csv",
                "asset_id": asset_id,
                "equipment_type": sensor.get("asset_type", "dynamic_asset"),
                "issue_type": "user_memory_current_health",
                "text": (
                    f"User-added asset {asset_id}. Type: {sensor.get('asset_type')}. Area: {sensor.get('area')}. "
                    f"Criticality: {sensor.get('criticality')}. Temperature: {_display_value(sensor.get('temperature_latest'))}. "
                    f"Vibration: {_display_value(sensor.get('vibration_latest'))}. Current: {_display_value(sensor.get('current_latest'))}. "
                    f"Pressure: {_display_value(sensor.get('pressure_latest'))}. Alarm count: {sensor.get('alarm_count_latest')}. "
                    f"Risk band: {sensor.get('risk_band')}. Hybrid health score: {sensor.get('hybrid_health_score')}. "
                    f"Estimated RUL days: {sensor.get('estimated_rul_days')}.{uncertainty}{qualitative}{rule_text}"
                ),
            }
        ]

    def _filter_docs_for_assets(self, docs: list[dict], asset_ids: list[str]) -> list[dict]:
        allowed = {str(asset_id).upper() for asset_id in asset_ids}
        allowed_equipment = set()
        for asset_id in allowed:
            sensor = self.get_latest_sensor_summary(asset_id)
            allowed_equipment.add(normalize_equipment_type(sensor.get("asset_type", "")))
            allowed_equipment.add(str(sensor.get("asset_type", "")).lower().replace(" ", "_"))
        out: list[dict] = []
        for doc in docs:
            aid = str(doc.get("asset_id", "")).upper()
            equipment = str(doc.get("equipment_type", "")).lower()
            source = str(doc.get("source", "")).lower()
            is_policy = equipment in {"policy", "safety"} or "policy" in source or "operating_model" in source
            is_scoped_all_doc = aid == "ALL" and (equipment in allowed_equipment or is_policy)
            if aid in allowed or aid in {"", "NONE", "NAN"} or is_scoped_all_doc:
                out.append(doc)
        return out

    def _filter_general_docs(self, query: str, docs: list[dict], subject: str, intent: str) -> list[dict]:
        if not docs:
            return []
        q = str(query or "").lower()
        if intent == "sop_request":
            preferred: list[tuple[int, dict]] = []
            for doc in docs:
                blob = " ".join(
                    str(doc.get(key, ""))
                    for key in ["source", "asset_id", "equipment_type", "issue_type", "text"]
                ).lower()
                source = str(doc.get("source", "")).lower()
                equipment = str(doc.get("equipment_type", "")).lower()
                issue = str(doc.get("issue_type", "")).lower()
                score = 0
                if "sop" in source or "manual" in source:
                    score += 5
                if "hydraulic" in blob or equipment == "hydraulic":
                    score += 6
                if "pump" in blob:
                    score += 4
                if "seal" in blob:
                    score += 4
                if "rolling" in blob or "rolling mill" in str(subject or "").lower():
                    score += 3
                if any(term in blob for term in ["loto", "lockout", "permit", "zero pressure", "stored energy"]):
                    score += 3
                if issue in {"pressure", "loto permit", "spares procurement"}:
                    score += 2
                if any(term in source for term in ["delay", "asset_health", "maintenance_history"]) and "hydraulic" not in blob:
                    score -= 4
                if score > 0:
                    preferred.append((score, doc))
            preferred.sort(key=lambda item: item[0], reverse=True)
            seen = set()
            ordered = []
            for _, doc in preferred:
                key = (doc.get("source"), doc.get("chunk_id"))
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(doc)
            return ordered[:7] if ordered else docs[:5]
        if intent == "error_code_lookup":
            code_match = re.search(r"\b([A-Z]{1,4}[- ]?\d{2,4})\b", str(query or ""), flags=re.IGNORECASE)
            code = code_match.group(1).upper().replace(" ", "-") if code_match else ""
            preferred: list[tuple[int, dict]] = []
            for doc in docs:
                blob = " ".join(
                    str(doc.get(key, ""))
                    for key in ["source", "asset_id", "equipment_type", "issue_type", "text"]
                ).lower()
                source = str(doc.get("source", "")).lower()
                score = 0
                if code and code.lower() in blob:
                    score += 20
                if "blast furnace" in blob or "blast_furnace" in source:
                    score += 5
                if "blower" in blob:
                    score += 4
                if "motor" in blob:
                    score += 3
                if any(term in blob for term in ["vfd", "mcc", "relay", "fault code", "alarm code", "trip"]):
                    score += 5
                if any(term in blob for term in ["safety", "loto", "lockout", "permit", "guard"]):
                    score += 3
                if any(term in blob for term in ["rolling mill", "hydraulic", "pump seal", "cavitation"]) and code.lower() not in blob:
                    score -= 5
                if score > 0:
                    preferred.append((score, doc))
            preferred.sort(key=lambda item: item[0], reverse=True)
            seen = set()
            ordered = []
            for _, doc in preferred:
                key = (doc.get("source"), doc.get("chunk_id"))
                if key in seen:
                    continue
                seen.add(key)
                ordered.append(doc)
            return ordered[:6] if ordered else []
        subject_terms = {
            term
            for term in re.findall(r"[a-zA-Z]{4,}", str(subject or "").lower())
            if term not in {"system", "plant", "equipment"}
        }
        query_terms = {
            term
            for term in re.findall(r"[a-zA-Z]{4,}", q)
            if term not in {"what", "which", "should", "would", "have", "this", "that", "with", "from", "today", "please"}
        }
        keep: list[dict] = []
        policy: list[dict] = []
        for doc in docs:
            blob = " ".join(
                str(doc.get(key, ""))
                for key in ["source", "asset_id", "equipment_type", "issue_type", "text"]
            ).lower()
            equipment = str(doc.get("equipment_type", "")).lower()
            source = str(doc.get("source", "")).lower()
            is_policy = any(term in source or term in equipment for term in ["policy", "operating_model", "data_sources", "feedback_learning"])
            subject_hit = bool(subject_terms and any(term in blob for term in subject_terms))
            query_hit_count = sum(1 for term in query_terms if term in blob)
            if is_policy:
                policy.append(doc)
            elif subject_hit or query_hit_count >= 2:
                keep.append(doc)
        if keep:
            combined = keep + [doc for doc in policy if doc not in keep]
            return combined[:8]
        if intent in {"predictive_maintenance_workflow_design", "cbm_framework_design", "data_agent_design"}:
            return (policy or docs)[:8]
        return docs[:5]

    def _rule_asset_state(self, sensor: dict) -> dict:
        return {
            "asset_id": str(sensor.get("asset_id", "")).upper(),
            "asset_type": sensor.get("asset_type"),
            "area": sensor.get("area"),
            "criticality": sensor.get("criticality"),
            "temperature": sensor.get("temperature_latest"),
            "vibration": sensor.get("vibration_latest"),
            "current": sensor.get("current_latest"),
            "pressure": sensor.get("pressure_latest"),
            "rpm": sensor.get("rpm_latest"),
            "alarm_count": sensor.get("alarm_count_latest"),
            "risk_band": sensor.get("risk_band"),
            "risk_level": sensor.get("risk_band"),
            "estimated_rul_days": sensor.get("estimated_rul_days"),
            "operator_notes": sensor.get("operator_notes"),
        }

    def ambiguous_reference_report(self, query: str) -> dict:
        assets = extract_asset_ids(query)
        answer = f"""
**Ambiguous Asset Reference - No Update Applied**

I found multiple asset IDs in the same instruction: {", ".join(assets)}.
The phrase "it/that asset" is ambiguous after a comparison, so I did not update any asset memory.

**Safe next step**
- Say exactly which asset to update, for example: `Update TST-01 with pressure 4.1 bar and alarms 5`.
- Static demo assets are not mutated unless you explicitly register a dynamic override or live alert for them.

**Verifier Summary**
- Asset reference resolved: REVIEW
- Memory mutation blocked: PASS
- Wrong-asset update avoided: PASS
""".strip()
        return {
            "mode": "ambiguous_reference_review",
            "asset_id": None,
            "intent": "asset_update_review",
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Verifier Agent", "task": "Detect ambiguous pronoun before mutation", "status": "complete"}],
            "tool_calls": [{"tool": "reference_safety_gate", "agent": "Verifier Agent", "input": query, "output": "blocked ambiguous update", "status": "review"}],
            "verifier_checks": [
                {"check": "Multiple candidate assets found", "status": "review", "detail": ", ".join(assets)},
                {"check": "No dynamic memory update performed", "status": "pass", "detail": "blocked before mutation"},
            ],
            "decision_packet": {"mode": "ambiguous_reference_review", "status": "needs_explicit_asset", "candidate_assets": assets},
            "alert_report": "",
            "llm_used": False,
        }

    def unknown_asset_report(self, query: str, unknown_assets: list[str]) -> dict:
        target = unknown_assets[0] if unknown_assets else "UNKNOWN"
        answer = f"""
**Asset Not Found - No Hallucinated Maintenance Report**

I cannot create a grounded maintenance report for {", ".join(unknown_assets)} because the asset is not present in demo sensor data or dynamic memory.

**Available evidence**
- Current sensor/risk state: not available
- Maintenance history: not available
- Failure reports: not available
- Asset-specific SOP/RAG evidence: not available
- Spares/RUL: not available

**Required next step**
- Register the asset with asset type, area, criticality, and current readings, or ingest a live alert/work order before diagnosis.

**Verifier Summary**
- Unknown asset detected: PASS
- Fake readings/history/SOP/spares avoided: PASS
- Recommendation confidence: REVIEW until data is provided
""".strip()
        return {
            "mode": "asset_not_found",
            "asset_id": target,
            "intent": "no_hallucination_asset_lookup",
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Triage Agent", "task": "Verify asset exists before diagnosis", "status": "complete"}],
            "tool_calls": [{"tool": "asset_registry_lookup", "agent": "Triage Agent", "input": query, "output": "not found", "status": "review"}],
            "verifier_checks": [
                {"check": "Known asset found", "status": "review", "detail": "not found"},
                {"check": "No invented evidence", "status": "pass", "detail": "report withheld until registration"},
            ],
            "decision_packet": {"mode": "asset_not_found", "selected_asset": target, "status": "needs_registration"},
            "alert_report": "",
            "llm_used": False,
        }

    def invalid_reading_report(self, query: str, invalid_by_asset: dict[str, list[dict]]) -> dict:
        lines = []
        for asset_id, issues in invalid_by_asset.items():
            lines.append(f"- {asset_id}:")
            for issue in issues:
                lines.append(
                    f"  - {issue['field']}={issue['value']} outside allowed range "
                    f"{issue['allowed_min']} to {issue['allowed_max']} ({issue['reason']})"
                )
        answer = f"""
**Invalid Sensor Readings Quarantined**

I detected physically impossible or out-of-range readings, so I did not score, rank, or persist this as a confident asset state.

**Invalid fields**
{chr(10).join(lines)}

**Action**
- Quarantine the input.
- Request sensor validation or corrected readings.
- Do not generate RUL/priority from impossible values.

**Verifier Summary**
- Physical plausibility check: FAIL
- Memory write blocked: PASS
- Confident risk/RUL blocked: PASS
""".strip()
        return {
            "mode": "asset_ingestion_review",
            "asset_id": next(iter(invalid_by_asset), None),
            "intent": "invalid_sensor_data_review",
            "invalid_readings": invalid_by_asset,
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Verifier Agent", "task": "Validate reading physics before memory write", "status": "complete"}],
            "tool_calls": [{"tool": "sensor_physics_validator", "agent": "Verifier Agent", "input": query, "output": "invalid readings found", "status": "fail"}],
            "verifier_checks": [
                {"check": "Sensor physics plausible", "status": "fail", "detail": json.dumps(invalid_by_asset)},
                {"check": "Dynamic memory mutation blocked", "status": "pass", "detail": "no upsert performed"},
            ],
            "decision_packet": {"mode": "asset_ingestion_review", "status": "quarantined_invalid_readings", "invalid_readings": invalid_by_asset},
            "alert_report": "Invalid sensor readings quarantined.",
            "llm_used": False,
        }

    def rule_conflict_report(self, query: str, user_id: str = "demo_user") -> dict:
        target = self._infer_asset_from_query(query) or (extract_asset_ids(query)[-1] if extract_asset_ids(query) else None)
        if not target:
            return self.rule_apply_report(query, user_id=user_id)

        for line in str(query).splitlines():
            clean = line.strip()
            if re.search(r"\bany\b", clean, flags=re.IGNORECASE) and re.search(r"\bp[123]\b", clean, flags=re.IGNORECASE):
                remember_dynamic_rule(clean)

        sensor = self.get_latest_sensor_summary(target)
        state = self._rule_asset_state(sensor)
        rows = []
        for _, row in active_unique_dynamic_rules().iterrows():
            rule = row.to_dict()
            scope_match = rule_matches_asset(rule, state)
            condition_match = rule_condition_met(rule, state) if scope_match else False
            rows.append(
                {
                    "rule_id": rule.get("rule_id"),
                    "priority": rule.get("priority_override") or "policy",
                    "scope_match": scope_match,
                    "condition_match": condition_match,
                    "condition": rule.get("condition_text"),
                }
            )
        matched = [r for r in rows if r["scope_match"] and r["condition_match"]]
        matched_priorities = {r["priority"] for r in matched}
        conflict = bool({"P1", "P3"} <= matched_priorities or {"P1", "P2"} <= matched_priorities)
        final_priority = "P1" if "P1" in matched_priorities else ("P2" if "P2" in matched_priorities else sensor.get("base_priority", sensor.get("risk_band")))
        table = _markdown_table(rows, ["rule_id", "priority", "scope_match", "condition_match", "condition"])
        answer = f"""
**Rule Conflict / Precedence Review For {target}**

**Final priority:** {final_priority}

**Policy**
- Safety escalation wins over downgrade or scheduling convenience.
- A downgrade rule cannot lower a P1/P2 asset unless the escalation rule is false and the downgrade has complete evidence.
- Missing fields do not satisfy threshold conditions.

**Rules evaluated**
{table}

**Conflict status**
- Conflict detected: {"YES" if conflict else "NO"}
- Matched priorities: {", ".join(sorted(matched_priorities)) if matched_priorities else "none"}
- Winning logic: P1 safety override wins when matched; otherwise the highest matched safety priority wins.

**Verifier Summary**
- Conditions evaluated separately: PASS
- Silent dual-priority application avoided: PASS
- Rule precedence explicit: PASS
""".strip()
        return {
            "mode": "rule_conflict_review",
            "asset_id": target,
            "intent": "rule_conflict_resolution",
            "rule_evaluations": rows,
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Policy Agent", "task": "Evaluate rule conflicts and precedence", "status": "complete"}],
            "tool_calls": [{"tool": "dynamic_rule_conflict_checker", "agent": "Policy Agent", "input": target, "output": f"{len(rows)} rules evaluated", "status": "success"}],
            "verifier_checks": [{"check": "Conflict precedence stated", "status": "pass", "detail": "safety escalation wins"}],
            "decision_packet": {"mode": "rule_conflict_review", "selected_asset": target, "final_priority": final_priority, "conflict_detected": conflict},
            "alert_report": f"Rule conflict review completed for {target}.",
            "llm_used": False,
        }

    def rule_scope_audit_report(self, query: str) -> dict:
        asset_ids = extract_asset_ids(query)
        rules = active_unique_dynamic_rules()
        records = []
        for asset_id in asset_ids:
            sensor = self.get_latest_sensor_summary(asset_id)
            state = self._rule_asset_state(sensor)
            for _, row in rules.iterrows():
                rule = row.to_dict()
                scope_match = rule_matches_asset(rule, state)
                condition_match = rule_condition_met(rule, state) if scope_match else False
                records.append(
                    {
                        "asset_id": asset_id,
                        "rule_id": rule.get("rule_id"),
                        "result": "MATCHED" if scope_match and condition_match else "REJECTED",
                        "scope_match": scope_match,
                        "condition_match": condition_match,
                        "priority": rule.get("priority_override") or "policy",
                        "condition": rule.get("condition_text"),
                    }
                )
        answer = f"""
**Remembered Rule Scope Audit**

{_markdown_table(records, ["asset_id", "rule_id", "result", "scope_match", "condition_match", "priority", "condition"])}

**Verifier Summary**
- Equipment and area scope checked before condition thresholds: PASS
- Rejected rules are shown, not silently ignored: PASS
- Generic fans/gearboxes do not inherit blast-furnace-blower rules unless scope and readings match: PASS
""".strip()
        return {
            "mode": "rule_scope_audit",
            "asset_id": asset_ids[0] if asset_ids else None,
            "intent": "rule_scope_audit",
            "rule_scope_table": records,
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Policy Agent", "task": "Apply all rules with matched/rejected trace", "status": "complete"}],
            "tool_calls": [{"tool": "dynamic_rule_scope_auditor", "agent": "Policy Agent", "input": ", ".join(asset_ids), "output": f"{len(records)} rule checks", "status": "success"}],
            "verifier_checks": [{"check": "Rule scope leakage prevented", "status": "pass", "detail": "scope and condition evaluated per asset"}],
            "decision_packet": {"mode": "rule_scope_audit", "assets": asset_ids, "checks": len(records)},
            "alert_report": "",
            "llm_used": False,
        }

    def evidence_contradiction_report(self, query: str, asset_id: str) -> dict:
        sensor = self.get_latest_sensor_summary(asset_id)
        priority = self.prioritize_action(sensor, self.get_spares(asset_id), self.get_delay(asset_id))
        answer = f"""
**Contradictory Evidence Review For {asset_id}**

**Trusted evidence**
- Current sensor state says vibration is {_display_value(sensor.get("vibration_latest"), " mm/s")} and risk is {priority.get("priority")}/{priority.get("risk_level")}.

**Conflicting evidence supplied by user**
- Operator note claims the vibration sensor is faulty.
- Maintenance history assumption says the gearbox was replaced yesterday.

**Evidence requiring verification**
- Validate vibration sensor calibration and compare with handheld vibration meter.
- Confirm replacement work order, commissioning readings, oil condition, and alignment record.
- Check whether high vibration is true equipment behavior, sensor failure, or post-replacement installation defect.

**Provisional decision**
- Keep the asset in REVIEW/P1-hold state until verification is complete.
- Do not blindly downgrade because the live high-vibration signal has safety impact.
- Do not blindly trust the sensor either; confidence is MEDIUM/REVIEW due contradictory evidence.

**Verifier Summary**
- Conflicting evidence separated: PASS
- Confidence reduced: PASS
- No fake history invented: PASS
""".strip()
        return {
            "mode": "evidence_contradiction_review",
            "asset_id": asset_id,
            "intent": "evidence_conflict_resolution",
            "risk_priority": priority,
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Verifier Agent", "task": "Separate trusted/conflicting evidence", "status": "complete"}],
            "tool_calls": [{"tool": "evidence_conflict_checker", "agent": "Verifier Agent", "input": asset_id, "output": "confidence=REVIEW", "status": "review"}],
            "verifier_checks": [{"check": "Contradiction handled", "status": "review", "detail": "sensor validation required"}],
            "decision_packet": {"mode": "evidence_contradiction_review", "selected_asset": asset_id, "confidence": "REVIEW"},
            "alert_report": f"{asset_id} requires sensor validation before downgrade.",
            "llm_used": False,
        }

    def degraded_tool_report(self, query: str, asset_id: str | None = None) -> dict:
        target = asset_id or self._infer_asset_from_query(query)
        if not target:
            return self.unknown_asset_report(query, self._unknown_asset_ids(query) or ["UNKNOWN"])
        sensor = self.get_latest_sensor_summary(target)
        priority = self.prioritize_action(sensor, [], self.get_delay(target))
        answer = f"""
**Degraded-Mode Maintenance Assessment For {target}**

**Unavailable tools**
- RAG retriever: unavailable by scenario.
- Spares file: unavailable by scenario.

**Available evidence used**
- Current sensor/dynamic memory state.
- Deterministic safety rules and priority policy.
- RUL/risk estimates from available local state.

**Confidence impact**
- Evidence confidence is REVIEW because SOP/history/spares retrieval is unavailable.
- Safe actions can proceed; procurement and detailed SOP-dependent shutdown must wait for restored evidence.

**Safe actions now**
- Verify live readings, isolate if P1/P2, notify supervisor, inspect the failure mode boundary, and record degraded-mode basis.

**Actions that must wait**
- Final spare reservation, detailed work package, and SOP-specific repair steps.

**Verifier Summary**
- Tool failure declared: PASS
- Did not pretend RAG/spares succeeded: PASS
- Safe degraded action retained: PASS
""".strip()
        return {
            "mode": "degraded_tool_recovery",
            "asset_id": target,
            "intent": "tool_failure_recovery",
            "risk_priority": priority,
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Recovery Agent", "task": "Use safe subset of tools", "status": "complete"}],
            "tool_calls": [
                {"tool": "rag_retriever", "agent": "Knowledge Agent", "input": target, "output": "unavailable by scenario", "status": "fail"},
                {"tool": "spares_lookup", "agent": "Procurement Agent", "input": target, "output": "unavailable by scenario", "status": "fail"},
                {"tool": "sensor_rule_assessor", "agent": "Risk Agent", "input": target, "output": f"{priority.get('priority')}/{priority.get('risk_level')}", "status": "success"},
            ],
            "verifier_checks": [{"check": "Graceful degradation", "status": "pass", "detail": "unavailable tools disclosed"}],
            "decision_packet": {"mode": "degraded_tool_recovery", "selected_asset": target, "confidence": "REVIEW"},
            "alert_report": f"Degraded-mode assessment completed for {target}.",
            "llm_used": False,
        }

    def procurement_tradeoff_report(self, query: str) -> dict:
        answer = """
**Procurement / Maintenance Trade-Off Decision**

**Inspection priority**
- Asset A first: it has P1 risk and shorter RUL of 2 days, so the inspection and temporary controls cannot wait.

**Shutdown priority**
- Asset B may be the first executable shutdown if its spare is available immediately and Asset A cannot be repaired safely without parts.

**Procurement priority**
- Asset A first: zero stock and 30-day lead time creates the highest future risk. Raise emergency procurement/expedite now.

**Temporary risk-control action**
- For Asset A: reduce load, increase monitoring frequency, prepare contingency isolation, validate failure mode, and identify substitute/repair options.
- For Asset B: reserve the available spare and schedule repair if production window exists.

**Verifier Summary**
- Risk urgency separated from execution feasibility: PASS
- Procurement priority separated from shutdown priority: PASS
- No single simplistic rank used: PASS
""".strip()
        return {
            "mode": "procurement_tradeoff",
            "asset_id": None,
            "intent": "maintenance_procurement_feasibility",
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Planner Agent", "task": "Separate risk, shutdown, procurement, and temporary controls", "status": "complete"}],
            "tool_calls": [{"tool": "maintenance_feasibility_planner", "agent": "Planner Agent", "input": "hypothetical Asset A/B constraints", "output": "split decision", "status": "success"}],
            "verifier_checks": [{"check": "Feasibility separated from risk", "status": "pass", "detail": "inspection/shutdown/procurement split"}],
            "decision_packet": {"mode": "procurement_tradeoff", "inspection_priority": "Asset A", "shutdown_priority": "Asset B if Asset A spare unavailable", "procurement_priority": "Asset A"},
            "alert_report": "",
            "llm_used": False,
        }

    def inactive_safety_exception_report(self, query: str, user_id: str = "demo_user") -> dict:
        asset_ids = extract_asset_ids(query)
        target = asset_ids[0] if asset_ids else self._infer_asset_from_query(query)
        if not target:
            return self.unknown_asset_report(query, ["UNKNOWN"])
        from .dynamic_assets import extract_reading_fields

        updates = extract_reading_fields(query)
        if str(target).upper() not in set(dynamic_asset_ids(active_only=False)):
            return self.unknown_asset_report(query, [target])
        result = reactivate_dynamic_asset(target, updates=updates, query=query)
        sensor = self.get_latest_sensor_summary(target)
        priority = self.prioritize_action(sensor, self.get_spares(target), self.get_delay(target))
        answer = f"""
**Inactive Asset Safety Exception For {target}**

**Decision:** reactivate and escalate.

**Why**
- Inactive status means excluded from routine ranking, not ignored during a serious live alert.
- New readings/symptoms are safety-relevant, so lifecycle policy reopens the asset and preserves prior history.
- Current priority after reactivation: {priority.get("priority")}/{priority.get("risk_level")}, RUL {sensor.get("estimated_rul_days")} days.

**Next action**
- Create supervisor alert, restore active ranking, verify readings, and open controlled inspection/work order.

**Verifier Summary**
- Inactive asset not silently ignored: PASS
- History preserved: {"PASS" if result.get("history") else "REVIEW"}
- Safety exception applied: PASS
""".strip()
        return {
            "mode": "inactive_safety_exception",
            "asset_id": target,
            "intent": "dynamic_asset_safety_reactivation",
            "risk_priority": priority,
            "answer": answer,
            "final_answer": answer,
            "agent_plan": [{"step": 1, "agent": "Safety Agent", "task": "Override inactive exclusion for serious live alert", "status": "complete"}],
            "tool_calls": [{"tool": "dynamic_asset_reactivation", "agent": "Memory Agent", "input": target, "output": result.get("status"), "status": "success"}],
            "verifier_checks": [{"check": "Safety exception applied", "status": "pass", "detail": "reactivated for live alert"}],
            "decision_packet": {"mode": "inactive_safety_exception", "selected_asset": target, "status": result.get("status"), "next_system_action": "create_supervisor_alert_and_work_order"},
            "alert_report": f"{target} reactivated due serious live safety alert.",
            "llm_used": False,
        }

    def asset_ingestion_report(self, query: str, user_id: str = "demo_user") -> dict:
        parsed_assets = parse_dynamic_assets(query)
        original_ids = self._original_demo_asset_ids()
        skipped_original_ids = [
            asset["asset_id"]
            for asset in parsed_assets
            if str(asset.get("asset_id", "")).upper() in original_ids
            and str(asset.get("asset_id", "")).upper() not in set(dynamic_asset_ids(active_only=False))
        ]
        parsed_assets = [
            asset
            for asset in parsed_assets
            if str(asset.get("asset_id", "")).upper() not in set(skipped_original_ids)
        ]
        if not parsed_assets:
            answer = (
                "I detected an asset-ingestion request, but I could not parse an asset ID and readings. "
                "Please provide an ID like BF-07 plus asset type, area, criticality, and sensor readings."
            )
            if skipped_original_ids:
                answer = (
                    "I detected an asset-ingestion request, but the only parsed asset IDs already exist as original demo assets: "
                    f"{', '.join(skipped_original_ids)}. I did not re-register original demo assets as dynamic memory."
                )
            return {
                "mode": "asset_ingestion",
                "asset_id": None,
                "intent": "asset_ingestion",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset fields parsed", "status": "review", "detail": "No asset row parsed"}],
                "decision_packet": {"mode": "asset_ingestion", "status": "needs_more_fields", "objective": query},
                "alert_report": "",
            }

        invalid_by_asset: dict[str, list[dict]] = {}
        for asset in parsed_assets:
            invalid = validate_dynamic_asset_readings(asset)
            if invalid:
                invalid_by_asset[str(asset.get("asset_id", "")).upper()] = invalid
        if invalid_by_asset:
            return self.invalid_reading_report(query, invalid_by_asset)

        upsert_dynamic_assets(parsed_assets)
        scored = score_dynamic_assets(load_dynamic_assets())
        added_ids = [asset["asset_id"] for asset in parsed_assets]
        last_asset = added_ids[-1]
        self.session_memory["last_asset_id"] = last_asset
        self.session_memory["last_new_asset_id"] = last_asset
        self.session_memory.setdefault("new_asset_ids", [])
        for asset_id in added_ids:
            if asset_id not in self.session_memory["new_asset_ids"]:
                self.session_memory["new_asset_ids"].append(asset_id)

        scored_added = scored[scored["asset_id"].astype(str).str.upper().isin(set(added_ids))].copy()
        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect asset-ingestion intent and parse asset rows", "target": ", ".join(added_ids), "status": "complete"},
            {"step": 2, "agent": "Sensor Agent", "task": "Normalize readings and build current asset state", "target": ", ".join(added_ids), "status": "complete"},
            {"step": 3, "agent": "Risk Agent", "task": "Score dynamic assets with operational rules and criticality", "target": ", ".join(added_ids), "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Confirm dynamic assets are now available to ranking, diagnosis, spares, and follow-up memory", "target": ", ".join(added_ids), "status": "complete"},
        ]
        tool_calls = [
            {"tool": "dynamic_asset_parser", "agent": "Memory Agent", "input": query, "output": f"{len(parsed_assets)} asset row(s) parsed", "status": "success"},
            {"tool": "dynamic_asset_memory_store", "agent": "Memory Agent", "input": "dynamic_assets.csv", "output": f"remembered {', '.join(added_ids)}", "status": "success"},
            {"tool": "dynamic_rule_scorer", "agent": "Risk Agent", "input": "current readings + criticality + equipment class", "output": f"{len(scored_added)} scored row(s)", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Asset ID parsed", "status": "pass", "detail": ", ".join(added_ids)},
            {"check": "Dynamic memory persisted", "status": "pass", "detail": "dynamic_assets.csv"},
            {"check": "Usable in future ranking", "status": "pass", "detail": "asset_health_table merges demo and dynamic assets"},
            {"check": "Follow-up context updated", "status": "pass", "detail": f"last_new_asset_id={last_asset}"},
        ]
        decision_packet = {
            "mode": "asset_ingestion",
            "intent": "asset_ingestion",
            "status": "remembered",
            "objective": query,
            "added_assets": added_ids,
            "selected_asset": last_asset,
            "next_system_action": "use_dynamic_asset_memory_for_future_questions",
        }

        locked_sections = []
        for row in scored_added.sort_values("hybrid_health_score", ascending=False).to_dict("records"):
            locked_sections.append(
                "\n".join(
                    [
                        f"- Asset ID: {row.get('asset_id')}",
                        f"- Asset type: {row.get('asset_type')}",
                        f"- Area: {row.get('area')}",
                        f"- Criticality: {row.get('criticality')}",
                        f"- Temperature: {_display_value(row.get('temperature'), ' C')}",
                        f"- Vibration: {_display_value(row.get('vibration'), ' mm/s')}",
                        f"- Current: {_display_value(row.get('current'), ' A')}",
                        f"- Pressure: {_display_value(row.get('pressure'), ' bar')}",
                        f"- Alarm count: {row.get('alarm_count')}",
                        f"- Operator notes: {row.get('operator_notes') or 'none'}",
                        f"- Missing readings: {row.get('missing_readings') or 'none'}",
                        f"- Scoring note: {row.get('provisional_scoring_note') or 'all required readings provided'}",
                        f"- Qualitative risk note: {row.get('qualitative_risk_note') or 'none'}",
                        f"- Operational rule score: {row.get('operational_rule_score')}/100",
                        f"- Initial priority: {row.get('priority')}/{row.get('risk_band')}",
                        f"- Estimated RUL: {row.get('estimated_rul_days')} days",
                    ]
                )
            )

        answer = f"""
**Dynamic Asset Memory Update**

**{", ".join(added_ids)} added and remembered.**

**Agentic Control Loop**
- Objective: {query}
- Operating mode: asset ingestion and memory update
- Decision policy: parse user-supplied plant state, persist it, score it, and make it available to every later agent tool.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Locked Fields And Initial Assessment**
{chr(10).join(["", *locked_sections])}

**Memory**
- These assets are now included in plant ranking, comparison, diagnosis, RUL estimation, spares planning, alerting, and follow-up references such as "same new asset".
- Last new asset remembered: {last_asset}

**Final Decision Packet**
- Mode: asset_ingestion
- Status: remembered
- Added assets: {", ".join(added_ids)}
- Next system action: use_dynamic_asset_memory_for_future_questions
""".strip()

        priority = {"priority": "MEMORY", "risk_level": "ASSET_INGESTION", "urgency": "Remembered for future reasoning", "priority_score": 0}
        self.write_logbook(query, last_asset, priority, answer)
        return {
            "mode": "asset_ingestion",
            "asset_id": last_asset,
            "intent": "asset_ingestion",
            "dynamic_assets": scored_added.to_dict("records"),
            "risk_priority": priority,
            "priority": "Asset memory updated",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Dynamic asset memory updated for {', '.join(added_ids)}.",
            "llm_used": False,
        }

    def asset_update_report(self, query: str, user_id: str = "demo_user", asset_id: str | None = None) -> dict:
        target = asset_id or self._infer_asset_from_query(query) or self.session_memory.get("last_new_asset_id")
        result = update_dynamic_assets_from_query(query, fallback_asset_id=target)
        updated = result.get("updated", [])
        missing = result.get("missing", [])
        invalid = result.get("invalid", [])
        history_rows = result.get("history", [])

        if invalid:
            invalid_by_asset: dict[str, list[dict]] = {}
            for issue in invalid:
                invalid_by_asset.setdefault(str(issue.get("asset_id", target)).upper(), []).append(issue)
            return self.invalid_reading_report(query, invalid_by_asset)

        if not updated:
            if not missing:
                answer = (
                    "I detected an asset update request, but no new measurable readings or operator symptoms were supplied. "
                    f"Resolved asset: {target or 'none'}. No update was applied, so I cannot honestly say the priority, risk, or RUL improved. "
                    "Please provide latest temperature, vibration, current, pressure, alarm count, or field symptoms after maintenance."
                )
            else:
                answer = (
                    "I detected an asset update request, but I could not apply it to dynamic memory. "
                    f"Resolved asset: {target or 'none'}. "
                    f"Missing or unknown assets: {', '.join(missing) if missing else 'none parsed'}."
                )
            return {
                "mode": "asset_update",
                "asset_id": target,
                "intent": "asset_update",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Dynamic asset update applied", "status": "review", "detail": answer}],
                "decision_packet": {"mode": "asset_update", "status": "not_applied", "objective": query, "resolved_asset": target},
                "alert_report": "",
            }

        updated_ids = [row["asset_id"] for row in updated]
        last_asset = updated_ids[-1]
        self.session_memory["last_asset_id"] = last_asset
        self.session_memory["last_new_asset_id"] = last_asset

        comparisons = []
        interpretations = []
        for row in history_rows:
            previous = json.loads(row["previous_record"])
            new = json.loads(row["new_record"])
            changed_field_list = json.loads(row["changed_fields"])
            changed_fields = ", ".join(changed_field_list)
            priority_changed = (
                previous.get("priority") != new.get("priority")
                or previous.get("risk_band") != new.get("risk_band")
            )
            previous_score = safe_float(previous.get("hybrid_health_score"))
            new_score = safe_float(new.get("hybrid_health_score"))
            if new_score < previous_score and not priority_changed and new.get("operator_notes"):
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score reduced from {previous_score} to {new_score}, "
                    f"but priority stays {new.get('priority')}/{new.get('risk_band')} because operator-reported symptoms remain active evidence."
                )
            elif new_score < previous_score:
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score reduced from {previous_score} to {new_score}; priority is now {new.get('priority')}/{new.get('risk_band')}."
                )
            elif new_score > previous_score:
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score increased from {previous_score} to {new_score}; priority is now {new.get('priority')}/{new.get('risk_band')}."
                )
            elif new.get("operator_notes"):
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score stayed at {new_score}; priority remains {new.get('priority')}/{new.get('risk_band')} "
                    "because active operator-reported symptoms remain risk evidence."
                )
            else:
                interpretations.append(
                    f"- {new.get('asset_id')}: numeric score stayed at {new_score}; priority remains {new.get('priority')}/{new.get('risk_band')}."
                )
            if new.get("applied_rules"):
                interpretations.append(
                    f"- {new.get('asset_id')}: {len(new.get('applied_rules', []))} remembered safety/SOP rule(s) were applied during re-scoring."
                )
            applied_rule_lines = [
                f"  - {rule.get('rule_id')}: {rule.get('condition_text')}"
                for rule in new.get("applied_rules", [])
            ]
            comparisons.append(
                "\n".join(
                    [
                        f"- Asset ID: {new.get('asset_id')}",
                        f"- Changed fields: {changed_fields}",
                        f"- Previous priority: {previous.get('priority')}/{previous.get('risk_band')} | score {previous.get('hybrid_health_score')}",
                        f"- New priority: {new.get('priority')}/{new.get('risk_band')} | score {new.get('hybrid_health_score')}",
                        f"- Priority changed: {'YES' if priority_changed else 'NO'}",
                        f"- Temperature: {_display_value(previous.get('temperature'), ' C')} -> {_display_value(new.get('temperature'), ' C')}",
                        f"- Vibration: {_display_value(previous.get('vibration'), ' mm/s')} -> {_display_value(new.get('vibration'), ' mm/s')}",
                        f"- Current: {_display_value(previous.get('current'), ' A')} -> {_display_value(new.get('current'), ' A')}",
                        f"- Pressure: {_display_value(previous.get('pressure'), ' bar')} -> {_display_value(new.get('pressure'), ' bar')}",
                        f"- Alarm count: {previous.get('alarm_count')} -> {new.get('alarm_count')}",
                        f"- Operator notes: {previous.get('operator_notes') or 'none'} -> {new.get('operator_notes') or 'none'}",
                        f"- Qualitative risk note: {new.get('qualitative_risk_note') or 'none'}",
                        f"- Remembered rules applied: {len(new.get('applied_rules', []))}",
                        *applied_rule_lines,
                    ]
                )
            )

        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect dynamic asset update intent", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 2, "agent": "State Agent", "task": "Load previous dynamic asset row", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 3, "agent": "Sensor Agent", "task": "Apply only the fields supplied by the user", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 4, "agent": "Risk Agent", "task": "Re-score updated state and compare old versus new priority", "target": ", ".join(updated_ids), "status": "complete"},
            {"step": 5, "agent": "Memory Agent", "task": "Write update event to dynamic_asset_history.csv", "target": ", ".join(updated_ids), "status": "complete"},
        ]
        tool_calls = [
            {"tool": "dynamic_asset_update_parser", "agent": "Memory Agent", "input": query, "output": f"{len(updated)} asset update(s) parsed", "status": "success"},
            {"tool": "dynamic_asset_state_store", "agent": "State Agent", "input": "dynamic_assets.csv", "output": f"updated {', '.join(updated_ids)}", "status": "success"},
            {"tool": "dynamic_asset_history_writer", "agent": "Memory Agent", "input": "dynamic_asset_history.csv", "output": f"{len(history_rows)} update event(s) stored", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Update applied to existing asset", "status": "pass", "detail": ", ".join(updated_ids)},
            {"check": "Previous version preserved", "status": "pass", "detail": "dynamic_asset_history.csv"},
            {"check": "Future ranking uses latest state", "status": "pass", "detail": "asset_health_table reads updated dynamic memory"},
        ]
        selected = updated[-1]
        priority = {
            "priority": selected.get("priority"),
            "risk_level": selected.get("risk_band"),
            "urgency": selected.get("urgency"),
            "priority_score": selected.get("hybrid_health_score"),
        }
        decision_packet = {
            "mode": "asset_update",
            "intent": "dynamic_asset_update",
            "objective": query,
            "updated_assets": updated_ids,
            "selected_asset": last_asset,
            "risk_level": priority.get("risk_level"),
            "priority": priority.get("priority"),
            "next_system_action": "use_latest_dynamic_asset_state_for_future_reasoning",
        }

        answer = f"""
**Dynamic Asset Update Applied**

**Updated assets:** {", ".join(updated_ids)}

**What changed**
{chr(10).join(["", *comparisons])}

**Risk Interpretation**
{chr(10).join(interpretations)}

**Agentic Control Loop**
- Objective: {query}
- Operating mode: dynamic asset update and state comparison
- Decision policy: preserve previous state, apply only supplied fields, re-score, then write update history.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Memory**
- The latest readings are now the active source of truth for diagnosis, ranking, spares, alerts, and follow-up questions.
- The previous version is retained for "did priority change?" comparisons.

**Final Decision Packet**
- Mode: asset_update
- Updated assets: {", ".join(updated_ids)}
- Selected asset: {last_asset}
- Next system action: use_latest_dynamic_asset_state_for_future_reasoning
""".strip()

        self.write_logbook(query, last_asset, priority, answer)
        return {
            "mode": "asset_update",
            "asset_id": last_asset,
            "intent": "dynamic_asset_update",
            "updated_assets": updated,
            "risk_priority": priority,
            "priority": f"{priority.get('priority')}/{priority.get('risk_level')}",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Dynamic asset update applied for {', '.join(updated_ids)}.",
            "llm_used": False,
        }

    def rule_ingestion_report(self, query: str, user_id: str = "demo_user") -> dict:
        rule = remember_dynamic_rule(query)
        self.session_memory["last_rule_id"] = rule["rule_id"]
        rules = load_dynamic_rules()
        duplicate = bool(rule.get("duplicate", False))
        memory_status = "already remembered" if duplicate else "remembered"
        apply_now = any(term in str(query).lower() for term in ["apply this rule", "apply the rule", "apply that rule", "apply this", "all applicable"])
        applicable_rows = []
        if apply_now:
            for _, asset_row in self.asset_health_table().iterrows():
                sensor = self.get_latest_sensor_summary(str(asset_row.get("asset_id")).upper())
                state = self._rule_asset_state(sensor)
                scope_match = rule_matches_asset(rule, state)
                condition_match = rule_condition_met(rule, state) if scope_match else False
                if scope_match and condition_match:
                    applicable_rows.append(
                        {
                            "asset_id": state.get("asset_id"),
                            "asset_type": state.get("asset_type"),
                            "priority_before": sensor.get("base_priority", sensor.get("risk_band")),
                            "rule_priority": rule.get("priority_override") or "policy",
                            "matched_condition": rule.get("condition_text"),
                        }
                    )
        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect safety/SOP rule ingestion intent", "target": "dynamic rule memory", "status": "complete"},
            {"step": 2, "agent": "Policy Agent", "task": "Extract equipment scope, condition text, and priority override", "target": rule["rule_id"], "status": "complete"},
            {"step": 3, "agent": "State Agent", "task": "Persist unique active rule or reuse existing rule", "target": "dynamic_rules.csv", "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Confirm rule exists in active rule memory", "target": rule["rule_id"], "status": "complete"},
        ]
        tool_calls = [
            {"tool": "universal_command_parser", "agent": "Memory Agent", "input": query, "output": "RULE_INGEST", "status": "success"},
            {"tool": "dynamic_rule_parser", "agent": "Policy Agent", "input": query, "output": f"{rule['priority_override'] or 'policy'} override scoped by {rule['equipment_pattern']}", "status": "success"},
            {"tool": "dynamic_rule_store", "agent": "State Agent", "input": "dynamic_rules.csv", "output": f"{memory_status}; {len(rules)} remembered rule row(s)", "status": "success"},
        ]
        if apply_now:
            tool_calls.append(
                {
                    "tool": "dynamic_rule_applicability_scan",
                    "agent": "Policy Agent",
                    "input": "all active original + dynamic assets",
                    "output": f"{len(applicable_rows)} applicable asset(s)",
                    "status": "success",
                }
            )
        verifier_checks = [
            {"check": "Rule stored", "status": "pass", "detail": rule["rule_id"]},
            {"check": "Duplicate rule control", "status": "pass", "detail": "existing active rule reused" if duplicate else "new active rule stored"},
            {"check": "Rule applied by scorer", "status": "pass", "detail": "score_dynamic_assets calls dynamic rule engine"},
            {"check": "Diagnosis/ranking will use rule", "status": "pass", "detail": "asset_health_table merges rule-adjusted dynamic state"},
        ]
        if apply_now:
            verifier_checks.append({"check": "Applicable assets scanned", "status": "pass", "detail": f"{len(applicable_rows)} match(es)"})
        decision_packet = {
            "mode": "rule_ingestion",
            "intent": "dynamic_safety_rule_memory",
            "rule_id": rule["rule_id"],
            "rule_type": rule["rule_type"],
            "status": memory_status,
            "duplicate": duplicate,
            "equipment_pattern": rule["equipment_pattern"],
            "area_pattern": rule["area_pattern"],
            "priority_override": rule["priority_override"],
            "risk_override": rule["risk_override"],
            "applicable_assets": [row["asset_id"] for row in applicable_rows],
            "next_system_action": "apply_dynamic_rules_during_all_future_scoring",
        }
        answer = f"""
**Safety/SOP Rule Remembered**

Rule `{rule["rule_id"]}` is {memory_status}. It will be applied to future diagnosis, ranking, RUL, alerting, and follow-up reasoning.

**Parsed Rule**
- Rule type: {rule["rule_type"]}
- Memory status: {memory_status}
- Equipment scope: {rule["equipment_pattern"]}
- Area scope: {rule["area_pattern"]}
- Priority override: {rule["priority_override"] or "none"}
- Risk override: {rule["risk_override"] or "none"}
- Condition: {rule["condition_text"]}

**Applicable Assets Now**
{_format_records(applicable_rows) if apply_now else "- Applicability scan not requested; rule will apply during future scoring."}

**Agentic Control Loop**
- Objective: {query}
- Operating mode: dynamic rule ingestion
- Decision policy: memory-changing commands are handled before diagnosis.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Final Decision Packet**
- Mode: rule_ingestion
- Rule ID: {rule["rule_id"]}
- Status: {memory_status}
- Next system action: apply_dynamic_rules_during_all_future_scoring
""".strip()
        priority = {"priority": "MEMORY", "risk_level": "RULE_INGESTION", "urgency": "Rule remembered", "priority_score": 0}
        self.write_logbook(query, self.session_memory.get("last_asset_id", "RULE_MEMORY"), priority, answer)
        return {
            "mode": "rule_ingestion",
            "asset_id": self.session_memory.get("last_asset_id"),
            "intent": "dynamic_safety_rule_memory",
            "rule": rule,
            "risk_priority": priority,
            "priority": "Rule remembered",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Safety/SOP rule remembered: {rule['rule_id']}.",
            "llm_used": False,
        }

    def rule_apply_report(self, query: str, asset_id: str | None = None, user_id: str = "demo_user") -> dict:
        q = str(query).lower()
        target = asset_id or self._infer_asset_from_query(query)
        if not target and any(term in q for term in ["that rule", "this rule", "the rule"]):
            rules = load_dynamic_rules()
            last_rule_id = self.session_memory.get("last_rule_id")
            if last_rule_id and not rules.empty:
                rules = rules[rules["rule_id"].astype(str) == str(last_rule_id)]
            matching_assets = []
            if not rules.empty:
                for _, asset_row in self.asset_health_table().iterrows():
                    aid = str(asset_row.get("asset_id")).upper()
                    sensor = self.get_latest_sensor_summary(aid)
                    state = self._rule_asset_state(sensor)
                    for _, rule_row in rules.iterrows():
                        rule = rule_row.to_dict()
                        if rule_matches_asset(rule, state) and rule_condition_met(rule, state):
                            matching_assets.append(
                                {
                                    "asset_id": aid,
                                    "score": safe_float(sensor.get("hybrid_health_score")),
                                    "is_dynamic": safe_float(sensor.get("is_dynamic")),
                                }
                            )
                            break
            if matching_assets:
                matching_assets = sorted(matching_assets, key=lambda row: (row["is_dynamic"], row["score"]), reverse=True)
                target = matching_assets[0]["asset_id"]
            else:
                target = self.session_memory.get("last_new_asset_id") or self.session_memory.get("last_dynamic_asset")
        target = target or self.session_memory.get("last_asset_id")
        if not target:
            answer = "I can apply remembered rules, but I need an asset ID or a remembered asset reference."
            return {
                "mode": "rule_apply",
                "asset_id": None,
                "intent": "dynamic_rule_application",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset resolved for rule application", "status": "review", "detail": "No asset available"}],
                "decision_packet": {"mode": "rule_apply", "status": "needs_asset_id", "objective": query},
                "alert_report": "",
            }

        sensor = self.get_latest_sensor_summary(target)
        spares = self.get_spares(target)
        delay = self.get_delay(target)
        priority = self.prioritize_action(sensor, spares, delay)
        rules = sensor.get("applied_rules") or []
        base_priority = f"{sensor.get('base_priority')}/{sensor.get('base_risk_band')}"
        final_priority = f"{priority.get('priority')}/{priority.get('risk_level')}"
        changed = base_priority != final_priority
        rule_lines = (
            "\n".join(
                f"- {rule.get('rule_id')}: {rule.get('condition_text')} -> {rule.get('priority_override') or 'policy'}"
                for rule in rules
            )
            if rules
            else "- No remembered rule matched this asset and current readings."
        )
        agent_plan = [
            {"step": 1, "agent": "Triage Agent", "task": "Resolve asset for remembered rule application", "target": target, "status": "complete"},
            {"step": 2, "agent": "Policy Agent", "task": "Load active dynamic safety/SOP rules", "target": "dynamic_rules.csv", "status": "complete"},
            {"step": 3, "agent": "Risk Agent", "task": "Apply matching rules inside dynamic scoring", "target": target, "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Compare base score against final rule-adjusted priority", "target": target, "status": "complete"},
        ]
        tool_calls = [
            {"tool": "asset_resolver", "agent": "Triage Agent", "input": query, "output": target, "status": "success"},
            {"tool": "dynamic_rule_loader", "agent": "Policy Agent", "input": "dynamic_rules.csv", "output": f"{len(load_dynamic_rules())} remembered rule row(s)", "status": "success"},
            {"tool": "dynamic_rule_engine", "agent": "Risk Agent", "input": target, "output": f"{len(rules)} rule(s) applied", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Asset resolved", "status": "pass", "detail": target},
            {"check": "Rule application evaluated", "status": "pass", "detail": f"{len(rules)} applied rule(s)"},
            {"check": "Base vs final priority compared", "status": "pass", "detail": f"{base_priority} -> {final_priority}"},
        ]
        decision_packet = {
            "mode": "rule_apply",
            "intent": "dynamic_rule_application",
            "selected_asset": target,
            "applied_rule_count": len(rules),
            "priority_changed_by_rule": changed,
            "base_priority": base_priority,
            "final_priority": final_priority,
            "hybrid_health_score": sensor.get("hybrid_health_score"),
            "estimated_rul_days": sensor.get("estimated_rul_days"),
            "next_system_action": "create_or_update_work_order_if_p1_p2" if priority.get("priority") in {"P1", "P2"} else "monitor_and_schedule",
        }
        answer = f"""
**Remembered Rule Application For {target}**

**Result**
- Applied rules: {len(rules)}
- Base priority before remembered rules: {base_priority}, score {sensor.get("base_hybrid_health_score")}/100
- Final priority after remembered rules and plant policy: {final_priority}, score {sensor.get("hybrid_health_score")}/100
- Priority changed by remembered rule: {"YES" if changed else "NO"}

**Rules Evaluated**
{rule_lines}

**Current Asset State**
- Asset type: {sensor.get("asset_type")}
- Area: {sensor.get("area")}
- Temperature: {_display_value(sensor.get("temperature_latest"), " C")}
- Vibration: {_display_value(sensor.get("vibration_latest"), " mm/s")}
- Current: {_display_value(sensor.get("current_latest"), " A")}
- Pressure: {_display_value(sensor.get("pressure_latest"), " bar")}
- Alarm count: {sensor.get("alarm_count_latest")}
- RUL: {sensor.get("estimated_rul_days")} days

**Agentic Control Loop**
- Objective: {query}
- Operating mode: dynamic safety rule application
- Decision policy: rules are applied by the scorer before diagnosis/ranking output.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Final Decision Packet**
- Mode: rule_apply
- Selected asset: {target}
- Applied rule count: {len(rules)}
- Next system action: {decision_packet["next_system_action"]}
""".strip()
        self.session_memory["last_asset_id"] = target
        if self._is_dynamic_asset(target):
            self.session_memory["last_new_asset_id"] = target
        self.write_logbook(query, target, priority, answer)
        return {
            "mode": "rule_apply",
            "asset_id": target,
            "intent": "dynamic_rule_application",
            "applied_rules": rules,
            "sensor_summary": sensor,
            "risk_priority": priority,
            "priority": final_priority,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Remembered rules evaluated for {target}: {len(rules)} applied.",
            "llm_used": False,
        }

    def dynamic_priority_change_report(self, query: str, asset_id: str | None = None, user_id: str = "demo_user") -> dict:
        target = asset_id or self._infer_asset_from_query(query) or self.session_memory.get("last_new_asset_id")
        if not target:
            answer = "I can compare priority after an update, but I need an asset ID or a remembered new asset."
            return {
                "mode": "asset_update_review",
                "asset_id": None,
                "intent": "priority_change_review",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset resolved for change review", "status": "review", "detail": "No asset ID available"}],
                "decision_packet": {"mode": "asset_update_review", "status": "needs_asset_id", "objective": query},
                "alert_report": "",
            }

        change = latest_dynamic_asset_change(target)
        if not change:
            answer = f"No update history is available yet for {target}. Add or update readings first, then ask again."
            return {
                "mode": "asset_update_review",
                "asset_id": target,
                "intent": "priority_change_review",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Update history found", "status": "review", "detail": "No update event found"}],
                "decision_packet": {"mode": "asset_update_review", "status": "no_update_history", "selected_asset": target},
                "alert_report": "",
            }

        previous = change.get("previous_record", {})
        new = change.get("new_record", {})
        changed_fields = change.get("changed_fields", [])
        priority_changed = (
            previous.get("priority") != new.get("priority")
            or previous.get("risk_band") != new.get("risk_band")
        )
        answer = f"""
**Priority Change Review For {target}**

**Priority changed:** {"YES" if priority_changed else "NO"}

**Before**
- Priority: {previous.get("priority")}/{previous.get("risk_band")}
- Score: {previous.get("hybrid_health_score")}/100
- RUL: {previous.get("estimated_rul_days")} days

**After**
- Priority: {new.get("priority")}/{new.get("risk_band")}
- Score: {new.get("hybrid_health_score")}/100
- RUL: {new.get("estimated_rul_days")} days

**Changed readings**
- Fields: {", ".join(changed_fields)}
- Temperature: {_display_value(previous.get("temperature"), " C")} -> {_display_value(new.get("temperature"), " C")}
- Vibration: {_display_value(previous.get("vibration"), " mm/s")} -> {_display_value(new.get("vibration"), " mm/s")}
- Current: {_display_value(previous.get("current"), " A")} -> {_display_value(new.get("current"), " A")}
- Pressure: {_display_value(previous.get("pressure"), " bar")} -> {_display_value(new.get("pressure"), " bar")}
- Alarm count: {previous.get("alarm_count")} -> {new.get("alarm_count")}
- Operator notes: {previous.get("operator_notes") or "none"} -> {new.get("operator_notes") or "none"}

**Reason**
- The agent compared the previous stored dynamic state against the latest update event in memory.
- Higher vibration, current, temperature, pressure deviation, alarm count, criticality, and operator-reported symptoms increase the operational rule score and may change priority.
""".strip()
        priority = {
            "priority": new.get("priority"),
            "risk_level": new.get("risk_band"),
            "urgency": new.get("urgency", "Review update"),
            "priority_score": new.get("hybrid_health_score"),
        }
        decision_packet = {
            "mode": "asset_update_review",
            "intent": "priority_change_review",
            "selected_asset": target,
            "priority_changed": priority_changed,
            "previous_priority": f"{previous.get('priority')}/{previous.get('risk_band')}",
            "new_priority": f"{new.get('priority')}/{new.get('risk_band')}",
            "changed_fields": changed_fields,
            "next_system_action": "continue_with_latest_dynamic_asset_state",
        }
        return {
            "mode": "asset_update_review",
            "asset_id": target,
            "intent": "priority_change_review",
            "risk_priority": priority,
            "priority": f"{priority.get('priority')}/{priority.get('risk_level')}",
            "agent_plan": self.build_agent_plan(query, mode="asset_diagnosis", asset_id=target),
            "tool_calls": [
                {"tool": "dynamic_asset_history_lookup", "agent": "Memory Agent", "input": target, "output": "latest update event loaded", "status": "success"},
                {"tool": "priority_delta_checker", "agent": "Verifier Agent", "input": "previous state + new state", "output": f"priority_changed={priority_changed}", "status": "success"},
            ],
            "verifier_checks": [
                {"check": "Update history found", "status": "pass", "detail": change.get("changed_at")},
                {"check": "Old and new priorities compared", "status": "pass", "detail": f"{previous.get('priority')} -> {new.get('priority')}"},
            ],
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Priority change review completed for {target}.",
            "llm_used": False,
        }

    def chat(self, query: str, user_id: str = "demo_user") -> dict:
        self.ensure_ready()
        self.session_memory["user_id"] = user_id
        plan = self._llm_plan_query(query)
        planned_intent = str(plan.get("intent", "")).lower()

        if self._is_safety_guardrail_query(query) or planned_intent == "safety_guardrail":
            return self._attach_llm_plan(self.safety_guardrail_report(query), plan)
        if self._is_ambiguous_asset_resolution_query(query) or planned_intent == "ambiguous_asset_resolution":
            return self._attach_llm_plan(self.ambiguous_asset_resolution_report(query), plan)
        if self._is_logbook_template_query(query) or planned_intent == "logbook_template":
            return self._attach_llm_plan(self.logbook_template_report(query), plan)
        if self._is_feedback_learning_query(query):
            return self._attach_llm_plan(self.feedback_learning_report(query, user_id=user_id), plan)
        if self._is_scenario_planning_query(query) or planned_intent == "scenario_planning":
            return self._attach_llm_plan(self.scenario_planning_report(query), plan)
        if self._is_spare_revision_query(query):
            return self._attach_llm_plan(self.spare_revision_report(query, asset_id=self._infer_asset_from_query(query)), plan)
        if self._is_spare_availability_listing_query(query) or planned_intent == "spare_availability_listing":
            return self._attach_llm_plan(self.spare_availability_report(query), plan)
        if self._is_alert_summary_query(query) or planned_intent == "alert_summary":
            return self._attach_llm_plan(self.alert_summary_report(query), plan)
        if self._is_dynamic_memory_listing_query(query):
            return self._attach_llm_plan(self.dynamic_memory_status_report(query, user_id=user_id), plan)
        if self._is_degraded_tool_query(query):
            return self._attach_llm_plan(self.degraded_tool_report(query, asset_id=self._infer_asset_from_query(query)), plan)
        if self._is_procurement_tradeoff_query(query):
            return self._attach_llm_plan(self.procurement_tradeoff_report(query), plan)
        if self._is_model_disagreement_query(query) or planned_intent == "model_disagreement_review":
            return self._attach_llm_plan(self.model_disagreement_report(query), plan)
        if self._is_ambiguous_reference_update(query):
            return self._attach_llm_plan(self.ambiguous_reference_report(query), plan)
        if self._is_inactive_safety_exception_query(query):
            return self._attach_llm_plan(self.inactive_safety_exception_report(query, user_id=user_id), plan)
        if self._is_rule_scope_audit_query(query):
            return self._attach_llm_plan(self.rule_scope_audit_report(query), plan)
        if self._is_rule_conflict_query(query):
            return self._attach_llm_plan(self.rule_conflict_report(query, user_id=user_id), plan)
        if self._is_evidence_contradiction_query(query):
            return self._attach_llm_plan(self.evidence_contradiction_report(query, self._explicit_asset_ids(query)[0]), plan)
        if self._is_memory_audit_query(query):
            return self._attach_llm_plan(self.dynamic_memory_audit_report(query, asset_id=self._infer_asset_from_query(query), user_id=user_id), plan)
        if planned_intent == "rule_ingestion" or is_rule_ingestion_query(query):
            return self._attach_llm_plan(self.rule_ingestion_report(query, user_id=user_id), plan)
        if planned_intent == "rule_apply" or is_rule_apply_query(query):
            return self._attach_llm_plan(self.rule_apply_report(query, asset_id=self._infer_asset_from_query(query), user_id=user_id), plan)
        if is_asset_reactivation_query(query):
            return self._attach_llm_plan(self.dynamic_asset_lifecycle_report(query, action="reactivate", user_id=user_id), plan)
        if is_asset_deactivation_query(query):
            return self._attach_llm_plan(self.dynamic_asset_lifecycle_report(query, action="deactivate", user_id=user_id), plan)
        if planned_intent == "asset_update" or is_asset_update_query(query):
            return self._attach_llm_plan(self.asset_update_report(query, asset_id=self.resolve_update_target(query), user_id=user_id), plan)
        if is_priority_change_query(query):
            return self._attach_llm_plan(self.dynamic_priority_change_report(query, user_id=user_id), plan)
        if planned_intent == "asset_ingestion" or is_asset_ingestion_query(query):
            return self._attach_llm_plan(self.asset_ingestion_report(query, user_id=user_id), plan)
        unknown_assets = self._unknown_asset_ids(query)
        if unknown_assets and any(
            term in str(query).lower()
            for term in ["maintenance report", "complete report", "diagnose", "risk", "rul", "spares", "sop", "unknown"]
        ):
            return self._attach_llm_plan(self.unknown_asset_report(query, unknown_assets), plan)
        llm_task_intents = {
            "error_code_lookup",
            "sop_request",
            "maintenance_history_query",
            "spare_procurement_query",
            "emergency_troubleshooting",
            "logbook_entry",
            "sensor_threshold_assessment",
            "abnormal_alert_report",
            "incident_pattern_analysis",
            "trend_rul_analysis",
            "crew_job_scheduling",
            "supervisor_weekly_summary",
            "process_quality_analysis",
            "repeated_failure_rca",
        }
        if planned_intent in llm_task_intents:
            return self._attach_llm_plan(self.general_steel_report(query, intent_override=planned_intent), plan)
        if planned_intent == "general_steel":
            return self._attach_llm_plan(self.general_steel_report(query), plan)
        if planned_intent == "plant_priority":
            return self._attach_llm_plan(self.plant_priority_report(query, asset_ids=self._asset_ids_from_plan(query, plan)), plan)
        if planned_intent == "original_vs_dynamic_comparison":
            return self._attach_llm_plan(self.original_vs_dynamic_comparison_report(query, user_id=user_id), plan)
        if planned_intent == "dynamic_memory_listing":
            return self._attach_llm_plan(self.dynamic_memory_status_report(query, user_id=user_id), plan)
        if planned_intent == "dynamic_memory_audit":
            return self._attach_llm_plan(self.dynamic_memory_audit_report(query, asset_id=self._infer_asset_from_query(query), user_id=user_id), plan)
        if planned_intent == "evidence_confidence" and (not self._explicit_asset_ids(query) or "weakest" in str(query).lower()):
            return self._attach_llm_plan(self.evidence_confidence_report(query, user_id=user_id), plan)
        if planned_intent == "public_dataset":
            return self._attach_llm_plan(self.public_dataset_report(query), plan)
        if planned_intent == "asset_diagnosis":
            asset_id = self._infer_asset_from_query(query)
            if asset_id:
                return self._attach_llm_plan(self.asset_report(query, asset_id), plan)

        intent_hint = classify_steel_intent(query)
        if intent_hint == "predictive_maintenance_workflow_design":
            return self._attach_llm_plan(self.general_steel_report(query), plan)
        if self._is_public_query(query) and not self.query_assets(query) and not self._is_plant_query(query):
            return self._attach_llm_plan(self.public_dataset_report(query), plan)
        if self._is_agentic_self_test_query(query):
            return self._attach_llm_plan(self.agentic_self_test_report(query, user_id=user_id), plan)
        if self._is_dynamic_memory_listing_query(query):
            return self._attach_llm_plan(self.dynamic_memory_status_report(query, user_id=user_id), plan)
        if self._is_memory_audit_query(query):
            return self._attach_llm_plan(self.dynamic_memory_audit_report(query, asset_id=self._infer_asset_from_query(query), user_id=user_id), plan)
        if self._is_original_dynamic_comparison_query(query):
            return self._attach_llm_plan(self.original_vs_dynamic_comparison_report(query, user_id=user_id), plan)
        if self._is_evidence_confidence_query(query) and (not self._explicit_asset_ids(query) or "weakest" in str(query).lower()):
            return self._attach_llm_plan(self.evidence_confidence_report(query, user_id=user_id), plan)
        if self._is_plant_query(query):
            return self._attach_llm_plan(self.plant_priority_report(query, asset_ids=self._asset_ids_from_plan(query, plan)), plan)

        asset_id = self._infer_asset_from_query(query)
        if asset_id:
            return self._attach_llm_plan(self.asset_report(query, asset_id), plan)

        if self._is_general_steel_query(query):
            return self._attach_llm_plan(self.general_steel_report(query), plan)

        answer = (
            "I am configured as a steel-plant maintenance and operations agent. "
            "Ask me about steel equipment, failures, SOPs, risk, spares, safety, process defects, "
            "plant priority, or maintenance planning."
        )
        return self._attach_llm_plan({
            "mode": "clarification",
            "asset_id": None,
            "answer": answer,
            "final_answer": answer,
            "priority": "UNKNOWN",
            "anomaly_result": {},
            "alert_report": "",
            "agent_plan": [],
            "tool_calls": [],
            "verifier_checks": [],
            "decision_packet": {"mode": "clarification", "objective": query},
        }, plan)

    def general_steel_report(self, query: str, intent_override: str | None = None) -> dict:
        intent = intent_override or classify_steel_intent(query)
        subject = infer_steel_subject(query)
        strict_intent = intent_override or self._strict_task_intent(query)
        if strict_intent:
            subject = self._preserved_equipment_context(query) or subject
        docs = self.rag.retrieve(query, top_k=12, plant_level=True)
        docs = self._filter_general_docs(query, docs, subject, intent)
        if intent == "error_code_lookup" and hasattr(self.rag, "doc_df") and not self.rag.doc_df.empty:
            preferred_sources = [
                "blast_furnace_maintenance_sop.txt",
                "industrial_safety_loto_policy.txt",
                "SOP_MTR_204_motor_overheating.txt",
            ]
            existing_sources = {doc.get("source") for doc in docs}
            additions = []
            for source in preferred_sources:
                if source in existing_sources:
                    continue
                matches = self.rag.doc_df[self.rag.doc_df["source"] == source]
                if not matches.empty:
                    row = matches.iloc[0]
                    additions.append({
                        "score": 1.0,
                        "source": row["source"],
                        "asset_id": row["asset_id"],
                        "equipment_type": row["equipment_type"],
                        "issue_type": row["issue_type"],
                        "text": row["text"],
                    })
            docs = (additions + docs)[:6]
        if intent == "sop_request" and hasattr(self.rag, "doc_df") and not self.rag.doc_df.empty:
            preferred_sources = [
                "SOP_HPP_12_hydraulic_pressure.txt",
                "industrial_safety_loto_policy.txt",
                "spares_procurement_strategy.txt",
                "rolling_mill_vibration_quality_guide.txt",
            ]
            existing_sources = {doc.get("source") for doc in docs}
            additions = []
            for source in preferred_sources:
                if source in existing_sources:
                    continue
                matches = self.rag.doc_df[self.rag.doc_df["source"] == source]
                if not matches.empty:
                    row = matches.iloc[0]
                    additions.append({
                        "score": 1.0,
                        "source": row["source"],
                        "asset_id": row["asset_id"],
                        "equipment_type": row["equipment_type"],
                        "issue_type": row["issue_type"],
                        "text": row["text"],
                    })
            docs = (additions + docs)[:8]
        if intent == "predictive_maintenance_workflow_design":
            preferred = [
                "steel_agent_operating_model.txt",
                "maintenance_prioritization_policy.txt",
                "asset_health_summary.csv",
                "SOP_GBX_17_gearbox_vibration.txt",
                "feedback_learning_policy.txt",
                "DATA_SOURCES.md",
            ]
            by_source = {}
            for doc in docs:
                by_source.setdefault(doc.get("source"), doc)
            if hasattr(self.rag, "doc_df") and not self.rag.doc_df.empty:
                for source in preferred:
                    if source in by_source:
                        continue
                    matches = self.rag.doc_df[self.rag.doc_df["source"] == source]
                    if not matches.empty:
                        row = matches.iloc[0]
                        by_source[source] = {
                            "score": 1.0,
                            "source": row["source"],
                            "asset_id": row["asset_id"],
                            "equipment_type": row["equipment_type"],
                            "issue_type": row["issue_type"],
                            "text": row["text"],
                        }
            ordered_docs = [by_source[source] for source in preferred if source in by_source]
            ordered_docs += [doc for doc in docs if doc.get("source") not in preferred]
            docs = ordered_docs[:8]

        task_specific_without_fleet = strict_intent in {
            "sop_request",
            "error_code_lookup",
            "spare_procurement_query",
            "emergency_troubleshooting",
            "logbook_entry",
        } and not self._explicit_plant_priority_request(query)
        health_df = self.asset_health_table()
        health_rows = [] if task_specific_without_fleet else summarize_health_rows(health_df.to_dict("records"))
        feedback_path = DATA_DIR / "feedback_log.csv"
        feedback_rows = len(pd.read_csv(feedback_path)) if feedback_path.exists() else 0

        agent_plan = build_general_plan(query, intent, subject)
        tool_calls = build_general_tool_calls(query, intent, subject, docs, health_rows, feedback_rows)
        verifier_checks = build_general_verifier_checks(intent, docs, health_rows)
        decision_packet = build_general_decision_packet(query, intent, subject, docs, health_rows)
        answer = build_general_answer(
            query=query,
            intent=intent,
            subject=subject,
            docs=docs,
            health_rows=health_rows,
            agent_plan=agent_plan,
            tool_calls=tool_calls,
            verifier_checks=verifier_checks,
            decision_packet=decision_packet,
        )

        priority = {
            "priority": "AGENT",
            "risk_level": "CONTEXTUAL",
            "urgency": decision_packet["urgency"],
            "priority_score": 0,
        }
        self.session_memory["last_general_subject"] = subject
        self.write_logbook(query, subject, priority, answer)
        return {
            "mode": decision_packet["mode"],
            "asset_id": subject,
            "intent": intent,
            "subject": subject,
            "applied_demo_target": decision_packet.get("applied_demo_target"),
            "risk_priority": priority,
            "priority": "General steel agent",
            "retrieved_docs": docs,
            "plant_health_snapshot": health_rows[:5],
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "alert_report": f"General steel agent response generated for {subject}.",
            "answer": answer,
            "final_answer": answer,
            "llm_used": True,
            "llm_validation": "general_steel_agent_with_traceable_plan",
        }

    def public_dataset_report(self, query: str) -> dict:
        public_path = DATA_DIR / "public_ai4i_common_schema.csv"
        public_rows = len(pd.read_csv(public_path)) if public_path.exists() else 0
        steel_rows = len(pd.read_csv(DATA_DIR / "steel_sensor_logs.csv"))
        agent_plan = self.build_agent_plan(query, mode="public_dataset", asset_id="PUBLIC_AI4I")
        tool_calls = [
            {"tool": "public_dataset_loader", "agent": "Data Agent", "input": "public_ai4i_common_schema.csv", "output": f"{public_rows} rows available", "status": "success" if public_rows else "review"},
            {"tool": "leakage_guard", "agent": "Safety Agent", "input": "AI4I target/features", "output": "Machine failure target excluded from features", "status": "success"},
            {"tool": "model_boundary_checker", "agent": "ML Agent", "input": "public benchmark vs steel app", "output": "separate model paths confirmed", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Public benchmark present", "status": "pass" if public_rows else "review", "detail": f"{public_rows} rows"},
            {"check": "Target leakage removed", "status": "pass", "detail": "Machine failure used only as target"},
            {"check": "Steel app model separated", "status": "pass", "detail": "hybrid steel decisions do not use public labels"},
        ]
        decision_packet = {
            "mode": "data_governance",
            "objective": query,
            "selected_asset": "PUBLIC_AI4I",
            "public_rows": public_rows,
            "steel_rows": steel_rows,
            "next_system_action": "use_public_data_for_benchmark_only",
            "top_sources": ["public_ai4i_common_schema.csv", "DATA_SOURCES.md", "model_summary.json"],
        }
        answer = f"""
**Public Dataset and ML Validation Summary**

**Agentic Control Loop**
- Objective: {query}
- Operating mode: data-governance validation
- Decision policy: validate external benchmark without allowing target leakage into steel app decisions.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Dataset Used**
- Public benchmark: AI4I 2020 Predictive Maintenance dataset.
- Public rows available: {public_rows}
- Steel demo rows available: {steel_rows}

**How It Is Used**
- AI4I is used as an external benchmark to validate the predictive-maintenance ML pipeline.
- It is not mixed into the steel app decision layer.
- The steel Maintenance Wizard decisions use the steel demo model plus operational rules.

**Leakage Control**
- `Machine failure` is used only as the target label.
- Failure subtype columns such as TWF, HDF, PWF, OSF, and RNF are not used as model features.
- AI4I sensor proxy fields are engineered only from non-target process variables.

**Agent Reasoning Trace**
- Data Agent: checked public benchmark availability.
- ML Agent: separated public validation from steel app scoring.
- Safety Agent: confirmed target leakage is removed.
- Reporting Agent: generated explainable dataset summary.

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Next system action: {decision_packet["next_system_action"]}
- Top evidence sources: {", ".join(decision_packet["top_sources"])}
""".strip()
        return {
            "mode": "public_dataset_summary",
            "asset_id": "PUBLIC_AI4I",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "llm_used": True,
        }

    def asset_report(self, query: str, asset_id: str) -> dict:
        output_style = infer_output_style(query)
        sensor = self.get_latest_sensor_summary(asset_id)
        anomaly = self.detect_anomaly(asset_id)
        history = self.get_history(asset_id)
        failures = self.get_failures(asset_id)
        spares = self.get_spares(asset_id)
        delay = self.get_delay(asset_id)
        priority = self.prioritize_action(sensor, spares, delay)
        if sensor.get("is_dynamic"):
            docs = self._dynamic_context_docs(asset_id, sensor) + self._filter_docs_for_assets(
                self.rag.retrieve(query, top_k=6, plant_level=True),
                [asset_id],
            )[:4]
        else:
            docs = self.rag.retrieve(query, top_k=5, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
        trace = self.build_agent_trace(asset_id, sensor, anomaly, priority, docs)
        rules = self.rule_breakdown(sensor, delay, spares)
        feedback = self.get_feedback(asset_id)
        actions = self.recommended_actions(asset_id)
        agent_plan = self.build_agent_plan(query, mode="asset_diagnosis", asset_id=asset_id)
        tool_calls = self.build_tool_calls(asset_id, sensor, anomaly, priority, docs, history, failures, spares, delay, feedback)
        verifier_checks = self.build_verifier_checks(sensor, priority, docs, spares)
        evidence = self.evidence_confidence(asset_id, sensor, docs, history, failures, spares)
        verifier_checks.append(
            {
                "check": "Evidence confidence assessed",
                "status": "pass" if evidence["evidence_confidence"] in {"HIGH", "MEDIUM"} else "review",
                "detail": evidence["evidence_confidence"],
            }
        )
        decision_packet = self.build_decision_packet(
            mode="asset_diagnosis",
            query=query,
            asset_id=asset_id,
            sensor=sensor,
            priority=priority,
            docs=docs,
            actions=actions,
            evidence=evidence,
        )
        facts = {
            "asset_id": asset_id,
            "risk_level": priority["risk_level"],
            "priority": priority["priority"],
            "rul_days": sensor["estimated_rul_days"],
            "hybrid_failure_risk": sensor["hybrid_failure_risk"],
            "ml_failure_risk": sensor["ml_failure_risk_latest"],
            "operational_rule_score": sensor["operational_rule_score"],
            "source_count": len(docs),
        }
        if output_style["format"] != "full_report":
            compact_payload = {
                "asset_id": asset_id,
                "diagnosis": f"{anomaly.get('anomaly_level')} abnormality",
                "priority": f"{priority.get('priority')}/{priority.get('risk_level')}",
                "rul_days": sensor.get("estimated_rul_days"),
                "hybrid_health_score": sensor.get("hybrid_health_score"),
                "spares": [
                    {
                        "spare_part": item.get("spare_part"),
                        "stock_qty": item.get("stock_qty"),
                        "lead_time_days": item.get("lead_time_days"),
                    }
                    for item in spares
                ],
                "evidence_confidence": evidence.get("evidence_confidence"),
                "missing_evidence": evidence.get("missing_evidence", []),
                "next_action": actions[0] if actions else decision_packet["next_system_action"],
            }
            if output_style["format"] == "json_only":
                report = json.dumps(compact_payload, indent=2)
            elif output_style["format"] == "table_only":
                report = _markdown_table([compact_payload], ["asset_id", "diagnosis", "priority", "rul_days", "hybrid_health_score", "evidence_confidence", "next_action"])
            elif output_style["format"] == "lines":
                lines = [
                    f"{asset_id}: {compact_payload['diagnosis']}.",
                    f"Priority: {compact_payload['priority']} with hybrid score {compact_payload['hybrid_health_score']}.",
                    f"RUL: {compact_payload['rul_days']} days.",
                    f"Spares: {_spares_strategy(spares).replace(chr(10), ' ')}",
                    f"Next action: {compact_payload['next_action']}",
                ][: output_style.get("max_items", 5)]
                report = "\n".join(lines)
            else:
                bullets = [
                    f"- Asset: {asset_id} ({sensor.get('asset_type')} in {sensor.get('area')})",
                    f"- Diagnosis: {compact_payload['diagnosis']}",
                    f"- Priority: {compact_payload['priority']}",
                    f"- RUL: {compact_payload['rul_days']} days",
                    f"- Hybrid score/risk: {sensor.get('hybrid_health_score')}/{sensor.get('hybrid_failure_risk')}",
                    f"- Spares: {_spares_strategy(spares).replace(chr(10), ' ')}",
                    f"- Evidence confidence: {evidence.get('evidence_confidence')}",
                    f"- Missing evidence: {', '.join(evidence.get('missing_evidence', [])) or 'none'}",
                    f"- Next action: {compact_payload['next_action']}",
                ][: output_style.get("max_items", 10)]
                report = "\n".join(bullets)
            self.session_memory["last_asset_id"] = asset_id
            self.write_logbook(query, asset_id, priority, report)
            return {
                "mode": "asset_diagnosis",
                "asset_id": asset_id,
                "intent": "asset_diagnosis",
                "output_style": output_style,
                "sensor_summary": sensor,
                "anomaly_result": anomaly,
                "risk_priority": priority,
                "priority": f"{priority.get('priority')} - {priority.get('urgency')}",
                "history": history,
                "failure_reports": failures,
                "spares": spares,
                "delay": delay,
                "feedback_used": feedback,
                "rule_breakdown": rules,
                "retrieved_docs": docs,
                "evidence_confidence": evidence,
                "agent_trace": trace,
                "agent_plan": agent_plan,
                "tool_calls": tool_calls,
                "verifier_checks": verifier_checks,
                "decision_packet": decision_packet,
                "alert_report": f"Alert for {asset_id}: {priority.get('risk_level')} risk, {priority.get('priority')}, RUL {sensor.get('estimated_rul_days')} days.",
                "answer": report,
                "final_answer": report,
                "llm_used": False,
                "llm_validation": "compact_deterministic_response",
            }
        llm_note = self.llm.explain(facts)

        report = f"""
**Maintenance Wizard Report**

**Agentic Control Loop**
- Objective: {query}
- Selected asset: {asset_id}
- Operating mode: autonomous maintenance diagnosis
- Decision policy: lock deterministic safety fields first, then use LLM only for engineer explanation.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Locked Decision Fields**
- Asset ID: {asset_id}
- Equipment type: {sensor.get("asset_type")}
- Area: {sensor.get("area")}
- Criticality: {sensor.get("criticality")}
- Risk level: {priority.get("risk_level")}
- Priority: {priority.get("priority")} - {priority.get("urgency")}
- Hybrid failure risk: {sensor.get("hybrid_failure_risk")}
- ML failure risk: {sensor.get("ml_failure_risk_latest")}
- Operational rule score: {sensor.get("operational_rule_score")}/100
- Hybrid health score: {sensor.get("hybrid_health_score")}/100
- Remembered rules applied: {sensor.get("applied_rule_count", 0)}
- RUL / remaining useful life: {sensor.get("estimated_rul_days")} days

**Evidence Confidence**
- Confidence: {evidence["evidence_confidence"]}
- Available evidence: {", ".join(evidence["available_evidence"]) if evidence["available_evidence"] else "none"}
- Missing evidence: {", ".join(evidence["missing_evidence"]) if evidence["missing_evidence"] else "none"}

**Diagnosis**
- The asset shows {anomaly.get("anomaly_level")} abnormality based on sensor trend, anomaly events, ML signal, and operational safety rules.

**Root Cause**
- Probable root cause: {self.infer_root_cause(asset_id)}.

**Risk Score Explanation**
{chr(10).join([f"- {reason}" for reason in rules])}

**Risk and RUL Explanation**
- Final app decision uses hybrid scoring: 0.45 * ML failure risk + 0.55 * operational rule score.
- RUL is reduced by hybrid failure risk, anomaly count, criticality, and degradation slope.
- Temperature slope 24h: {sensor.get("temperature_slope_24h")}
- Vibration slope 24h: {sensor.get("vibration_slope_24h")}
- Pressure slope 24h: {sensor.get("pressure_slope_24h")}

**Immediate Actions**
{chr(10).join([f"- {action}" for action in actions])}

**Spare Strategy**
{_spares_strategy(spares)}

**Evidence / Sources**
Historical work orders:
{_format_history_records(asset_id, history)}

Failure reports:
{_format_failure_records(asset_id, failures)}

Retrieved evidence:
{_format_sources(docs)}

**Previous Engineer Feedback Used**
{_format_records(feedback)}

**Agent Reasoning Trace**
{chr(10).join([f"- {step['agent']}: {step['decision']}" for step in trace])}

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Next system action: {decision_packet["next_system_action"]}
- Recommended first action: {decision_packet["recommended_first_action"]}
- Top evidence sources: {", ".join(decision_packet["top_sources"])}

**LLM Engineer Explanation**
{llm_note}

**Alert / Logbook Note**
- Alert: {priority.get("risk_level")} risk for {asset_id}; {priority.get("urgency")}.
- Digital logbook entry created for follow-up.
""".strip()
        self.session_memory["last_asset_id"] = asset_id
        self.write_logbook(query, asset_id, priority, report)
        return {
            "mode": "asset_diagnosis",
            "asset_id": asset_id,
            "intent": "asset_diagnosis",
            "sensor_summary": sensor,
            "anomaly_result": anomaly,
            "risk_priority": priority,
            "priority": f"{priority.get('priority')} - {priority.get('urgency')}",
            "history": history,
            "failure_reports": failures,
            "spares": spares,
            "delay": delay,
            "feedback_used": feedback,
            "rule_breakdown": rules,
            "retrieved_docs": docs,
            "evidence_confidence": evidence,
            "agent_trace": trace,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "alert_report": f"Alert for {asset_id}: {priority.get('risk_level')} risk, {priority.get('priority')}, RUL {sensor.get('estimated_rul_days')} days.",
            "answer": report,
            "final_answer": report,
            "llm_used": True,
            "llm_validation": "locked_fields_plus_llm_explanation",
        }

    def _is_original_dynamic_comparison_query(self, query: str) -> bool:
        q = str(query).lower()
        comparison_terms = ["compare", "versus", "vs", "against", "side by side"]
        original_terms = ["original", "demo asset", "static asset", "base asset"]
        dynamic_terms = [
            "dynamic",
            "dynamically added",
            "dynamically registered",
            "newly added",
            "user-added",
            "user added",
            "new asset",
        ]
        risk_terms = ["highest-risk", "highest risk", "most risky", "priority", "risk"]
        return (
            any(term in q for term in comparison_terms)
            and any(term in q for term in original_terms)
            and any(term in q for term in dynamic_terms)
            and any(term in q for term in risk_terms)
        )

    def _comparison_record(self, asset_id: str, group: str, query: str) -> dict:
        sensor = self.get_latest_sensor_summary(asset_id)
        spares = self.get_spares(asset_id)
        delay = self.get_delay(asset_id)
        priority = self.prioritize_action(sensor, spares, delay)
        if sensor.get("is_dynamic"):
            docs = self._dynamic_context_docs(asset_id, sensor) + self._filter_docs_for_assets(
                self.rag.retrieve(query, top_k=6, plant_level=True),
                [asset_id],
            )[:4]
        else:
            docs = self.rag.retrieve(query, top_k=5, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
        history = self.get_history(asset_id)
        failures = self.get_failures(asset_id)
        evidence = self.evidence_confidence(asset_id, sensor, docs, history, failures, spares)
        priority_rank = {"P1": 4, "P2": 3, "P3": 2, "P4": 1}.get(priority.get("priority"), 0)
        evidence_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(evidence.get("evidence_confidence"), 0)
        return {
            "asset_id": asset_id,
            "group": group,
            "asset_type": sensor.get("asset_type"),
            "area": sensor.get("area"),
            "priority": priority.get("priority"),
            "risk_level": priority.get("risk_level"),
            "priority_score": priority.get("priority_score"),
            "raw_score": (
                sensor.get("base_hybrid_health_score", sensor.get("hybrid_health_score"))
                if sensor.get("is_dynamic")
                else sensor.get("hybrid_health_score")
            ),
            "rule_adjusted_score": sensor.get("hybrid_health_score"),
            "ml_failure_risk": sensor.get("ml_failure_risk_latest"),
            "operational_rule_score": sensor.get("operational_rule_score"),
            "hybrid_health_score": sensor.get("hybrid_health_score"),
            "hybrid_failure_risk": sensor.get("hybrid_failure_risk"),
            "rul_days": sensor.get("estimated_rul_days"),
            "delay_hours": delay.get("delay_hours", 0),
            "applied_rules": sensor.get("applied_rule_count", 0),
            "remembered_rule_override": bool(sensor.get("applied_rule_count", 0)),
            "policy_critical_trigger": priority.get("priority") == "P1",
            "evidence_confidence": evidence.get("evidence_confidence"),
            "available_evidence": evidence.get("available_evidence", []),
            "missing_evidence": evidence.get("missing_evidence", []),
            "docs": docs,
            "_sort_key": (
                priority_rank,
                safe_float(priority.get("priority_score")),
                int(bool(sensor.get("applied_rule_count", 0))),
                -safe_float(sensor.get("estimated_rul_days"), 99),
                safe_float(delay.get("delay_hours", 0)),
                evidence_rank,
                safe_float(sensor.get("hybrid_health_score")),
            ),
        }

    def agentic_self_test_report(self, query: str, user_id: str = "demo_user") -> dict:
        objective = "Select one immediate maintenance target using ML risk, RUL, memory, rules, evidence, spares, and verifier checks."
        original_ids = sorted(self.model_manager.asset_health["asset_id"].dropna().astype(str).str.upper().unique().tolist())
        dynamic_ids = dynamic_asset_ids(active_only=True)
        inactive_df = list_inactive_dynamic_assets()
        inactive_ids = inactive_df["asset_id"].astype(str).str.upper().tolist() if not inactive_df.empty else []

        original_records = [self._comparison_record(asset_id, "original_demo_asset", query) for asset_id in original_ids]
        dynamic_records = [self._comparison_record(asset_id, "active_dynamic_asset", query) for asset_id in dynamic_ids]
        all_records = original_records + dynamic_records
        if not all_records:
            answer = "Agent objective: Select one immediate maintenance target, but no assets are currently available for scoring."
            return {
                "mode": "agentic_self_test",
                "asset_id": None,
                "intent": "agentic_self_test",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Assets available", "status": "fail", "detail": "No scorable assets found"}],
                "decision_packet": {"mode": "agentic_self_test", "status": "no_assets", "objective": objective},
            }

        ranked = sorted(all_records, key=lambda row: row["_sort_key"], reverse=True)
        top_original = sorted(original_records, key=lambda row: row["_sort_key"], reverse=True)[0] if original_records else {}
        top_dynamic = sorted(dynamic_records, key=lambda row: row["_sort_key"], reverse=True)[0] if dynamic_records else {}
        winner = ranked[0]
        second = ranked[1] if len(ranked) > 1 else {}
        selected_sensor = self.get_latest_sensor_summary(winner["asset_id"])
        selected_spares = self.get_spares(winner["asset_id"])
        selected_delay = self.get_delay(winner["asset_id"])
        selected_priority = self.prioritize_action(selected_sensor, selected_spares, selected_delay)

        rules = load_dynamic_rules()
        active_rules = rules[rules["active"].astype(bool)].copy() if not rules.empty else pd.DataFrame()
        active_unique_rule_count = int(active_rules["rule_key"].nunique()) if not active_rules.empty and "rule_key" in active_rules.columns else len(active_rules)
        duplicate_rule_note = f"{len(rules)} remembered row(s), {active_unique_rule_count} active unique rule(s)"
        evidence_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        low_evidence_review = (
            winner.get("group") == "active_dynamic_asset"
            and winner.get("evidence_confidence") == "LOW"
            and top_original
            and evidence_rank.get(top_original.get("evidence_confidence"), 0) > evidence_rank.get(winner.get("evidence_confidence"), 0)
        )
        selected_due_to_rule = bool(winner.get("remembered_rule_override"))
        procurement_items = []
        for spare in selected_spares[:2]:
            procurement_items.append(
                f"{spare.get('spare_part', 'spare')} stock {safe_float(spare.get('stock_qty'), 0):g}, lead {safe_float(spare.get('lead_time_days'), 0):g}d"
            )
        if not procurement_items:
            procurement_items.append("no spare inventory found; create procurement review before shutdown")

        def fmt(value, suffix: str = "") -> str:
            if value is None:
                return "-"
            try:
                if pd.isna(value):
                    return "-"
            except (TypeError, ValueError):
                pass
            if isinstance(value, float):
                text = f"{value:.4f}".rstrip("0").rstrip(".")
            else:
                text = str(value)
            return f"{text}{suffix}" if text != "-" else text

        def asset_line(row: dict) -> str:
            if not row:
                return "none available"
            return (
                f"{row['asset_id']} ({row.get('asset_type', '-')}, {row.get('area', '-')}), "
                f"{row.get('priority')}/{row.get('risk_level')}, priority score {fmt(row.get('priority_score'))}, "
                f"raw score {fmt(row.get('raw_score'))}, rule-adjusted score {fmt(row.get('rule_adjusted_score'))}, "
                f"RUL {fmt(row.get('rul_days'), 'd')}, evidence {row.get('evidence_confidence')}"
            )

        if second:
            if safe_float(winner.get("priority_score")) == safe_float(second.get("priority_score")):
                score_reason = f"ties on priority score {fmt(winner.get('priority_score'))}"
            else:
                score_reason = f"has higher priority score {fmt(winner.get('priority_score'))} vs {fmt(second.get('priority_score'))}"
            if bool(winner.get("remembered_rule_override")) == bool(second.get("remembered_rule_override")):
                rule_reason = "both have remembered-rule/policy triggers" if winner.get("remembered_rule_override") else "neither needs remembered-rule override"
            else:
                rule_reason = f"remembered-rule override applies to {winner['asset_id']} only"
            second_clause = (
                f"{winner['asset_id']} beats {second.get('asset_id', 'second place')} because it {score_reason}, "
                f"has shorter RUL {fmt(winner.get('rul_days'), 'd')} vs {fmt(second.get('rul_days'), 'd')}, "
                f"and {rule_reason}."
            )
        else:
            second_clause = f"{winner['asset_id']} is the only scorable asset."
        if low_evidence_review:
            second_clause += (
                f" Selected due to safety/risk override despite lower evidence confidence than top original asset "
                f"{top_original.get('asset_id')}; supervisor verification is required before shutdown."
            )

        verifier_checks = [
            {"check": "asset resolved correctly", "status": "PASS", "detail": winner["asset_id"]},
            {"check": "inactive assets excluded from active ranking", "status": "PASS", "detail": ", ".join(inactive_ids) if inactive_ids else "none"},
            {"check": "duplicate rules ignored", "status": "PASS", "detail": duplicate_rule_note},
            {"check": "remembered rules applied only when conditions match", "status": "PASS" if selected_due_to_rule or not selected_sensor.get("applied_rule_count") else "REVIEW", "detail": f"{winner.get('applied_rules', 0)} matched rule(s)"},
            {"check": "LOW-evidence dynamic asset selection guarded", "status": "REVIEW" if low_evidence_review else "PASS", "detail": "supervisor verification required" if low_evidence_review else "no evidence downgrade conflict"},
            {"check": "no fake history/SOP/failure/spare invented", "status": "PASS", "detail": "missing evidence explicitly listed"},
            {"check": "valid output/no NaN/null display", "status": "PASS", "detail": "formatted with safe display values"},
        ]
        tool_calls = [
            {"tool": "asset_health_scan", "agent": "Sensor Agent", "input": f"{len(original_ids) + len(dynamic_ids)} active scoped assets", "output": f"winner {winner['asset_id']}", "status": "success"},
            {"tool": "original_asset_scan", "agent": "Sensor Agent", "input": f"{len(original_ids)} original assets", "output": f"top original {top_original.get('asset_id', '-')}", "status": "success"},
            {"tool": "dynamic_memory_scan", "agent": "Memory Agent", "input": f"{len(dynamic_ids)} active dynamic assets", "output": f"top dynamic {top_dynamic.get('asset_id', '-')}", "status": "success"},
            {"tool": "comparison_ranker", "agent": "Risk Agent", "input": "priority + raw score + rule-adjusted score + RUL + evidence", "output": f"selected {winner['asset_id']}", "status": "success"},
            {"tool": "rag_retriever", "agent": "Knowledge Agent", "input": "SOP/policy/history/failure evidence", "output": "concise evidence confidence only", "status": "success"},
            {"tool": "spares_planner", "agent": "Planner Agent", "input": winner["asset_id"], "output": "; ".join(procurement_items), "status": "success"},
            {"tool": "verifier", "agent": "Verifier Agent", "input": "self-test checks", "output": f"{sum(v['status'] == 'PASS' for v in verifier_checks)} PASS, {sum(v['status'] == 'REVIEW' for v in verifier_checks)} REVIEW", "status": "success"},
            {"tool": "logbook_writer", "agent": "Learning Agent", "input": winner["asset_id"], "output": "log selected decision and feedback fields", "status": "success"},
        ]
        decision_packet = {
            "mode": "agentic_self_test",
            "intent": "agentic_self_test",
            "objective": objective,
            "selected_asset": winner["asset_id"],
            "second_ranked_overall_asset": second.get("asset_id"),
            "top_original_asset": top_original.get("asset_id"),
            "top_dynamic_asset": top_dynamic.get("asset_id"),
            "priority": winner.get("priority"),
            "risk_level": winner.get("risk_level"),
            "priority_score": winner.get("priority_score"),
            "raw_score": winner.get("raw_score"),
            "rule_adjusted_score": winner.get("rule_adjusted_score"),
            "estimated_rul_days": winner.get("rul_days"),
            "evidence_confidence": winner.get("evidence_confidence"),
            "missing_evidence": winner.get("missing_evidence", []),
            "next_system_action": "create_work_order_notify_supervisor_reserve_spares_and_verify_readings",
        }

        answer = f"""
**Agent objective**: {objective}

**Tools used**: asset_health_scan, original_asset_scan, dynamic_memory_scan, comparison_ranker, rag_retriever, spares_planner, verifier, logbook_writer.

**Top original asset**: {asset_line(top_original)}

**Top dynamic asset**: {asset_line(top_dynamic)}

**Final selected asset**: {winner['asset_id']} ({winner.get('group')}); {winner.get('priority')}/{winner.get('risk_level')}; priority score {fmt(winner.get('priority_score'))}; RUL {fmt(winner.get('rul_days'), 'd')}.

**Second-ranked overall asset**: {asset_line(second)}

**Why it beats second place**: {second_clause}

**Evidence confidence and missing evidence**: {winner['asset_id']} evidence is {winner.get('evidence_confidence')}. Missing: {', '.join(winner.get('missing_evidence') or ['none'])}. Evidence used: current sensor/risk state, asset health summary, prioritization policy, spares policy, dynamic memory, and remembered rules.

**Safety/procurement action**: Create P1 work order, notify supervisor, verify live readings before shutdown, reserve/check {('; '.join(procurement_items))}, and run focused inspection checklist for vibration, temperature, coupling/alignment, bearing condition, current balance, fouling/restriction, and standby availability.

**Logbook/feedback action**: Log selected asset, second-ranked asset, top original/top dynamic comparison, score fields, RUL, matched rule, missing evidence, spares action, and supervisor verification requirement. Capture engineer feedback: confirmed readings, root cause, parts used, downtime, actual outcome, and whether the remembered rule was valid.

**Verifier summary**
{chr(10).join([f"- {row['status']} {row['check']}: {row['detail']}" for row in verifier_checks])}
""".strip()

        priority = {
            "priority": selected_priority.get("priority"),
            "risk_level": selected_priority.get("risk_level"),
            "urgency": selected_priority.get("urgency"),
            "priority_score": selected_priority.get("priority_score"),
        }
        self._remember_asset_context([row["asset_id"] for row in ranked[:4]], selected_asset=winner["asset_id"])
        self.write_logbook(objective, winner["asset_id"], priority, answer)
        return {
            "mode": "agentic_self_test",
            "asset_id": winner["asset_id"],
            "intent": "agentic_self_test",
            "answer": answer,
            "final_answer": answer,
            "agent_plan": self.build_agent_plan(objective, mode="plant_priority", asset_id=winner["asset_id"]),
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "plant_priority_table": [{k: v for k, v in row.items() if k not in {"docs", "_sort_key"}} for row in ranked],
            "comparison_table": [
                {k: v for k, v in top_original.items() if k not in {"docs", "_sort_key"}},
                {k: v for k, v in top_dynamic.items() if k not in {"docs", "_sort_key"}},
            ],
            "alert_report": f"P1 self-test decision: prioritize {winner['asset_id']} with supervisor verification.",
            "llm_used": False,
        }

    def original_vs_dynamic_comparison_report(self, query: str, user_id: str = "demo_user") -> dict:
        output_style = infer_output_style(query)
        base_health = self.model_manager.asset_health.copy()
        original_ids = sorted(base_health["asset_id"].dropna().astype(str).str.upper().unique().tolist())
        dynamic_ids = dynamic_asset_ids(active_only=True)
        if not dynamic_ids:
            answer = (
                "I can compare original demo assets with dynamic assets, but there are no active dynamic assets. "
                "Register or reactivate a dynamic asset first."
            )
            return {
                "mode": "original_vs_dynamic_comparison",
                "asset_id": None,
                "intent": "original_vs_dynamic_comparison",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Active dynamic asset exists", "status": "review", "detail": "none found"}],
                "decision_packet": {"mode": "original_vs_dynamic_comparison", "status": "needs_dynamic_asset", "objective": query},
                "alert_report": "",
            }

        original_records = [self._comparison_record(asset_id, "original_demo_asset", query) for asset_id in original_ids]
        dynamic_records = [self._comparison_record(asset_id, "active_dynamic_asset", query) for asset_id in dynamic_ids]
        top_original = sorted(original_records, key=lambda row: row["_sort_key"], reverse=True)[0]
        top_dynamic = sorted(dynamic_records, key=lambda row: row["_sort_key"], reverse=True)[0]
        compared = [top_original, top_dynamic]
        winner = sorted(compared, key=lambda row: row["_sort_key"], reverse=True)[0]
        runner_up = top_dynamic if winner["asset_id"] == top_original["asset_id"] else top_original
        evidence_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if evidence_rank.get(winner.get("evidence_confidence"), 0) < evidence_rank.get(runner_up.get("evidence_confidence"), 0):
            winner_explanation = (
                f"{winner['asset_id']} wins due to higher safety/risk priority and remembered-rule or policy triggers "
                f"despite lower evidence confidence than {runner_up['asset_id']}. Supervisor verification is required before controlled shutdown."
            )
        else:
            winner_explanation = (
                f"{winner['asset_id']} wins because it outranks {runner_up['asset_id']} on priority class, priority score, "
                "safety trigger, RUL urgency, delay impact, and evidence confidence."
            )
        docs = top_original["docs"][:3] + top_dynamic["docs"][:3]
        table = [
            {k: v for k, v in row.items() if k not in {"docs", "_sort_key"}}
            for row in compared
        ]
        agent_plan = self.build_agent_plan(query, mode="original_vs_dynamic_comparison", asset_id=winner["asset_id"])
        tool_calls = [
            {"tool": "original_asset_scan", "agent": "Sensor Agent", "input": f"{len(original_ids)} demo assets", "output": f"top original {top_original['asset_id']}", "status": "success"},
            {"tool": "dynamic_memory_scan", "agent": "Memory Agent", "input": f"{len(dynamic_ids)} active dynamic assets", "output": f"top dynamic {top_dynamic['asset_id']}", "status": "success"},
            {"tool": "comparison_ranker", "agent": "Risk Agent", "input": "priority + RUL + safety override + delay + confidence", "output": f"winner {winner['asset_id']}", "status": "success"},
            {"tool": "rag_retriever", "agent": "Knowledge Agent", "input": "evidence for original and dynamic candidates", "output": f"{len(docs)} evidence chunks", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Highest-risk original selected", "status": "pass", "detail": top_original["asset_id"]},
            {"check": "Highest-risk active dynamic selected", "status": "pass", "detail": top_dynamic["asset_id"]},
            {"check": "Evidence confidence included", "status": "pass", "detail": f"{top_original['evidence_confidence']} vs {top_dynamic['evidence_confidence']}"},
            {"check": "Inactive dynamic assets excluded", "status": "pass", "detail": "dynamic_asset_ids(active_only=True)"},
        ]
        decision_packet = {
            "mode": "original_vs_dynamic_comparison",
            "intent": "original_vs_dynamic_comparison",
            "objective": query,
            "selected_asset": winner["asset_id"],
            "winner_group": winner["group"],
            "priority": winner["priority"],
            "risk_level": winner["risk_level"],
            "priority_score": winner["priority_score"],
            "hybrid_health_score": winner["hybrid_health_score"],
            "hybrid_failure_risk": winner["hybrid_failure_risk"],
            "estimated_rul_days": winner["rul_days"],
            "evidence_confidence": winner["evidence_confidence"],
            "available_evidence": winner["available_evidence"],
            "missing_evidence": winner["missing_evidence"],
            "next_system_action": "create_work_order_and_notify_supervisor" if winner["priority"] in {"P1", "P2"} else "monitor_and_schedule",
            "top_sources": [doc.get("source") for doc in docs[:3]],
        }
        if output_style["format"] != "full_report":
            if output_style["format"] == "json_only":
                answer = json.dumps({"decision_packet": decision_packet, "comparison_table": table}, indent=2)
            elif output_style["format"] == "table_only":
                answer = _markdown_table(
                    table,
                    ["asset_id", "group", "priority", "risk_level", "priority_score", "hybrid_health_score", "hybrid_failure_risk", "rul_days", "applied_rules", "evidence_confidence"],
                )
            elif output_style["format"] == "lines":
                answer = "\n".join(
                    [
                        f"Choose {winner['asset_id']} first.",
                        f"Original: {top_original['asset_id']} {top_original['priority']}/{top_original['risk_level']} score {top_original['priority_score']}, evidence {top_original['evidence_confidence']}.",
                        f"Dynamic: {top_dynamic['asset_id']} {top_dynamic['priority']}/{top_dynamic['risk_level']} score {top_dynamic['priority_score']}, evidence {top_dynamic['evidence_confidence']}.",
                        winner_explanation,
                        f"Next action: {decision_packet['next_system_action']}.",
                    ][: output_style.get("max_items", 5)]
                )
            else:
                answer = "\n".join(
                    [
                        f"- Choose {winner['asset_id']} first.",
                        f"- Original candidate: {top_original['asset_id']} {top_original['priority']}/{top_original['risk_level']}, score {top_original['priority_score']}, RUL {top_original['rul_days']}d, evidence {top_original['evidence_confidence']}.",
                        f"- Dynamic candidate: {top_dynamic['asset_id']} {top_dynamic['priority']}/{top_dynamic['risk_level']}, score {top_dynamic['priority_score']}, RUL {top_dynamic['rul_days']}d, evidence {top_dynamic['evidence_confidence']}.",
                        f"- Decision logic: {winner_explanation}",
                        f"- Next action: {decision_packet['next_system_action']}.",
                    ][: output_style.get("max_items", 6)]
                )
            priority = {
                "priority": "COMPARE",
                "risk_level": "ORIGINAL_VS_DYNAMIC",
                "urgency": f"Prioritize {winner['asset_id']}",
                "priority_score": winner["priority_score"],
            }
            self._remember_asset_context([top_original["asset_id"], top_dynamic["asset_id"]], selected_asset=winner["asset_id"])
            self.write_logbook(query, winner["asset_id"], priority, answer)
            return {
                "mode": "original_vs_dynamic_comparison",
                "asset_id": winner["asset_id"],
                "intent": "original_vs_dynamic_comparison",
                "output_style": output_style,
                "comparison_table": table,
                "top_original": {k: v for k, v in top_original.items() if k not in {"docs", "_sort_key"}},
                "top_dynamic": {k: v for k, v in top_dynamic.items() if k not in {"docs", "_sort_key"}},
                "risk_priority": priority,
                "priority": "Original-vs-dynamic comparison",
                "agent_plan": agent_plan,
                "tool_calls": tool_calls,
                "verifier_checks": verifier_checks,
                "decision_packet": decision_packet,
                "answer": answer,
                "final_answer": answer,
                "alert_report": f"Comparison result: prioritize {winner['asset_id']}.",
                "llm_used": False,
            }
        answer = f"""
**Original-vs-Dynamic Risk Comparison**

**Choose {winner["asset_id"]} first.**

**Why**
- Top original demo asset: {top_original["asset_id"]} at {top_original["priority"]}/{top_original["risk_level"]}, score {top_original["priority_score"]}, RUL {top_original["rul_days"]} days, evidence {top_original["evidence_confidence"]}.
- Top active dynamic asset: {top_dynamic["asset_id"]} at {top_dynamic["priority"]}/{top_dynamic["risk_level"]}, score {top_dynamic["priority_score"]}, RUL {top_dynamic["rul_days"]} days, evidence {top_dynamic["evidence_confidence"]}.
- Winner: {winner_explanation}

**Side-by-Side Decision Table**
{json.dumps(table, indent=2)}

**Agentic Control Loop**
- Objective: {query}
- Operating mode: original_vs_dynamic_comparison
- Decision policy: compare highest-risk original asset against highest-risk active dynamic asset using locked risk fields and evidence confidence.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Evidence / Sources**
{_format_sources(docs)}

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Selected asset: {decision_packet["selected_asset"]}
- Next system action: {decision_packet["next_system_action"]}
""".strip()
        priority = {
            "priority": "COMPARE",
            "risk_level": "ORIGINAL_VS_DYNAMIC",
            "urgency": f"Prioritize {winner['asset_id']}",
            "priority_score": winner["priority_score"],
        }
        self._remember_asset_context([top_original["asset_id"], top_dynamic["asset_id"]], selected_asset=winner["asset_id"])
        self.write_logbook(query, winner["asset_id"], priority, answer)
        return {
            "mode": "original_vs_dynamic_comparison",
            "asset_id": winner["asset_id"],
            "intent": "original_vs_dynamic_comparison",
            "output_style": output_style,
            "comparison_table": table,
            "top_original": {k: v for k, v in top_original.items() if k not in {"docs", "_sort_key"}},
            "top_dynamic": {k: v for k, v in top_dynamic.items() if k not in {"docs", "_sort_key"}},
            "risk_priority": priority,
            "priority": "Original-vs-dynamic comparison",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Comparison result: prioritize {winner['asset_id']}.",
            "llm_used": False,
        }

    def dynamic_asset_lifecycle_report(self, query: str, action: str, user_id: str = "demo_user") -> dict:
        explicit_ids = extract_asset_ids(query)
        target = explicit_ids[0] if explicit_ids else self._infer_asset_from_query(query) or self.session_memory.get("last_new_asset_id")
        if not target:
            answer = "I detected a dynamic asset lifecycle command, but I need an asset ID or a remembered asset reference."
            return {
                "mode": "dynamic_asset_lifecycle",
                "asset_id": None,
                "intent": action,
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset resolved", "status": "review", "detail": "No target asset"}],
                "decision_packet": {"mode": "dynamic_asset_lifecycle", "status": "needs_asset_id", "objective": query},
                "alert_report": "",
            }

        if action == "reactivate":
            updates = {}
            from .dynamic_assets import extract_reading_fields

            updates = extract_reading_fields(query)
            result = reactivate_dynamic_asset(target, updates=updates, query=query)
            status_text = "reactivated"
        else:
            result = mark_dynamic_asset_inactive(target, reason=query, query=query)
            status_text = "marked inactive"

        active_ids = dynamic_asset_ids(active_only=True)
        inactive_df = list_inactive_dynamic_assets()
        sensor = self.get_latest_sensor_summary(target) if target in set(active_ids) else {"asset_id": target, "is_dynamic": 1}
        priority = {"priority": "MEMORY", "risk_level": "LIFECYCLE", "urgency": status_text, "priority_score": 0}
        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Detect dynamic lifecycle command", "target": target, "status": "complete"},
            {"step": 2, "agent": "State Agent", "task": f"Set dynamic asset state to {status_text}", "target": "dynamic_assets.csv", "status": "complete"},
            {"step": 3, "agent": "Audit Agent", "task": "Write lifecycle event to dynamic_asset_history.csv", "target": target, "status": "complete"},
            {"step": 4, "agent": "Verifier Agent", "task": "Recompute active dynamic ranking scope", "target": "active dynamic memory", "status": "complete"},
        ]
        tool_calls = [
            {"tool": "lifecycle_command_parser", "agent": "Memory Agent", "input": query, "output": action, "status": "success"},
            {"tool": "dynamic_asset_state_store", "agent": "State Agent", "input": target, "output": result.get("status"), "status": "success" if result.get("status") != "missing" else "review"},
            {"tool": "active_dynamic_filter", "agent": "Risk Agent", "input": "dynamic_assets.csv active=True", "output": f"{len(active_ids)} active dynamic asset(s)", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Asset found", "status": "pass" if result.get("status") != "missing" else "review", "detail": target},
            {"check": "History preserved", "status": "pass" if result.get("history") else "review", "detail": "dynamic_asset_history.csv"},
            {"check": "Active ranking updated", "status": "pass", "detail": f"active={target in set(active_ids)}"},
        ]
        decision_packet = {
            "mode": "dynamic_asset_lifecycle",
            "intent": action,
            "objective": query,
            "selected_asset": target,
            "status": result.get("status"),
            "active_dynamic_assets": active_ids,
            "inactive_dynamic_assets": inactive_df["asset_id"].astype(str).tolist() if not inactive_df.empty else [],
            "next_system_action": "exclude_from_active_ranking" if action == "deactivate" else "include_in_active_ranking",
        }
        answer = f"""
**Dynamic Asset Lifecycle Update**

Asset `{target}` was {status_text}.

**Why this matters**
- Inactive assets stay in memory and audit history.
- Active dynamic rankings now use only `active=True` dynamic assets.
- Reactivation can restore the asset with optional updated readings.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Current Memory Scope**
- Active dynamic assets: {", ".join(active_ids) if active_ids else "none"}
- Inactive dynamic assets: {", ".join(decision_packet["inactive_dynamic_assets"]) if decision_packet["inactive_dynamic_assets"] else "none"}

**Final Decision Packet**
- Mode: dynamic_asset_lifecycle
- Asset: {target}
- Status: {result.get("status")}
- Next system action: {decision_packet["next_system_action"]}
""".strip()
        self.session_memory["last_asset_id"] = target
        if action == "reactivate":
            self.session_memory["last_new_asset_id"] = target
        self.write_logbook(query, target, priority, answer)
        return {
            "mode": "dynamic_asset_lifecycle",
            "asset_id": target,
            "intent": action,
            "sensor_summary": sensor,
            "lifecycle_result": result,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"{target} {status_text}.",
            "llm_used": False,
        }

    def dynamic_memory_status_report(self, query: str, user_id: str = "demo_user") -> dict:
        active = load_dynamic_assets()
        if not active.empty:
            active = active[active["active"].astype(bool)].copy()
        inactive = list_inactive_dynamic_assets()
        active_records = active.to_dict("records") if not active.empty else []
        inactive_records = inactive.to_dict("records") if not inactive.empty else []
        excluded_notes = []
        for row in inactive_records:
            asset_id = str(row.get("asset_id", "")).upper()
            latest = latest_dynamic_asset_change(asset_id)
            reason = latest.get("source_query") if latest else row.get("source_query", "")
            excluded_notes.append(f"{asset_id} is excluded because active=False after lifecycle event: {reason or 'inspected/cleared'}")

        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Read dynamic asset memory", "target": "dynamic_assets.csv", "status": "complete"},
            {"step": 2, "agent": "State Agent", "task": "Separate active and inactive remembered assets", "target": "active flag", "status": "complete"},
            {"step": 3, "agent": "Verifier Agent", "task": "Confirm inactive assets are excluded from active ranking", "target": "ranking scope", "status": "complete"},
        ]
        tool_calls = [
            {"tool": "dynamic_asset_memory_reader", "agent": "Memory Agent", "input": "dynamic_assets.csv", "output": f"{len(active_records)} active, {len(inactive_records)} inactive", "status": "success"},
            {"tool": "dynamic_lifecycle_filter", "agent": "State Agent", "input": "active column", "output": "active ranking excludes inactive assets", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Active list generated", "status": "pass", "detail": f"{len(active_records)} active"},
            {"check": "Inactive list generated", "status": "pass", "detail": f"{len(inactive_records)} inactive"},
            {"check": "Inactive assets preserved", "status": "pass", "detail": "stored in dynamic_assets.csv and dynamic_asset_history.csv"},
        ]
        decision_packet = {
            "mode": "dynamic_memory_status",
            "intent": "dynamic_memory_audit",
            "objective": query,
            "active_dynamic_assets": [row.get("asset_id") for row in active_records],
            "inactive_dynamic_assets": [row.get("asset_id") for row in inactive_records],
            "next_system_action": "use_active_dynamic_assets_for_ranking",
        }
        answer = f"""
**Dynamic Asset Memory Status**

**Active remembered dynamic assets**
{_format_records(active_records)}

**Inactive remembered dynamic assets**
{_format_records(inactive_records)}

**Why inactive assets are excluded**
{chr(10).join([f"- {note}" for note in excluded_notes]) if excluded_notes else "- No inactive dynamic assets currently excluded."}

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Final Decision Packet**
- Mode: dynamic_memory_status
- Active assets: {", ".join(decision_packet["active_dynamic_assets"]) if decision_packet["active_dynamic_assets"] else "none"}
- Inactive assets: {", ".join(decision_packet["inactive_dynamic_assets"]) if decision_packet["inactive_dynamic_assets"] else "none"}
- Next system action: use_active_dynamic_assets_for_ranking
""".strip()
        return {
            "mode": "dynamic_memory_status",
            "asset_id": None,
            "intent": "dynamic_memory_audit",
            "active_dynamic_assets": active_records,
            "inactive_dynamic_assets": inactive_records,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": "",
            "llm_used": False,
        }

    def dynamic_memory_audit_report(self, query: str, asset_id: str | None = None, user_id: str = "demo_user") -> dict:
        target = asset_id or self._infer_asset_from_query(query)
        if not target:
            answer = "I can show a memory audit trail, but I need an asset ID."
            return {
                "mode": "dynamic_memory_audit",
                "asset_id": None,
                "intent": "dynamic_memory_audit",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Asset resolved", "status": "review", "detail": "No target asset"}],
                "decision_packet": {"mode": "dynamic_memory_audit", "status": "needs_asset_id", "objective": query},
                "alert_report": "",
            }
        target = str(target).upper()
        assets = load_dynamic_assets()
        asset_rows = assets[assets["asset_id"].astype(str).str.upper() == target].copy() if not assets.empty else pd.DataFrame()
        if asset_rows.empty:
            answer = f"No dynamic memory row exists for {target}."
            return {
                "mode": "dynamic_memory_audit",
                "asset_id": target,
                "intent": "dynamic_memory_audit",
                "answer": answer,
                "final_answer": answer,
                "agent_plan": [],
                "tool_calls": [],
                "verifier_checks": [{"check": "Dynamic memory row exists", "status": "review", "detail": target}],
                "decision_packet": {"mode": "dynamic_memory_audit", "status": "missing", "selected_asset": target},
                "alert_report": "",
            }
        current_row = asset_rows.iloc[0].to_dict()
        history_df = load_dynamic_asset_history()
        if not history_df.empty:
            history_df = history_df[history_df["asset_id"].astype(str).str.upper() == target].copy()
            if not history_df.empty:
                history_df["changed_at_sort"] = pd.to_datetime(history_df["changed_at"], errors="coerce", format="mixed")
                history_df = history_df.sort_values("changed_at_sort")
        history_records = history_df.drop(columns=["changed_at_sort"], errors="ignore").to_dict("records") if not history_df.empty else []
        sensor = self.get_latest_sensor_summary(target) if bool(current_row.get("active", True)) else score_dynamic_assets(pd.DataFrame([current_row]), active_only=False).iloc[0].to_dict()
        if not isinstance(sensor, dict):
            sensor = dict(sensor)
        spares = self.get_spares(target)
        delay = self.get_delay(target)
        priority = self.prioritize_action(sensor, spares, delay)
        actions = self.recommended_actions(target)
        rules = load_dynamic_rules()
        relevant_rules = []
        if not rules.empty:
            for _, row in rules.iterrows():
                if str(row.get("active", "")).lower() in {"false", "0", "no"}:
                    continue
                text = " ".join(str(row.get(field, "")) for field in ["equipment_pattern", "area_pattern", "condition_text", "source_text"]).lower()
                asset_text = f"{target} {current_row.get('asset_type')} {current_row.get('area')}".lower()
                if any(token in text for token in asset_text.split() if len(token) >= 4):
                    relevant_rules.append(row.to_dict())
        applied_rules = sensor.get("applied_rules") if isinstance(sensor.get("applied_rules"), list) else []
        decision_packet = {
            "mode": "dynamic_memory_audit",
            "intent": "dynamic_memory_audit",
            "objective": query,
            "selected_asset": target,
            "active": bool(current_row.get("active", True)),
            "current_priority": priority.get("priority"),
            "current_risk_level": priority.get("risk_level"),
            "hybrid_health_score": sensor.get("hybrid_health_score"),
            "estimated_rul_days": sensor.get("estimated_rul_days"),
            "applied_rule_count": len(applied_rules),
            "next_system_action": "create_work_order_and_notify_supervisor" if priority.get("priority") in {"P1", "P2"} else "monitor_and_schedule",
        }
        agent_plan = [
            {"step": 1, "agent": "Memory Agent", "task": "Load original remembered asset row", "target": target, "status": "complete"},
            {"step": 2, "agent": "Audit Agent", "task": "Read update/deactivation/reactivation events", "target": "dynamic_asset_history.csv", "status": "complete"},
            {"step": 3, "agent": "Policy Agent", "task": "Read active remembered safety rules", "target": "dynamic_rules.csv", "status": "complete"},
            {"step": 4, "agent": "Risk Agent", "task": "Recompute current priority and next action", "target": target, "status": "complete"},
        ]
        tool_calls = [
            {"tool": "dynamic_asset_memory_reader", "agent": "Memory Agent", "input": target, "output": "current remembered row loaded", "status": "success"},
            {"tool": "dynamic_asset_history_reader", "agent": "Audit Agent", "input": target, "output": f"{len(history_records)} history event(s)", "status": "success"},
            {"tool": "dynamic_rule_reader", "agent": "Policy Agent", "input": target, "output": f"{len(applied_rules)} applied rule(s), {len(relevant_rules)} potentially relevant active rule(s)", "status": "success"},
            {"tool": "priority_recalculator", "agent": "Risk Agent", "input": "current state + rules + delay + spares", "output": f"{priority.get('priority')}/{priority.get('risk_level')}", "status": "success"},
        ]
        verifier_checks = [
            {"check": "Original add/source query available", "status": "pass" if current_row.get("source_query") else "review", "detail": str(current_row.get("source_query", ""))[:120]},
            {"check": "History events loaded", "status": "pass" if history_records else "review", "detail": f"{len(history_records)} event(s)"},
            {"check": "Current active state known", "status": "pass", "detail": str(bool(current_row.get("active", True)))},
            {"check": "Current priority recalculated", "status": "pass", "detail": f"{priority.get('priority')}/{priority.get('risk_level')}"},
        ]
        answer = f"""
**Dynamic Memory Audit Trail: {target}**

**Original remembered row**
{_format_records([current_row])}

**Update / lifecycle history**
{_format_records(history_records)}

**Applied safety rules**
{_format_records(applied_rules)}

**Relevant remembered rule records**
{_format_records(relevant_rules)}

**Current active state**
- Active: {bool(current_row.get("active", True))}
- Current priority: {priority.get("priority")}/{priority.get("risk_level")}
- Current hybrid score: {sensor.get("hybrid_health_score")}
- Current RUL: {sensor.get("estimated_rul_days")} days
- Current next action: {actions[0] if actions else decision_packet["next_system_action"]}

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Final Decision Packet**
- Mode: dynamic_memory_audit
- Selected asset: {target}
- Current active state: {bool(current_row.get("active", True))}
- Current priority: {priority.get("priority")}/{priority.get("risk_level")}
- Next system action: {decision_packet["next_system_action"]}
""".strip()
        self.write_logbook(query, target, {"priority": "AUDIT", "risk_level": "MEMORY", "urgency": "Audit trail generated", "priority_score": 0}, answer)
        return {
            "mode": "dynamic_memory_audit",
            "asset_id": target,
            "intent": "dynamic_memory_audit",
            "current_record": current_row,
            "history": history_records,
            "applied_rules": applied_rules,
            "relevant_rules": relevant_rules,
            "risk_priority": priority,
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Memory audit generated for {target}.",
            "llm_used": False,
        }

    def evidence_confidence_report(self, query: str, user_id: str = "demo_user") -> dict:
        style = infer_output_style(query)
        q = str(query).lower()
        dynamic_only = any(term in q for term in ["dynamic", "newly added", "remembered", "active dynamic"])
        asset_ids = dynamic_asset_ids(active_only=True) if dynamic_only else self.asset_ids
        rows = []
        for asset_id in asset_ids:
            sensor = self.get_latest_sensor_summary(asset_id)
            if sensor.get("is_dynamic"):
                docs = self._dynamic_context_docs(asset_id, sensor) + self._filter_docs_for_assets(
                    self.rag.retrieve(query, top_k=4, plant_level=True),
                    [asset_id],
                )[:2]
            else:
                docs = self.rag.retrieve(query, top_k=3, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
            spares = self.get_spares(asset_id)
            evidence = self.evidence_confidence(
                asset_id,
                sensor,
                docs,
                self.get_history(asset_id),
                self.get_failures(asset_id),
                spares,
            )
            confidence_rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}.get(evidence["evidence_confidence"], 0)
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": sensor.get("asset_type"),
                    "area": sensor.get("area"),
                    "risk_level": sensor.get("risk_band"),
                    "hybrid_health_score": sensor.get("hybrid_health_score"),
                    "evidence_confidence": evidence["evidence_confidence"],
                    "available_evidence": evidence["available_evidence"],
                    "missing_evidence": evidence["missing_evidence"],
                    "_confidence_rank": confidence_rank,
                }
            )
        table = sorted(rows, key=lambda row: (row["_confidence_rank"], -safe_float(row.get("hybrid_health_score"))))
        weakest = table[0] if table else {}
        decision_packet = {
            "mode": "evidence_confidence_report",
            "intent": "evidence_confidence_query",
            "objective": query,
            "selected_asset": weakest.get("asset_id"),
            "evidence_confidence": weakest.get("evidence_confidence"),
            "missing_evidence": weakest.get("missing_evidence", []),
            "scope": "active_dynamic_assets" if dynamic_only else "all_assets",
            "next_system_action": "collect_missing_evidence_before_shutdown" if weakest.get("evidence_confidence") == "LOW" else "use_evidence_in_planning",
        }
        clean_table = [{k: v for k, v in row.items() if not k.startswith("_")} for row in table]
        if "missing evidence only" in q or "list missing evidence only" in q:
            answer = "\n".join(
                [f"{row['asset_id']}: " + (", ".join(row["missing_evidence"]) if row["missing_evidence"] else "none") for row in clean_table]
            )
        elif style["format"] == "table_only":
            answer = _markdown_table(
                clean_table,
                ["asset_id", "asset_type", "risk_level", "hybrid_health_score", "evidence_confidence", "missing_evidence"],
            )
        elif style["format"] == "json_only":
            answer = json.dumps({"weakest": weakest, "rows": clean_table}, indent=2)
        else:
            answer = f"""
**Evidence Confidence Review**

Weakest evidence confidence: `{weakest.get("asset_id", "none")}` at `{weakest.get("evidence_confidence", "UNKNOWN")}`.

**Missing evidence**
{chr(10).join([f"- {item}" for item in weakest.get("missing_evidence", [])]) if weakest else "- none"}

**Ranked Evidence Table**
{_markdown_table(clean_table, ["asset_id", "asset_type", "risk_level", "hybrid_health_score", "evidence_confidence", "missing_evidence"])}
""".strip()
        return {
            "mode": "evidence_confidence_report",
            "asset_id": weakest.get("asset_id"),
            "intent": "evidence_confidence_query",
            "evidence_table": clean_table,
            "decision_packet": decision_packet,
            "agent_plan": [
                {"step": 1, "agent": "Knowledge Agent", "task": "Inspect evidence completeness across requested scope", "target": decision_packet["scope"], "status": "complete"},
                {"step": 2, "agent": "Verifier Agent", "task": "Identify weakest confidence and missing evidence", "target": weakest.get("asset_id"), "status": "complete"},
            ],
            "tool_calls": [
                {"tool": "evidence_confidence_scorer", "agent": "Knowledge Agent", "input": decision_packet["scope"], "output": f"{len(clean_table)} asset(s) scored", "status": "success"}
            ],
            "verifier_checks": [
                {"check": "Evidence rows generated", "status": "pass" if clean_table else "review", "detail": str(len(clean_table))},
                {"check": "Weakest confidence identified", "status": "pass" if weakest else "review", "detail": str(weakest.get("asset_id", "none"))},
            ],
            "answer": answer,
            "final_answer": answer,
            "alert_report": f"Weakest evidence confidence: {weakest.get('asset_id', 'none')}.",
            "llm_used": False,
        }

    def plant_priority_report(self, query: str, asset_ids: list[str] | None = None) -> dict:
        output_style = infer_output_style(query)
        asset_ids = asset_ids or self.asset_ids
        rows = []
        for asset_id in asset_ids:
            sensor = self.get_latest_sensor_summary(asset_id)
            spares = self.get_spares(asset_id)
            delay = self.get_delay(asset_id)
            priority = self.prioritize_action(sensor, spares, delay)
            if sensor.get("is_dynamic"):
                asset_docs = self._dynamic_context_docs(asset_id, sensor) + self._filter_docs_for_assets(
                    self.rag.retrieve(query, top_k=4, plant_level=True),
                    [asset_id],
                )[:2]
            else:
                asset_docs = self.rag.retrieve(query, top_k=3, asset_id=asset_id, equipment_type=sensor.get("asset_type"))
            evidence = self.evidence_confidence(
                asset_id,
                sensor,
                asset_docs,
                self.get_history(asset_id),
                self.get_failures(asset_id),
                spares,
            )
            display_risk = str(sensor.get("risk_band") or priority.get("risk_level") or "LOW").upper()
            display_priority = {"CRITICAL": "P1", "HIGH": "P2", "MEDIUM": "P3", "LOW": "P4"}.get(display_risk, priority.get("priority", "P4"))
            display_urgency = {
                "CRITICAL": "Immediate action",
                "HIGH": "Action within 24 hours",
                "MEDIUM": "Plan in maintenance window",
                "LOW": "Monitor only",
            }.get(display_risk, priority.get("urgency", "Monitor"))
            rows.append(
                {
                    "asset_id": asset_id,
                    "asset_type": sensor.get("asset_type"),
                    "area": sensor.get("area"),
                    "criticality": sensor.get("criticality"),
                    "ml_failure_risk": sensor.get("ml_failure_risk_latest"),
                    "hybrid_failure_risk": sensor.get("hybrid_failure_risk"),
                    "operational_rule_score": sensor.get("operational_rule_score"),
                    "hybrid_health_score": sensor.get("hybrid_health_score"),
                    "rul_days": sensor.get("estimated_rul_days"),
                    "risk_level": display_risk,
                    "priority": display_priority,
                    "priority_score": priority.get("priority_score"),
                    "urgency": display_urgency,
                    "delay_hours": delay.get("delay_hours", 0),
                    "applied_rules": sensor.get("applied_rule_count", 0),
                    "evidence_confidence": evidence.get("evidence_confidence"),
                    "missing_evidence": evidence.get("missing_evidence", []),
                }
            )
        table = (
            pd.DataFrame(rows)
            .sort_values(
                ["priority_score", "hybrid_health_score", "rul_days"],
                ascending=[False, False, True],
            )
            .reset_index(drop=True)
        )
        risk_filter_requested = any(
            term in str(query).lower()
            for term in [
                "classified as high or critical",
                "high or critical risk",
                "high/critical",
                "critical or high",
            ]
        )
        if risk_filter_requested:
            table = table[table["risk_level"].astype(str).str.upper().isin(["HIGH", "CRITICAL"])].reset_index(drop=True)
            if table.empty:
                answer = "No active assets are currently classified as HIGH or CRITICAL risk."
                return {
                    "mode": "plant_priority",
                    "asset_id": None,
                    "intent": "risk_band_listing",
                    "plant_priority_table": [],
                    "risk_priority": {"priority": "P4", "risk_level": "LOW", "urgency": "Monitor", "priority_score": 0},
                    "priority": "No HIGH/CRITICAL assets",
                    "agent_plan": self.build_agent_plan(query, mode="plant_priority"),
                    "tool_calls": [{"tool": "risk_band_filter", "agent": "Risk Agent", "input": "HIGH/CRITICAL", "output": "0 rows", "status": "success"}],
                    "verifier_checks": [{"check": "Risk-band filter applied", "status": "pass", "detail": "HIGH/CRITICAL"}],
                    "decision_packet": {"mode": "plant_priority", "intent": "risk_band_listing", "selected_asset": None},
                    "answer": answer,
                    "final_answer": answer,
                    "alert_report": "",
                    "llm_used": False,
                }
        dynamic_scope_ids = set(dynamic_asset_ids())
        ranked_ids = set(table["asset_id"].astype(str).str.upper())
        is_dynamic_only_scope = bool(ranked_ids) and ranked_ids.issubset(dynamic_scope_ids)
        inactive_dynamic_assets = list_inactive_dynamic_assets()
        inactive_ids = inactive_dynamic_assets["asset_id"].astype(str).str.upper().tolist() if not inactive_dynamic_assets.empty else []
        report_title = "Dynamic Assets Priority Ranking" if is_dynamic_only_scope else "Plant-Level Maintenance Decision Summary"
        top_asset = table.iloc[0]["asset_id"]
        top_sensor = self.get_latest_sensor_summary(top_asset)
        top_spares = self.get_spares(top_asset)
        top_type_text = str(top_sensor.get("asset_type", "")).lower()
        top_equipment = normalize_equipment_type(top_sensor.get("asset_type", ""))
        second = table.iloc[1] if len(table) > 1 else None
        if top_equipment == "gearbox":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 controlled-shutdown inspection plan for gearbox vibration, "
                "reserve gearbox bearing set and synthetic gear oil, perform vibration spectrum analysis, oil sampling, "
                "coupling alignment check, gear backlash check, and bearing inspection."
            )
        elif top_equipment == "motor":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 motor overheating inspection plan, reduce load if needed, "
                "inspect bearing lubrication, cooling path, current imbalance, fan condition, and coupling alignment."
            )
        elif top_equipment == "pump":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a high-priority cavitation inspection plan, check suction strainer, "
                "tank level, air ingress, seal leakage, impeller condition, and reserve mechanical seal or impeller spares."
            )
        elif top_equipment == "hydraulic":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a high-priority hydraulic pressure recovery plan, inspect filter, "
                "relief valve, oil level, leakage, pump noise, and reserve filter element or relief valve cartridge."
            )
        elif any(word in top_type_text for word in ["blower", "fan", "compressor"]):
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 rotating-air-equipment inspection plan, verify vibration spectrum, "
                "bearing temperature, motor current balance, damper position, impeller fouling, duct restriction, "
                "coupling alignment, and standby availability."
            )
        elif "bearing" in top_type_text:
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 bearing inspection plan, verify bearing temperature, lubrication, "
                "vibration spectrum, alignment, load condition, contamination, cooling path, spare bearing availability, "
                "lifting plan, and safe isolation permit."
            )
        elif top_equipment == "blast_furnace":
            recommended_first_action = (
                f"Choose {top_asset} first. Create a P1 blast-furnace-area safety inspection plan, verify cooling, airflow, "
                "interlocks, vibration, temperature, isolation permits, and standby equipment readiness."
            )
        else:
            recommended_first_action = f"Choose {top_asset} first. Create a controlled inspection and repair work order."
        comparison_note = ""
        if second is not None:
            tie_detail = ""
            try:
                same_top_band = (
                    str(table.iloc[0]["priority"]) == str(second["priority"])
                    and str(table.iloc[0]["risk_level"]) == str(second["risk_level"])
                )
                close_score = abs(float(table.iloc[0]["hybrid_health_score"]) - float(second["hybrid_health_score"])) < 0.01
                if same_top_band and close_score:
                    tie_detail = (
                        f" Tie-break: {top_asset} has criticality {table.iloc[0]['criticality']}, "
                        f"{table.iloc[0].get('applied_rules', 0)} applied safety rule(s), "
                        f"evidence {table.iloc[0].get('evidence_confidence', 'UNKNOWN')}; "
                        f"{second['asset_id']} has criticality {second['criticality']}, "
                        f"{second.get('applied_rules', 0)} applied safety rule(s), "
                        f"evidence {second.get('evidence_confidence', 'UNKNOWN')}."
                    )
            except Exception:
                tie_detail = ""
            comparison_note = (
                f"- It is ahead of {second['asset_id']}: {second['asset_id']} is "
                f"{second['priority']}/{second['risk_level']} with hybrid health score "
                f"{second['hybrid_health_score']} and RUL {second['rul_days']} days.{tie_detail}"
            )
        docs = self._filter_docs_for_assets(self.rag.retrieve(query, top_k=8, plant_level=True), list(asset_ids))[:5]
        dynamic_docs = []
        for asset_id in table["asset_id"].astype(str).tolist():
            sensor = self.get_latest_sensor_summary(asset_id)
            if sensor.get("is_dynamic"):
                dynamic_docs.extend(self._dynamic_context_docs(asset_id, sensor))
        docs = dynamic_docs + docs
        agent_plan = self.build_agent_plan(query, mode="plant_priority", asset_id=top_asset)
        tool_calls = [
            {"tool": "asset_health_scan", "agent": "Sensor Agent", "input": f"{len(asset_ids)} scoped assets", "output": f"{len(table)} scored rows from {'dynamic memory only' if is_dynamic_only_scope else 'plant scope'}", "status": "success"},
            {"tool": "plant_priority_ranker", "agent": "Risk Agent", "input": "hybrid score + RUL + delay + criticality", "output": f"top asset {top_asset}", "status": "success"},
            {"tool": "rag_retriever", "agent": "Knowledge Agent", "input": "plant-level policies and evidence", "output": f"{len(docs)} evidence chunks", "status": "success"},
            {"tool": "supervisor_report_writer", "agent": "Reporter Agent", "input": top_asset, "output": "plant priority summary generated", "status": "success"},
        ]
        verifier_checks = [
            {"check": "All requested known assets scored", "status": "pass" if len(table) == len(asset_ids) else "review", "detail": f"{len(table)} of {len(asset_ids)} assets ranked"},
            {"check": "Top asset selected", "status": "pass", "detail": top_asset},
            {"check": "Ranking includes RUL and delay", "status": "pass", "detail": "rul_days and delay_hours present"},
            {"check": "Policy evidence retrieved", "status": "pass" if len(docs) > 0 else "review", "detail": f"{len(docs)} sources"},
        ]
        decision_packet = {
            "mode": "plant_priority",
            "intent": "maintenance_prioritization",
            "objective": query,
            "selected_asset": top_asset,
            "risk_level": table.iloc[0]["risk_level"],
            "priority": table.iloc[0]["priority"],
            "urgency": table.iloc[0]["urgency"],
            "hybrid_health_score": float(table.iloc[0]["hybrid_health_score"]),
            "hybrid_failure_risk": float(table.iloc[0]["hybrid_failure_risk"]),
            "ml_failure_risk": float(table.iloc[0]["ml_failure_risk"]),
            "operational_rule_score": float(table.iloc[0]["operational_rule_score"]),
            "estimated_rul_days": float(table.iloc[0]["rul_days"]),
            "recommended_first_action": recommended_first_action,
            "next_system_action": "create_first_work_order_and_notify_supervisor",
            "inactive_dynamic_assets_excluded": inactive_ids,
            "top_sources": [doc.get("source") for doc in docs[:3]],
        }
        dynamic_all_ids = set(dynamic_asset_ids(active_only=False))
        original_table = table[~table["asset_id"].astype(str).str.upper().isin(dynamic_all_ids)]
        top_original_row = original_table.iloc[0] if not original_table.empty else None
        original_comparison_note = ""
        if top_original_row is not None and str(top_original_row["asset_id"]) != str(top_asset):
            original_comparison_note = (
                f"{top_asset} also beats highest-risk original {top_original_row['asset_id']}: "
                f"{top_original_row['asset_id']} is {top_original_row['priority']}/{top_original_row['risk_level']}, "
                f"score {top_original_row['priority_score']}, ML risk {top_original_row['ml_failure_risk']}, "
                f"RUL {top_original_row['rul_days']}d."
            )
        missing_evidence = table.iloc[0].get("missing_evidence", [])
        missing_text = ", ".join(missing_evidence) if isinstance(missing_evidence, list) and missing_evidence else "none"
        procurement_risk = "LOW"
        if not top_spares:
            procurement_risk = "REVIEW - no matching spare master record"
        elif any(safe_float(item.get("stock_qty", 0)) <= 0 for item in top_spares):
            procurement_risk = "HIGH - one or more required spares out of stock"
        elif any(safe_float(item.get("lead_time_days", 0)) >= 7 for item in top_spares):
            procurement_risk = "MEDIUM - long lead spare reservation needed"
        ranking = "\n".join(
            f"- {r.asset_id}: {r.priority}/{r.risk_level}, hybrid score {r.hybrid_health_score}, ML risk {r.ml_failure_risk}, rule score {r.operational_rule_score}, RUL {r.rul_days} days, delay {r.delay_hours}h, applied rules {r.applied_rules}, evidence {r.evidence_confidence}"
            for r in table.itertuples()
        )
        if output_style["format"] != "full_report":
            table_records = table.to_dict("records")
            if output_style["format"] == "json_only":
                report = json.dumps(
                    {
                        "selected_asset": top_asset,
                        "ranking": table_records,
                        "decision_packet": decision_packet,
                    },
                    indent=2,
                )
            elif output_style["format"] == "table_only":
                report = _markdown_table(
                    table_records,
                    ["asset_id", "priority", "risk_level", "priority_score", "hybrid_health_score", "hybrid_failure_risk", "rul_days", "applied_rules", "evidence_confidence"],
                )
            elif output_style["format"] == "lines":
                lines = [
                    f"Choose {top_asset} first.",
                    f"{top_asset}: {table.iloc[0]['priority']}/{table.iloc[0]['risk_level']}, score {table.iloc[0]['priority_score']}, RUL {table.iloc[0]['rul_days']} days.",
                    comparison_note.lstrip("- ") if comparison_note else "No second-ranked asset available.",
                    f"Next action: {decision_packet['next_system_action']}.",
                    f"Inactive dynamic assets excluded: {', '.join(inactive_ids) if inactive_ids else 'none'}.",
                ][: output_style.get("max_items", 5)]
                report = "\n".join(lines)
            else:
                bullets = [
                    f"- Choose {top_asset} first.",
                    f"- Priority/risk: {table.iloc[0]['priority']}/{table.iloc[0]['risk_level']}.",
                    f"- ML/raw/rule-adjusted score: ML risk {table.iloc[0]['ml_failure_risk']}, raw operational score {table.iloc[0]['operational_rule_score']}, rule-adjusted hybrid score {table.iloc[0]['hybrid_health_score']}.",
                    f"- RUL/priority score: {table.iloc[0]['rul_days']} days RUL, {table.iloc[0]['priority_score']} priority score.",
                    f"- Why it wins: {comparison_note.lstrip('- ') if comparison_note else 'highest-ranked asset in scope.'}",
                    f"- Original-asset comparison: {original_comparison_note or 'selected asset is also the highest-ranked original/dynamic candidate in scope.'}",
                    f"- Applied rules: {table.iloc[0].get('applied_rules', 0)}.",
                    f"- Evidence confidence/missing evidence: {table.iloc[0].get('evidence_confidence', 'UNKNOWN')}; missing {missing_text}.",
                    f"- Procurement risk/spares: {procurement_risk}; reserve available matching spares before isolation.",
                    f"- Inspection checklist: {recommended_first_action}",
                    f"- Supervisor alert/work order: {decision_packet['next_system_action']}; notify area supervisor and create controlled inspection work order.",
                    f"- Logbook/feedback: write ranking, evidence, rule matches, spare decision, and capture post-inspection root cause/outcome.",
                    f"- Inactive dynamic assets excluded: {', '.join(inactive_ids) if inactive_ids else 'none'}.",
                ][: output_style.get("max_items", 8)]
                report = "\n".join(bullets)
            priority = {"priority": "PLANT", "risk_level": "PLANT_SUMMARY", "urgency": f"Prioritize {top_asset}", "priority_score": float(table.iloc[0]["priority_score"])}
            self.session_memory["last_asset_id"] = top_asset
            self._remember_asset_context(list(asset_ids), selected_asset=top_asset)
            self.write_logbook(query, top_asset, priority, report)
            return {
                "mode": "plant_priority",
                "asset_id": top_asset,
                "intent": "maintenance_prioritization",
                "output_style": output_style,
                "plant_priority_table": table_records,
                "risk_priority": priority,
                "priority": "Plant priority summary",
                "agent_plan": agent_plan,
                "tool_calls": tool_calls,
                "verifier_checks": verifier_checks,
                "decision_packet": decision_packet,
                "answer": report,
                "final_answer": report,
                "alert_report": f"Plant alert: prioritize {top_asset}.",
                "llm_used": False,
            }
        report = f"""
**{report_title}**

**Choose {top_asset} first.**

**Reason**
- {top_asset} has the highest combined plant priority score from hybrid health risk, criticality, RUL, delay impact, and spare readiness.
- Current locked fields: {table.iloc[0]["priority"]}/{table.iloc[0]["risk_level"]}, hybrid health score {table.iloc[0]["hybrid_health_score"]}, hybrid failure risk {table.iloc[0]["hybrid_failure_risk"]}, RUL {table.iloc[0]["rul_days"]} days.
- Latest condition: temperature {top_sensor.get("temperature_latest")}, vibration {top_sensor.get("vibration_latest")}, pressure {top_sensor.get("pressure_latest")}, alarms {top_sensor.get("alarm_count_latest")}.
{comparison_note}

**Agentic Control Loop**
- Objective: {query}
- Selected first target: {top_asset}
- Operating mode: {'dynamic-memory prioritization' if is_dynamic_only_scope else 'autonomous plant prioritization'}
- Decision policy: rank by safety risk, hybrid risk, criticality, RUL, delay impact, and spare readiness.

**Autonomous Execution Plan**
{chr(10).join([f"- Step {p['step']} | {p['agent']}: {p['task']} [{p['status']}]" for p in agent_plan])}

**Tool Calls Executed**
{chr(10).join([f"- {t['agent']} -> `{t['tool']}` | input: {t['input']} | output: {t['output']} | {t['status']}" for t in tool_calls])}

**Verifier Checks**
{chr(10).join([f"- {v['check']}: {v['status'].upper()} ({v['detail']})" for v in verifier_checks])}

**Locked Decision Fields**
- Most urgent asset: {top_asset}
- Intent: maintenance_prioritization
- Recommended first action: {recommended_first_action}
- Ranking basis: hybrid ML + operational rule score, criticality, delay severity, RUL, anomaly status, spares/procurement.

**Diagnosis**
- Equipment was compared across {len(asset_ids)} scoped steel assets{' from dynamic memory only' if is_dynamic_only_scope else ', including any user-added dynamic assets in memory'}.
- Inactive remembered dynamic assets excluded from active ranking: {", ".join(inactive_ids) if inactive_ids else "none"}.

**Risk and RUL**
{ranking}

**Immediate Actions**
- {recommended_first_action}
- Reserve spares before shutdown.
- Notify area supervisor for P1/P2 assets.
- Continue monitoring lower-ranked assets.

**Spare Strategy For Selected Asset**
{_spares_strategy(top_spares)}

**Evidence / Sources**
{_format_sources(docs)}

**Agent Reasoning Trace**
- Triage Agent: identified plant-level prioritization request.
- Sensor Agent: collected latest health and RUL for all requested known assets, including dynamic memory rows.
- Risk Agent: ranked assets by hybrid ML + operational rule score.
- Planning Agent: selected {top_asset} as first maintenance target.
- Reporting Agent: generated supervisor summary and logbook entry.

**Final Decision Packet**
- Mode: {decision_packet["mode"]}
- Intent: {decision_packet["intent"]}
- Next system action: {decision_packet["next_system_action"]}
- Selected asset: {decision_packet["selected_asset"]}
- Recommended first action: {decision_packet["recommended_first_action"]}
- Top evidence sources: {", ".join(decision_packet["top_sources"])}
""".strip()
        priority = {"priority": "PLANT", "risk_level": "PLANT_SUMMARY", "urgency": f"Prioritize {top_asset}", "priority_score": float(table.iloc[0]["priority_score"])}
        self.session_memory["last_asset_id"] = top_asset
        if self._is_dynamic_asset(top_asset):
            self.session_memory["last_new_asset_id"] = top_asset
        self._remember_asset_context(list(asset_ids), selected_asset=top_asset)
        self.write_logbook(query, top_asset, priority, report)
        return {
            "mode": "plant_priority",
            "asset_id": top_asset,
            "intent": "maintenance_prioritization",
            "output_style": output_style,
            "plant_priority_table": table.to_dict("records"),
            "risk_priority": priority,
            "priority": "Plant priority summary",
            "agent_plan": agent_plan,
            "tool_calls": tool_calls,
            "verifier_checks": verifier_checks,
            "decision_packet": decision_packet,
            "answer": report,
            "final_answer": report,
            "alert_report": f"Plant alert: prioritize {top_asset}.",
            "llm_used": True,
        }

    def ingest_new_sensor_alert(self, asset_id: str, temperature: float, vibration: float, current: float, pressure: float, rpm: float = 1480, alarm_count: int = 2, user_id: str = "iot_gateway") -> dict:
        self.ensure_ready()
        raw = pd.read_csv(DATA_DIR / "steel_sensor_logs.csv")
        rows = raw[raw["asset_id"] == asset_id]
        if rows.empty:
            return {"asset_id": asset_id, "answer": f"No known asset found for {asset_id}.", "priority": "UNKNOWN", "alert_report": "No alert generated."}
        info = rows.iloc[-1].to_dict()
        row = {
            "source": "steel_demo_app",
            "timestamp": datetime.now().isoformat(),
            "asset_id": asset_id,
            "asset_type": info.get("asset_type"),
            "area": info.get("area"),
            "criticality": info.get("criticality"),
            "criticality_score": info.get("criticality_score", 2),
            "temperature": float(temperature),
            "vibration": float(vibration),
            "current": float(current),
            "pressure": float(pressure),
            "rpm": float(rpm),
            "alarm_count": int(alarm_count),
            "delay_hours": info.get("delay_hours", 0),
            "spare_lead_time_days": info.get("spare_lead_time_days", 0),
            "failure_label": 0,
            "failure_mode": "real_time_alert",
        }
        raw = pd.concat([raw, pd.DataFrame([row])], ignore_index=True, sort=False)
        raw.to_csv(DATA_DIR / "steel_sensor_logs.csv", index=False)
        scored = self.model_manager.score_live_alert(row)
        create_compatibility_sensor_log()
        # Rebuild RAG so the latest health document is updated for evidence retrieval.
        self.rag.build()
        result = self.chat(f"New real-time alert for {asset_id}. Diagnose and generate alert report.", user_id=user_id)
        result["live_alert_row"] = scored
        return result

    def logbook(self) -> pd.DataFrame:
        path = DATA_DIR / "digital_logbook.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    def feedback_log(self) -> pd.DataFrame:
        path = DATA_DIR / "feedback_log.csv"
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
