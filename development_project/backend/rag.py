"""Metadata-filtered RAG retrieval for manuals, history, failures, spares, and policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, DOC_DIR, EMBED_MODEL_ID


def row_get(row, names, default="Not available"):
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return row[name]
    return default


def normalize_equipment_type(value: str) -> str:
    text = str(value).lower()
    if "motor" in text:
        return "motor"
    if "gearbox" in text or "gear" in text:
        return "gearbox"
    if "pump" in text:
        return "pump"
    if "hydraulic" in text or "hpp" in text:
        return "hydraulic"
    if "public" in text or "ai4i" in text:
        return "public_dataset"
    if "policy" in text:
        return "policy"
    if "caster" in text or "casting" in text or "tundish" in text or "mold" in text or "mould" in text:
        return "caster"
    if "blast furnace" in text or "bf" == text.strip():
        return "blast_furnace"
    if "rolling" in text or "hot strip" in text or "plate mill" in text or "finishing mill" in text:
        return "rolling_mill"
    if "crane" in text or "hoist" in text:
        return "crane"
    if "conveyor" in text or "belt" in text:
        return "conveyor"
    if "safety" in text or "loto" in text or "permit" in text:
        return "safety"
    if "quality" in text or "defect" in text:
        return "quality"
    return text.replace(" ", "_").strip() or "general"


def infer_doc_metadata(filename: str, text: str) -> tuple[str, str]:
    name = str(filename).lower()
    filename_map = {
        "sop_mtr_204": ("motor", "overheating"),
        "sop_gbx_17": ("gearbox", "vibration"),
        "sop_pmp_09": ("pump", "cavitation"),
        "sop_hpp_12": ("hydraulic", "pressure"),
        "maintenance_prioritization": ("policy", "priority"),
        "feedback_learning": ("policy", "feedback"),
        "steel_agent_operating_model": ("policy", "agent workflow"),
        "blast_furnace": ("blast_furnace", "maintenance"),
        "continuous_caster": ("caster", "breakout prevention"),
        "rolling_mill": ("rolling_mill", "vibration quality"),
        "industrial_safety": ("safety", "loto permit"),
        "spares_procurement": ("policy", "spares procurement"),
        "data_sources": ("public_dataset", "data source"),
    }
    for key, value in filename_map.items():
        if key in name:
            return value

    text_l = str(text).lower()
    if "asset mtr" in text_l or "induction motor" in text_l:
        return "motor", "overheating"
    if "asset gbx" in text_l or "gbx-17" in text_l or "gearbox" in text_l:
        return "gearbox", "vibration"
    if "asset pmp" in text_l or "pmp-09" in text_l or "cooling water pump" in text_l:
        return "pump", "cavitation"
    if "asset hpp" in text_l or "hpp-12" in text_l or "hydraulic power pack" in text_l:
        return "hydraulic", "pressure"
    if "priority" in text_l or "prioritize" in text_l:
        return "policy", "priority"
    if "feedback" in text_l:
        return "policy", "feedback"
    if "ai4i" in text_l or "public benchmark" in text_l:
        return "public_dataset", "data source"
    if "continuous caster" in text_l or "caster" in text_l or "tundish" in text_l or "mold" in text_l:
        return "caster", "breakout prevention"
    if "blast furnace" in text_l or "tuyere" in text_l or "stave" in text_l:
        return "blast_furnace", "maintenance"
    if "rolling mill" in text_l or "hot strip" in text_l or "roll chock" in text_l:
        return "rolling_mill", "vibration quality"
    if "loto" in text_l or "permit" in text_l or "isolation" in text_l:
        return "safety", "loto permit"
    if "quality defect" in text_l or "surface defect" in text_l:
        return "quality", "defect"
    return "general", "general"


def chunk_text(text: str, chunk_size: int = 120, overlap: int = 25) -> list[str]:
    words = str(text).split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += max(chunk_size - overlap, 1)
    return chunks or [str(text).strip()]


class RAGIndex:
    def __init__(self):
        self.doc_df: pd.DataFrame = pd.DataFrame()
        self.asset_ids: list[str] = []
        self.backend = "uninitialized"
        self.embedder = None
        self.embeddings = None

    def build(self) -> "RAGIndex":
        records = []

        for path in DOC_DIR.glob("*.txt"):
            text = path.read_text(encoding="utf-8")
            equipment_type, issue_type = infer_doc_metadata(path.name, text)
            for i, chunk in enumerate(chunk_text(text)):
                records.append(
                    {
                        "source": path.name,
                        "chunk_id": int(i),
                        "asset_id": "ALL",
                        "equipment_type": equipment_type,
                        "issue_type": issue_type,
                        "text": chunk,
                    }
                )

        self._append_structured_records(records)

        data_sources_path = DATA_DIR.parent / "DATA_SOURCES.md"
        if data_sources_path.exists():
            text = data_sources_path.read_text(encoding="utf-8")
            equipment_type, issue_type = infer_doc_metadata("DATA_SOURCES.md", text)
            records.append(
                {
                    "source": "DATA_SOURCES.md",
                    "chunk_id": 0,
                    "asset_id": "ALL",
                    "equipment_type": equipment_type,
                    "issue_type": issue_type,
                    "text": text,
                }
            )

        self.doc_df = pd.DataFrame(records).dropna(subset=["text"]).reset_index(drop=True)
        self.doc_df["text"] = self.doc_df["text"].astype(str)
        self.doc_df.to_csv(DATA_DIR / "rag_documents.csv", index=False)

        asset_source = DATA_DIR / "asset_health_summary.csv"
        if asset_source.exists():
            self.asset_ids = sorted(pd.read_csv(asset_source)["asset_id"].unique().tolist())
        else:
            self.asset_ids = sorted(pd.read_csv(DATA_DIR / "asset_master.csv")["asset_id"].unique().tolist())

        self._fit_embeddings()
        return self

    def _append_structured_records(self, records: list[dict]) -> None:
        history_df = pd.read_csv(DATA_DIR / "maintenance_history.csv")
        failure_df = pd.read_csv(DATA_DIR / "failure_reports.csv")
        spares_df = pd.read_csv(DATA_DIR / "spares_inventory.csv")
        delay_df = pd.read_csv(DATA_DIR / "delay_logs.csv")
        incident_path = DATA_DIR / "incident_records.csv"
        incident_df = pd.read_csv(incident_path) if incident_path.exists() else pd.DataFrame()
        health_path = DATA_DIR / "asset_health_summary.csv"
        asset_map_df = pd.read_csv(health_path) if health_path.exists() else pd.read_csv(DATA_DIR / "asset_master.csv")
        asset_type_map = asset_map_df.set_index("asset_id")["asset_type"].to_dict()

        def asset_equipment(asset_id):
            return normalize_equipment_type(asset_type_map.get(asset_id, "unknown"))

        for idx, row in history_df.iterrows():
            aid = row_get(row, ["asset_id"])
            records.append(
                {
                    "source": "maintenance_history.csv",
                    "chunk_id": int(idx),
                    "asset_id": aid,
                    "equipment_type": asset_equipment(aid),
                    "issue_type": "history",
                    "text": (
                        f"Historical maintenance record for {aid}. Date: {row_get(row, ['timestamp', 'date'])}. "
                        f"Issue: {row_get(row, ['issue', 'failure_mode', 'problem'])}. "
                        f"Action: {row_get(row, ['action_taken', 'action', 'corrective_action'])}. "
                        f"Result: {row_get(row, ['result', 'outcome'])}. Downtime hours: {row_get(row, ['downtime_hours'], 0)}."
                    ),
                }
            )

        for idx, row in failure_df.iterrows():
            aid = row_get(row, ["asset_id"])
            records.append(
                {
                    "source": "failure_reports.csv",
                    "chunk_id": int(idx),
                    "asset_id": aid,
                    "equipment_type": asset_equipment(aid),
                    "issue_type": "failure report",
                    "text": (
                        f"Failure report for {aid}. Date: {row_get(row, ['timestamp', 'date'])}. "
                        f"Failure mode: {row_get(row, ['failure_mode', 'issue'])}. "
                        f"Root cause: {row_get(row, ['root_cause', 'cause'])}. "
                        f"Corrective action: {row_get(row, ['corrective_action', 'action_taken', 'action'])}. "
                        f"Business impact: {row_get(row, ['business_impact', 'impact'])}."
                    ),
                }
            )

        for idx, row in spares_df.iterrows():
            aid = row_get(row, ["asset_id"])
            records.append(
                {
                    "source": "spares_inventory.csv",
                    "chunk_id": int(idx),
                    "asset_id": aid,
                    "equipment_type": asset_equipment(aid),
                    "issue_type": "spares",
                    "text": (
                        f"Spare inventory for {aid}. Part: {row_get(row, ['spare_part', 'part_name'])}. "
                        f"Stock quantity: {row_get(row, ['stock_qty', 'quantity'])}. "
                        f"Procurement lead time days: {row_get(row, ['lead_time_days', 'procurement_lead_time_days'])}. "
                        f"Spare criticality: {row_get(row, ['spare_criticality', 'criticality'])}. "
                        f"Unit cost INR: {row_get(row, ['unit_cost_inr', 'cost'])}."
                    ),
                }
            )

        for idx, row in delay_df.iterrows():
            aid = row_get(row, ["asset_id"])
            records.append(
                {
                    "source": "delay_logs.csv",
                    "chunk_id": int(idx),
                    "asset_id": aid,
                    "equipment_type": asset_equipment(aid),
                    "issue_type": "delay",
                    "text": (
                        f"Delay log for {aid}. Area: {row_get(row, ['area'])}. "
                        f"Delay hours: {row_get(row, ['delay_hours'], 0)}. "
                        f"Production impact or delay reason: {row_get(row, ['delay_reason', 'production_impact', 'impact'])}."
                    ),
                }
            )

        for idx, row in incident_df.iterrows():
            aid = row_get(row, ["asset_id"])
            records.append(
                {
                    "source": "incident_records.csv",
                    "chunk_id": int(idx),
                    "asset_id": aid,
                    "equipment_type": asset_equipment(aid),
                    "issue_type": "incident",
                    "text": (
                        f"Incident record for {aid}. Severity: {row_get(row, ['severity'])}. "
                        f"Incident: {row_get(row, ['incident_summary', 'summary', 'incident'])}. "
                        f"Recommended response: {row_get(row, ['recommended_response', 'response', 'action'])}."
                    ),
                }
            )

        if health_path.exists():
            for idx, row in pd.read_csv(health_path).iterrows():
                aid = row_get(row, ["asset_id"])
                records.append(
                    {
                        "source": "asset_health_summary.csv",
                        "chunk_id": int(idx),
                        "asset_id": aid,
                        "equipment_type": asset_equipment(aid),
                        "issue_type": "current health",
                        "text": (
                            f"Latest asset health for {aid}. Risk band: {row_get(row, ['risk_band'])}. "
                            f"Hybrid failure risk: {row_get(row, ['failure_risk', 'hybrid_failure_risk'])}. "
                            f"ML failure risk: {row_get(row, ['ml_failure_risk'])}. "
                            f"Operational rule score: {row_get(row, ['operational_rule_score'])}. "
                            f"Hybrid health score: {row_get(row, ['hybrid_health_score'])}. "
                            f"Estimated RUL days: {row_get(row, ['estimated_rul_days'])}. "
                            f"Temperature: {row_get(row, ['temperature'])}. Vibration: {row_get(row, ['vibration'])}. "
                            f"Pressure: {row_get(row, ['pressure'])}. Alarm count: {row_get(row, ['alarm_count'])}."
                        ),
                    }
                )

    def _fit_embeddings(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self.backend = "sentence_transformer"
            self.embedder = SentenceTransformer(EMBED_MODEL_ID, device="cpu")
            self.embeddings = np.array(
                self.embedder.encode(
                    self.doc_df["text"].tolist(),
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    batch_size=16,
                    device="cpu",
                )
            ).astype("float32")
        except Exception:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize

            self.backend = "tfidf"
            self.embedder = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=6000)
            self.embeddings = normalize(self.embedder.fit_transform(self.doc_df["text"].tolist()))

    def query_assets(self, query: str) -> list[str]:
        q = str(query).upper()
        return [asset for asset in self.asset_ids if asset.upper() in q]

    def query_equipment_type(self, query: str) -> str | None:
        q = str(query).lower()
        if any(word in q for word in ["mtr-204", "motor", "overheat", "overheating", "current"]):
            return "motor"
        if any(word in q for word in ["gbx-17", "gearbox", "gear", "vibration", "bearing"]):
            return "gearbox"
        if any(word in q for word in ["pmp-09", "pump", "cavitation", "suction"]):
            return "pump"
        if any(word in q for word in ["hpp-12", "hydraulic", "pressure", "filter", "relief"]):
            return "hydraulic"
        if any(word in q for word in ["public", "dataset", "ai4i", "uci", "source"]):
            return "public_dataset"
        if any(word in q for word in ["caster", "casting", "tundish", "mold", "mould", "breakout"]):
            return "caster"
        if any(word in q for word in ["blast furnace", "tuyere", "stave", "burden", "hot blast"]):
            return "blast_furnace"
        if any(word in q for word in ["rolling", "hot strip", "plate mill", "roll chock", "flatness", "camber"]):
            return "rolling_mill"
        if any(word in q for word in ["safety", "loto", "permit", "isolation", "hazard"]):
            return "safety"
        if any(word in q for word in ["defect", "quality", "surface crack", "scale"]):
            return "quality"
        return None

    def retrieve(self, query: str, top_k: int = 5, asset_id: str | None = None, equipment_type: str | None = None, plant_level: bool = False) -> list[dict]:
        asset_ids = [asset_id] if asset_id else self.query_assets(query)
        equipment_type = normalize_equipment_type(equipment_type) if equipment_type else self.query_equipment_type(query)
        candidates = self.doc_df.copy()

        if not plant_level:
            mask = candidates["equipment_type"].isin(["policy", "public_dataset"])
            if asset_ids:
                mask = mask | candidates["asset_id"].isin(asset_ids)
            if equipment_type:
                mask = mask | candidates["equipment_type"].eq(equipment_type)
            candidates = candidates[mask].copy()

        if candidates.empty:
            candidates = self.doc_df.copy()

        cand_idx = candidates.index.values
        if self.backend == "sentence_transformer":
            q_emb = np.array(self.embedder.encode([query], normalize_embeddings=True, device="cpu")).astype("float32")[0]
            scores = self.embeddings[cand_idx] @ q_emb
        else:
            q_vec = self.embedder.transform([query])
            scores = (self.embeddings[cand_idx] @ q_vec.T).toarray().ravel()

        order = np.argsort(scores)[::-1][: min(top_k, len(scores))]
        results = []
        for pos in order:
            idx = int(cand_idx[pos])
            row = self.doc_df.loc[idx]
            results.append(
                {
                    "score": float(scores[pos]),
                    "source": row["source"],
                    "asset_id": row["asset_id"],
                    "equipment_type": row["equipment_type"],
                    "issue_type": row["issue_type"],
                    "text": row["text"],
                }
            )
        return results
