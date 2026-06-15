"""FastAPI service for the Maintenance Wizard backend."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel

from .agent import MaintenanceWizard


app = FastAPI(title="Tata Steel Maintenance Wizard", version="0.1.0")
wizard = MaintenanceWizard()


class ChatRequest(BaseModel):
    query: str
    user_id: str = "demo_user"


class FeedbackRequest(BaseModel):
    user_id: str = "maintenance_engineer_01"
    asset_id: str
    query: str
    feedback_type: str = "accepted"
    feedback_text: str
    corrected_action: str = ""
    outcome: str = ""


class AlertRequest(BaseModel):
    asset_id: str
    temperature: float
    vibration: float
    current: float
    pressure: float
    rpm: Optional[float] = 1480
    alarm_count: int = 2
    user_id: str = "iot_gateway"


@app.on_event("startup")
def startup() -> None:
    wizard.initialize(load_llm=False)


@app.get("/health")
def health() -> dict:
    wizard.ensure_ready()
    return {
        "status": "ok",
        "assets": wizard.asset_ids,
        "llm_loaded": wizard.llm.available,
        "rag_backend": wizard.rag.backend,
    }


@app.get("/assets")
def assets() -> list[dict]:
    wizard.ensure_ready()
    return wizard.asset_health_table().to_dict("records")


@app.get("/assets/{asset_id}")
def asset(asset_id: str) -> dict:
    wizard.ensure_ready()
    return wizard.get_latest_sensor_summary(asset_id)


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    wizard.ensure_ready()
    return wizard.chat(req.query, user_id=req.user_id)


@app.get("/priority")
def priority() -> dict:
    wizard.ensure_ready()
    return wizard.plant_priority_report("Compare all equipment and tell supervisor what to prioritize today.")


@app.post("/alerts")
def alerts(req: AlertRequest) -> dict:
    wizard.ensure_ready()
    return wizard.ingest_new_sensor_alert(
        asset_id=req.asset_id,
        temperature=req.temperature,
        vibration=req.vibration,
        current=req.current,
        pressure=req.pressure,
        rpm=req.rpm or 1480,
        alarm_count=req.alarm_count,
        user_id=req.user_id,
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest) -> dict:
    wizard.ensure_ready()
    return wizard.save_feedback(
        user_id=req.user_id,
        asset_id=req.asset_id,
        query=req.query,
        feedback_type=req.feedback_type,
        feedback_text=req.feedback_text,
        corrected_action=req.corrected_action,
        outcome=req.outcome,
    )


@app.get("/logbook")
def logbook() -> list[dict]:
    wizard.ensure_ready()
    return wizard.logbook().tail(100).to_dict("records")
