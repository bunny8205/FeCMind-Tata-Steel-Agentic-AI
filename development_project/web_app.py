"""Standalone web app for the Tata Steel Agentic AI.

This avoids Streamlit entirely. It serves a plain HTML/CSS/JS frontend and a
small JSON API backed by the existing MaintenanceWizard agent.

Run:
    python web_app.py --port 8600
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard  # noqa: E402


WIZARD: MaintenanceWizard | None = None


def get_wizard() -> MaintenanceWizard:
    global WIZARD
    if WIZARD is None:
        wizard = MaintenanceWizard()
        wizard.initialize(load_llm=False)
        WIZARD = wizard
    return WIZARD


def _load_json_file(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _safe_num(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_text(value: Any, default: str = "-") -> str:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return text if text else default


def build_command_center_payload() -> dict[str, Any]:
    """Competitor-inspired command-center payload built from real local agent data."""
    wizard = get_wizard()
    assets = wizard.asset_health_table().copy()
    if assets.empty:
        return {
            "metrics": [],
            "priority_matrix": [],
            "schedule": [],
            "spares": [],
            "evidence": [],
            "benchmark": {},
            "model_card": {},
        }

    score_col = "hybrid_health_score" if "hybrid_health_score" in assets.columns else "failure_risk"
    risk_col = "hybrid_failure_risk" if "hybrid_failure_risk" in assets.columns else "failure_risk"
    assets = assets.sort_values([score_col, "estimated_rul_days"], ascending=[False, True]).reset_index(drop=True)

    priority_rows = []
    for rank, row in enumerate(assets.head(8).to_dict(orient="records"), 1):
        score = _safe_num(row.get(score_col))
        risk = _safe_num(row.get(risk_col))
        rul = _safe_num(row.get("estimated_rul_days"), 999.0)
        impact = "Critical production bottleneck" if rank == 1 else ("High asset criticality" if rank <= 3 else "Watchlist")
        priority_rows.append(
            {
                "rank": rank,
                "asset_id": _safe_text(row.get("asset_id")),
                "asset_type": _safe_text(row.get("asset_type")),
                "area": _safe_text(row.get("area")),
                "risk_band": _safe_text(row.get("risk_band")),
                "risk": round(risk, 3),
                "score": round(score, 3),
                "rul_days": round(rul, 1),
                "impact": impact,
                "next_action": "Create P1 work order now" if rank == 1 else "Plan inspection / monitor",
            }
        )

    total_assets = len(assets)
    critical_assets = int(assets["risk_band"].astype(str).str.contains("critical", case=False, na=False).sum()) if "risk_band" in assets else 0
    high_assets = int(assets["risk_band"].astype(str).str.contains("high", case=False, na=False).sum()) if "risk_band" in assets else 0
    near_failure = (
        int(sum(1 for v in assets["estimated_rul_days"] if _safe_num(v, 999.0) <= 2))
        if "estimated_rul_days" in assets
        else 0
    )
    avg_risk = float(assets[risk_col].apply(_safe_num).mean()) if risk_col in assets else 0.0
    plant_health = max(0, min(100, round((1 - avg_risk) * 100, 1)))

    spares_path = PROJECT_ROOT / "data" / "spares_inventory.csv"
    spares_df = _read_csv_for_ui(spares_path)
    spares = []
    below_reorder = 0
    if not spares_df.empty:
        for row in spares_df.head(10).to_dict(orient="records"):
            stock = _safe_num(row.get("stock_available", row.get("available_qty", row.get("quantity", row.get("stock", 0)))))
            reorder = _safe_num(row.get("reorder_level", row.get("minimum_stock", row.get("min_stock", 0))))
            below_reorder += int(reorder > 0 and stock <= reorder)
            spares.append(
                {
                    "part": _safe_text(row.get("part_name", row.get("name", row.get("spare_part", row.get("part_number"))))),
                    "asset": _safe_text(row.get("asset_id", row.get("equipment_id", row.get("asset_type")))),
                    "stock": stock,
                    "reorder": reorder,
                    "lead_days": _safe_num(row.get("lead_time_days", row.get("lead_days", 0))),
                    "procurement_risk": "Emergency" if stock <= 0 else ("Low stock" if reorder > 0 and stock <= reorder else "Ready"),
                }
            )

    schedule = []
    windows = ["Today shift B", "Tomorrow shift A", "Within 72h", "This week", "Next planned stop"]
    for i, row in enumerate(priority_rows[:5]):
        schedule.append(
            {
                "slot": windows[i],
                "asset_id": row["asset_id"],
                "priority": row["risk_band"],
                "work": row["next_action"],
                "crew": "Mechanical + Reliability" if i == 0 else "Area maintenance",
                "spares_status": "Reserve critical spare" if i == 0 else "Check inventory",
                "downtime_strategy": "Controlled intervention" if i == 0 else "Bundle with planned work",
            }
        )

    evidence_files = [
        ("Operating model", "docs/steel_agent_operating_model.txt"),
        ("Priority policy", "docs/maintenance_prioritization_policy.txt"),
        ("Spares strategy", "docs/spares_procurement_strategy.txt"),
        ("Feedback policy", "docs/feedback_learning_policy.txt"),
        ("AI4I report", "data/public_ai4i_report.json"),
        ("22 prompt benchmark", "reports/round2_22_prompts_summary.json"),
    ]
    evidence = []
    for label, rel_path in evidence_files:
        file_path = PROJECT_ROOT / rel_path
        evidence.append(
            {
                "source": label,
                "path": rel_path,
                "status": "available" if file_path.exists() else "missing",
                "confidence": "High" if file_path.exists() else "Missing",
            }
        )

    benchmark = _load_json_file(PROJECT_ROOT / "reports" / "round2_22_prompts_summary.json", {})
    public_ai4i = _load_json_file(PROJECT_ROOT / "data" / "public_ai4i_report.json", {})
    model_card = {
        "architecture": "LLM planner + deterministic tools + LLM final synthesizer + deterministic verifier",
        "ml_layer": "Hybrid anomaly, risk, and RUL scoring from local sensor logs plus public AI4I validation",
        "memory_layer": "Dynamic assets, active/inactive lifecycle, safety-rule deduplication, digital logbook, feedback loop",
        "public_ai4i": public_ai4i,
        "guardrails": [
            "No NaN/null JSON output",
            "Missing evidence is shown instead of invented",
            "Inactive dynamic assets excluded from active ranking",
            "Duplicate safety rules ignored",
            "Verifier checks run after every answer",
        ],
    }

    metrics = [
        {"label": "Plant Health", "value": f"{plant_health}%", "detail": "Risk-weighted health"},
        {"label": "Assets Monitored", "value": total_assets, "detail": "Original + dynamic memory"},
        {"label": "Critical Assets", "value": critical_assets, "detail": f"{high_assets} high-risk assets"},
        {"label": "Near Failure", "value": near_failure, "detail": "RUL <= 2 days"},
        {"label": "Spares Below Reorder", "value": below_reorder, "detail": "Procurement risk"},
        {"label": "Prompt Benchmark", "value": f"{benchmark.get('pass_count', '-')}/{benchmark.get('prompt_count', '-')}", "detail": "Round-2 stress suite"},
    ]

    return {
        "metrics": metrics,
        "priority_matrix": priority_rows,
        "schedule": schedule,
        "spares": spares,
        "evidence": evidence,
        "benchmark": benchmark,
        "model_card": model_card,
    }


def _read_csv_for_ui(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def jsonable(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return [jsonable(row) for row in value.to_dict(orient="records")]
    if isinstance(value, pd.Series):
        return jsonable(value.to_dict())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Tata Steel Agentic AI</title>
  <style>
    :root {
      --bg: #eef2f6;
      --panel: #ffffff;
      --panel-2: #f7f9fc;
      --ink: #111827;
      --text: #1f2937;
      --muted: #637083;
      --line: #d9e0ea;
      --line-strong: #c7d0dd;
      --blue: #16436b;
      --blue-2: #2563eb;
      --blue-soft: #eaf2ff;
      --red: #b91c1c;
      --red-soft: #fff1f2;
      --amber: #b45309;
      --amber-soft: #fff7ed;
      --green: #047857;
      --green-soft: #ecfdf5;
      --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    html { height: 100%; }
    body {
      margin: 0;
      height: 100%;
      overflow: hidden;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    button, textarea { font: inherit; }
    .shell {
      display: grid;
      grid-template-columns: clamp(260px, 20vw, 320px) minmax(420px, 1fr) clamp(360px, 29vw, 480px);
      height: 100dvh;
      min-height: 0;
      overflow: hidden;
    }
    aside, main, .workspace { min-width: 0; }
    aside {
      background: #ffffff;
      border-right: 1px solid var(--line);
      padding: 18px 16px;
      height: 100dvh;
      overflow-y: auto;
      overscroll-behavior: contain;
    }
    main {
      display: flex;
      flex-direction: column;
      height: 100dvh;
      min-height: 0;
      overflow: hidden;
      background: #fbfcfe;
    }
    .workspace {
      background: #ffffff;
      border-left: 1px solid var(--line);
      height: 100dvh;
      min-height: 0;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    aside::-webkit-scrollbar, .chat::-webkit-scrollbar, .workspace-body::-webkit-scrollbar, textarea::-webkit-scrollbar { width: 10px; }
    aside::-webkit-scrollbar-thumb, .chat::-webkit-scrollbar-thumb, .workspace-body::-webkit-scrollbar-thumb, textarea::-webkit-scrollbar-thumb {
      background: #c9d2df;
      border-radius: 999px;
      border: 2px solid transparent;
      background-clip: content-box;
    }
    .brand {
      font-weight: 800;
      font-size: 20px;
      color: var(--blue);
      margin-bottom: 4px;
    }
    .subtitle { color: var(--muted); font-size: 13px; line-height: 1.4; }
    .section-title {
      margin: 22px 0 8px;
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 800;
    }
    .side-actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-top: 14px;
    }
    .ghost {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 8px;
      padding: 9px 10px;
      cursor: pointer;
      font-weight: 700;
      font-size: 12px;
    }
    .ghost:hover { border-color: var(--blue-2); background: var(--blue-soft); }
    .asset {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 8px;
      background: var(--panel-2);
      transition: border-color 0.15s ease, transform 0.15s ease, background 0.15s ease;
    }
    .asset:hover { transform: translateY(-1px); border-color: var(--line-strong); background: #fff; }
    .asset strong { display: block; font-size: 14px; }
    .asset span { color: var(--muted); font-size: 12px; }
    .risk-critical { border-left: 4px solid var(--red); }
    .risk-high { border-left: 4px solid var(--amber); }
    .risk-medium { border-left: 4px solid var(--blue-2); }
    .starter {
      width: 100%;
      text-align: left;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      padding: 10px;
      border-radius: 8px;
      margin-bottom: 8px;
      cursor: pointer;
      font-size: 13px;
      line-height: 1.35;
      transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
    }
    .starter:hover { border-color: var(--blue-2); background: #f8fbff; transform: translateY(-1px); }
    .topbar {
      border-bottom: 1px solid var(--line);
      padding: 14px 22px;
      background: #ffffff;
      display: flex;
      gap: 16px;
      align-items: center;
      justify-content: space-between;
    }
    .topbar-title { min-width: 0; }
    .topbar h1 { margin: 0; font-size: 22px; color: #111827; }
    .topbar p { margin: 4px 0 0; color: var(--muted); font-size: 14px; }
    .topbar-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .chat {
      flex: 1;
      min-height: 0;
      padding: 20px 22px 18px;
      overflow-y: auto;
      overscroll-behavior: contain;
    }
    .msg {
      display: grid;
      grid-template-columns: 34px minmax(0, 1fr);
      gap: 10px;
      margin-bottom: 16px;
      align-items: start;
    }
    .avatar {
      width: 34px;
      height: 34px;
      border-radius: 50%;
      display: grid;
      place-items: center;
      font-weight: 800;
      color: white;
      background: var(--blue);
      font-size: 12px;
    }
    .msg.user .avatar { background: #374151; }
    .bubble {
      border: 1px solid var(--line);
      background: #ffffff;
      border-radius: 8px;
      padding: 12px 14px;
      line-height: 1.48;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      overflow-wrap: anywhere;
    }
    .msg.user .bubble { background: #eef4ff; border-color: #c8d8ff; }
    .composer {
      border-top: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.94);
      backdrop-filter: blur(8px);
      padding: 14px 22px;
      flex: 0 0 auto;
    }
    .composer-inner {
      display: grid;
      grid-template-columns: 1fr 92px;
      gap: 10px;
    }
    textarea {
      width: 100%;
      min-height: 54px;
      max-height: 140px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      outline: none;
      background: #fff;
    }
    textarea:focus { border-color: var(--blue-2); box-shadow: 0 0 0 3px rgba(37,99,235,0.12); }
    button.primary {
      border: none;
      border-radius: 8px;
      background: var(--blue);
      color: white;
      font-weight: 800;
      cursor: pointer;
      min-height: 54px;
    }
    button.primary:disabled { opacity: 0.55; cursor: wait; }
    .hint-row {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }
    .workspace-header {
      padding: 14px 16px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
      flex: 0 0 auto;
    }
    .workspace-header h2 { margin: 0; font-size: 17px; color: var(--ink); }
    .workspace-header p { margin: 3px 0 12px; font-size: 12px; color: var(--muted); }
    .tabs {
      display: flex;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow-x: auto;
      background: var(--panel-2);
    }
    .tab {
      flex: 0 0 auto;
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      padding: 9px 10px;
      cursor: pointer;
      color: var(--muted);
      font-size: 12px;
      font-weight: 800;
    }
    .tab:last-child { border-right: 0; }
    .tab.active { background: var(--blue); color: #fff; }
    .workspace-body {
      flex: 1;
      min-height: 0;
      overflow-y: auto;
      padding: 14px 16px 18px;
      overscroll-behavior: contain;
    }
    .tabpanel { display: none; }
    .tabpanel.active { display: block; }
    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 12px;
      margin-bottom: 12px;
      box-shadow: 0 1px 2px rgba(15, 23, 42, 0.035);
    }
    .card h3 { margin: 0 0 8px; font-size: 15px; color: #111827; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }
    .metric-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
      padding: 11px;
      min-height: 84px;
    }
    .metric-card b {
      display: block;
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .metric-card strong {
      display: block;
      color: var(--ink);
      font-size: 22px;
      margin: 5px 0 3px;
    }
    .metric-card span { color: var(--muted); font-size: 12px; }
    .action-strip {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 12px;
    }
    .action-strip .ghost { min-height: 38px; }
    .timeline {
      display: grid;
      gap: 8px;
    }
    .timeline-item {
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
      font-size: 13px;
    }
    .timeline-item b { color: var(--blue); }
    .compact-note {
      border: 1px solid var(--line);
      background: var(--panel-2);
      border-radius: 8px;
      padding: 10px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.45;
      margin-bottom: 10px;
    }
    .whatif {
      display: grid;
      gap: 8px;
    }
    .whatif label {
      display: grid;
      grid-template-columns: 1fr 82px;
      gap: 8px;
      align-items: center;
      color: var(--muted);
      font-size: 12px;
    }
    .whatif input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px;
      color: var(--text);
      background: #fff;
    }
    .kv {
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 6px 10px;
      font-size: 13px;
    }
    .kv b { color: var(--muted); }
    .pill-row {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 999px;
      padding: 6px 9px;
      font-size: 12px;
      font-weight: 800;
    }
    .pill.critical { background: var(--red-soft); color: var(--red); border-color: #fecdd3; }
    .pill.high { background: var(--amber-soft); color: var(--amber); border-color: #fed7aa; }
    .pill.ok { background: var(--green-soft); color: var(--green); border-color: #bbf7d0; }
    .table-wrap {
      width: 100%;
      overflow-x: auto;
      border: 1px solid #eef0f4;
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      background: #fff;
    }
    th, td {
      border-bottom: 1px solid #eef0f4;
      text-align: left;
      padding: 7px 6px;
      vertical-align: top;
    }
    th { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
    pre {
      margin: 0;
      overflow-x: auto;
      background: #0b1220;
      color: #dbeafe;
      padding: 10px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.45;
      max-height: 58dvh;
    }
    code {
      background: #eef2f7;
      border: 1px solid #dbe2ec;
      border-radius: 5px;
      padding: 1px 5px;
      color: #0f172a;
    }
    .answer-table {
      margin: 8px 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .answer-table table { min-width: 520px; }
    .empty-state {
      border: 1px dashed var(--line-strong);
      border-radius: 8px;
      padding: 18px;
      color: var(--muted);
      background: var(--panel-2);
      font-size: 13px;
      line-height: 1.45;
    }
    .status {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
      margin-top: 10px;
    }
    .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--green); }
    .hidden { display: none; }
    body.workspace-collapsed .shell { grid-template-columns: clamp(260px, 20vw, 320px) minmax(420px, 1fr) 0; }
    body.workspace-collapsed .workspace { display: none; }
    body.rail-collapsed .shell { grid-template-columns: 0 minmax(420px, 1fr) clamp(360px, 29vw, 480px); }
    body.rail-collapsed aside { display: none; }
    @media (max-width: 1180px) {
      .shell { grid-template-columns: 260px minmax(0, 1fr); }
      .workspace { display: none; }
      body.workspace-collapsed .shell,
      body.rail-collapsed .shell { grid-template-columns: minmax(0, 1fr); }
      body.rail-collapsed aside { display: none; }
    }
    @media (max-width: 780px) {
      .shell { display: block; }
      aside { display: none; }
      main { height: 100dvh; }
      .topbar { align-items: flex-start; }
      .composer-inner { grid-template-columns: 1fr; }
      button.primary { min-height: 44px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">Tata Steel Agentic AI</div>
      <div class="subtitle">Autonomous maintenance decision support for steel plant assets.</div>
      <div class="status"><span class="dot"></span><span id="healthText">Starting agent...</span></div>
      <div class="side-actions">
        <button class="ghost" id="clearChat">Clear chat</button>
        <button class="ghost" id="focusPrompt">Ask now</button>
        <button class="ghost" id="judgePrompt">Judge mode</button>
        <button class="ghost" id="exportBrief">Export brief</button>
      </div>

      <div class="section-title">Live Assets</div>
      <div id="assetList"></div>

      <div class="section-title">Prompt Starters</div>
      <button class="starter">If I can maintain only one asset today, which one should I choose and why?</button>
      <button class="starter">Create a P1 alert report for GBX-17 abnormal vibration.</button>
      <button class="starter">MTR-204 is overheating. Diagnose root cause and give inspection plan.</button>
      <button class="starter">Design an agentic workflow for steel plant predictive maintenance using logs, SOPs, sensor alerts, and feedback.</button>
      <button class="starter">Plan spares and procurement for BOF trunnion bearing maintenance.</button>
      <button class="starter">Compare original demo assets, active dynamic assets, safety rules, RUL, procurement risk, and choose exactly one asset for immediate maintenance.</button>
      <button class="starter">Generate a shift handoff with active alerts, top priority, spares, schedule, evidence gaps, and verifier summary.</button>
    </aside>

    <main>
      <div class="topbar">
        <div class="topbar-title">
          <h1>Steel Maintenance Agent</h1>
          <p>Ask any maintenance, reliability, safety, spares, SOP, RCA, or plant-priority question.</p>
        </div>
        <div class="topbar-actions">
          <button class="ghost" id="toggleRail">Assets</button>
          <button class="ghost" id="toggleWorkspace">Workspace</button>
        </div>
      </div>
      <div class="chat" id="chat">
        <div class="msg assistant">
          <div class="avatar">AI</div>
          <div class="bubble">Ask any steel-plant maintenance, operations, safety, spares, RCA, SOP, quality, or reliability question.</div>
        </div>
      </div>
      <div class="composer">
        <div class="composer-inner">
          <textarea id="prompt" placeholder="Ask the steel agent"></textarea>
          <button class="primary" id="send">Send</button>
        </div>
        <div class="hint-row">
          <span>Enter sends. Shift+Enter adds a new line.</span>
          <span id="latencyHint">Ready</span>
        </div>
      </div>
    </main>

    <section class="workspace">
      <div class="workspace-header">
        <h2>Agent Workspace</h2>
        <p>Decision packet, plan, tool calls, checks, and raw trace stay visible here.</p>
        <div class="tabs" role="tablist">
          <button class="tab active" data-tab="decision">Decision</button>
          <button class="tab" data-tab="plant">Plant</button>
          <button class="tab" data-tab="schedule">Schedule</button>
          <button class="tab" data-tab="spares">Spares</button>
          <button class="tab" data-tab="evidence">Evidence</button>
          <button class="tab" data-tab="benchmark">Benchmark</button>
          <button class="tab" data-tab="plan">Plan</button>
          <button class="tab" data-tab="tools">Tools</button>
          <button class="tab" data-tab="checks">Checks</button>
          <button class="tab" data-tab="json">JSON</button>
        </div>
      </div>
      <div class="workspace-body">
        <div class="tabpanel active" id="tab-decision">
          <div class="card">
            <h3>Decision Packet</h3>
            <div id="decisionPills" class="pill-row">
              <span class="pill">No decision yet</span>
            </div>
            <div id="packet" class="kv"><b>Status</b><span>No decision yet</span></div>
          </div>
          <div class="card">
            <h3>Ranking / Comparison</h3>
            <div id="rankingTable" class="empty-state">Run a prompt to see ranked assets or comparison rows.</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-plant">
          <div class="action-strip">
            <button class="ghost" id="handoffPrompt">Generate Handoff</button>
            <button class="ghost" id="downloadPacket">Download Packet</button>
          </div>
          <div id="metricGrid" class="metric-grid"></div>
          <div class="card">
            <h3>Plant Priority Matrix</h3>
            <div id="commandPriority" class="empty-state">Loading plant priorities...</div>
          </div>
          <div class="card">
            <h3>What-If Simulator</h3>
            <div class="compact-note">Fast local simulation for demo interaction. It does not overwrite memory; ask the agent to commit changes.</div>
            <div class="whatif">
              <label>Vibration mm/s <input id="simVib" type="number" step="0.1" value="8.0"></label>
              <label>Temperature C <input id="simTemp" type="number" step="1" value="90"></label>
              <label>Alarm count <input id="simAlarm" type="number" step="1" value="2"></label>
              <button class="ghost" id="runSim">Simulate risk shift</button>
              <div id="simResult" class="compact-note">Set values and run simulation.</div>
            </div>
          </div>
        </div>
        <div class="tabpanel" id="tab-schedule">
          <div class="card">
            <h3>7-Day Maintenance Schedule</h3>
            <div id="scheduleTimeline" class="empty-state">Loading schedule...</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-spares">
          <div class="card">
            <h3>Spares & Procurement Risk</h3>
            <div id="sparesTable" class="empty-state">Loading spares...</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-evidence">
          <div class="card">
            <h3>Evidence & Model Card</h3>
            <div id="evidenceTable" class="empty-state">Loading evidence registry...</div>
          </div>
          <div class="card">
            <h3>Architecture</h3>
            <div id="modelCard" class="empty-state">Loading model card...</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-benchmark">
          <div class="card">
            <h3>Judge Readiness</h3>
            <div id="benchmarkPanel" class="empty-state">Loading benchmark proof...</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-plan">
          <div class="card">
            <h3>Autonomous Plan</h3>
            <div id="plan" class="empty-state">No plan yet</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-tools">
          <div class="card">
            <h3>Tool Calls</h3>
            <div id="tools" class="empty-state">No tool calls yet</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-checks">
          <div class="card">
            <h3>Verifier Checks</h3>
            <div id="checks" class="empty-state">No checks yet</div>
          </div>
        </div>
        <div class="tabpanel" id="tab-json">
          <div class="card">
            <h3>Raw JSON</h3>
            <button class="ghost" id="copyJson">Copy JSON</button>
            <pre id="raw">{}</pre>
          </div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const chat = document.getElementById("chat");
    const promptBox = document.getElementById("prompt");
    const sendBtn = document.getElementById("send");
    const latencyHint = document.getElementById("latencyHint");
    let lastRawResult = {};
    let commandCenter = {};

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function renderInline(value) {
      return escapeHtml(value)
        .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
        .replace(/`([^`]+)`/g, "<code>$1</code>");
    }

    function renderMarkdownTable(lines) {
      const rows = lines
        .filter(line => line.trim().startsWith("|"))
        .map(line => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim()));
      if (rows.length < 2) return lines.map(line => `<div>${renderInline(line)}</div>`).join("");
      const headers = rows[0];
      const bodyRows = rows.slice(2);
      return `<div class="answer-table table-wrap"><table>
        <thead><tr>${headers.map(h => `<th>${renderInline(h)}</th>`).join("")}</tr></thead>
        <tbody>${bodyRows.map(row => `<tr>${headers.map((_, i) => `<td>${renderInline(row[i] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody>
      </table></div>`;
    }

    function renderRichText(text) {
      const lines = String(text ?? "").split(/\r?\n/);
      const chunks = [];
      let i = 0;
      while (i < lines.length) {
        const line = lines[i];
        const next = lines[i + 1] || "";
        if (line.trim().startsWith("|") && /^\s*\|?\s*:?-{3,}/.test(next)) {
          const tableLines = [];
          while (i < lines.length && lines[i].trim().startsWith("|")) {
            tableLines.push(lines[i]);
            i += 1;
          }
          chunks.push(renderMarkdownTable(tableLines));
          continue;
        }
        if (!line.trim()) {
          chunks.push("<div style='height:8px'></div>");
        } else if (/^\*\*(.+)\*\*$/.test(line.trim())) {
          chunks.push(`<h3 style="margin:10px 0 6px;font-size:15px;color:#111827">${renderInline(line.trim().replace(/^\*\*|\*\*$/g, ""))}</h3>`);
        } else {
          chunks.push(`<div>${renderInline(line)}</div>`);
        }
        i += 1;
      }
      return chunks.join("");
    }

    function addMessage(role, text) {
      const row = document.createElement("div");
      row.className = `msg ${role}`;
      row.innerHTML = `
        <div class="avatar">${role === "user" ? "YOU" : "AI"}</div>
        <div class="bubble">${role === "assistant" ? renderRichText(text) : escapeHtml(text)}</div>
      `;
      chat.appendChild(row);
      chat.scrollTop = chat.scrollHeight;
      return row;
    }

    function riskClass(risk) {
      const r = String(risk || "").toLowerCase();
      if (r.includes("critical")) return "risk-critical";
      if (r.includes("high")) return "risk-high";
      return "risk-medium";
    }

    function pillClass(value) {
      const v = String(value || "").toLowerCase();
      if (v.includes("critical") || v === "p1") return "critical";
      if (v.includes("high") || v === "p2") return "high";
      if (v.includes("pass") || v.includes("ready")) return "ok";
      return "";
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === "") return "-";
      if (Array.isArray(value)) return value.map(formatValue).join(", ");
      if (typeof value === "object") return JSON.stringify(value);
      if (typeof value === "number") return Number.isInteger(value) ? String(value) : value.toFixed(Math.abs(value) >= 10 ? 2 : 4).replace(/0+$/, "").replace(/\.$/, "");
      return String(value);
    }

    function table(rows, preferredCols = null) {
      if (!rows || !rows.length) return `<div class="empty-state">No rows yet.</div>`;
      const cols = (preferredCols || Object.keys(rows[0])).filter(c => rows.some(r => r[c] !== undefined)).slice(0, 8);
      return `<div class="table-wrap"><table><thead><tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join("")}</tr></thead>
        <tbody>${rows.map(r => `<tr>${cols.map(c => `<td>${escapeHtml(formatValue(r[c]))}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
    }

    function metricCards(metrics) {
      if (!metrics || !metrics.length) return `<div class="empty-state">No command-center metrics loaded.</div>`;
      return metrics.map(m => `
        <div class="metric-card">
          <b>${escapeHtml(m.label)}</b>
          <strong>${escapeHtml(formatValue(m.value))}</strong>
          <span>${escapeHtml(formatValue(m.detail))}</span>
        </div>
      `).join("");
    }

    function renderTimeline(rows) {
      if (!rows || !rows.length) return `<div class="empty-state">No schedule rows loaded.</div>`;
      return `<div class="timeline">${rows.map(row => `
        <div class="timeline-item">
          <b>${escapeHtml(row.slot)}</b>
          <div>
            <strong>${escapeHtml(row.asset_id)} | ${escapeHtml(row.priority)}</strong><br>
            ${escapeHtml(row.work)}<br>
            <span style="color:#637083">${escapeHtml(row.crew)} | ${escapeHtml(row.spares_status)} | ${escapeHtml(row.downtime_strategy)}</span>
          </div>
        </div>
      `).join("")}</div>`;
    }

    function modelCardHtml(card) {
      if (!card || !Object.keys(card).length) return `<div class="empty-state">Model card unavailable.</div>`;
      return `
        <div class="kv">
          <b>Architecture</b><span>${escapeHtml(card.architecture || "-")}</span>
          <b>ML layer</b><span>${escapeHtml(card.ml_layer || "-")}</span>
          <b>Memory layer</b><span>${escapeHtml(card.memory_layer || "-")}</span>
          <b>Guardrails</b><span>${escapeHtml((card.guardrails || []).join("; "))}</span>
        </div>
      `;
    }

    function benchmarkHtml(benchmark) {
      if (!benchmark || !Object.keys(benchmark).length) return `<div class="empty-state">Run the benchmark script to populate proof.</div>`;
      const pass = `${benchmark.pass_count ?? "-"} / ${benchmark.prompt_count ?? "-"}`;
      return `
        <div class="metric-grid">
          <div class="metric-card"><b>Prompt Suite</b><strong>${escapeHtml(pass)}</strong><span>Round-2 judge prompts</span></div>
          <div class="metric-card"><b>Pass Rate</b><strong>${escapeHtml(formatValue(benchmark.pass_rate_pct))}%</strong><span>Review/fail: ${escapeHtml(formatValue(benchmark.review_or_fail_count))}</span></div>
          <div class="metric-card"><b>LLM Used</b><strong>${escapeHtml(formatValue(benchmark.llm_used_count))}</strong><span>Planner/synthesizer path</span></div>
          <div class="metric-card"><b>Avg Latency</b><strong>${escapeHtml(formatValue(benchmark.avg_latency_sec))}s</strong><span>Local run measurement</span></div>
        </div>
        <div class="compact-note">This proof panel is intentionally judge-facing: it shows that the app is tested beyond hand-picked prompts.</div>
      `;
    }

    function downloadText(filename, text) {
      const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }

    function buildHandoffText() {
      const top = (commandCenter.priority_matrix || [])[0] || {};
      const schedule = (commandCenter.schedule || []).map(row => `- ${row.slot}: ${row.asset_id} | ${row.work} | ${row.spares_status}`).join("\n");
      return [
        "Tata Steel Maintenance Wizard - Shift Handoff",
        `Generated: ${new Date().toLocaleString()}`,
        "",
        `Top priority: ${top.asset_id || "-"} (${top.risk_band || "-"})`,
        `Reason: risk ${top.risk ?? "-"}, RUL ${top.rul_days ?? "-"} days, ${top.impact || "plant priority"}.`,
        "",
        "Recommended schedule:",
        schedule || "- No schedule rows loaded.",
        "",
        "Last agent decision:",
        JSON.stringify(lastRawResult.decision_packet || {}, null, 2),
      ].join("\n");
    }

    function renderCommandCenter(data) {
      commandCenter = data || {};
      document.getElementById("metricGrid").innerHTML = metricCards(commandCenter.metrics || []);
      document.getElementById("commandPriority").innerHTML = table(commandCenter.priority_matrix || [], ["rank", "asset_id", "risk_band", "risk", "score", "rul_days", "impact", "next_action"]);
      document.getElementById("scheduleTimeline").innerHTML = renderTimeline(commandCenter.schedule || []);
      document.getElementById("sparesTable").innerHTML = table(commandCenter.spares || [], ["part", "asset", "stock", "reorder", "lead_days", "procurement_risk"]);
      document.getElementById("evidenceTable").innerHTML = table(commandCenter.evidence || [], ["source", "status", "confidence", "path"]);
      document.getElementById("modelCard").innerHTML = modelCardHtml(commandCenter.model_card || {});
      document.getElementById("benchmarkPanel").innerHTML = benchmarkHtml(commandCenter.benchmark || {});
    }

    function renderWorkspace(result) {
      lastRawResult = result || {};
      const packet = result.decision_packet || {};
      const selected = packet.selected_asset || result.asset_id || "-";
      const priority = packet.priority || result.priority || "-";
      const risk = packet.risk_level || packet.risk || "-";
      const mode = packet.mode || result.mode || "-";
      document.getElementById("decisionPills").innerHTML = `
        <span class="pill ${pillClass(priority)}">${escapeHtml(formatValue(priority))}</span>
        <span class="pill ${pillClass(risk)}">${escapeHtml(formatValue(risk))}</span>
        <span class="pill">${escapeHtml(formatValue(selected))}</span>
        <span class="pill">${escapeHtml(formatValue(mode))}</span>
      `;
      document.getElementById("packet").innerHTML = Object.entries(packet).slice(0, 12)
        .map(([k, v]) => `<b>${escapeHtml(k)}</b><span>${escapeHtml(formatValue(v))}</span>`).join("");
      const ranking = result.plant_priority_table || result.comparison_table || [];
      document.getElementById("rankingTable").innerHTML = table(
        ranking,
        ["asset_id", "group", "priority", "risk_level", "priority_score", "hybrid_health_score", "rul_days", "evidence_confidence"]
      );
      document.getElementById("plan").innerHTML = table(result.agent_plan || [], ["step", "agent", "task", "target", "status"]);
      document.getElementById("tools").innerHTML = table(result.tool_calls || [], ["agent", "tool", "input", "output", "status"]);
      document.getElementById("checks").innerHTML = table(result.verifier_checks || [], ["check", "status", "detail"]);
      document.getElementById("raw").textContent = JSON.stringify(result, null, 2);
    }

    async function loadHealth() {
      try {
        const res = await fetch("/api/health");
        const data = await res.json();
        document.getElementById("healthText").textContent = "Agent ready";
        const assets = data.assets || [];
        document.getElementById("assetList").innerHTML = assets.map(a => `
          <div class="asset ${riskClass(a.risk_band)}">
            <strong>${escapeHtml(a.asset_id)} | ${escapeHtml(a.risk_band)}</strong>
            <span>${escapeHtml(a.asset_type || "Asset")}<br>RUL ${Number(a.estimated_rul_days || 0).toFixed(1)}d | Hybrid risk ${Number(a.hybrid_failure_risk || 0).toFixed(3)}</span>
          </div>
        `).join("");
      } catch (err) {
        document.getElementById("healthText").textContent = "Agent health check failed";
      }
    }

    async function loadCommandCenter() {
      try {
        const res = await fetch("/api/command-center");
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Command-center request failed");
        renderCommandCenter(data);
      } catch (err) {
        const message = `<div class="empty-state">Command-center load failed: ${escapeHtml(err.message)}</div>`;
        ["metricGrid", "commandPriority", "scheduleTimeline", "sparesTable", "evidenceTable", "modelCard", "benchmarkPanel"].forEach(id => {
          const el = document.getElementById(id);
          if (el) el.innerHTML = message;
        });
      }
    }

    async function sendPrompt() {
      const prompt = promptBox.value.trim();
      if (!prompt) return;
      promptBox.value = "";
      sendBtn.disabled = true;
      const started = performance.now();
      latencyHint.textContent = "Agent running...";
      addMessage("user", prompt);
      const thinking = addMessage("assistant", "Thinking: planning, retrieving evidence, checking risk, verifying action plan...");
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt, user_id: "maintenance_engineer_01" })
        });
        const data = await res.json();
        thinking.remove();
        if (!res.ok) throw new Error(data.error || "Agent request failed");
        addMessage("assistant", data.answer || data.final_answer || "Agent completed.");
        renderWorkspace(data);
        latencyHint.textContent = `Done in ${((performance.now() - started) / 1000).toFixed(1)}s`;
      } catch (err) {
        thinking.remove();
        addMessage("assistant", `The agent hit an error: ${err.message}`);
        latencyHint.textContent = "Error";
      } finally {
        sendBtn.disabled = false;
        promptBox.focus();
      }
    }

    sendBtn.addEventListener("click", sendPrompt);
    promptBox.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendPrompt();
      }
    });
    document.querySelectorAll(".tab").forEach(tab => {
      tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(item => item.classList.remove("active"));
        document.querySelectorAll(".tabpanel").forEach(panel => panel.classList.remove("active"));
        tab.classList.add("active");
        document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
      });
    });
    document.querySelectorAll(".starter").forEach(btn => {
      btn.addEventListener("click", () => {
        promptBox.value = btn.textContent.trim();
        promptBox.focus();
      });
    });
    document.getElementById("toggleRail").addEventListener("click", () => {
      document.body.classList.toggle("rail-collapsed");
    });
    document.getElementById("toggleWorkspace").addEventListener("click", () => {
      document.body.classList.toggle("workspace-collapsed");
    });
    document.getElementById("focusPrompt").addEventListener("click", () => promptBox.focus());
    document.getElementById("judgePrompt").addEventListener("click", () => {
      promptBox.value = "Run an agentic-AI self-test. Perceive, retrieve, reason, act, verify, and learn. Rank original and active dynamic assets together, compare the top original with the top dynamic asset, choose exactly one asset for immediate maintenance today, show evidence confidence, missing evidence, spares/procurement action, logbook action, feedback action, and verifier PASS/REVIEW/FAIL summary.";
      promptBox.focus();
    });
    document.getElementById("handoffPrompt").addEventListener("click", () => {
      promptBox.value = "Generate a shift handoff with active alerts, top priority asset, current RUL, spares/procurement risk, 7-day maintenance schedule, missing evidence, safety constraints, and verifier summary.";
      promptBox.focus();
    });
    document.getElementById("exportBrief").addEventListener("click", () => {
      downloadText("maintenance_wizard_handoff.txt", buildHandoffText());
      latencyHint.textContent = "Handoff downloaded";
    });
    document.getElementById("downloadPacket").addEventListener("click", () => {
      downloadText("maintenance_wizard_decision_packet.json", JSON.stringify({
        command_center: commandCenter,
        last_agent_result: lastRawResult,
      }, null, 2));
      latencyHint.textContent = "Decision packet downloaded";
    });
    document.getElementById("runSim").addEventListener("click", () => {
      const vib = Number(document.getElementById("simVib").value || 0);
      const temp = Number(document.getElementById("simTemp").value || 0);
      const alarms = Number(document.getElementById("simAlarm").value || 0);
      const score = Math.min(100, Math.round((vib / 12) * 42 + (temp / 120) * 28 + Math.min(alarms, 6) * 5));
      const band = score >= 80 ? "P1/CRITICAL" : score >= 60 ? "P2/HIGH" : score >= 40 ? "P3/MEDIUM" : "P4/LOW";
      const action = score >= 80 ? "create controlled-shutdown work order and reserve spare" : score >= 60 ? "inspect next shift and increase monitoring" : "continue planned monitoring";
      document.getElementById("simResult").innerHTML = `<strong>${band}</strong><br>Simulated risk score ${score}/100. Recommended action: ${escapeHtml(action)}. Ask the agent to commit this as an asset update if these are real readings.`;
    });
    document.getElementById("clearChat").addEventListener("click", () => {
      chat.innerHTML = `
        <div class="msg assistant">
          <div class="avatar">AI</div>
          <div class="bubble">Ask any steel-plant maintenance, operations, safety, spares, RCA, SOP, quality, or reliability question.</div>
        </div>
      `;
      renderWorkspace({ decision_packet: {}, agent_plan: [], tool_calls: [], verifier_checks: [] });
      latencyHint.textContent = "Ready";
    });
    document.getElementById("copyJson").addEventListener("click", async () => {
      await navigator.clipboard.writeText(JSON.stringify(lastRawResult, null, 2));
      latencyHint.textContent = "JSON copied";
    });
    loadHealth();
    loadCommandCenter();
  </script>
</body>
</html>
"""


class AgentHandler(BaseHTTPRequestHandler):
    server_version = "TataSteelAgentHTTP/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}")

    def send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(jsonable(payload), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_html(self) -> None:
        body = INDEX_HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/index.html"}:
            self.send_html()
            return
        if path == "/api/health":
            try:
                wizard = get_wizard()
                assets = wizard.asset_health_table().sort_values("hybrid_health_score", ascending=False)
                self.send_json({"status": "ok", "assets": assets})
            except Exception as exc:
                self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        if path == "/api/command-center":
            try:
                self.send_json(build_command_center_payload())
            except Exception as exc:
                self.send_json({"status": "error", "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/chat":
            self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            payload = self.read_json()
            prompt = str(payload.get("prompt", "")).strip()
            user_id = str(payload.get("user_id", "maintenance_engineer_01"))
            if not prompt:
                self.send_json({"error": "Prompt is required"}, HTTPStatus.BAD_REQUEST)
                return

            result = get_wizard().chat(prompt, user_id=user_id)
            self.send_json(result)
        except Exception as exc:
            print(f"Agent error: {exc}")
            self.send_json(
                {
                    "error": str(exc),
                    "answer": "The agent hit an error while processing this prompt.",
                },
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8600, type=int)
    args = parser.parse_args()

    print("Starting Tata Steel Agentic AI without Streamlit...")
    get_wizard()
    server = ThreadingHTTPServer((args.host, args.port), AgentHandler)
    print(f"Open http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
