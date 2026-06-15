"""Hugging Face Space frontend for the Tata Steel Maintenance Wizard.

This Space defaults to a fast Qwen3-0.6B triage profile for responsive demos,
with Qwen3-8B + LoRA available for high-fidelity maintenance reasoning. The
backend combines RAG, ML scoring, dynamic memory, safety rules, spares,
logbooks, verifier checks, and Qwen final natural-language synthesis.
"""

from __future__ import annotations

import json
import math
import gc
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
import types
import zipfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MW_USE_LLM", "1")
os.environ.setdefault("MW_LLM_PROVIDER", "local_gpu")
os.environ.setdefault("MW_LLM_MODEL_ID", "Qwen/Qwen3-0.6B")
os.environ.setdefault("MW_LLM_LAZY_LOAD", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard
from backend.dynamic_assets import list_inactive_dynamic_assets, load_dynamic_assets, load_dynamic_rules


BASE_MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen3-0.6B")
ADAPTER_REPO_ID = os.getenv("ADAPTER_REPO_ID", "rn8205/qwen38bfinetuned")
ADAPTER_REPO_TYPE = os.getenv("ADAPTER_REPO_TYPE", "dataset")
ADAPTER_FILENAME = os.getenv("ADAPTER_FILENAME", "qwen3_8b_steel_maintenance_lora.zip")
DEFAULT_MODEL_LABEL = os.getenv("DEFAULT_MODEL_LABEL", "Qwen3-0.6B instant triage")
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "8192"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "1000"))
HARD_MAX_NEW_TOKENS = int(os.getenv("HARD_MAX_NEW_TOKENS", "1100"))
REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "180"))

MODEL_OPTIONS: dict[str, dict[str, Any]] = {
    "Qwen3-8B LoRA fine-tuned": {
        "model_id": "Qwen/Qwen3-8B",
        "adapter": True,
        "provider": "local_gpu_lora",
        "description": "Highest-fidelity fine-tuned maintenance copilot for RCA, SOPs, prioritization, safety reasoning and executive-grade decisions.",
    },
    "Qwen3-4B balanced analysis": {
        "model_id": "Qwen/Qwen3-4B",
        "adapter": False,
        "provider": "local_gpu_base",
        "description": "Balanced Qwen3 reasoning profile for responsive maintenance analysis, troubleshooting and planning demos.",
    },
    "Qwen3-1.7B fast analysis": {
        "model_id": "Qwen/Qwen3-1.7B",
        "adapter": False,
        "provider": "local_gpu_base",
        "description": "Fast Qwen3 analysis profile for quick triage, asset summaries, ranking checks and operator-facing responses.",
    },
    "Qwen3-0.6B instant triage": {
        "model_id": "Qwen/Qwen3-0.6B",
        "adapter": False,
        "provider": "local_gpu_base",
        "description": "Ultra-responsive Qwen3 profile for lightweight checks, fast demos and instant maintenance assistant interactions.",
    },
}
if DEFAULT_MODEL_LABEL not in MODEL_OPTIONS:
    DEFAULT_MODEL_LABEL = "Qwen3-0.6B instant triage"
DEFAULT_MODEL_CONFIG = MODEL_OPTIONS[DEFAULT_MODEL_LABEL]

INIT_LOCK = threading.Lock()
WIZARD: MaintenanceWizard | None = None
INIT_STATUS: dict[str, Any] = {
    "loaded": False,
    "model_id": DEFAULT_MODEL_CONFIG["model_id"],
    "model_label": DEFAULT_MODEL_LABEL,
    "adapter_repo": ADAPTER_REPO_ID,
    "adapter_file": ADAPTER_FILENAME,
    "adapter_enabled": bool(DEFAULT_MODEL_CONFIG.get("adapter")),
    "device": "not loaded",
    "error": "",
}


def json_safe(value: Any) -> Any:
    """Recursively convert objects into strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return json_safe(value.to_dict())
    try:
        import numpy as np

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            value = float(value)
        if isinstance(value, (np.ndarray,)):
            return json_safe(value.tolist())
    except Exception:
        pass
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _download_and_extract_adapter() -> Path:
    from huggingface_hub import hf_hub_download

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    zip_path = hf_hub_download(
        repo_id=ADAPTER_REPO_ID,
        filename=ADAPTER_FILENAME,
        repo_type=ADAPTER_REPO_TYPE,
        token=token or None,
    )
    extract_root = Path(tempfile.gettempdir()) / "qwen3_8b_steel_maintenance_lora"
    adapter_config = list(extract_root.rglob("adapter_config.json")) if extract_root.exists() else []
    if not adapter_config:
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_root)
        adapter_config = list(extract_root.rglob("adapter_config.json"))
    if not adapter_config:
        raise FileNotFoundError("adapter_config.json was not found inside the LoRA zip.")
    return adapter_config[0].parent


def _first_model_device(model: Any):
    try:
        return next(model.parameters()).device
    except Exception:
        return "cuda"


def _selected_model_config(self) -> tuple[str, dict[str, Any]]:
    label = getattr(self, "selected_model_label", DEFAULT_MODEL_LABEL)
    if label not in MODEL_OPTIONS:
        label = DEFAULT_MODEL_LABEL
    return label, MODEL_OPTIONS[label]


def _load_selected_qwen(self):
    """Patch LocalLLM.load so the backend can switch between Qwen models."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        label, config = _selected_model_config(self)
        model_id = config["model_id"]
        use_adapter = bool(config.get("adapter"))
        adapter_dir = None
        # Use the base Qwen tokenizer for deployment. Some exported LoRA zips
        # include tokenizer metadata/chat-template variants that can trip newer
        # Transformers with errors such as "'list' object has no attribute
        # 'keys'". The adapter only changes model weights; it is compatible with
        # the base tokenizer used during fine-tuning.
        tokenizer_source = model_id

        self.model_id = model_id
        self.provider = config["provider"]
        self.is_encoder_decoder = False
        self.load_error = ""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=True,
                use_fast=True,
            )
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source,
                trust_remote_code=True,
                use_fast=False,
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        compute_dtype = torch.float16
        if torch.cuda.is_available():
            try:
                if torch.cuda.is_bf16_supported():
                    compute_dtype = torch.bfloat16
            except Exception:
                compute_dtype = torch.float16

        base_kwargs = {
            "device_map": "auto",
            "torch_dtype": compute_dtype,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        load_attempts: list[tuple[str, dict[str, Any]]] = []
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            load_attempts.append(("4bit_nf4", {**base_kwargs, "quantization_config": bnb_config}))
        load_attempts.append(("native_dtype", base_kwargs))

        base_model = None
        attempt_errors: dict[str, str] = {}
        for load_mode, load_kwargs in load_attempts:
            try:
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        attn_implementation="sdpa",
                        **load_kwargs,
                    )
                except TypeError:
                    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
                INIT_STATUS["load_mode"] = load_mode
                break
            except Exception as exc:
                attempt_errors[load_mode] = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                base_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    try:
                        torch.cuda.ipc_collect()
                    except Exception:
                        pass
        if base_model is None:
            raise RuntimeError(f"{model_id} failed to load. Attempts: {attempt_errors}")

        if use_adapter:
            adapter_dir = _download_and_extract_adapter()
            self.model = PeftModel.from_pretrained(base_model, str(adapter_dir), is_trainable=False)
        else:
            self.model = base_model
        self.model.eval()
        self.load_error = ""
        INIT_STATUS.update(
            {
                "loaded": True,
                "model_label": label,
                "model_id": model_id,
                "adapter_repo": ADAPTER_REPO_ID,
                "adapter_file": ADAPTER_FILENAME,
                "adapter_enabled": use_adapter,
                "adapter_dir": str(adapter_dir) if adapter_dir else "",
                "tokenizer_source": tokenizer_source,
                "device": str(_first_model_device(self.model)),
                "load_mode": INIT_STATUS.get("load_mode", ""),
                "error": "",
                "traceback": "",
            }
        )
    except Exception as exc:
        self.tokenizer = None
        self.model = None
        self.load_error = traceback.format_exc()
        INIT_STATUS.update({"loaded": False, "error": str(exc), "traceback": self.load_error[-3000:]})
    return self


def _generate_qwen3_lora(self, prompt: str, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, max_time: float = 60.0, min_new_tokens: int = 0) -> str:
    if not self.ensure_available():
        return ""
    try:
        import torch

        max_new_tokens = max(1, min(int(max_new_tokens), HARD_MAX_NEW_TOKENS))
        max_time = max(3.0, float(max_time or 30.0))
        user_prompt = f"{prompt.strip()}\n/no_think"
        messages = [{"role": "user", "content": user_prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        device = _first_model_device(self.model)
        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 5,
            "max_time": max_time,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if min_new_tokens > 0:
            generate_kwargs["min_new_tokens"] = min_new_tokens
        with torch.no_grad():
            output = self.model.generate(**inputs, **generate_kwargs)
        generated = output[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        return text
    except Exception as exc:
        self.load_error = str(exc)
        INIT_STATUS.update({"error": str(exc)})
        return ""


def patch_llm_for_qwen(wizard: MaintenanceWizard, model_label: str = DEFAULT_MODEL_LABEL) -> None:
    if model_label not in MODEL_OPTIONS:
        model_label = DEFAULT_MODEL_LABEL
    config = MODEL_OPTIONS[model_label]
    wizard.llm.enabled = True
    wizard.llm.selected_model_label = model_label
    wizard.llm.model_id = config["model_id"]
    wizard.llm.provider = config["provider"]
    wizard.llm.load = types.MethodType(_load_selected_qwen, wizard.llm)
    wizard.llm.generate = types.MethodType(_generate_qwen3_lora, wizard.llm)


def update_pending_model_status(model_label: str) -> None:
    if model_label not in MODEL_OPTIONS:
        model_label = DEFAULT_MODEL_LABEL
    config = MODEL_OPTIONS[model_label]
    INIT_STATUS.update(
        {
            "loaded": False,
            "model_label": model_label,
            "model_id": config["model_id"],
            "adapter_enabled": bool(config.get("adapter")),
            "adapter_repo": ADAPTER_REPO_ID if config.get("adapter") else "",
            "adapter_file": ADAPTER_FILENAME if config.get("adapter") else "",
            "adapter_dir": "",
            "device": "loading",
            "error": "",
            "traceback": "",
        }
    )


def unload_llm(wizard: MaintenanceWizard) -> None:
    try:
        if getattr(wizard.llm, "model", None) is not None:
            del wizard.llm.model
        if getattr(wizard.llm, "tokenizer", None) is not None:
            del wizard.llm.tokenizer
        wizard.llm.model = None
        wizard.llm.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
    except Exception:
        pass


def runtime_status(wizard: MaintenanceWizard) -> str:
    status = {
        **INIT_STATUS,
        "selected_model": getattr(wizard.llm, "selected_model_label", DEFAULT_MODEL_LABEL),
        "available": bool(wizard.llm.available),
        "load_error": wizard.llm.load_error,
        "provider": wizard.llm.provider,
        "model_options": list(MODEL_OPTIONS.keys()),
    }
    return json.dumps(json_safe(status), indent=2)


def get_wizard(model_label: str | None = None) -> MaintenanceWizard:
    global WIZARD
    if model_label and model_label not in MODEL_OPTIONS:
        model_label = DEFAULT_MODEL_LABEL
    if WIZARD is not None:
        if model_label and getattr(WIZARD.llm, "selected_model_label", DEFAULT_MODEL_LABEL) != model_label:
            unload_llm(WIZARD)
            patch_llm_for_qwen(WIZARD, model_label)
            update_pending_model_status(model_label)
            WIZARD.llm.load()
        return WIZARD
    with INIT_LOCK:
        if WIZARD is None:
            wizard = MaintenanceWizard()
            wizard.initialize(load_llm=False)
            selected = model_label or DEFAULT_MODEL_LABEL
            patch_llm_for_qwen(wizard, selected)
            update_pending_model_status(selected)
            wizard.llm.load()
            WIZARD = wizard
    return WIZARD


def switch_model(model_label: str) -> str:
    global WIZARD
    if model_label not in MODEL_OPTIONS:
        model_label = DEFAULT_MODEL_LABEL
    if WIZARD is None:
        WIZARD = MaintenanceWizard()
        WIZARD.initialize(load_llm=False)
    else:
        unload_llm(WIZARD)
    patch_llm_for_qwen(WIZARD, model_label)
    update_pending_model_status(model_label)
    WIZARD.llm.load()
    return runtime_status(WIZARD)


def warmup_model(model_label: str | None = None, force_retry: bool = False) -> str:
    wizard = get_wizard(model_label if model_label in MODEL_OPTIONS else None)
    if force_retry or not bool(getattr(wizard.llm, "available", False)):
        try:
            wizard.llm.load()
        except Exception:
            pass
    return runtime_status(wizard)


def compact_result(result: dict) -> dict:
    keys = [
        "mode",
        "asset_id",
        "priority",
        "llm_used",
        "llm_validation",
        "llm_planner",
        "llm_synthesizer",
        "decision_packet",
        "evidence_confidence",
        "alert_report",
    ]
    return {key: json_safe(result.get(key)) for key in keys if key in result}


def activity_stages_for_prompt(message: str) -> list[str]:
    q = str(message or "").lower()
    if "sop" in q or "standard operating procedure" in q or ("replace" in q and any(x in q for x in ["seal", "pump", "bearing", "gearbox", "motor"])):
        return [
            "LLM planner: classify as SOP/checklist request",
            "Retrieval: prioritize SOP/manual/safety evidence",
            "Safety: add LOTO, zero-energy and permit hold points",
            "Procedure: assemble step-by-step field checklist",
            "Verifier: block invented torque, limits or part numbers",
            "Qwen writer: produce engineer-facing SOP answer",
        ]
    if re.search(r"\b(?:error|fault)\s+code\b", q) or re.search(r"\be[- ]?\d{2,4}\b", q):
        return [
            "LLM planner: classify as error-code lookup",
            "Code guard: preserve the exact fault code",
            "Retrieval: search OEM/manual/code evidence",
            "Safety: prepare no-blind-reset guidance",
            "Verifier: reject unsupported code meanings",
            "Qwen writer: produce concise answer-first response",
        ]
    if any(x in q for x in ["only one", "choose", "rank", "priority", "prioritize", "immediate maintenance"]):
        return [
            "LLM planner: classify plant-priority scope",
            "Sensor tools: score health, anomaly and RUL",
            "Memory tools: include active dynamic assets only",
            "Rules tools: apply matching safety rules",
            "Procurement tools: check spares and lead-time risk",
            "Qwen writer: explain winner and second-place tradeoff",
        ]
    if any(x in q for x in ["spare", "procurement", "lead time", "inventory", "stock"]):
        return [
            "LLM planner: classify spares/procurement request",
            "Retrieval: identify equipment boundary",
            "Inventory tools: check stock and lead-time fields",
            "Risk tools: estimate delay and substitute exposure",
            "Verifier: avoid invented part numbers or stock",
            "Qwen writer: produce procurement-ready answer",
        ]
    if any(x in q for x in ["stopped", "tripped", "right now", "first checks", "walk me through"]):
        return [
            "LLM planner: classify emergency troubleshooting",
            "Safety tools: check stop-work and LOTO conditions",
            "Retrieval: gather equipment-class guidance",
            "Reasoning: order immediate checks before RCA",
            "Verifier: keep advice field-safe",
            "Qwen writer: produce action-first checklist",
        ]
    return [
        "LLM planner: understand objective and route",
        "Retrieval: search RAG, SOPs, history and policies",
        "Tool layer: calculate any risk, RUL, spares or memory facts",
        "Safety: check LOTO, escalation and missing evidence",
        "Verifier: preserve locked facts and reject hallucinations",
        "Qwen writer: produce final natural-language answer",
    ]


def format_activity(
    result: dict | None = None,
    running: bool = False,
    message: str = "",
    stage: str = "",
    stages: list[str] | None = None,
) -> str:
    if running:
        current_stage = stage or "Planning route"
        stage_items = stages or activity_stages_for_prompt(message)
        stage_lines = []
        for item in stage_items:
            marker = "&#9658;" if item == current_stage else "&rarr;"
            strong = "**" if item == current_stage else ""
            stage_lines.append(f"{marker} {strong}{item}{strong}")
        return f"""
### Agent Activity

<span class="activity-hint">Current step: {current_stage}</span>

{chr(10).join(stage_lines)}

<span class="activity-hint">Running now for: {message[:180]}</span>
"""
    if not result:
        return """
### Agent Activity

Ready. Ask a maintenance, reliability, safety, spares, SOP, RCA, quality or operations question.
"""
    plan = result.get("agent_plan") if isinstance(result.get("agent_plan"), list) else []
    calls = result.get("tool_calls") if isinstance(result.get("tool_calls"), list) else []
    checks = result.get("verifier_checks") if isinstance(result.get("verifier_checks"), list) else []
    lines = ["### Agent Activity"]
    if plan:
        lines.append("\n**Plan executed**")
        for item in plan[:8]:
            step = item.get("step", "-")
            agent = item.get("agent", "Agent")
            task = item.get("task", item.get("objective", "ran step"))
            status = item.get("status", "complete")
            lines.append(f"&rarr; `{step}` **{agent}** {task}  \n   `{status}`")
    if calls:
        lines.append("\n**Tool calls**")
        for call in calls[:8]:
            agent = call.get("agent", "Tool Agent")
            tool = call.get("tool", "tool")
            output = str(call.get("output", ""))[:120]
            status = call.get("status", "success")
            lines.append(f"&rarr; **{agent}** ran `{tool}`  \n   `{status}` - {output}")
    if checks:
        pass_count = sum(1 for c in checks if str(c.get("status", "")).lower() == "pass")
        review_count = sum(1 for c in checks if str(c.get("status", "")).lower() == "review")
        fail_count = sum(1 for c in checks if str(c.get("status", "")).lower() == "fail")
        lines.append(f"\n**Verifier summary**  \n&rarr; PASS `{pass_count}` - REVIEW `{review_count}` - FAIL `{fail_count}`")
        for check in checks[:5]:
            lines.append(f"&rarr; {check.get('check', 'check')} - `{check.get('status', '')}`")
    synth = result.get("llm_synthesizer")
    if isinstance(synth, dict):
        lines.append(f"\n**Qwen final writer**  \n&rarr; `{synth.get('status', 'not attempted')}` - `{synth.get('model_id', BASE_MODEL_ID)}`")
    return "\n".join(lines)


def format_answer_phase(result: dict | None = None) -> str:
    """Keep the activity panel quiet once the user-facing answer starts typing."""
    selected = "plant context"
    priority = "contextual"
    if isinstance(result, dict):
        packet = result.get("decision_packet") if isinstance(result.get("decision_packet"), dict) else {}
        selected = str(
            packet.get("selected_asset")
            or packet.get("asset_id")
            or result.get("asset_id")
            or selected
        )
        priority = str(
            packet.get("priority")
            or packet.get("risk_band")
            or result.get("priority")
            or priority
        )
    return f"""
### Agent Activity

Answer is being written in the chat. Live execution steps are hidden now so the final response stays clean.

<span class="activity-hint">Resolved context: {selected} | Priority: {priority}</span>
"""


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        if path.exists() and path.stat().st_size > 0:
            return pd.read_csv(path)
    except Exception:
        pass
    return pd.DataFrame()


def _num(series_or_value: Any, default: float = 0.0):
    try:
        if isinstance(series_or_value, pd.Series):
            return pd.to_numeric(series_or_value, errors="coerce").fillna(default)
        value = float(series_or_value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def asset_intelligence() -> pd.DataFrame:
    decision_path = PROJECT_ROOT / "data" / "asset_decision_intelligence.csv"
    health_path = PROJECT_ROOT / "data" / "asset_health_summary.csv"
    df = _read_csv(decision_path)
    if df.empty:
        df = _read_csv(health_path)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if "active" not in df.columns:
        df["active"] = True
    if "priority" not in df.columns:
        df["priority"] = df.get("risk_band", "UNKNOWN")
    if "decision_score" not in df.columns:
        df["decision_score"] = df.get("hybrid_health_score", df.get("failure_risk", 0))
    if "hybrid_health_score" not in df.columns:
        df["hybrid_health_score"] = df["decision_score"]
    if "estimated_rul_days" not in df.columns:
        df["estimated_rul_days"] = 0
    if "procurement_risk" not in df.columns:
        df["procurement_risk"] = "UNKNOWN"
    if "evidence_confidence" not in df.columns:
        df["evidence_confidence"] = "REVIEW"
    if "delay_cost_impact_inr" not in df.columns:
        df["delay_cost_impact_inr"] = 0
    for col in [
        "decision_score",
        "hybrid_health_score",
        "hybrid_failure_risk",
        "failure_risk",
        "estimated_rul_days",
        "delay_cost_impact_inr",
        "spare_stock_qty",
        "max_spare_lead_time_days",
    ]:
        if col in df.columns:
            df[col] = _num(df[col])
    df["active"] = df["active"].astype(str).str.lower().isin(["true", "1", "yes", "active"])
    return df.sort_values(["decision_score", "estimated_rul_days"], ascending=[False, True]).reset_index(drop=True)


def _risk_color(value: Any) -> str:
    text = str(value).upper()
    if "CRITICAL" in text or text == "P1":
        return "#ef4444"
    if "HIGH" in text or text == "P2":
        return "#f59e0b"
    if "MEDIUM" in text or text == "P3":
        return "#06b6d4"
    return "#22c55e"


def dashboard_summary() -> str:
    df = asset_intelligence()
    if df.empty:
        return "<div class='metric-grid'><div class='metric-card'>No asset data loaded</div></div>"
    active = df[df["active"]].copy()
    critical = int(active["risk_band"].astype(str).str.contains("critical", case=False, na=False).sum()) if "risk_band" in active else 0
    high = int(active["risk_band"].astype(str).str.contains("high", case=False, na=False).sum()) if "risk_band" in active else 0
    near = int((_num(active["estimated_rul_days"], 999) <= 2).sum()) if "estimated_rul_days" in active else 0
    delay = float(_num(active.get("delay_cost_impact_inr", pd.Series(dtype=float))).sum())
    top = active.iloc[0] if not active.empty else df.iloc[0]
    top_asset = str(top.get("asset_id", "-"))
    top_score = _num(top.get("decision_score", top.get("hybrid_health_score", 0)))
    top_rul = _num(top.get("estimated_rul_days", 0))
    return f"""
    <div class="metric-grid">
      <div class="metric-card"><div class="metric-label">Top asset</div><div class="metric-value">{top_asset}</div><div class="metric-sub">score {top_score:.1f} · RUL {top_rul:.1f}d</div></div>
      <div class="metric-card danger"><div class="metric-label">Critical assets</div><div class="metric-value">{critical}</div><div class="metric-sub">{high} high-risk watchlist</div></div>
      <div class="metric-card amber"><div class="metric-label">RUL ≤ 2 days</div><div class="metric-value">{near}</div><div class="metric-sub">immediate planning queue</div></div>
      <div class="metric-card sky"><div class="metric-label">Delay exposure</div><div class="metric-value">₹{delay/1_000_000:.1f}M</div><div class="metric-sub">current estimated impact</div></div>
    </div>
    """


def risk_bar_plot():
    df = asset_intelligence().head(12)
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df,
        x="asset_id",
        y="decision_score",
        color="risk_band" if "risk_band" in df.columns else "priority",
        hover_data=[col for col in ["asset_type", "area", "estimated_rul_days", "evidence_confidence", "procurement_risk"] if col in df.columns],
        title="Priority Score by Asset",
        color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#06b6d4", "LOW": "#22c55e"},
    )
    fig.update_layout(template="plotly_dark", height=390, margin=dict(l=20, r=20, t=55, b=30), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig


def rul_risk_scatter():
    df = asset_intelligence().copy()
    if df.empty:
        return go.Figure()
    y_col = "decision_score"
    size_col = "delay_cost_impact_inr" if "delay_cost_impact_inr" in df.columns else None
    fig = px.scatter(
        df,
        x="estimated_rul_days",
        y=y_col,
        size=size_col,
        color="evidence_confidence",
        hover_name="asset_id",
        hover_data=[col for col in ["asset_type", "area", "risk_band", "procurement_risk", "missing_evidence"] if col in df.columns],
        title="RUL vs Priority Score with Evidence Confidence",
        size_max=34,
    )
    fig.update_xaxes(autorange="reversed", title="Estimated RUL days (lower is more urgent)")
    fig.update_yaxes(title="Decision score")
    fig.update_layout(template="plotly_dark", height=390, margin=dict(l=20, r=20, t=55, b=30), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig


def rul_bar_plot():
    df = asset_intelligence().head(12)
    if df.empty:
        return go.Figure()
    fig = px.bar(
        df.sort_values("estimated_rul_days", ascending=True),
        x="estimated_rul_days",
        y="asset_id",
        orientation="h",
        color="risk_band" if "risk_band" in df.columns else "priority",
        title="Remaining Useful Life Watchlist",
        hover_data=[col for col in ["asset_type", "area", "decision_score", "next_system_action"] if col in df.columns],
        color_discrete_map={"CRITICAL": "#ef4444", "HIGH": "#f59e0b", "MEDIUM": "#06b6d4", "LOW": "#22c55e"},
    )
    fig.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=55, b=30), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig


def spares_plot():
    spares = _read_csv(PROJECT_ROOT / "data" / "spares_inventory.csv")
    if spares.empty:
        return go.Figure()
    spares = spares.copy()
    spares["lead_time_days"] = _num(spares.get("lead_time_days", 0))
    spares["stock_qty"] = _num(spares.get("stock_qty", 0))
    fig = px.scatter(
        spares,
        x="lead_time_days",
        y="stock_qty",
        color="spare_criticality",
        size="unit_cost_inr" if "unit_cost_inr" in spares else None,
        hover_name="spare_part",
        hover_data=["asset_id"],
        title="Spares Availability vs Procurement Lead Time",
        color_discrete_map={"Critical": "#ef4444", "High": "#f59e0b", "Medium": "#06b6d4", "Low": "#22c55e"},
    )
    fig.update_layout(template="plotly_dark", height=390, margin=dict(l=20, r=20, t=55, b=30), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig


def _empty_plot(title: str = "No visual required"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=360,
        margin=dict(l=20, r=20, t=55, b=30),
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
    )
    return fig


def original_vs_dynamic_plot():
    df = asset_intelligence().copy()
    if df.empty:
        return _empty_plot("No asset comparison data loaded")
    if "active" in df.columns:
        df = df[df["active"]].copy()
    if df.empty:
        return _empty_plot("No active assets to compare")
    if "is_dynamic" in df.columns:
        dynamic_mask = df["is_dynamic"].astype(str).str.lower().isin(["1", "true", "yes"])
        original = df[~dynamic_mask].head(1)
        dynamic = df[dynamic_mask].head(1)
        view = pd.concat([original, dynamic], ignore_index=True)
        if not view.empty:
            view["group"] = view["is_dynamic"].map(lambda x: "Dynamic memory" if str(x).lower() in ["1", "true", "yes"] else "Original demo")
        else:
            view = df.head(6).copy()
            view["group"] = view.get("data_origin", "asset")
    else:
        view = df.head(6).copy()
        view["group"] = view.get("data_origin", "asset")
    fig = px.bar(
        view,
        x="asset_id",
        y="decision_score",
        color="group",
        barmode="group",
        hover_data=[col for col in ["risk_band", "estimated_rul_days", "evidence_confidence", "procurement_risk", "next_system_action"] if col in view.columns],
        title="Original vs Dynamic Asset Priority",
    )
    fig.update_layout(template="plotly_dark", height=390, margin=dict(l=20, r=20, t=55, b=30), paper_bgcolor="#0f172a", plot_bgcolor="#0f172a")
    return fig


def agent_visual_for_query(message: str, result: dict | None = None):
    q = (message or "").lower()
    packet = result.get("decision_packet", {}) if isinstance(result, dict) and isinstance(result.get("decision_packet"), dict) else {}
    mode = str(packet.get("mode", result.get("mode", "") if isinstance(result, dict) else "")).lower()
    wants_visual = any(
        token in q
        for token in [
            "graph",
            "chart",
            "plot",
            "visual",
            "compare",
            "side-by-side",
            "rank",
            "ranking",
            "which one",
            "only one asset",
            "priority",
            "rul",
            "remaining useful",
            "spare",
            "procurement",
            "lead time",
        ]
    ) or any(token in mode for token in ["comparison", "priority", "ranking"])
    if not wants_visual:
        return gr.update(visible=False)
    if any(token in q for token in ["spare", "procurement", "lead time", "inventory", "stock"]):
        fig = spares_plot()
    elif any(token in q for token in ["compare", "side-by-side", "dynamic", "original"]):
        fig = original_vs_dynamic_plot()
    elif any(token in q for token in ["rul", "remaining useful", "life", "trend"]):
        fig = rul_risk_scatter()
    else:
        fig = risk_bar_plot()
    return gr.update(value=fig, visible=True)


def model_diagnostics() -> str:
    report_path = PROJECT_ROOT / "data" / "public_ai4i_report.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    except Exception:
        report = {}
    df = asset_intelligence()
    active = df[df["active"]] if not df.empty and "active" in df.columns else df
    rows = int(report.get("public_rows", 0))
    max_risk = float(_num(active.get("failure_risk", active.get("hybrid_failure_risk", pd.Series([0])))).max()) if not active.empty else 0.0
    return f"""
### Model Credibility

- **LLM:** default Qwen3-0.6B instant triage for responsive demos; selectable Qwen3-8B + Tata Steel maintenance LoRA for high-fidelity reasoning.
- **Agent loop:** LLM planner → deterministic ML/RAG/memory/tools → Qwen final synthesis → deterministic verifier.
- **Predictive stack:** hybrid failure risk, anomaly flags, RUL estimation, delay impact, procurement risk and evidence confidence.
- **Public benchmark:** AI4I public dataset available: `{bool(report.get("public_ai4i_available", False))}`; rows: `{rows}`.
- **Leakage control:** machine failure used only as target: `{report.get("leakage_control", {}).get("machine_failure_used_only_as_target", "unknown")}`.
- **Current active assets:** `{len(active)}`; max failure risk: `{max_risk:.3f}`.
"""


def knowledge_table() -> pd.DataFrame:
    rows = []
    for file in sorted((PROJECT_ROOT / "docs").glob("*.txt")):
        text = file.read_text(encoding="utf-8", errors="ignore")
        rows.append({"source": file.name, "type": "SOP / policy", "chars": len(text), "preview": " ".join(text.split())[:180]})
    rag = _read_csv(PROJECT_ROOT / "data" / "rag_documents.csv")
    if not rag.empty:
        for row in rag.head(15).to_dict(orient="records"):
            rows.append({
                "source": str(row.get("source", "rag_documents.csv")),
                "type": str(row.get("issue_type", "RAG")),
                "chars": len(str(row.get("text", ""))),
                "preview": " ".join(str(row.get("text", "")).split())[:180],
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"source": "No evidence files found"}])


def project_file_inventory() -> pd.DataFrame:
    rows = []
    allowed_suffixes = {".csv", ".txt", ".json", ".md", ".yaml", ".yml"}
    for root_name in ["data", "docs", "backend"]:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for file in sorted(root.rglob("*")):
            if not file.is_file() or file.suffix.lower() not in allowed_suffixes:
                continue
            try:
                rel = file.relative_to(PROJECT_ROOT).as_posix()
            except Exception:
                rel = str(file)
            rows.append(
                {
                    "file": rel,
                    "type": file.suffix.lower().lstrip("."),
                    "size_kb": round(file.stat().st_size / 1024, 2),
                    "agent_access": "available",
                }
            )
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"file": "No project files found"}])


def workplan_table() -> pd.DataFrame:
    df = asset_intelligence().copy()
    if df.empty:
        return pd.DataFrame([{"status": "No workplan data loaded"}])
    cols = [
        "asset_id",
        "priority",
        "risk_band",
        "decision_score",
        "estimated_rul_days",
        "procurement_risk",
        "delay_cost_impact_inr",
        "spare_plan",
        "next_system_action",
    ]
    cols = [col for col in cols if col in df.columns]
    return df[cols].head(15)


def run_stop_controls(is_running: bool):
    if is_running:
        return gr.update(value="Run Agent", visible=False), gr.update(value="Stop", visible=True, variant="stop")
    return gr.update(value="Run Agent", visible=True, variant="primary"), gr.update(value="Stop", visible=False)


def ask_agent(
    message: str,
    history: list[dict] | None,
    operator_role: str = "Maintenance Engineer",
    model_label: str = DEFAULT_MODEL_LABEL,
):
    history = history or []
    message = (message or "").strip()
    if not message:
        empty = json.dumps({}, indent=2)
        yield history, "", gr.update(value="", visible=False), gr.update(visible=False), format_activity(), *run_stop_controls(False), empty, "[]", "[]", "[]", asset_table(), memory_table()
        return

    running_stages = activity_stages_for_prompt(message)
    running_history = history + [(message, "●")]
    empty_packet = json.dumps({}, indent=2)
    assets_snapshot = asset_table()
    memory_snapshot = memory_table()
    loading_bubble = "<div class='typing-dots'><span></span><span></span><span></span></div>"
    running_history[-1] = (message, loading_bubble)
    yield (
        running_history,
        "",
        gr.update(value=format_activity(running=True, message=message, stage=running_stages[0], stages=running_stages), visible=True),
        gr.update(visible=False),
        format_activity(running=True, message=message, stage=running_stages[0], stages=running_stages),
        *run_stop_controls(True),
        empty_packet,
        "[]",
        "[]",
        "[]",
        assets_snapshot,
        memory_snapshot,
    )

    result_box: dict[str, Any] = {}
    error_box: dict[str, str] = {}

    def run_agent() -> None:
        try:
            wizard = get_wizard(model_label)
            wizard.session_memory["operator_role"] = operator_role
            wizard.session_memory["role_duties"] = ROLE_DUTIES.get(operator_role, ROLE_DUTIES["Maintenance Engineer"])
            if is_casual_chat(message):
                result_box["result"] = casual_chat_result(wizard, message, operator_role)
            else:
                result_box["result"] = wizard.chat(message, user_id="hf_space_user")
        except Exception as exc:
            error_box["error"] = str(exc)
            error_box["traceback"] = traceback.format_exc()

    worker = threading.Thread(target=run_agent, daemon=True)
    worker.start()
    started_at = time.monotonic()
    tick = 0
    while worker.is_alive():
        elapsed = time.monotonic() - started_at
        if elapsed > REQUEST_TIMEOUT_SECONDS:
            timeout_answer = (
                "This request took too long while the GPU model was loading or generating. "
                "The app stopped waiting so the UI does not hang. Please retry once the model is warm, "
                "or press **Warm Up / Check Model** first."
            )
            timeout_history = history + [(message, timeout_answer)]
            timeout_activity = format_activity(
                running=True,
                message=message,
                stage=f"Timed out after {int(elapsed)} seconds; GPU/model may still be warming",
                stages=running_stages,
            )
            timeout_packet = json.dumps(
                {
                    "mode": "request_timeout",
                    "objective": message,
                    "elapsed_seconds": round(elapsed, 1),
                    "advice": "Retry after warmup or reduce prompt size.",
                },
                indent=2,
            )
            yield timeout_history, "", gr.update(value="", visible=False), gr.update(visible=False), timeout_activity, *run_stop_controls(False), timeout_packet, "[]", "[]", "[]", assets_snapshot, memory_snapshot
            return
        running_history[-1] = (message, loading_bubble)
        yield (
            running_history,
            "",
            gr.update(
                value=format_activity(
                    running=True,
                    message=message,
                    stage=running_stages[tick % len(running_stages)],
                    stages=running_stages,
                ),
                visible=True,
            ),
            gr.update(visible=False),
            format_activity(
                running=True,
                message=message,
                stage=running_stages[tick % len(running_stages)],
                stages=running_stages,
            ),
            *run_stop_controls(True),
            empty_packet,
            "[]",
            "[]",
            "[]",
            assets_snapshot,
            memory_snapshot,
        )
        tick += 1
        time.sleep(0.55)
    worker.join()

    if error_box:
        result = {
            "answer": f"The agent hit an error while generating the answer: {error_box.get('error', 'unknown error')}",
            "final_answer": f"The agent hit an error while generating the answer: {error_box.get('error', 'unknown error')}",
            "agent_plan": [],
            "tool_calls": [],
            "verifier_checks": [{"check": "Agent execution completed", "status": "fail", "detail": error_box.get("error", "")}],
            "decision_packet": {"mode": "runtime_error", "objective": message},
        }
    else:
        result = result_box.get("result", {})
    answer = str(result.get("final_answer") or result.get("answer") or "").strip()
    if not answer:
        answer = "I could not generate a safe answer from the available tools. Please add the asset ID, readings, or the exact maintenance objective."
    answer = clean_visible_answer(answer)

    plan = json_safe(result.get("agent_plan", []))
    calls = json_safe(result.get("tool_calls", []))
    checks = json_safe(result.get("verifier_checks", []))
    decision = json.dumps(compact_result(result), indent=2, ensure_ascii=False)
    plan_text = json.dumps(plan, indent=2, ensure_ascii=False)
    calls_text = json.dumps(calls, indent=2, ensure_ascii=False)
    checks_text = json.dumps(checks, indent=2, ensure_ascii=False)
    assets = asset_table()
    memory = memory_table()
    visual_update = agent_visual_for_query(message, result)
    answer_activity = format_answer_phase(result)

    typed_history = history + [(message, "")]
    chunk_size = 1 if len(answer) <= 2600 else 2
    for end in range(chunk_size, len(answer) + chunk_size, chunk_size):
        partial = answer[:end]
        cursor = "" if end >= len(answer) else "▌"
        typed_history[-1] = (message, partial + cursor)
        yield typed_history, "", gr.update(value="", visible=False), visual_update, answer_activity, *run_stop_controls(False), decision, plan_text, calls_text, checks_text, assets, memory
        if len(answer) < 4000:
            time.sleep(0.004)

    typed_history[-1] = (message, answer)
    yield (
        typed_history,
        "",
        gr.update(value="", visible=False),
        visual_update,
        answer_activity,
        *run_stop_controls(False),
        decision,
        plan_text,
        calls_text,
        checks_text,
        assets,
        memory,
    )


def ask_agent_from_demo(
    demo_prompt: str,
    history: list[dict] | None,
    operator_role: str = "Maintenance Engineer",
    model_label: str = DEFAULT_MODEL_LABEL,
):
    yield from ask_agent(demo_prompt, history, operator_role, model_label)


def stop_agent_run(history: list[dict] | None):
    history = history or []
    if history:
        try:
            user_msg, _assistant_msg = history[-1]
            history[-1] = (user_msg, "Run stopped by operator. You can adjust the prompt or start another run.")
        except Exception:
            pass
    stopped_activity = """
### Agent Activity

Stopped by operator. Ready for the next maintenance question.
"""
    run_update, stop_update = run_stop_controls(False)
    return history, gr.update(value="", visible=False), gr.update(visible=False), stopped_activity, run_update, stop_update


def asset_table() -> pd.DataFrame:
    try:
        df = asset_intelligence().copy()
        preferred = [
            "asset_id",
            "asset_type",
            "area",
            "priority",
            "risk_band",
            "decision_score",
            "hybrid_health_score",
            "hybrid_failure_risk",
            "estimated_rul_days",
            "evidence_confidence",
            "procurement_risk",
        ]
        cols = [col for col in preferred if col in df.columns]
        return df[cols].head(20) if cols else df.head(20)
    except Exception as exc:
        return pd.DataFrame([{"error": str(exc)}])


def memory_table() -> pd.DataFrame:
    try:
        active_raw = load_dynamic_assets()
        inactive_raw = list_inactive_dynamic_assets()
        active = active_raw.copy() if isinstance(active_raw, pd.DataFrame) else pd.DataFrame(active_raw or [])
        inactive_df = inactive_raw.copy() if isinstance(inactive_raw, pd.DataFrame) else pd.DataFrame(inactive_raw or [])
        if not active.empty:
            active = active.copy()
            if "active" in active.columns:
                active["memory_state"] = active["active"].map(lambda x: "active" if str(x).lower() in ["true", "1", "yes", "active"] else "inactive")
            else:
                active["memory_state"] = "active"
        if not inactive_df.empty:
            inactive_df["memory_state"] = "inactive"
        rows = pd.concat([active, inactive_df], ignore_index=True) if not inactive_df.empty else active
        if rows.empty:
            return pd.DataFrame([{"status": "No dynamic assets remembered yet."}])
        preferred = ["asset_id", "asset_type", "area", "memory_state", "risk_band", "hybrid_health_score", "estimated_rul_days"]
        cols = [col for col in preferred if col in rows.columns]
        return rows[cols].drop_duplicates().head(30) if cols else rows.head(30)
    except Exception as exc:
        return pd.DataFrame([{"error": str(exc)}])


def rules_table() -> pd.DataFrame:
    try:
        rules = load_dynamic_rules()
        if rules.empty:
            return pd.DataFrame([{"status": "No remembered safety rules yet."}])
        if "active" in rules.columns and "condition_text" in rules.columns:
            rules = rules.copy()
            rules["_active_sort"] = rules["active"].astype(str).str.lower().isin(["true", "1", "yes", "active"])
            dedup_cols = [col for col in ["condition_text", "priority_override"] if col in rules.columns]
            rules = rules.sort_values("_active_sort", ascending=False).drop_duplicates(subset=dedup_cols, keep="first").drop(columns=["_active_sort"])
        preferred = ["rule_id", "equipment_type", "condition_text", "priority_override", "active"]
        cols = [col for col in preferred if col in rules.columns]
        return rules[cols].drop_duplicates().head(50) if cols else rules.head(50)
    except Exception as exc:
        return pd.DataFrame([{"error": str(exc)}])


EXAMPLES = [
    "If I can maintain only one asset today, which one should I choose and why?",
    "What does error code E-045 mean on the blast furnace blower motor, and what steps should I take immediately?",
    "Show me the standard SOP for replacing a hydraulic pump seal on the rolling mill.",
    "What spare parts will I need if I replace the ladle car wheel assembly this weekend? What is the procurement lead time?",
    "Our conveyor belt on line 3 just stopped. Walk me through the first checks I should do right now.",
    "Generate a digital logbook entry for today's planned maintenance on the EAF transformer cooling system. Technician: R. Kumar. Work done: oil top-up and fan belt inspection.",
    "Rank active dynamic assets only. Show priority, risk, RUL, applied rules, and evidence confidence.",
]


ROLE_DUTIES = {
    "Maintenance Engineer": "focus on diagnosis, inspection order, isolation, repair steps, acceptance checks, and logbook closure.",
    "Reliability Engineer": "focus on failure modes, recurrence prevention, RUL, condition monitoring, RCA evidence, and long-term reliability controls.",
    "Operations Supervisor": "focus on safe continuity of operations, production impact, shift handoff, escalation, crew sequencing, and practical next actions.",
    "Safety Officer": "focus on LOTO, stored energy, stop-work triggers, permit controls, unsafe restart prevention, and escalation criteria.",
    "Plant Head / Executive": "focus on business impact, downtime exposure, priority tradeoffs, risk ownership, decision rationale, and resource allocation.",
    "Procurement Planner": "focus on spare availability, lead time, substitutes, vendor risk, reservation priority, and procurement action.",
}


def is_casual_chat(message: str) -> bool:
    q = re.sub(r"\s+", " ", str(message or "").strip().lower())
    clean = re.sub(r"[^\w\s]", "", q).strip()
    exact = {
        "hi",
        "hii",
        "hello",
        "hey",
        "heyy",
        "yo",
        "good morning",
        "good afternoon",
        "good evening",
        "thanks",
        "thank you",
        "ok",
        "okay",
    }
    if clean in exact:
        return True
    if len(clean.split()) <= 5 and clean in {"who are you", "what can you do", "help me", "help"}:
        return True
    return False


def _short_llm_chat(wizard: MaintenanceWizard, prompt: str) -> str:
    llm = getattr(wizard, "llm", None)
    if not llm or not getattr(llm, "available", False):
        return ""
    try:
        text = llm.generate(prompt, max_new_tokens=150, max_time=10.0, min_new_tokens=0)
    except TypeError:
        text = llm.generate(prompt, max_new_tokens=150, max_time=10.0)
    except Exception:
        return ""
    text = str(text or "").strip()
    text = re.split(r"Assistant:|Final answer:", text, flags=re.IGNORECASE)[-1].strip()
    text = clean_visible_answer(text)
    if "steel plant agent response" in text.lower():
        return ""
    return text


def casual_chat_result(wizard: MaintenanceWizard, message: str, operator_role: str) -> dict[str, Any]:
    role = operator_role if operator_role in ROLE_DUTIES else "Maintenance Engineer"
    role_hint = ROLE_DUTIES.get(role, ROLE_DUTIES["Maintenance Engineer"])
    prompt = f"""
You are FeCMind, a friendly but serious agentic AI maintenance copilot for steel manufacturing.
The current user role is {role}. Use that only to choose a helpful tone and examples.
Do not mention hidden role duties, internal routing, tool traces, JSON, templates, or implementation details.
Reply naturally to this short conversational message in 2-4 sentences.
Invite the user to ask about asset comparison, RCA, SOPs, RUL, spares, safety, logbook entries, or maintenance planning.

Role emphasis: {role_hint}
User message: {message}
""".strip()
    answer = _short_llm_chat(wizard, prompt)
    used_llm = bool(answer)
    if not answer:
        answer = (
            "Hi, I’m FeCMind. I can help you compare steel-plant assets, diagnose faults, draft SOPs, "
            "estimate RUL, plan spares, create logbook entries, and explain maintenance priorities. "
            "Tell me the asset, symptom, alarm, or decision you want to work through."
        )
    return {
        "mode": "casual_chat",
        "intent": "conversation",
        "asset_id": None,
        "answer": answer,
        "final_answer": answer,
        "agent_plan": [],
        "tool_calls": [],
        "verifier_checks": [
            {
                "check": "Casual conversation handled without maintenance pipeline",
                "status": "pass",
                "detail": "No hidden role prompt or asset fallback exposed.",
            }
        ],
        "decision_packet": {
            "mode": "casual_chat",
            "intent": "conversation",
            "objective": str(message or "").strip(),
            "selected_asset": None,
            "operator_role": role,
            "next_system_action": "await_user_maintenance_question",
        },
        "llm_used": used_llm,
    }


def clean_visible_answer(answer: str) -> str:
    text = str(answer or "").strip()
    if not text:
        return text
    text = re.sub(r"(?im)^\s*Acting user role:.*\n?", "", text)
    text = re.sub(r"(?im)^\s*Role duties and decision lens:.*\n?", "", text)
    text = re.sub(r"(?im)^\s*User request:\s*$\n?", "", text)
    text = re.sub(r"(?im)^\s*Role emphasis:.*\n?", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def build_role_prompt(message: str, operator_role: str) -> str:
    """Legacy helper kept only for older imports; do not wrap user prompts with it."""
    role = operator_role if operator_role in ROLE_DUTIES else "Maintenance Engineer"
    duties = ROLE_DUTIES[role]
    return f"""
Acting user role: {role}
Role duties and decision lens: {duties}

Answer the request for this role. Emphasize what this role must decide, verify, communicate, or execute.
Keep the same technical truth; change the depth, priorities, language, and next action to fit the role.

User request:
{message}
""".strip()


CSS = """
.gradio-container { max-width: 1500px !important; background: #0b1120 !important; }
.hero {
  padding: 24px 28px;
  border-radius: 16px;
  background: radial-gradient(circle at top left, #155e75 0%, #111827 42%, #020617 100%);
  color: white;
  margin-bottom: 16px;
  border: 1px solid rgba(148, 163, 184, 0.25);
}
.hero h1 { margin: 0 0 6px 0; font-size: 30px; }
.hero p { margin: 0; opacity: 0.92; }
.metric-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 10px 0 16px; }
.metric-card { border: 1px solid rgba(148,163,184,.22); background: #0f172a; border-radius: 14px; padding: 14px 16px; }
.metric-card.danger { border-color: rgba(239,68,68,.45); }
.metric-card.amber { border-color: rgba(245,158,11,.45); }
.metric-card.sky { border-color: rgba(14,165,233,.45); }
.metric-label { font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: .08em; }
.metric-value { margin-top: 5px; color: #f8fafc; font-size: 28px; font-weight: 800; }
.metric-sub { margin-top: 4px; color: #94a3b8; font-size: 12px; }
.feature-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 8px 0 14px; }
.feature-chip { border: 1px solid rgba(14,165,233,.22); background: rgba(15,23,42,.92); color: #e2e8f0; border-radius: 14px; padding: 12px 13px; min-height: 74px; }
.feature-chip b { display: block; color: #f8fafc; margin-bottom: 4px; font-size: 13px; }
.feature-chip span { color: #94a3b8; font-size: 12px; line-height: 1.35; }
.chat-working-panel { border: 1px solid rgba(14,165,233,.32); background: rgba(14,165,233,.08); color: #dbeafe; border-radius: 14px; padding: 12px 14px; margin-bottom: 10px; }
.demo-prompt-list { gap: 8px; }
.demo-prompt-btn { text-align: left !important; justify-content: flex-start !important; white-space: normal !important; min-height: 44px; border-radius: 12px !important; }
.status-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: #e7f7ef;
  color: #11623b;
  font-size: 13px;
  font-weight: 700;
}
.panel-note { color: #475569; font-size: 13px; }
.activity-hint { display: block; margin-top: 8px; color: #94a3b8; font-size: 12px; }
.module-note { border: 1px solid rgba(14,165,233,.25); background: rgba(14,165,233,.07); padding: 12px 14px; border-radius: 12px; color: #cbd5e1; font-size: 13px; }
.typing-dots { display: inline-flex; align-items: center; gap: 7px; min-height: 24px; padding: 8px 2px; }
.typing-dots span { width: 10px; height: 10px; border-radius: 999px; background: #111827; display: inline-block; animation: dotPulse 1s infinite ease-in-out; }
.typing-dots span:nth-child(2) { animation-delay: .16s; }
.typing-dots span:nth-child(3) { animation-delay: .32s; }
@keyframes dotPulse { 0%, 80%, 100% { transform: scale(.75); opacity: .38; } 40% { transform: scale(1.25); opacity: 1; } }
@media (max-width: 900px) { .metric-grid, .feature-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
@media (max-width: 560px) { .metric-grid, .feature-grid { grid-template-columns: 1fr; } }
"""


with gr.Blocks(title="FeCMind: Tata Steel Agentic AI", css=CSS, theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div class="hero">
          <h1>FeCMind: Tata Steel Agentic Maintenance AI</h1>
          <p>Default fast Qwen3-0.6B triage with selectable Qwen3-8B + LoRA high-fidelity mode, grounded by ML risk scoring, RAG evidence, dynamic memory, safety rules, spares, logbook, and verifier checks.</p>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("Agent Chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### Maintenance Copilot")
                    gr.HTML(
                        """
                        <div class="feature-grid">
                          <div class="feature-chip"><b>Industrial Qwen profiles</b><span>0.6B opens by default for fast demos; 8B LoRA is available for deeper maintenance reasoning.</span></div>
                          <div class="feature-chip"><b>RAG + SOP evidence</b><span>Grounded answers from SOPs, policies, history, failure reports and plant records.</span></div>
                          <div class="feature-chip"><b>ML risk + RUL</b><span>Hybrid priority scoring, anomaly signals, remaining useful life and delay impact.</span></div>
                          <div class="feature-chip"><b>Agent memory</b><span>Dynamic assets, remembered safety rules, spares, logbook and feedback loop.</span></div>
                        </div>
                        """
                    )
                    chat_activity_md = gr.Markdown(value="", visible=False, elem_classes=["chat-working-panel"])
                    chatbot = gr.Chatbot(height=620, show_copy_button=True, label="Agent conversation")
                    chat_visual = gr.Plot(label="Agent visual", visible=False)
                    with gr.Row():
                        prompt_box = gr.Textbox(
                            placeholder="Ask any steel-plant maintenance, operations, safety, spares, RCA, SOP, quality, or reliability question...",
                            lines=3,
                            scale=6,
                            show_label=False,
                        )
                        send_btn = gr.Button("Run Agent", variant="primary", scale=1)
                        stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)
                    gr.Markdown("### One-click demo prompts")
                    demo_prompt_buttons: list[tuple[gr.Button, gr.State]] = []
                    with gr.Column(elem_classes=["demo-prompt-list"]):
                        for example_prompt in EXAMPLES:
                            prompt_state = gr.State(example_prompt)
                            prompt_button = gr.Button(example_prompt, variant="secondary", size="sm", elem_classes=["demo-prompt-btn"])
                            demo_prompt_buttons.append((prompt_button, prompt_state))
                with gr.Column(scale=1):
                    gr.Markdown("### Session Controls")
                    role_dropdown = gr.Dropdown(
                        label="Role context",
                        choices=[
                            "Maintenance Engineer",
                            "Reliability Engineer",
                            "Operations Supervisor",
                            "Safety Officer",
                            "Plant Head / Executive",
                            "Procurement Planner",
                        ],
                        value="Maintenance Engineer",
                    )
                    gr.Markdown("### Model Status")
                    gr.Markdown('<span class="status-pill">Selectable Qwen3 local GPU model</span>')
                    model_dropdown = gr.Dropdown(
                        label="LLM model",
                        choices=list(MODEL_OPTIONS.keys()),
                        value=DEFAULT_MODEL_LABEL,
                    )
                    model_note = gr.Markdown(MODEL_OPTIONS[DEFAULT_MODEL_LABEL]["description"])
                    with gr.Row():
                        switch_model_btn = gr.Button("Switch / Load Model", variant="secondary")
                        warmup_btn = gr.Button("Warm Up / Check Model")
                    warmup_out = gr.Code(label="Runtime status", language="json", lines=12)
                    gr.Markdown(
                        """
                        <div class="panel-note">
                        The app opens on Qwen3-0.6B for a responsive first demo. Switch to Qwen3-8B LoRA when you want the deepest maintenance reasoning.
                        The dashboard loads from local plant data first; the selected LLM warms only for chat/synthesis.
                        </div>
                        """
                    )

        with gr.Tab("Command Center"):
            gr.HTML(dashboard_summary())
            gr.HTML(
                """
                <div class="module-note">
                Control-room view inspired by industrial predictive-maintenance demos: hybrid risk, RUL, evidence confidence,
                delay exposure, dynamic memory and procurement constraints are shown before the agent writes a recommendation.
                </div>
                """
            )
            with gr.Row():
                gr.Plot(value=risk_bar_plot(), label="Priority score")
                gr.Plot(value=rul_risk_scatter(), label="RUL vs risk")
            gr.Markdown("### Live Agent Execution")
            activity_md = gr.Markdown(format_activity())
            assets_df = gr.Dataframe(label="Fleet priority matrix", value=asset_table(), interactive=False, wrap=True, height=360)

        with gr.Tab("Predictive Analytics"):
            with gr.Row():
                gr.Plot(value=rul_bar_plot(), label="RUL watchlist")
                gr.Plot(value=spares_plot(), label="Spares lead time")
            gr.Markdown(model_diagnostics())

        with gr.Tab("Spares & Work Planning"):
            with gr.Row():
                spares_df = gr.Dataframe(label="Spares inventory", value=_read_csv(PROJECT_ROOT / "data" / "spares_inventory.csv"), interactive=False, wrap=True, height=300)
                workplan_df = gr.Dataframe(label="Work-order / procurement queue", value=workplan_table(), interactive=False, wrap=True, height=300)

        with gr.Tab("Evidence, Memory & Trace"):
            with gr.Row():
                memory_df = gr.Dataframe(label="Remembered dynamic assets", value=memory_table(), interactive=False, wrap=True, height=280)
                rules_df = gr.Dataframe(label="Remembered safety rules", value=rules_table(), interactive=False, wrap=True, height=280)
            knowledge_df = gr.Dataframe(label="Knowledge base / RAG evidence", value=knowledge_table(), interactive=False, wrap=True, height=300)
            files_df = gr.Dataframe(label="Accessible project files", value=project_file_inventory(), interactive=False, wrap=True, height=260)
            with gr.Accordion("Internal verifier trace", open=False, visible=False):
                with gr.Tabs():
                    with gr.Tab("Decision Packet"):
                        decision_json = gr.Code(label="Locked decision + LLM validation", language="json", lines=18)
                    with gr.Tab("Agent Plan"):
                        plan_json = gr.Code(label="Planner steps", language="json", lines=18)
                    with gr.Tab("Tool Calls"):
                        calls_json = gr.Code(label="Tool calls", language="json", lines=18)
                    with gr.Tab("Verifier"):
                        checks_json = gr.Code(label="Verifier checks", language="json", lines=18)

    agent_outputs = [chatbot, prompt_box, chat_activity_md, chat_visual, activity_md, send_btn, stop_btn, decision_json, plan_json, calls_json, checks_json, assets_df, memory_df]
    running_events = []

    send_event = send_btn.click(
        ask_agent,
        inputs=[prompt_box, chatbot, role_dropdown, model_dropdown],
        outputs=agent_outputs,
        api_name=False,
    )
    running_events.append(send_event)

    submit_event = prompt_box.submit(
        ask_agent,
        inputs=[prompt_box, chatbot, role_dropdown, model_dropdown],
        outputs=agent_outputs,
        api_name=False,
    )
    running_events.append(submit_event)

    for prompt_button, prompt_state in demo_prompt_buttons:
        demo_event = prompt_button.click(
            ask_agent_from_demo,
            inputs=[prompt_state, chatbot, role_dropdown, model_dropdown],
            outputs=agent_outputs,
            api_name=False,
        )
        running_events.append(demo_event)

    stop_btn.click(
        stop_agent_run,
        inputs=[chatbot],
        outputs=[chatbot, chat_activity_md, chat_visual, activity_md, send_btn, stop_btn],
        cancels=running_events,
        api_name=False,
    )
    model_dropdown.change(
        lambda label: MODEL_OPTIONS.get(label, MODEL_OPTIONS[DEFAULT_MODEL_LABEL])["description"],
        inputs=model_dropdown,
        outputs=model_note,
        api_name=False,
    )
    switch_model_btn.click(switch_model, inputs=model_dropdown, outputs=warmup_out, api_name=False)
    warmup_btn.click(switch_model, inputs=model_dropdown, outputs=warmup_out, api_name=False)
    demo.load(asset_table, outputs=assets_df, api_name=False)
    demo.load(memory_table, outputs=memory_df, api_name=False)
    demo.load(rules_table, outputs=rules_df, api_name=False)
    demo.load(project_file_inventory, outputs=files_df, api_name=False)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1, max_size=8).launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
    )
