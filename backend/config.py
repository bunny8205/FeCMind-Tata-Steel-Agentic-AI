"""Project paths and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DOC_DIR = PROJECT_ROOT / "docs"
PUBLIC_DIR = PROJECT_ROOT / "public_datasets"
REPORT_DIR = PROJECT_ROOT / "reports"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"

for directory in [DATA_DIR, DOC_DIR, PUBLIC_DIR, REPORT_DIR, ARTIFACT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


RANDOM_STATE = 42
USE_LOCAL_LLM = os.getenv("MW_USE_LLM", "1").strip().lower() not in {"0", "false", "no"}
LOCAL_LLM_MODEL_ID = os.getenv("MW_LLM_MODEL_ID", "distilgpt2")
LLM_LAZY_LOAD = os.getenv("MW_LLM_LAZY_LOAD", "1").strip().lower() not in {"0", "false", "no"}
EMBED_MODEL_ID = os.getenv("MW_EMBED_MODEL_ID", "BAAI/bge-small-en-v1.5")

# Final-answer LLM. Use an OpenAI-compatible Qwen endpoint for serious demos.
# Examples:
#   MW_LLM_PROVIDER=qwen_api
#   MW_QWEN_API_BASE=https://openrouter.ai/api/v1
#   MW_QWEN_API_MODEL=qwen/qwen-2.5-32b-instruct
#   MW_QWEN_API_KEY=...
LLM_PROVIDER = os.getenv("MW_LLM_PROVIDER", "auto").strip().lower()
QWEN_API_BASE = os.getenv("MW_QWEN_API_BASE", os.getenv("OPENAI_BASE_URL", "")).strip().rstrip("/")
QWEN_API_KEY = os.getenv(
    "MW_QWEN_API_KEY",
    os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", os.getenv("TOGETHER_API_KEY", ""))),
).strip()
QWEN_API_MODEL = os.getenv("MW_QWEN_API_MODEL", os.getenv("MW_QWEN_MODEL_ID", "qwen/qwen-2.5-32b-instruct")).strip()
QWEN_API_REFERER = os.getenv("MW_QWEN_REFERER", "http://127.0.0.1:8610").strip()
QWEN_API_TITLE = os.getenv("MW_QWEN_TITLE", "Tata Steel Maintenance Wizard").strip()
