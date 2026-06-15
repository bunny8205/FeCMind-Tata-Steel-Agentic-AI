# FeCMind: Tata Steel Agentic Maintenance AI

FeCMind is an agentic AI maintenance decision-support system for steel manufacturing. It combines Qwen model profiles, RAG evidence retrieval, predictive maintenance scoring, dynamic asset memory, safety-rule memory, spares/procurement planning, digital logbooks, and verifier checks.

## Live App

- Hugging Face Space: https://huggingface.co/spaces/rn8205/FeCMind

## What Is Included

- `app.py` - Hugging Face Gradio application.
- `backend/` - agent orchestration, steel-domain agent logic, RAG, ML/risk scoring, dynamic memory, LLM loading, data setup, and APIs.
- `data/` - demo plant data, scored asset tables, spares, logbook, feedback, dynamic memory, and RAG documents.
- `docs/` - SOPs, policies, steel-agent operating model notes, procurement strategy, and safety guidance.
- `notebooks/` - project notebooks included in the app package.
- `notebooks_external/` - Kaggle/backend/fine-tuning notebooks copied from the development workflow.
- `development_project/` - fuller local development project with frontend/backend/scripts/reports/public dataset artifacts where available.

## Models

The app defaults to `Qwen3-0.6B instant triage` for fast live demos. The selector also includes Qwen3-1.7B, Qwen3-4B, and Qwen3-8B LoRA high-fidelity mode. Large model weights and LoRA adapters are intentionally not committed to GitHub; they are hosted externally on Hugging Face.

## Public Dataset

The predictive maintenance validation uses the public AI4I 2020 Predictive Maintenance dataset, along with synthetic steel-plant data for assets, sensors, SOPs, failure records, spares, logbook and feedback behavior.

## Run Locally

CPU-only local runs are for UI/backend inspection. Full model inference is intended for GPU environments such as Hugging Face Spaces or Kaggle.

```bash
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:7860
```

## GPU Deployment

For the full demo, deploy this repo as a Hugging Face Gradio Space and select GPU hardware. The app opens in the fast Qwen profile by default and can switch models from the UI.

## Notes

- Do not commit Hugging Face or GitHub tokens.
- Large model files (`*.safetensors`, `*.bin`, `*.pt`, `*.zip`) are excluded from Git and should remain on Hugging Face model/dataset storage.
