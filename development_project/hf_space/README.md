---
title: FeCMind
emoji: 🏭
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
---

# FeCMind: Tata Steel Agentic Maintenance AI

This Hugging Face Space runs the Tata Steel Maintenance Wizard as a full Gradio demo.

Runtime stack:

- Qwen/Qwen3-8B base model
- 4-bit GPU loading with bitsandbytes
- LoRA adapter downloaded from `rn8205/qwen38bfinetuned`
- Maintenance Wizard backend with ML risk scoring, RAG, memory, rules, spares, logbook, and deterministic verifier

Recommended Space settings:

- SDK: Gradio
- Hardware: L4
- Sleep time: 10-15 minutes
- Concurrency: 1

Optional Space secrets:

- `HF_TOKEN` if the LoRA dataset or base model requires authentication
- `BASE_MODEL_ID` if you want to override `Qwen/Qwen3-8B`
- `ADAPTER_REPO_ID`, `ADAPTER_FILENAME` if you move the adapter

The first request after the Space wakes may be slow because the GPU loads Qwen3-8B and attaches the LoRA adapter.

Included reference notebook:

- `notebooks/tatasteel-qwen3-8b-kaggle-backend-lora.ipynb`

Push example:

```bash
git clone https://huggingface.co/spaces/rn8205/FeCMind
cp -r hf_space/* FeCMind/
cd FeCMind
git add .
git commit -m "Deploy FeCMind Qwen3 LoRA app"
git push
```
