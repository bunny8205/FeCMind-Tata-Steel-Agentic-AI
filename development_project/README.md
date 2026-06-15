# Tata Steel Agentic AI - Maintenance Wizard

Chat-first agentic AI prototype for steel-plant maintenance decision support.

The system accepts broad natural-language steel prompts and routes them through an agent loop:
perceive, retrieve, reason, act, verify, log, and learn.

## What It Does

- Diagnoses known demo assets with ML risk, RUL, anomaly signals, RAG evidence, spares, and logbook output.
- Handles broader steel prompts even when no asset ID is provided.
- Generates RCA, SOP, work-order plans, safety controls, spares strategy, plant priority summaries, and incident reports.
- Retrieves evidence from SOPs, maintenance history, failure reports, spares, policies, and steel-process guides.
- Writes digital logbook entries and supports feedback learning.

## Main Entry Points

- Frontend: `frontend/streamlit_app.py`
- Backend agent: `backend/agent.py`
- General steel agent layer: `backend/steel_agent.py`
- Data and docs setup: `backend/data_setup.py`
- ML risk and RUL logic: `backend/models.py`
- RAG retrieval: `backend/rag.py`
- Terminal chat: `scripts/agent_cli.py`

## Run

```powershell
python -m pip install -r requirements.txt
python scripts\initialize_project.py
python -m streamlit run app.py
```

Open:

```text
http://127.0.0.1:8501
```

For online deployment, see `DEPLOYMENT.md`. Streamlit Cloud main file path is `app.py`.

## Example Prompts

- `A continuous caster mold temperature is rising and breakout alarms are appearing. Build an action plan.`
- `Create an RCA for repeated bearing failures in a hot strip mill gearbox.`
- `Generate a safe SOP for hydraulic pressure loss on a plate mill power pack.`
- `Plan spares and procurement for BOF trunnion bearing maintenance.`
- `GBX-17 abnormal vibration. Diagnose root cause, RUL, spares, and alert.`
