# Online Streamlit Deployment

This project is ready for Streamlit Community Cloud.

## Files Streamlit Cloud Uses

- Main file: `app.py`
- Dependencies: `requirements.txt`
- Python version: `runtime.txt`
- Streamlit settings: `.streamlit/config.toml`

## Deploy Steps

1. Create a GitHub repository, for example:

   ```text
   tatasteel-agentic-ai
   ```

2. Push this project folder to that repository.

3. Open Streamlit Community Cloud:

   ```text
   https://share.streamlit.io
   ```

4. Click **New app**.

5. Select:

   ```text
   Repository: your-username/tatasteel-agentic-ai
   Branch: main
   Main file path: app.py
   ```

6. Click **Deploy**.

## Local Test Before Deploy

```powershell
cd C:\Users\omshr\PycharmProjects\PythonProject1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Demo Prompts

```text
If I can maintain only one asset today, which one should I choose and why?
```

```text
Design an agentic workflow for steel plant predictive maintenance using logs, SOPs, sensor alerts, and feedback.
```

```text
A continuous caster mold temperature is rising and breakout alarms are appearing. Build an action plan.
```

```text
Create an RCA for repeated bearing failures in a hot strip mill gearbox.
```

```text
GBX-17 abnormal vibration. Diagnose root cause, RUL, spares, and alert.
```

## Notes

The cloud app uses lightweight TF-IDF retrieval if sentence-transformer models are not installed. This keeps startup fast and avoids large GPU/LLM dependencies on Streamlit Cloud.
