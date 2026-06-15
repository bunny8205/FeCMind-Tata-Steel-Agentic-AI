"""Chat-first frontend for the Tata Steel Steel Plant Agent."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agent import MaintenanceWizard  # noqa: E402


st.set_page_config(
    page_title="Tata Steel Agentic AI",
    page_icon="TS",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Starting steel agent...")
def get_wizard() -> MaintenanceWizard:
    wizard = MaintenanceWizard()
    wizard.initialize(load_llm=False)
    return wizard

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1320px; }
    [data-testid="stSidebar"] { background: #f7f8fb; }
    .agent-title { font-size: 1.35rem; font-weight: 760; color: #111827; margin-bottom: 0.2rem; }
    .agent-subtitle { color: #4b5563; font-size: 0.9rem; margin-bottom: 1rem; }
    .status-pill {
        display: inline-block;
        border: 1px solid #d7dde8;
        border-radius: 6px;
        padding: 0.22rem 0.45rem;
        margin: 0.08rem;
        font-size: 0.78rem;
        background: #ffffff;
        color: #1f2937;
    }
    .compact-note {
        border-left: 3px solid #2563eb;
        padding: 0.55rem 0.75rem;
        background: #f8fbff;
        color: #1f2937;
        font-size: 0.88rem;
        margin-bottom: 0.7rem;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8dde6;
        border-radius: 8px;
        padding: 0.7rem 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="agent-title">Tata Steel Agentic AI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="agent-subtitle">Steel maintenance, reliability, safety, spares, RCA, SOP, and plant-priority agent.</div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Loading steel agent, ML health tables, and retrieval index..."):
        wizard = get_wizard()
except Exception as exc:
    st.error("The Steel Plant Agent backend failed to start.")
    st.exception(exc)
    st.stop()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "Ask any steel-plant maintenance, operations, safety, spares, RCA, SOP, "
                    "quality, or reliability question."
                ),
            }
        ]
    if "last_result" not in st.session_state:
        st.session_state.last_result = None


def render_sidebar() -> None:
    st.sidebar.markdown('<div class="agent-title">Steel Plant Agent</div>', unsafe_allow_html=True)
    st.sidebar.markdown(
        '<div class="agent-subtitle">Maintenance Wizard for industrial equipment decisions.</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Restart Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        wizard.session_memory.clear()
        st.rerun()

    if st.sidebar.button("Reload Data", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption("Live assets")
    for row in wizard.asset_health_table().sort_values("hybrid_health_score", ascending=False).itertuples():
        label = f"{row.asset_id} | {row.risk_band} | RUL {round(float(row.estimated_rul_days), 1)}d"
        st.sidebar.markdown(f'<span class="status-pill">{label}</span>', unsafe_allow_html=True)

    st.sidebar.divider()
    st.sidebar.caption("Prompt starters")
    starters = [
        "A continuous caster mold temperature is rising. Build an action plan and breakout prevention checklist.",
        "Create an RCA for repeated bearing failures in a hot strip mill gearbox.",
        "Generate a safe SOP for hydraulic pressure loss on a plate mill power pack.",
        "Compare all current assets and tell the supervisor what to do this shift.",
        "Plan spares and procurement for BOF trunnion bearing maintenance.",
    ]
    selected = st.sidebar.selectbox("Insert prompt", [""] + starters, label_visibility="collapsed")
    if selected and st.sidebar.button("Use Prompt", use_container_width=True):
        st.session_state.pending_prompt = selected
        st.rerun()


def render_workspace(result: dict | None) -> None:
    if not result:
        return

    packet = result.get("decision_packet", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Mode", packet.get("mode", result.get("mode", "-")))
    with c2:
        if packet.get("applied_demo_target"):
            st.metric("Applied Target", packet.get("applied_demo_target"))
        else:
            st.metric("Subject", packet.get("selected_asset", result.get("asset_id", "-")))
    with c3:
        st.metric("Intent", packet.get("intent", result.get("intent", "-")))
    with c4:
        st.metric("Priority", packet.get("priority", result.get("priority", "-")))

    with st.expander("Agent Workspace", expanded=True):
        left, right = st.columns(2)
        with left:
            st.markdown("**Plan**")
            plan = result.get("agent_plan", [])
            if plan:
                st.dataframe(pd.DataFrame(plan), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Verifier Checks**")
            checks = result.get("verifier_checks", [])
            if checks:
                st.dataframe(pd.DataFrame(checks), use_container_width=True, hide_index=True)

        st.markdown("**Tool Calls**")
        calls = result.get("tool_calls", [])
        if calls:
            st.dataframe(pd.DataFrame(calls), use_container_width=True, hide_index=True)

        st.markdown("**Decision Packet JSON**")
        st.code(json.dumps(packet, indent=2, default=str), language="json")

    sources = result.get("retrieved_docs", [])
    if sources:
        with st.expander("Retrieved Evidence", expanded=False):
            for doc in sources[:8]:
                title = f"{doc.get('source')} | {doc.get('equipment_type')} | {doc.get('issue_type')}"
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(doc.get("text", ""))


def run_prompt(prompt: str) -> None:
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        with st.spinner("Agent is planning, retrieving, verifying, and writing..."):
            result = wizard.chat(prompt, user_id="maintenance_engineer_01")
    except Exception as exc:
        result = {
            "answer": "The agent hit an error while processing this prompt.",
            "decision_packet": {"mode": "error", "objective": prompt, "error": str(exc)},
            "agent_plan": [],
            "tool_calls": [],
            "verifier_checks": [],
        }
        st.session_state.last_result = result
        st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        st.error("Prompt execution failed.")
        st.exception(exc)
        return

    answer = result.get("answer", "").strip() or "The agent completed but returned an empty answer."
    st.session_state.last_result = result
    st.session_state.messages.append({"role": "assistant", "content": answer})


init_state()
render_sidebar()

if st.session_state.get("pending_prompt"):
    prompt = st.session_state.pop("pending_prompt")
    run_prompt(prompt)

if prompt := st.chat_input("Ask the steel agent"):
    run_prompt(prompt)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

render_workspace(st.session_state.last_result)
