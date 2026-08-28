"""Conversational, evidence-grounded explanation layer for current FPL decisions."""

from pathlib import Path
import sys

import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from fpl_predictor.analyst.citations import freshness_label  # noqa: E402
from fpl_predictor.analyst.provider import provider_from_config  # noqa: E402
from fpl_predictor.analyst.service import AnalystService  # noqa: E402
from fpl_predictor.ui.components import (  # noqa: E402
    analyst_evidence_text, analyst_provider_config, analyst_suggested_questions, cached_analyst_context,
    configure_page, render_sidebar,
)
from fpl_predictor.ui.data import dashboard_summary  # noqa: E402


configure_page("Ask FPL AI")
settings, bundle = render_sidebar()
summary = dashboard_summary(bundle)
st.title("Ask FPL AI")
st.caption("The analyst explains current structured outputs. It is not a separate prediction engine.")

status = freshness_label(bundle.status.stale)
timestamp = bundle.status.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if bundle.status.timestamp else "Unavailable"
columns = st.columns(4)
columns[0].metric("Data", status)
columns[1].metric("Target GW", f"GW{summary['target_gw']}" if summary.get("target_gw") else "—")
columns[2].metric("Squad source", summary.get("squad_source") or "unavailable")
columns[3].metric("Last refresh", timestamp)
if bundle.status.stale:
    st.warning("STALE DATA — analyst answers inherit this warning and may not reflect the latest FPL state.")

provider = provider_from_config(analyst_provider_config())
st.info("AI provider: Disabled · Using deterministic analyst" if provider.name == "disabled"
        else f"AI provider: {provider.name.title()} · Grounding validation enabled")

st.markdown("**Suggested questions**")
suggestion_columns = st.columns(3)
pending = None
for index, question in enumerate(analyst_suggested_questions()):
    if suggestion_columns[index % 3].button(question, key=f"analyst_suggestion_{index}", width="stretch"):
        pending = question

history = st.session_state.setdefault("analyst_messages", [])
for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Confidence: {message['confidence']} · {message['mode']}")
            with st.expander("Evidence used", expanded=False):
                st.write(analyst_evidence_text(message.get("evidence", [])))
                for label, value in message.get("evidence_details", {}).items():
                    st.write(f"**{label}:** {value if value is not None else 'Unavailable'}")

question = pending or st.chat_input("Ask about your squad, transfers, captaincy, players, fixtures, or chips")
if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    overrides = {chip: st.session_state.get(f"chip_{chip}", "unknown") for chip in
                 ("wildcard", "free_hit", "bench_boost", "triple_captain")}
    with st.chat_message("assistant"):
        with st.spinner("Checking structured FPL evidence…"):
            context = cached_analyst_context(question, bundle, settings, tuple(overrides.items()),
                                             st.session_state.get("refresh_generation", 0))
            universe = set(bundle.predictions.player.astype(str)) if not bundle.predictions.empty else set()
            response = AnalystService(provider).answer_context(question, context, universe)
        st.markdown(response.answer)
        mode = "Deterministic fallback" if response.fallback_used else "Grounded AI explanation"
        st.caption(f"Confidence: {response.confidence} · {mode}")
        with st.expander("Evidence used", expanded=False):
            st.write(analyst_evidence_text(response.evidence))
            for label, value in response.evidence_details.items():
                st.write(f"**{label}:** {value if value is not None else 'Unavailable'}")
    history.append({"role": "assistant", "content": response.answer, "confidence": response.confidence,
                    "mode": mode, "evidence": list(response.evidence),
                    "evidence_details": response.evidence_details})

st.caption("No conversation is persisted externally. The analyst never logs in, transfers players, or activates chips.")
