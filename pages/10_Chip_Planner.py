"""Read-only Wildcard, Free Hit, Bench Boost, and Triple Captain planner."""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path: sys.path.insert(0, str(SRC_ROOT))

from fpl_predictor.chips import (  # noqa: E402
    budget_legal_chip_gains, build_chip_plan, resolve_chip_states,
)
from fpl_predictor.config import RAW_DATA_DIR  # noqa: E402
from fpl_predictor.ui.components import chip_card, configure_page, render_sidebar, signal_badge  # noqa: E402
from fpl_predictor.ui.data import dashboard_summary, data_status  # noqa: E402

configure_page("Chip Planner")
settings, bundle = render_sidebar()
st.title("Chip Planner")
st.warning("Advisory only — this app never activates a chip, submits transfers, or logs in to FPL.")
target_gw = dashboard_summary(bundle)["target_gw"]
fixture_status = data_status(RAW_DATA_DIR / "fixtures.json", settings.refresh_ttl)
timestamp = fixture_status.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if fixture_status.timestamp else "Unavailable"
st.caption(f"Fixture data updated: {timestamp}")
if fixture_status.stale: st.warning("STALE FIXTURE DATA")
if target_gw is None or bundle.fixture_calendar.empty or bundle.predictions.empty or bundle.squad.empty:
    st.error("Chip planning requires current fixtures, player projections, and a valid squad.")
    st.stop()

overrides = {chip: st.session_state.get(f"chip_{chip}", "unknown") for chip in
             ("wildcard", "free_hit", "bench_boost", "triple_captain")}
history = {"chips": bundle.live_summary.get("entry_history_chips", [])}
states = resolve_chip_states(history, overrides)
state_table = pd.DataFrame([{"Chip": chip.replace("_", " ").title(), "State": state.state.upper(),
                             "Source": state.source} for chip, state in states.items()])
st.dataframe(state_table, width="stretch", hide_index=True)

@st.cache_data(ttl=600, show_spinner=False)
def _plan(players, squad, calendar, bank, start_gw, state_values, used_chips):
    resolved = resolve_chip_states({"chips": list(used_chips)}, dict(state_values))
    wildcard_gain, free_hit_gains = budget_legal_chip_gains(players, squad, bank, start_gw)
    return build_chip_plan(calendar, squad, players, resolved, start_gw, 8, wildcard_gain, free_hit_gains)

plan = _plan(bundle.predictions, bundle.squad, bundle.fixture_calendar, settings.bank, int(target_gw),
             tuple(overrides.items()), tuple(history["chips"]))
selected_gw = st.selectbox("Selected Gameweek", plan.gw.tolist())
row = plan[plan.gw.eq(selected_gw)].iloc[0]
columns = st.columns(4)
with columns[0]: chip_card("Wildcard", row.wildcard_signal, row.wildcard_score, row.wildcard_reason)
with columns[1]: chip_card("Free Hit", row.free_hit_signal, row.free_hit_score, row.free_hit_reason)
with columns[2]: chip_card("Bench Boost", row.bench_boost_signal, row.bench_boost_score, row.bench_boost_reason)
with columns[3]: chip_card("Triple Captain", row.triple_captain_signal, row.triple_captain_score, row.triple_captain_reason)

st.subheader("Chip comparison")
comparison = plan[["gw", "wildcard_signal", "wildcard_score", "free_hit_signal", "free_hit_score",
                   "bench_boost_signal", "bench_boost_score", "triple_captain_signal", "triple_captain_score"]].copy()
for chip in ("wildcard", "free_hit", "bench_boost", "triple_captain"):
    comparison[chip.replace("_", " ").title()] = comparison.apply(
        lambda item: f"{signal_badge(item[f'{chip}_signal'])} · " +
                     ("—" if pd.isna(item[f"{chip}_score"]) else f"{item[f'{chip}_score']:.2f}"), axis=1)
comparison = comparison.rename(columns={"gw": "GW"})
st.dataframe(comparison[["GW", "Wildcard", "Free Hit", "Bench Boost", "Triple Captain"]],
             width="stretch", hide_index=True)

with st.expander("Methodology and reasons", expanded=False):
    for item in plan.itertuples(index=False):
        st.markdown(f"**GW{item.gw}**")
        st.write(f"Wildcard: {item.wildcard_reason}")
        st.write(f"Free Hit: {item.free_hit_reason}")
        st.write(f"Bench Boost: {item.bench_boost_reason}")
        st.write(f"Triple Captain: {item.triple_captain_reason}")
st.download_button("Download chip_plan.csv", plan.to_csv(index=False), file_name="chip_plan.csv", mime="text/csv")
st.caption("GREEN is a strong modeled opportunity, YELLOW is plausible, RED is poor timing, and GREY means used/unknown/insufficient data.")
