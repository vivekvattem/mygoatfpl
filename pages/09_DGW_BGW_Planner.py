"""Confirmed official fixture calendar and Double/Blank Gameweek planner."""

import pandas as pd
import streamlit as st

from fpl_predictor.config import LIVE_DATA_DIR, RAW_DATA_DIR  # noqa: E402
from fpl_predictor.fixture_calendar import fixture_matrix  # noqa: E402
from fpl_predictor.ui.components import configure_page, fixture_alert, render_sidebar, signal_badge  # noqa: E402
from fpl_predictor.ui.data import dashboard_summary, data_status  # noqa: E402

configure_page("DGW/BGW Planner")
settings, bundle = render_sidebar()
st.title("Double & Blank Gameweek Planner")
target_gw = dashboard_summary(bundle)["target_gw"]
fixture_status = data_status(RAW_DATA_DIR / "fixtures.json", settings.refresh_ttl)
timestamp = fixture_status.timestamp.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z") if fixture_status.timestamp else "Unavailable"
st.caption(f"Fixture data updated: {timestamp}")
if fixture_status.stale: st.warning("STALE FIXTURE DATA — refresh before acting on schedule alerts.")

calendar = bundle.fixture_calendar
if calendar.empty or target_gw is None:
    st.error("Confirmed fixture calendar is unavailable. Refresh official FPL data.")
    st.stop()

st.subheader("Upcoming Schedule Alerts")
alerts = calendar[calendar.schedule_label.ne("NORMAL")]
if alerts.empty:
    st.success("No confirmed Double, Blank, or Triple Gameweek alerts in the next 10 Gameweeks.")
else:
    for (gw, label), group in alerts.groupby(["gw", "schedule_label"], sort=True):
        fixture_alert(int(gw), str(label), group.team.tolist(), "CONFIRMED")

st.subheader("Team fixture matrix")
window = st.radio("Matrix horizon", [5, 10], horizontal=True)
st.dataframe(fixture_matrix(calendar, int(target_gw), window), width="stretch", hide_index=True)

st.subheader("Team fixture signals")
signals = bundle.team_fixture_signals.copy()
if signals.empty:
    st.info("Fixture signals are unavailable.")
else:
    selected = st.multiselect("Signal", ["GREEN", "YELLOW", "RED"], default=[])
    if selected: signals = signals[signals.fixture_signal.isin(selected)]
    signals["Fixture Signal"] = signals.fixture_signal.map(signal_badge)
    display = signals.rename(columns={"team": "Team", "average_fdr_5": "Average FDR",
                                      "fixtures_next_5": "Fixtures next 5", "fixture_reason": "Reason"})
    st.dataframe(display[["Team", "Fixture Signal", "Average FDR", "Fixtures next 5", "Reason"]],
                 width="stretch", hide_index=True)

download = calendar.copy()
for column in ("opponents", "home_away", "official_fdr", "kickoff_times", "fixture_ids"):
    download[column] = download[column].map(lambda values: " | ".join(map(str, values)))
st.download_button("Download fixture_calendar.csv", download.to_csv(index=False),
                   file_name="fixture_calendar.csv", mime="text/csv")
st.caption("Only fixtures currently scheduled in the official FPL API are classified as CONFIRMED.")
