import json
import streamlit as st

from fpl_predictor.config import RAW_DATA_DIR
from fpl_predictor.loaders import load_players
from fpl_predictor.squad_update import update_manual_squad
from fpl_predictor.ui.components import (
    configure_page, render_sidebar,
)
from fpl_predictor.ui.state import (
    activate_uploaded_squad, active_squad_source, runtime_squad_path, write_uploaded_squad_to_runtime,
)

configure_page("Settings")
settings, bundle = render_sidebar()
st.title("Settings")
st.caption("Settings are runtime-only. No passwords, private FPL sessions, or API secrets are used.")
if message := st.session_state.pop("last_squad_upload_message", None):
    st.success(message)

st.subheader("Transfer scenario")
st.number_input("Minimum expected gain", min_value=0.0, step=0.1, key="minimum_gain")
st.number_input("Live cache TTL (seconds)", min_value=60, max_value=3600, step=60, key="refresh_ttl")
st.caption("Scenario Mode is controlled from the shared sidebar: Assume current price = selling price.")
if st.session_state.assume_selling_price_current:
    st.warning("SCENARIO MODE will be clearly marked on transfer outputs.")

st.subheader("Reliability alerts")
st.checkbox("Show reliability alerts", key="widget_show_reliability_alerts")
st.checkbox("Show owned-player change alerts", key="widget_show_player_change_alerts")
st.caption("Automatic refresh is active-session only. It stops when no browser session is running.")

st.subheader("Chip availability")
st.caption("Public pre-deadline chip availability is not assumed. Choose a manual state only when known.")
chip_columns = st.columns(2)
for index, (key, label) in enumerate((("chip_wildcard", "Wildcard"), ("chip_free_hit", "Free Hit"),
                                      ("chip_bench_boost", "Bench Boost"),
                                      ("chip_triple_captain", "Triple Captain"))):
    chip_columns[index % 2].selectbox(label, ["unknown", "available", "used"], key=key)

st.subheader("Manual squad upload")
st.caption(f"Current active squad: **{active_squad_source(st.session_state)}**")
if settings.squad_file is not None:
    st.caption(f"Active file: {settings.squad_file.name}")
uploaded = st.file_uploader("Upload manual_squad.json", type=["json"], key="squad_file_upload")
if uploaded is not None and st.button("Use uploaded squad for this session", key="activate_uploaded_squad"):
    try:
        bootstrap = json.loads((RAW_DATA_DIR / "bootstrap_static.json").read_text(encoding="utf-8"))
        runtime = write_uploaded_squad_to_runtime(
            uploaded.getvalue(), load_players(bootstrap), runtime_squad_path(st.session_state)
        )
        activate_uploaded_squad(st.session_state, runtime)
        # Only invalidates outputs derived from the selected squad; cached live API
        # snapshots and model artifacts stay intact.
        st.session_state.personalized_generation += 1
        st.session_state.analyst_generation += 1
        st.session_state.last_squad_upload_message = (
            "Uploaded squad validated and activated for this session. "
            "Personalized dashboard views were rebuilt from current live predictions."
        )
        st.rerun()
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(f"Manual squad upload failed; the previous valid squad remains active: {exc}")

st.subheader("Update Manual Squad")
if bundle.squad.empty:
    st.info("Load a valid manual squad and live player universe first.")
else:
    names = sorted(bundle.squad.player.dropna().tolist())
    player_out = st.selectbox("Player OUT", names)
    position = bundle.squad.loc[bundle.squad.player.eq(player_out), "position"].iloc[0]
    incoming = bundle.predictions[(bundle.predictions.position.eq(position)) & ~bundle.predictions.player_id.isin(bundle.squad.player_id)]
    player_in = st.selectbox("Player IN", sorted(incoming.player.dropna().tolist()))
    outgoing = bundle.squad[bundle.squad.player.eq(player_out)].iloc[0]
    captain = None; vice = None
    if bool(outgoing.get("is_captain", False)):
        captain = st.selectbox("New captain", [name for name in names if name != player_out])
    if bool(outgoing.get("is_vice_captain", False)):
        vice = st.selectbox("New vice-captain", [name for name in names if name != player_out])
    confirmed = st.checkbox("I confirm this local/session squad update")
    if st.button("Confirm Squad Update", disabled=not confirmed):
        try:
            if settings.squad_file is None:
                raise ValueError("No active manual squad file is available to update")
            bootstrap = json.loads((RAW_DATA_DIR / "bootstrap_static.json").read_text(encoding="utf-8"))
            backup, _ = update_manual_squad(settings.squad_file, load_players(bootstrap), player_out, player_in,
                                             captain=captain, vice_captain=vice)
            st.session_state.personalized_generation += 1
            st.session_state.analyst_generation += 1
            st.success(f"Squad updated and validated. Backup: {backup}")
        except (ValueError, OSError) as exc:
            st.error(f"Squad update failed: {exc}")

st.divider()
st.write("The dashboard never logs in to FPL, submits transfers, or changes your public team.")
st.info("On Streamlit Community Cloud, local filesystem changes are ephemeral. Uploads and squad edits may disappear when the app restarts; use the CLI for persistent local updates.")
