import json
from pathlib import Path

import streamlit as st

from fpl_predictor.config import LIVE_DATA_DIR, RAW_DATA_DIR
from fpl_predictor.entry import load_manual_squad
from fpl_predictor.loaders import load_players
from fpl_predictor.squad_update import update_manual_squad
from fpl_predictor.ui.components import configure_page, initialize_session, render_sidebar

configure_page("Settings")
settings, bundle = render_sidebar()
st.title("Settings")
st.caption("Settings are runtime-only. No passwords, private FPL sessions, or API secrets are used.")

st.subheader("Transfer scenario")
st.number_input("Minimum expected gain", min_value=0.0, step=0.1, key="minimum_gain")
st.checkbox("Assume unknown selling prices equal current prices", key="assume_selling_price_current",
            help="Explicit scenario mode only; this is not authoritative financial state.")
st.number_input("Live cache TTL (seconds)", min_value=60, max_value=3600, step=60, key="refresh_ttl")
if st.session_state.assume_selling_price_current:
    st.warning("SCENARIO MODE will be clearly marked on transfer outputs.")

st.subheader("Manual squad upload")
uploaded = st.file_uploader("Upload manual_squad.json", type="json")
if uploaded is not None and st.button("Use uploaded squad for this session"):
    try:
        payload = json.loads(uploaded.getvalue())
        bootstrap = json.loads((RAW_DATA_DIR / "bootstrap_static.json").read_text(encoding="utf-8"))
        runtime = LIVE_DATA_DIR / "session_manual_squad.json"
        runtime.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        load_manual_squad(runtime, load_players(bootstrap))
        st.session_state.squad_file = str(runtime)
        st.cache_data.clear()
        st.success("Uploaded squad validated. It is stored only in this runtime filesystem/session.")
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        st.error(f"Manual squad upload failed: {exc}")

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
            bootstrap = json.loads((RAW_DATA_DIR / "bootstrap_static.json").read_text(encoding="utf-8"))
            backup, _ = update_manual_squad(settings.squad_file, load_players(bootstrap), player_out, player_in,
                                             captain=captain, vice_captain=vice)
            st.cache_data.clear()
            st.success(f"Squad updated and validated. Backup: {backup}")
        except (ValueError, OSError) as exc:
            st.error(f"Squad update failed: {exc}")

st.divider()
st.write("The dashboard never logs in to FPL, submits transfers, or changes your public team.")
st.info("On Streamlit Community Cloud, local filesystem changes are ephemeral. Uploads and squad edits may disappear when the app restarts; use the CLI for persistent local updates.")
