from __future__ import annotations

import streamlit as st

import play_ui_common as ui

st.set_page_config(page_title="Liar's Poker - Command Palette", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    header {visibility:hidden;}
    .block-container {max-width: 100% !important; padding:.5rem .9rem !important;}
    div[data-testid="column"] {padding:0 .12rem !important;}
    .shell {background:#0f172a; color:#e5e7eb; border-radius:9px; padding:.65rem .75rem; font-family:Consolas, monospace; min-height:116px;}
    .prompt {color:#38bdf8; font-weight:900; font-size:.76rem; text-transform:uppercase; letter-spacing:.08em;}
    .command {font-size:1.45rem; font-weight:900; margin:.16rem 0;}
    .line {font-size:.85rem; color:#cbd5e1;}
    .lp-hand {display:flex; gap:.3rem; margin-top:.28rem;}
    .lp-card {width:54px; height:70px; border:2px solid #94a3b8; border-radius:7px; display:flex; justify-content:center; align-items:center; background:#f8fafc; color:#0f172a; font:900 1.5rem Consolas, monospace;}
    .lp-card.hidden {background:repeating-linear-gradient(45deg,#334155,#334155 8px,#64748b 8px,#64748b 16px); color:transparent;}
    .pad {border:1px solid #cbd5e1; border-radius:8px; padding:.55rem; background:#f8fafc;}
    .pad-title {font-size:.74rem; text-transform:uppercase; letter-spacing:.09em; font-weight:900; color:#475569; margin-bottom:.3rem;}
    .stButton > button {height:34px; min-height:34px; border-radius:5px; font:800 .82rem Consolas, monospace;}
    .stButton > button[kind="primary"] {background:#ef4444; border-color:#dc2626; color:white;}
    .stButton > button[kind="secondary"] {background:white; border:1px solid #94a3b8; color:#0f172a;}
    .stButton > button[kind="secondary"]:disabled {color:#cbd5e1; background:#f1f5f9; border-color:#e2e8f0;}
    .lp-log {height:310px; overflow:auto; border:1px solid #dbe3eb; border-radius:8px; padding:.35rem; background:#fff;}
    .lp-log-row {display:flex; justify-content:space-between; padding:.34rem .45rem; margin-bottom:.24rem; border-radius:5px; font-size:.82rem;}
    .lp-log-row.human {background:#eff6ff; color:#1d4ed8;}
    .lp-log-row.bot {background:#fff1f2; color:#be123c;}
    .lp-log-row.system {justify-content:center; color:#64748b;}
    div[data-baseweb="select"] {font-family:Consolas, monospace;}
    </style>
    """,
    unsafe_allow_html=True,
)

ui.ensure_state()
ui.render_loader_bar("Command palette")
ctx = ui.bootstrap_app()

show_bot = st.session_state.reveal_bot or ctx.game_over
st.markdown(
    f"""
    <div class="shell">
      <div class="prompt">liars-poker / {ctx.human_label} / {"terminal" if ctx.game_over else "live"}</div>
      <div class="command">{ui.last_claim(ctx)}</div>
      <div class="line">Score {st.session_state.score_human}-{st.session_state.score_bot} | {"your turn" if ctx.human_turn else "bot turn"} | {len(ctx.legal)} legal actions</div>
    </div>
    """,
    unsafe_allow_html=True,
)

hand_col, bot_col, controls_col = st.columns([1, 1, 1.4], gap="small")
with hand_col:
    st.caption("Your hand")
    st.markdown(ui.hand_html(ctx.human_hand, ctx.spec), unsafe_allow_html=True)
with bot_col:
    st.caption("Bot hand")
    st.markdown(ui.hand_html(ctx.bot_hand, ctx.spec, hidden=not show_bot), unsafe_allow_html=True)
with controls_col:
    ui.controls_row(ctx, compact=True)

left, right = st.columns([1.8, 1.0], gap="medium")

with right:
    st.markdown("<div class='pad-title'>Recent history</div>", unsafe_allow_html=True)
    st.markdown(ui.log_html(ctx, max_items=20), unsafe_allow_html=True)

with left:
    if not ctx.game_started:
        st.info("Start the hand to open the command palette.")
    elif ctx.game_over:
        ui.result_panel(ctx)
    elif not ctx.human_turn:
        st.info("Bot is choosing.")
    else:
        legal_claims = ui.legal_claims(ctx)
        legal_options = {
            f"{ui.action_long_label(ctx.rules, action)}   [{ui.action_label(ctx.rules, action)}]": action
            for action in legal_claims
        }
        st.markdown("<div class='pad-title'>Command entry</div>", unsafe_allow_html=True)
        c1, c2 = st.columns([2.7, .75], gap="small")
        selected = c1.selectbox(
            "Legal claim",
            list(legal_options.keys()),
            index=0 if legal_options else None,
            label_visibility="collapsed",
            placeholder="Choose a claim",
        )
        with c2:
            if st.button("PLAY", type="primary", disabled=not selected, use_container_width=True):
                ui.submit_action(legal_options[selected])
        st.caption("Keyboard-friendly mode: open the dropdown and type part of a claim.")

        quick = legal_claims[: min(18, len(legal_claims))]
        st.markdown("<div class='pad-title'>Next legal claims</div>", unsafe_allow_html=True)
        if quick:
            cols = st.columns(6, gap="small")
            for i, action in enumerate(quick):
                with cols[i % 6]:
                    ui.button_for_action(ctx, action, key=f"palette_quick_{action}", label=ui.action_label(ctx.rules, action))
        ui.call_button(ctx, key="palette_call", label="CALL / challenge")
