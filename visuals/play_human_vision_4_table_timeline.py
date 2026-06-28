from __future__ import annotations

import streamlit as st

import play_ui_common as ui

st.set_page_config(page_title="Liar's Poker - Table Timeline", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    header {visibility:hidden;}
    .block-container {max-width: 100% !important; padding:.45rem .8rem !important;}
    div[data-testid="column"] {padding:0 .12rem !important;}
    .table {display:grid; grid-template-columns: 1fr 270px 1fr; gap:.7rem; align-items:center; margin-bottom:.4rem;}
    .seat {border:1px solid #d1d5db; border-radius:9px; padding:.55rem; background:#fff; min-height:112px;}
    .pot {border-radius:12px; padding:.75rem; text-align:center; background:linear-gradient(135deg,#0f766e,#115e59); color:white; min-height:112px;}
    .seat-title {font-size:.74rem; text-transform:uppercase; letter-spacing:.1em; color:#6b7280; font-weight:900;}
    .pot .seat-title {color:#ccfbf1;}
    .claim {font:900 1.35rem Consolas, monospace; margin:.2rem 0;}
    .lp-hand {display:flex; gap:.35rem; margin-top:.3rem; align-items:center;}
    .lp-card {width:56px; height:72px; border:2px solid #1f2937; border-radius:8px; background:#fff; display:flex; align-items:center; justify-content:center; font:900 1.55rem Consolas, monospace; box-shadow:2px 2px 0 #9ca3af;}
    .lp-card.hidden {background:repeating-linear-gradient(45deg,#0f172a,#0f172a 9px,#475569 9px,#475569 18px); color:transparent;}
    .lp-timeline {display:flex; gap:.35rem; overflow-x:auto; padding:.4rem .1rem .5rem; border-bottom:1px solid #e5e7eb; margin-bottom:.5rem;}
    .lp-chip {display:flex; flex-direction:column; min-width:72px; border-radius:7px; padding:.32rem .45rem; border:1px solid #e5e7eb; background:#f9fafb;}
    .lp-chip span {font-size:.62rem; text-transform:uppercase; color:#6b7280; font-weight:900;}
    .lp-chip b {font:900 .8rem Consolas, monospace; color:#111827;}
    .lp-chip.human {background:#eff6ff; border-color:#bfdbfe;}
    .lp-chip.bot {background:#fff1f2; border-color:#fecdd3;}
    .rail {border:1px solid #d8dee4; border-radius:9px; background:#f8fafc; padding:.5rem;}
    .rail-title {font-size:.75rem; text-transform:uppercase; letter-spacing:.1em; font-weight:900; color:#4b5563; margin-bottom:.3rem;}
    .stButton > button {height:35px; min-height:35px; border-radius:999px; font:800 .82rem Consolas, monospace;}
    .stButton > button[kind="primary"] {background:#be123c; color:white; border-color:#9f1239;}
    .stButton > button[kind="secondary"] {background:white; border:1px solid #aeb7c2; color:#111827;}
    .stButton > button[kind="secondary"]:disabled {background:#f3f4f6; color:#c7cbd1; border-color:#e5e7eb;}
    .lp-log {height:372px; overflow:auto; border:1px solid #e5e7eb; border-radius:8px; background:#fff; padding:.35rem;}
    .lp-log-row {display:flex; justify-content:space-between; padding:.32rem .4rem; margin-bottom:.22rem; border-radius:5px; font-size:.78rem;}
    .lp-log-row.human {background:#eff6ff; color:#1d4ed8;}
    .lp-log-row.bot {background:#fff1f2; color:#be123c;}
    .lp-log-row.system {justify-content:center; color:#6b7280;}
    </style>
    """,
    unsafe_allow_html=True,
)

ui.ensure_state()
ui.render_loader_bar("Table timeline")
ctx = ui.bootstrap_app()

show_bot = st.session_state.reveal_bot or ctx.game_over
st.markdown(
    f"""
    <div class="table">
      <div class="seat">
        <div class="seat-title">You ({ctx.human_label})</div>
        {ui.hand_html(ctx.human_hand, ctx.spec)}
        <div class="claim">Score {st.session_state.score_human}</div>
      </div>
      <div class="pot">
        <div class="seat-title">Public state</div>
        <div class="claim">{ui.last_claim(ctx)}</div>
        <div>{"Your turn" if ctx.human_turn else "Bot turn"} | {len(ctx.legal)} legal</div>
      </div>
      <div class="seat">
        <div class="seat-title">Bot ({ctx.bot_label})</div>
        {ui.hand_html(ctx.bot_hand, ctx.spec, hidden=not show_bot)}
        <div class="claim">Score {st.session_state.score_bot}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ui.controls_row(ctx, compact=True)
st.markdown(ui.timeline_html(ctx), unsafe_allow_html=True)

action_col, info_col = st.columns([2.4, 1.0], gap="medium")
with info_col:
    st.markdown("<div class='rail-title'>Full log</div>", unsafe_allow_html=True)
    st.markdown(ui.log_html(ctx, max_items=16, compact=True), unsafe_allow_html=True)

with action_col:
    if not ctx.game_started:
        st.info("Start the hand.")
    elif ctx.game_over:
        ui.result_panel(ctx)
    elif not ctx.human_turn:
        st.info("Bot is choosing.")
    else:
        top_row = st.columns([.65, 2.35], gap="small")
        with top_row[0]:
            ui.call_button(ctx, key="timeline_call")
        with top_row[1]:
            mode = st.radio(
                "Action rail",
                ["Next claims", "All rank claims", "Combinations"],
                horizontal=True,
                label_visibility="collapsed",
            )
        if mode == "Next claims":
            st.markdown("<div class='rail-title'>Next legal claims</div>", unsafe_allow_html=True)
            claims = ui.legal_claims(ctx)[:24]
            cols = st.columns(8, gap="small")
            for i, action in enumerate(claims):
                with cols[i % 8]:
                    ui.button_for_action(ctx, action, key=f"timeline_next_{action}", label=ui.action_label(ctx.rules, action))
        elif mode == "All rank claims":
            for kind in [k for k in ("RankHigh", "Pair", "Trips", "Quads") if k in ctx.spec.claim_kinds]:
                st.markdown(f"<div class='rail-title'>{kind}</div>", unsafe_allow_html=True)
                cols = st.columns(ctx.spec.ranks, gap="small")
                for rank in range(1, ctx.spec.ranks + 1):
                    with cols[rank - 1]:
                        action = ui.single_kind_action(ctx, kind, rank)
                        ui.button_for_action(ctx, action, key=f"timeline_{kind}_{rank}", label=f"{rank}{ui.SINGLE_KIND_ALIASES[kind]}")
        else:
            if "TwoPair" in ctx.spec.claim_kinds:
                st.markdown("<div class='rail-title'>Two pair</div>", unsafe_allow_html=True)
                claims = [a for a in ui.legal_claims(ctx) if ctx.rules.claims[a][0] == "TwoPair"]
                cols = st.columns(8, gap="small")
                for i, action in enumerate(claims[:32]):
                    with cols[i % 8]:
                        ui.button_for_action(ctx, action, key=f"timeline_tp_{action}", label=ui.action_label(ctx.rules, action))
            if "FullHouse" in ctx.spec.claim_kinds:
                st.markdown("<div class='rail-title'>Full house</div>", unsafe_allow_html=True)
                claims = [a for a in ui.legal_claims(ctx) if ctx.rules.claims[a][0] == "FullHouse"]
                cols = st.columns(8, gap="small")
                for i, action in enumerate(claims[:40]):
                    with cols[i % 8]:
                        ui.button_for_action(ctx, action, key=f"timeline_fh_{action}", label=ui.action_label(ctx.rules, action))
