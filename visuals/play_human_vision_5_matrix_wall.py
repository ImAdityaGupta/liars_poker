from __future__ import annotations

import streamlit as st

import play_ui_common as ui

st.set_page_config(page_title="Liar's Poker - Matrix Wall", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    header {visibility:hidden;}
    .block-container {max-width: 100% !important; padding:.35rem .65rem !important;}
    div[data-testid="column"] {padding:0 .08rem !important;}
    .hud {position:sticky; top:0; z-index:99; display:grid; grid-template-columns: 150px 1fr 1fr 210px; gap:.4rem; background:#f8fafc; padding:.25rem 0 .42rem; border-bottom:1px solid #dbe3eb;}
    .hud-box {border:1px solid #cbd5e1; border-radius:6px; background:#fff; padding:.35rem .45rem; min-height:78px;}
    .k {font-size:.62rem; text-transform:uppercase; letter-spacing:.1em; font-weight:900; color:#64748b;}
    .v {font:900 1.05rem Consolas, monospace; color:#0f172a;}
    .lp-hand {display:flex; gap:.22rem; margin-top:.18rem;}
    .lp-card {width:42px; height:54px; border:2px solid #111827; border-radius:5px; display:flex; align-items:center; justify-content:center; background:#fff; font:900 1.15rem Consolas, monospace; box-shadow:1px 1px 0 #94a3b8;}
    .lp-card.hidden {background:repeating-linear-gradient(45deg,#1e293b,#1e293b 7px,#64748b 7px,#64748b 14px); color:transparent;}
    .wall {border:1px solid #cbd5e1; border-radius:7px; background:#f8fafc; padding:.38rem; margin-top:.45rem;}
    .wall-title {font-size:.68rem; text-transform:uppercase; letter-spacing:.1em; color:#475569; font-weight:900; margin:.15rem 0 .25rem;}
    .row-label {height:25px; display:flex; align-items:center; font:900 .66rem Consolas, monospace; color:#475569;}
    .head {height:15px; text-align:center; font:800 .58rem Consolas, monospace; color:#64748b;}
    .stButton > button {height:25px; min-height:25px; padding:0 .08rem; border-radius:3px; font:900 .66rem Consolas, monospace;}
    .stButton > button[kind="secondary"] {background:#fff; border:1px solid #94a3b8; color:#111827;}
    .stButton > button[kind="secondary"]:hover {border-color:#334155;}
    .stButton > button[kind="secondary"]:disabled {background:#eef2f7; border-color:#e2e8f0; color:#cbd5e1;}
    .stButton > button[kind="primary"] {height:35px; background:#e11d48; border-color:#be123c; color:white; font-size:.82rem;}
    .lp-log {height:430px; overflow:auto; border:1px solid #dbe3eb; border-radius:6px; padding:.28rem; background:#fff;}
    .lp-log-row {display:flex; justify-content:space-between; padding:.26rem .35rem; margin-bottom:.17rem; border-radius:4px; font-size:.72rem;}
    .lp-log-row.human {background:#eff6ff; color:#1d4ed8;}
    .lp-log-row.bot {background:#fff1f2; color:#be123c;}
    .lp-log-row.system {justify-content:center; color:#64748b;}
    .mini-caption {font-size:.7rem; color:#64748b;}
    </style>
    """,
    unsafe_allow_html=True,
)

ui.ensure_state()
ui.render_loader_bar("Matrix wall")
ctx = ui.bootstrap_app()

show_bot = st.session_state.reveal_bot or ctx.game_over
st.markdown(
    f"""
    <div class="hud">
      <div class="hud-box"><div class="k">Score</div><div class="v">{st.session_state.score_human}-{st.session_state.score_bot}</div></div>
      <div class="hud-box"><div class="k">Your hand</div>{ui.hand_html(ctx.human_hand, ctx.spec, tiny=True)}</div>
      <div class="hud-box"><div class="k">Bot hand</div>{ui.hand_html(ctx.bot_hand, ctx.spec, hidden=not show_bot, tiny=True)}</div>
      <div class="hud-box"><div class="k">Last claim</div><div class="v">{ui.last_claim(ctx)}</div><div class="mini-caption">{"You act" if ctx.human_turn else "Bot acts"}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

ui.controls_row(ctx, compact=True)

board, rail = st.columns([3.25, .95], gap="small")
with rail:
    if ctx.game_started and not ctx.game_over and ctx.human_turn:
        ui.call_button(ctx, key="wall_call", label="CALL")
    st.markdown("<div class='wall-title'>Log</div>", unsafe_allow_html=True)
    st.markdown(ui.log_html(ctx, max_items=18, compact=True), unsafe_allow_html=True)

with board:
    if not ctx.game_started:
        st.info("Start the hand.")
    elif ctx.game_over:
        ui.result_panel(ctx)
    elif not ctx.human_turn:
        st.info("Bot is choosing.")
    else:
        single_kinds = [k for k in ("RankHigh", "Pair", "Trips", "Quads") if k in ctx.spec.claim_kinds]
        if single_kinds:
            st.markdown("<div class='wall-title'>Rank matrix</div>", unsafe_allow_html=True)
            header = st.columns([.6] + [1] * ctx.spec.ranks, gap="small")
            header[0].markdown("<div class='head'>K</div>", unsafe_allow_html=True)
            for r in range(1, ctx.spec.ranks + 1):
                header[r].markdown(f"<div class='head'>{r}</div>", unsafe_allow_html=True)
            for kind in single_kinds:
                cols = st.columns([.6] + [1] * ctx.spec.ranks, gap="small")
                cols[0].markdown(f"<div class='row-label'>{ui.SINGLE_KIND_ALIASES[kind]}</div>", unsafe_allow_html=True)
                for rank in range(1, ctx.spec.ranks + 1):
                    with cols[rank]:
                        action = ui.single_kind_action(ctx, kind, rank)
                        ui.button_for_action(ctx, action, key=f"wall_{kind}_{rank}", label=f"{rank}{ui.SINGLE_KIND_ALIASES[kind]}")

        combo_a, combo_b = st.columns([1, 1.1], gap="small")
        if "TwoPair" in ctx.spec.claim_kinds:
            with combo_a:
                st.markdown("<div class='wall-title'>Two pair</div>", unsafe_allow_html=True)
                for high in range(2, ctx.spec.ranks + 1):
                    cols = st.columns([.45] + [1] * (ctx.spec.ranks - 1), gap="small")
                    cols[0].markdown(f"<div class='row-label'>{high}</div>", unsafe_allow_html=True)
                    for low in range(1, ctx.spec.ranks):
                        with cols[low]:
                            action = ui.two_pair_action(ctx, low, high) if low < high else None
                            ui.button_for_action(ctx, action, key=f"wall_tp_{high}_{low}", label=f"{high}{low}" if low < high else "")
        if "FullHouse" in ctx.spec.claim_kinds:
            with combo_b:
                st.markdown("<div class='wall-title'>Full house</div>", unsafe_allow_html=True)
                for trip in range(1, ctx.spec.ranks + 1):
                    cols = st.columns([.45] + [1] * ctx.spec.ranks, gap="small")
                    cols[0].markdown(f"<div class='row-label'>{trip}</div>", unsafe_allow_html=True)
                    for pair in range(1, ctx.spec.ranks + 1):
                        with cols[pair]:
                            action = ui.full_house_action(ctx, trip, pair) if pair != trip else None
                            ui.button_for_action(ctx, action, key=f"wall_fh_{trip}_{pair}", label=f"{trip}/{pair}" if pair != trip else "")
