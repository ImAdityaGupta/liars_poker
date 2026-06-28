from __future__ import annotations

import streamlit as st

import play_ui_common as ui

st.set_page_config(page_title="Liar's Poker - Calculator", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    header {visibility:hidden;}
    .block-container {max-width: 100% !important; padding: .45rem .75rem .55rem !important;}
    section[data-testid="stSidebar"] {width: 280px !important;}
    section[data-testid="stSidebar"] > div {width: 280px !important;}
    div[data-testid="column"] {padding: 0 .12rem !important;}
    .lp-top {display:grid; grid-template-columns: 185px 1fr 1fr 225px; gap:.55rem; align-items:stretch;}
    .lp-panel {border:1px solid #d8dce0; border-radius:7px; background:#fff; padding:.45rem .55rem; box-shadow:0 1px 0 #e9ecef;}
    .lp-label {font-size:.72rem; text-transform:uppercase; letter-spacing:.08em; color:#5f6368; font-weight:800; margin-bottom:.25rem;}
    .lp-score {display:flex; justify-content:space-around; align-items:center; height:58px;}
    .lp-score div {text-align:center;}
    .lp-score span {display:block; color:#6b7280; font-size:.68rem; text-transform:uppercase;}
    .lp-score b {font-size:1.45rem;}
    .lp-hand {display:flex; gap:.34rem; min-height:68px; align-items:center;}
    .lp-card {width:50px; height:66px; border:2px solid #222; border-radius:6px; display:flex; align-items:center; justify-content:center; font-family:Consolas, monospace; font-weight:900; font-size:1.45rem; box-shadow:2px 2px 0 #888; background:#fff;}
    .lp-card.hidden {background:repeating-linear-gradient(45deg,#374151,#374151 8px,#5865a8 8px,#5865a8 16px); color:transparent;}
    .lp-status {display:flex; gap:.45rem; align-items:center; font-family:Consolas, monospace;}
    .lp-status .claim {font-size:1.05rem; font-weight:900; color:#111827;}
    .lp-status .turn {font-size:.8rem; color:#4b5563;}
    .lp-grid-title {font-size:.76rem; text-transform:uppercase; letter-spacing:.08em; font-weight:900; color:#374151; margin:.15rem 0 .28rem;}
    .lp-row-label {height:28px; display:flex; align-items:center; font-size:.72rem; font-weight:900; color:#4b5563;}
    .lp-head {height:18px; text-align:center; color:#6b7280; font:700 .68rem Consolas, monospace;}
    .stButton > button {height:29px; min-height:29px; padding:0 .15rem; border-radius:4px; font:800 .78rem Consolas, monospace;}
    .stButton > button[kind="secondary"] {background:linear-gradient(#fff,#f1f3f5); border:1px solid #adb5bd; color:#1f2937; box-shadow:0 2px 0 #cbd5e1;}
    .stButton > button[kind="secondary"]:disabled {background:#f1f3f5; border-color:#e5e7eb; color:#c3c8cf; box-shadow:none;}
    .stButton > button[kind="primary"] {background:#e03131; border-color:#c92a2a; color:#fff; box-shadow:0 3px 0 #a61e1e; font-size:.9rem;}
    .lp-log {height:315px; overflow:auto; border:1px solid #dde2e7; border-radius:6px; padding:.35rem; background:#f8fafc;}
    .lp-log-row {display:flex; justify-content:space-between; gap:.5rem; padding:.28rem .4rem; border-radius:4px; margin-bottom:.22rem; font-size:.78rem;}
    .lp-log-row.human {background:#e7f5ff; color:#1864ab;}
    .lp-log-row.bot {background:#fff0f0; color:#c92a2a;}
    .lp-log-row.system {justify-content:center; color:#868e96; font-style:italic;}
    .muted {color:#868e96; font-size:.78rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ui.ensure_state()
ui.render_loader_bar("Calculator arena")
ctx = ui.bootstrap_app()

st.markdown(
    f"""
    <div class="lp-top">
      <div class="lp-panel"><div class="lp-label">Score</div>
        <div class="lp-score">
          <div><span>You</span><b>{st.session_state.score_human}</b></div>
          <div><span>Bot</span><b>{st.session_state.score_bot}</b></div>
        </div>
      </div>
      <div class="lp-panel"><div class="lp-label">Your hand</div>{ui.hand_html(ctx.human_hand, ctx.spec)}</div>
      <div class="lp-panel"><div class="lp-label">Bot hand</div>{ui.hand_html(ctx.bot_hand, ctx.spec, hidden=not (st.session_state.reveal_bot or ctx.game_over))}</div>
      <div class="lp-panel"><div class="lp-label">State</div>
        <div class="lp-status"><span class="claim">{ui.last_claim(ctx)}</span><span class="turn">{"Your turn" if ctx.human_turn else "Bot turn"}</span></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

ui.controls_row(ctx, compact=True)

left, right = st.columns([3.1, 1.0], gap="small")

with right:
    st.markdown("<div class='lp-grid-title'>Log</div>", unsafe_allow_html=True)
    st.markdown(ui.log_html(ctx, max_items=16, compact=True), unsafe_allow_html=True)

with left:
    if not ctx.game_started:
        st.info("Press Start.")
    elif ctx.game_over:
        ui.result_panel(ctx)
    elif not ctx.human_turn:
        st.info("Bot is choosing.")
    else:
        call_col, hint_col = st.columns([.55, 3.45], gap="small")
        with call_col:
            ui.call_button(ctx, key="calc_call")
        with hint_col:
            st.caption("Calculator mode: every compact legal claim is one key.")

        single_kinds = [k for k in ("RankHigh", "Pair", "Trips", "Quads") if k in ctx.spec.claim_kinds]
        if single_kinds:
            st.markdown("<div class='lp-grid-title'>Rank claims</div>", unsafe_allow_html=True)
            heads = st.columns([.8] + [1] * ctx.spec.ranks, gap="small")
            heads[0].markdown("<div class='lp-head'>TYPE</div>", unsafe_allow_html=True)
            for r in range(1, ctx.spec.ranks + 1):
                heads[r].markdown(f"<div class='lp-head'>{r}</div>", unsafe_allow_html=True)
            for kind in single_kinds:
                cols = st.columns([.8] + [1] * ctx.spec.ranks, gap="small")
                cols[0].markdown(f"<div class='lp-row-label'>{ui.SINGLE_KIND_ALIASES[kind]}</div>", unsafe_allow_html=True)
                for r in range(1, ctx.spec.ranks + 1):
                    with cols[r]:
                        action = ui.single_kind_action(ctx, kind, r)
                        ui.button_for_action(ctx, action, key=f"calc_{kind}_{r}", label=f"{r}{ui.SINGLE_KIND_ALIASES[kind]}")

        pair_col, house_col = st.columns([1, 1], gap="small")
        if "TwoPair" in ctx.spec.claim_kinds:
            with pair_col:
                st.markdown("<div class='lp-grid-title'>Two pair</div>", unsafe_allow_html=True)
                for high in range(2, ctx.spec.ranks + 1):
                    cols = st.columns([.55] + [1] * (ctx.spec.ranks - 1), gap="small")
                    cols[0].markdown(f"<div class='lp-row-label'>{high}</div>", unsafe_allow_html=True)
                    for low in range(1, ctx.spec.ranks):
                        with cols[low]:
                            action = ui.two_pair_action(ctx, low, high) if low < high else None
                            ui.button_for_action(ctx, action, key=f"calc_tp_{high}_{low}", label=f"{high}-{low}" if low < high else "")

        if "FullHouse" in ctx.spec.claim_kinds:
            with house_col:
                st.markdown("<div class='lp-grid-title'>Full house</div>", unsafe_allow_html=True)
                for trip in range(1, ctx.spec.ranks + 1):
                    cols = st.columns([.55] + [1] * ctx.spec.ranks, gap="small")
                    cols[0].markdown(f"<div class='lp-row-label'>{trip}</div>", unsafe_allow_html=True)
                    for pair in range(1, ctx.spec.ranks + 1):
                        with cols[pair]:
                            action = ui.full_house_action(ctx, trip, pair) if pair != trip else None
                            ui.button_for_action(ctx, action, key=f"calc_fh_{trip}_{pair}", label=f"{trip}/{pair}" if pair != trip else "")
