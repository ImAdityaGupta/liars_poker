from __future__ import annotations

import streamlit as st

import play_ui_common as ui

st.set_page_config(page_title="Liar's Poker - Claim Drawer", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    header {visibility:hidden;}
    .block-container {max-width: 100% !important; padding:.55rem .8rem !important;}
    div[data-testid="column"] {padding:0 .15rem !important;}
    .hero {display:grid; grid-template-columns: 260px 1fr; gap:.65rem; align-items:center; margin-bottom:.45rem;}
    .tile {background:#111827; color:#f9fafb; border-radius:8px; padding:.55rem .65rem; min-height:88px;}
    .tile.light {background:#fff; color:#111827; border:1px solid #d7dce2;}
    .eyebrow {font-size:.7rem; text-transform:uppercase; letter-spacing:.11em; color:#9ca3af; font-weight:900;}
    .tile.light .eyebrow {color:#6b7280;}
    .last-claim {font-size:1.55rem; font-family:Consolas, monospace; font-weight:900; line-height:1.15;}
    .sub {font-size:.78rem; color:#9ca3af;}
    .lp-hand {display:flex; gap:.35rem; align-items:center; margin-top:.25rem;}
    .lp-card {width:54px; height:70px; border:2px solid #1f2937; border-radius:7px; display:flex; align-items:center; justify-content:center; background:#fff; color:#111827; box-shadow:2px 2px 0 #9ca3af; font:900 1.5rem Consolas, monospace;}
    .lp-card.hidden {background:repeating-linear-gradient(45deg,#111827,#111827 8px,#4b5563 8px,#4b5563 16px); color:transparent;}
    .drawer {border:1px solid #d0d7de; border-radius:9px; padding:.55rem; background:#f8fafc;}
    .drawer-title {font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; font-weight:900; color:#374151; margin-bottom:.4rem;}
    .stButton > button {height:36px; min-height:36px; padding:.05rem .3rem; border-radius:5px; font:800 .86rem Consolas, monospace;}
    .stButton > button[kind="secondary"] {border:1px solid #aeb7c2; background:#fff; color:#111827;}
    .stButton > button[kind="secondary"]:disabled {border-color:#e5e7eb; color:#c7cbd1; background:#f3f4f6;}
    .stButton > button[kind="primary"] {background:#dc2626; color:#fff; border-color:#b91c1c; font-weight:900;}
    .lp-log {height:360px; overflow:auto; border:1px solid #dbe3eb; background:#fff; border-radius:8px; padding:.35rem;}
    .lp-log-row {display:flex; justify-content:space-between; margin-bottom:.25rem; padding:.38rem .45rem; border-radius:5px; font-size:.82rem;}
    .lp-log-row.human {background:#eff6ff; color:#1d4ed8;}
    .lp-log-row.bot {background:#fef2f2; color:#b91c1c;}
    .lp-log-row.system {justify-content:center; color:#6b7280;}
    .kind-note {font-size:.78rem; color:#6b7280; margin:.2rem 0 .45rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

ui.ensure_state()
ui.render_loader_bar("Claim drawer")
ctx = ui.bootstrap_app()

show_bot = st.session_state.reveal_bot or ctx.game_over
st.markdown(
    f"""
    <div class="hero">
      <div class="tile light"><div class="eyebrow">Score</div>
        <div class="last-claim">{st.session_state.score_human} - {st.session_state.score_bot}</div>
        <div class="sub">You vs Bot</div>
      </div>
      <div class="tile"><div class="eyebrow">Public claim</div>
        <div class="last-claim">{ui.last_claim(ctx)}</div>
        <div class="sub">{"Your decision" if ctx.human_turn else "Bot decision"}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 1, 1], gap="small")
with c1:
    st.markdown("<div class='eyebrow'>Your hand</div>", unsafe_allow_html=True)
    st.markdown(ui.hand_html(ctx.human_hand, ctx.spec), unsafe_allow_html=True)
with c2:
    st.markdown("<div class='eyebrow'>Bot hand</div>", unsafe_allow_html=True)
    st.markdown(ui.hand_html(ctx.bot_hand, ctx.spec, hidden=not show_bot), unsafe_allow_html=True)
with c3:
    ui.controls_row(ctx, compact=True)

main, side = st.columns([2.3, 1.0], gap="medium")
with side:
    st.markdown("<div class='drawer-title'>History</div>", unsafe_allow_html=True)
    st.markdown(ui.log_html(ctx, max_items=18), unsafe_allow_html=True)

with main:
    if not ctx.game_started:
        st.info("Start the hand when ready.")
    elif ctx.game_over:
        ui.result_panel(ctx)
    elif not ctx.human_turn:
        st.info("Bot is choosing.")
    else:
        call_col, kind_col = st.columns([.55, 2.4], gap="small")
        with call_col:
            ui.call_button(ctx, key="drawer_call")
        grouped = ui.actions_by_kind(ctx)
        kinds = [k for k in ctx.spec.claim_kinds if k in grouped]
        current_kind = kind_col.radio(
            "Claim kind",
            kinds,
            horizontal=True,
            label_visibility="collapsed",
            key="drawer_kind",
        )
        st.markdown(f"<div class='drawer-title'>{current_kind}</div>", unsafe_allow_html=True)
        st.markdown("<div class='kind-note'>Only this claim family is open; switch family above to reveal another drawer.</div>", unsafe_allow_html=True)

        if current_kind in ui.SINGLE_KIND_ALIASES:
            cols = st.columns(ctx.spec.ranks, gap="small")
            for r in range(1, ctx.spec.ranks + 1):
                with cols[r - 1]:
                    action = ui.single_kind_action(ctx, current_kind, r)
                    ui.button_for_action(ctx, action, key=f"drawer_{current_kind}_{r}", label=f"{r}{ui.SINGLE_KIND_ALIASES[current_kind]}")
        elif current_kind == "TwoPair":
            for high in range(2, ctx.spec.ranks + 1):
                cols = st.columns(high - 1, gap="small")
                for low in range(1, high):
                    with cols[low - 1]:
                        action = ui.two_pair_action(ctx, low, high)
                        ui.button_for_action(ctx, action, key=f"drawer_tp_{high}_{low}", label=f"{high}-{low}")
        elif current_kind == "FullHouse":
            for trip in range(1, ctx.spec.ranks + 1):
                cols = st.columns(ctx.spec.ranks - 1, gap="small")
                out_col = 0
                for pair in range(1, ctx.spec.ranks + 1):
                    if pair == trip:
                        continue
                    with cols[out_col]:
                        action = ui.full_house_action(ctx, trip, pair)
                        ui.button_for_action(ctx, action, key=f"drawer_fh_{trip}_{pair}", label=f"{trip}/{pair}")
                    out_col += 1
        else:
            actions = grouped[current_kind]
            cols = st.columns(min(8, max(1, len(actions))), gap="small")
            for i, action in enumerate(actions):
                with cols[i % len(cols)]:
                    ui.button_for_action(ctx, action, key=f"drawer_generic_{action}")
