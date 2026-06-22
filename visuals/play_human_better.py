import os
import sys
import random
import time
from typing import Dict, List, Tuple

import streamlit as st

# --- PATH SETUP ---
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from liars_poker.core import card_display, card_rank
from liars_poker.env import Env, rules_for_spec
from liars_poker.infoset import CALL, NO_CLAIM, InfoSet
from liars_poker.serialization import load_policy

# --- CONSTANTS ---
HUMAN_LABEL = "P1"
BOT_LABEL = "P2"

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Liar's Poker Arena",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS INJECTION ---
st.markdown(
    """
    <style>
    /* 1. LAYOUT & WHITESPACE CONTROL */
    .block-container {
        padding: 0.65rem 1.25rem 0.75rem !important;
        max-width: 100% !important;
    }
    header {visibility: hidden;}
    section[data-testid="stSidebar"] {
        width: 285px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 285px !important;
        padding-top: 1rem !important;
    }
    hr {
        margin: 0.55rem 0 !important;
    }
    
    /* Remove padding between columns for the grid */
    div[data-testid="column"] {
        padding: 0px !important;
        gap: 4px !important; 
    }
    
    /* 2. CARD-STYLE BUTTONS (The Input Grid) */
    .stButton button[kind="secondary"] {
        background-color: #ffffff;
        border: 1px solid #d0d0d0;
        border-radius: 6px;
        height: 42px;
        width: 100%;
        margin: 1px 0px;
        color: #333;
        font-family: 'Courier New', monospace;
        font-weight: 900;
        font-size: 0.95rem;
        box-shadow: 1px 1px 0px #e0e0e0;
        transition: all 0.1s;
    }
    
    .stButton button[kind="secondary"]:hover {
        border-color: #aaa;
        transform: translateY(-1px);
        box-shadow: 2px 3px 0px #ccc;
        color: #000;
    }
    
    .stButton button[kind="secondary"]:disabled {
        background-color: #f9f9f9;
        color: #ddd;
        border-color: #eee;
        box-shadow: none;
    }

    /* 3. CALL BUTTON (Primary) */
    div[data-testid="stButton"] button[kind="primary"] {
        background-color: #ff4b4b;
        color: white;
        height: 44px;
        font-size: 1rem;
        font-weight: bold;
        border-radius: 8px;
        margin-top: 2px;
        box-shadow: 0px 3px 0px #cc3a3a;
    }
    div[data-testid="stButton"] button[kind="primary"]:active {
        transform: translateY(2px);
        box-shadow: 0px 2px 0px #cc3a3a;
    }

    /* 4. VISUAL CARDS */
    .card-container {
        display: flex;
        gap: 5px;
        margin-top: 3px;
    }
    .display-card {
        background: white;
        border: 2px solid #333;
        border-radius: 5px;
        width: 40px;
        height: 54px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 17px;
        box-shadow: 2px 2px 0px #888;
    }
    .display-card.hidden {
        background: repeating-linear-gradient(
          45deg,
          #606dbc,
          #606dbc 10px,
          #465298 10px,
          #465298 20px
        );
        color: transparent;
        border-color: #222;
    }

    /* 5. TICKER / LOG */
    .ticker-box {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 8px;
        height: 370px;
        overflow-y: auto;
        font-family: 'Segoe UI', sans-serif;
        font-size: 0.85rem;
        display: flex;
        flex-direction: column;
        gap: 5px;
    }
    .log-entry {
        padding: 6px 8px;
        border-radius: 6px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .log-entry.bot { 
        background: #fff0f0; 
        border-left: 4px solid #ff4b4b;
        color: #c0392b;
    }
    .log-entry.human { 
        background: #f0f7ff; 
        border-left: 4px solid #3498db;
        color: #2980b9;
        flex-direction: row-reverse;
    }
    .log-entry.system { 
        color: #95a5a6; 
        font-style: italic; 
        font-size: 0.85rem;
        justify-content: center; 
        border-bottom: 1px solid #eee;
        padding-bottom: 4px;
    }
    .log-entry span { font-weight: 600; }
    
    /* 6. UTILITY */
    .big-label { 
        font-size: 0.9rem; 
        font-weight: 700; 
        color: #555; 
        text-transform: uppercase; 
        letter-spacing: 1px;
    }
    .scoreboard {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 4px 8px;
        display: flex;
        align-items: center;
        justify-content: space-around;
        font-family: 'Segoe UI', sans-serif;
        box-shadow: 2px 2px 0px #e0e0e0;
    }
    .score-item {
        text-align: center;
    }
    .score-label {
        font-size: 0.75rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .score-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #222;
    }
    /* 7. INPUT GRID SIZE */
    div[data-testid="stForm"] {
        padding: 0.7rem 0.8rem 0.8rem !important;
        border: 1px solid #aeb3b8 !important;
        border-radius: 5px !important;
        background: #f3f4f5 !important;
        box-shadow: inset 0 1px 0 #ffffff, 0 2px 0 #d3d6d8;
    }
    form[data-testid="stForm"] .stButton button[kind="secondary"] {
        height: 34px;
        width: 100%;
        margin: 2px auto;
        padding: 0.1rem 0.25rem;
        border: 1px solid #9aa0a6;
        border-radius: 3px;
        background: linear-gradient(#ffffff, #eceff1);
        color: #202124;
        font-family: 'Courier New', monospace;
        font-size: 0.86rem;
        font-weight: 800;
        box-shadow: 0 2px 0 #c4c7c9;
    }
    form[data-testid="stForm"] .stButton button[kind="secondary"]:hover {
        border-color: #555b60;
        background: #ffffff;
        transform: translateY(-1px);
        box-shadow: 0 3px 0 #b7bbbe;
    }
    form[data-testid="stForm"] .stButton button[kind="secondary"]:disabled {
        border-color: #d4d7d9;
        background: #e9ebec;
        color: #afb3b6;
        box-shadow: none;
    }
    .calc-title {
        margin: 0 0 0.35rem;
        color: #3c4043;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .calc-label {
        min-height: 34px;
        display: flex;
        align-items: center;
        color: #5f6368;
        font-family: 'Courier New', monospace;
        font-size: 0.78rem;
        font-weight: 800;
    }
    .calc-header {
        min-height: 20px;
        text-align: center;
        color: #777;
        font-family: 'Courier New', monospace;
        font-size: 0.72rem;
        font-weight: 700;
    }
    div[data-testid="stAlert"] {
        padding: 0.55rem 0.75rem !important;
    }
    .section-rule {
        height: 1px;
        background: #e8e8e8;
        margin: 0.55rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- HELPERS ---

def render_display_card(rank_str: str, hidden: bool = False) -> str:
    cls = "display-card hidden" if hidden else "display-card"
    content = "" if hidden else rank_str
    return f"<div class='{cls}'>{content}</div>"

def render_hand_html(hand: Tuple[int, ...], spec, hidden: bool = False) -> str:
    html = "<div class='card-container'>"
    if hidden:
        for _ in hand:
            html += render_display_card("?", hidden=True)
    else:
        for c in hand:
            html += render_display_card(card_display(c, spec))
    html += "</div>"
    return html

def render_ticker(history: List[int], rules, human_label: str) -> str:
    # Build list of strings to avoid indentation issues in Markdown
    lines = ["<div class='ticker-box'>"]
    
    if not history:
        lines.append("<div class='log-entry system'>Game Started</div>")
    
    for i, action in enumerate(history):
        actor_label = "P1" if i % 2 == 0 else "P2"
        is_human = (actor_label == human_label)
        actor = "You" if is_human else "Bot"
        cls = "human" if is_human else "bot"
        text = rules.render_action(action)
        
        # Single line HTML to prevent Markdown code block detection
        entry = f"<div class='log-entry {cls}'><span>{actor}</span><span>{text}</span></div>"
        lines.append(entry)
    
    lines.append("</div>")
    return "".join(lines)

def get_truth_summary(spec, rules, history, p1, p2):
    last_idx = InfoSet.last_claim_idx(history)
    if last_idx == NO_CLAIM: return None
    
    kind, rank = rules.claims[last_idx]
    if kind == "RankHigh":
        needed = 1
        count = sum(1 for c in p1 + p2 if card_rank(c, spec) == rank)
        return rules.render_action(last_idx), count, needed
    if kind == "Pair":
        needed = 2
        count = sum(1 for c in p1 + p2 if card_rank(c, spec) == rank)
        return rules.render_action(last_idx), count, needed
    if kind == "Trips":
        needed = 3
        count = sum(1 for c in p1 + p2 if card_rank(c, spec) == rank)
        return rules.render_action(last_idx), count, needed
    if kind == "TwoPair":
        low, high = rules.two_pair_ranks[rank]
        count_low = sum(1 for c in p1 + p2 if card_rank(c, spec) == low)
        count_high = sum(1 for c in p1 + p2 if card_rank(c, spec) == high)
        count = f"{high}:{count_high}, {low}:{count_low}"
        needed = "2 each"
        return rules.render_action(last_idx), count, needed
    if kind == "FullHouse":
        trip, pair = rules.full_house_ranks[rank]
        count_trip = sum(1 for c in p1 + p2 if card_rank(c, spec) == trip)
        count_pair = sum(1 for c in p1 + p2 if card_rank(c, spec) == pair)
        count = f"{trip}:{count_trip}, {pair}:{count_pair}"
        needed = "3+2"
        return rules.render_action(last_idx), count, needed
    if kind == "Quads":
        needed = 4
        count = sum(1 for c in p1 + p2 if card_rank(c, spec) == rank)
        return rules.render_action(last_idx), count, needed
    return rules.render_action(last_idx), 0, 0

def other_label(label: str) -> str:
    return "P2" if label == "P1" else "P1"

def hand_for_label(env: Env, label: str) -> Tuple[int, ...]:
    return env._p1_hand if label == "P1" else env._p2_hand


# --- STATE MANAGEMENT ---

def init_state():
    if "env" not in st.session_state:
        st.session_state.env = None
        st.session_state.policy = None
        st.session_state.spec = None
        st.session_state.obs = None
        st.session_state.game_over = False
        st.session_state.bot_rng = random.Random(time.time())
    if "human_label" not in st.session_state:
        st.session_state.human_label = HUMAN_LABEL
    if "game_started" not in st.session_state:
        st.session_state.game_started = False
    if "score_human" not in st.session_state:
        st.session_state.score_human = 0
        st.session_state.score_bot = 0
        st.session_state.score_recorded = False
    if "reveal_bot" not in st.session_state:
        st.session_state.reveal_bot = False

def reset_game(*, begin_episode: bool = True) -> None:
    if st.session_state.env:
        st.session_state.obs = st.session_state.env.reset()
        if begin_episode:
            st.session_state.policy.begin_episode(st.session_state.bot_rng)
        st.session_state.game_over = False
        st.session_state.score_recorded = False
        st.session_state.reveal_bot = False

init_state()

# --- SIDEBAR: CONFIG ---
with st.sidebar:
    st.title("⚙️ Settings")
    default_dir = os.path.join(ROOT, "artifacts", "my_dense_policy")
    policy_dir = st.text_input("Policy Path", value=default_dir)
    
    if st.button("Load Bot", type="primary"):
        try:
            p, s = load_policy(policy_dir)
            st.session_state.policy = p
            st.session_state.spec = s
            st.session_state.env = Env(s)
            reset_game(begin_episode=False)
            st.session_state.game_started = False
            st.session_state.score_human = 0
            st.session_state.score_bot = 0
            st.session_state.score_recorded = False
            st.toast("Bot Loaded Successfully!", icon="🤖")
        except Exception as e:
            st.error(f"Failed: {e}")

    if st.session_state.spec:
        s = st.session_state.spec
        st.divider()
        # Expanded Info Display
        st.markdown(f"""
        **Ranks:** {s.ranks}  
        **Suits:** {s.suits}  
        **Hand Size:** {s.hand_size}  
        **Claims:** {", ".join(s.claim_kinds)}
        """)
        st.divider()
        st.subheader("Who Starts")
        game_in_progress = False
        if st.session_state.get("env") and st.session_state.get("obs") is not None:
            game_in_progress = (
                st.session_state.game_started
                and not st.session_state.obs["terminal"]
            )
        start_cols = st.columns(2)
        if start_cols[0].button("You", disabled=game_in_progress, use_container_width=True):
            st.session_state.human_label = "P1"
        if start_cols[1].button("Bot", disabled=game_in_progress, use_container_width=True):
            st.session_state.human_label = "P2"
        current_starter = "You" if st.session_state.human_label == "P1" else "Bot"
        st.caption(f"Current starter: {current_starter}")

# --- MAIN APP ---

if not st.session_state.env:
    st.warning("Please load a policy from the sidebar to start.")
    st.stop()

env = st.session_state.env
spec = st.session_state.spec
rules = rules_for_spec(spec)
obs = st.session_state.obs
policy = st.session_state.policy
human_label = st.session_state.human_label
bot_label = other_label(human_label)
game_started = st.session_state.game_started

# -- GAME LOGIC --
human_turn = (env.current_player() == human_label)
game_over = obs["terminal"]
if game_started and game_over and not st.session_state.score_recorded:
    winner = obs["winner"]
    if winner == human_label:
        st.session_state.score_human += 1
    elif winner == bot_label:
        st.session_state.score_bot += 1
    st.session_state.score_recorded = True

if game_started and not game_over and not human_turn:
    # Bot Move
    bot_iset = env.infoset_key(bot_label)
    act = policy.sample(bot_iset, st.session_state.bot_rng)
    obs = env.step(act)
    st.session_state.obs = obs
    st.rerun()

# --- COMPACT STATUS BAR ---
human_display = f"You ({human_label})"
bot_display = f"Bot ({bot_label})"
human_hand = hand_for_label(env, human_label)
bot_hand = hand_for_label(env, bot_label)

score_col, human_col, bot_col, controls_col = st.columns(
    [1.0, 1.25, 1.25, 0.65],
    gap="medium",
)
with score_col:
    st.markdown("<div class='big-label'>Score</div>", unsafe_allow_html=True)
    score_html = f"""
    <div class='scoreboard'>
        <div class='score-item'>
            <div class='score-label'>{human_display}</div>
            <div class='score-value'>{st.session_state.score_human}</div>
        </div>
        <div class='score-item'>
            <div class='score-label'>{bot_display}</div>
            <div class='score-value'>{st.session_state.score_bot}</div>
        </div>
    </div>
    """
    st.markdown(score_html, unsafe_allow_html=True)
with human_col:
    st.markdown("<div class='big-label'>Your Hand</div>", unsafe_allow_html=True)
    st.markdown(render_hand_html(human_hand, spec), unsafe_allow_html=True)
with bot_col:
    st.markdown("<div class='big-label'>Bot</div>", unsafe_allow_html=True)
    show_bot = st.session_state.reveal_bot or game_over
    st.markdown(render_hand_html(bot_hand, spec, hidden=not show_bot), unsafe_allow_html=True)
with controls_col:
    st.markdown("<div class='big-label'>Game</div>", unsafe_allow_html=True)
    if st.button("Reset Score", use_container_width=True):
        st.session_state.score_human = 0
        st.session_state.score_bot = 0
    if game_over:
        reveal_label = "Cards Revealed"
    else:
        reveal_label = "Hide Cards" if st.session_state.reveal_bot else "Reveal Cards"
    if st.button(
        reveal_label,
        disabled=game_over or not game_started,
        use_container_width=True,
    ):
        st.session_state.reveal_bot = not st.session_state.reveal_bot
        st.rerun()
    if not game_started:
        if st.button("Start Game", type="primary", use_container_width=True):
            reset_game(begin_episode=True)
            st.session_state.game_started = True
            st.rerun()

st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)

# 2. MAIN ARENA
col_game, col_log = st.columns([2.1, 0.9], gap="medium")

# --- RIGHT: LOG ---
with col_log:
    st.markdown("<div class='big-label'>Game Log</div>", unsafe_allow_html=True)
    history = list(env._history)
    st.markdown(render_ticker(history, rules, human_label), unsafe_allow_html=True)
    
    if game_over:
        winner = obs["winner"]
        res_type = "success" if winner == human_label else "error"
        msg = "🎉 YOU WIN!" if winner == human_label else "💀 BOT WINS"
        
        st.divider()
        if res_type == "success": st.success(msg)
        else: st.error(msg)
        
        t = get_truth_summary(spec, rules, tuple(history), env._p1_hand, env._p2_hand)
        if t:
            claim_txt, count, needed = t
            st.info(f"Claim: **{claim_txt}**\n\nActual Count: **{count}** (Needed {needed})")
            
        if st.button("🔄 Play Again", type="primary", use_container_width=True):
            reset_game(begin_episode=True)
            st.session_state.game_started = True
            st.rerun()

# --- LEFT: INPUT GRID ---
with col_game:
    if not game_started:
        st.markdown("<div class='big-label'>Input</div>", unsafe_allow_html=True)
        st.info("Press Start Game to begin.")
    elif not game_over:
        st.markdown("<div class='big-label'>Input</div>", unsafe_allow_html=True)
        
        if not human_turn:
            st.info("Bot is thinking...")
        else:
            legal = set(env.legal_actions())
            claim_kinds = spec.claim_kinds

            single_kinds = [k for k in ("RankHigh", "Pair", "Trips", "Quads") if k in claim_kinds]
            has_two_pair = "TwoPair" in claim_kinds
            has_full_house = "FullHouse" in claim_kinds

            claim_map = {}
            for idx, (k, r) in enumerate(rules.claims):
                if k in single_kinds:
                    claim_map[(k, r)] = idx

            can_call = CALL in legal
            hint_col, call_col = st.columns([4, 1])
            with hint_col:
                st.caption("Choose a higher claim, or challenge the current claim.")
            with call_col:
                if st.button(
                    "CALL",
                    key="btn_call",
                    type="primary",
                    disabled=not can_call,
                    use_container_width=True,
                ):
                    obs = env.step(CALL)
                    st.session_state.obs = obs
                    st.rerun()

            with st.form(key="input-grid", clear_on_submit=False):
                has_combo_claims = has_two_pair or has_full_house
                if single_kinds and has_combo_claims:
                    rank_col, combo_col = st.columns([1.25, 1.0], gap="medium")
                elif single_kinds:
                    rank_col = st.container()
                    combo_col = None
                else:
                    rank_col = None
                    combo_col = st.container()

                if rank_col is not None:
                    with rank_col:
                        st.markdown("<div class='calc-title'>Rank claims</div>", unsafe_allow_html=True)
                        header_cols = st.columns(
                            [1.25] + [1.0] * spec.ranks,
                            gap="small",
                        )
                        header_cols[0].markdown(
                            "<div class='calc-header'>TYPE</div>",
                            unsafe_allow_html=True,
                        )
                        for rank in range(1, spec.ranks + 1):
                            header_cols[rank].markdown(
                                f"<div class='calc-header'>{rank}</div>",
                                unsafe_allow_html=True,
                            )

                        for kind in single_kinds:
                            row_cols = st.columns(
                                [1.25] + [1.0] * spec.ranks,
                                gap="small",
                            )
                            row_cols[0].markdown(
                                f"<div class='calc-label'>{kind}</div>",
                                unsafe_allow_html=True,
                            )
                            for rank in range(1, spec.ranks + 1):
                                idx = claim_map.get((kind, rank))
                                if row_cols[rank].form_submit_button(
                                    str(rank),
                                    key=f"btn_{idx}",
                                    disabled=idx not in legal,
                                    use_container_width=True,
                                ):
                                    obs = env.step(idx)
                                    st.session_state.obs = obs
                                    st.rerun()

                if combo_col is not None:
                    with combo_col:
                        if has_two_pair:
                            st.markdown("<div class='calc-title'>Two pair</div>", unsafe_allow_html=True)
                            pair_map = {}
                            for idx, (kind, rank_idx) in enumerate(rules.claims):
                                if kind == "TwoPair":
                                    low, high = rules.two_pair_ranks[rank_idx]
                                    pair_map[(low, high)] = idx

                            header_cols = st.columns(
                                [0.85] + [1.0] * max(1, spec.ranks - 1),
                                gap="small",
                            )
                            header_cols[0].markdown(
                                "<div class='calc-header'>H \\ L</div>",
                                unsafe_allow_html=True,
                            )
                            for low in range(1, spec.ranks):
                                header_cols[low].markdown(
                                    f"<div class='calc-header'>{low}</div>",
                                    unsafe_allow_html=True,
                                )

                            for high in range(2, spec.ranks + 1):
                                row_cols = st.columns(
                                    [0.85] + [1.0] * max(1, spec.ranks - 1),
                                    gap="small",
                                )
                                row_cols[0].markdown(
                                    f"<div class='calc-label'>{high}</div>",
                                    unsafe_allow_html=True,
                                )
                                for low in range(1, spec.ranks):
                                    if low >= high:
                                        row_cols[low].write("")
                                        continue
                                    idx = pair_map.get((low, high))
                                    if row_cols[low].form_submit_button(
                                        f"{high}-{low}",
                                        key=f"btn_tp_{high}_{low}",
                                        disabled=idx not in legal,
                                        use_container_width=True,
                                    ):
                                        obs = env.step(idx)
                                        st.session_state.obs = obs
                                        st.rerun()

                        if has_full_house:
                            st.markdown("<div class='calc-title'>Full house</div>", unsafe_allow_html=True)
                            fh_map = {}
                            for idx, (kind, rank_idx) in enumerate(rules.claims):
                                if kind == "FullHouse":
                                    trip, pair = rules.full_house_ranks[rank_idx]
                                    fh_map[(trip, pair)] = idx

                            header_cols = st.columns(
                                [0.85] + [1.0] * spec.ranks,
                                gap="small",
                            )
                            header_cols[0].markdown(
                                "<div class='calc-header'>T \\ P</div>",
                                unsafe_allow_html=True,
                            )
                            for pair in range(1, spec.ranks + 1):
                                header_cols[pair].markdown(
                                    f"<div class='calc-header'>{pair}</div>",
                                    unsafe_allow_html=True,
                                )

                            for trip in range(1, spec.ranks + 1):
                                row_cols = st.columns(
                                    [0.85] + [1.0] * spec.ranks,
                                    gap="small",
                                )
                                row_cols[0].markdown(
                                    f"<div class='calc-label'>{trip}</div>",
                                    unsafe_allow_html=True,
                                )
                                for pair in range(1, spec.ranks + 1):
                                    if pair == trip:
                                        row_cols[pair].write("")
                                        continue
                                    idx = fh_map.get((trip, pair))
                                    if row_cols[pair].form_submit_button(
                                        f"{trip}/{pair}",
                                        key=f"btn_fh_{trip}_{pair}",
                                        disabled=idx not in legal,
                                        use_container_width=True,
                                    ):
                                        obs = env.step(idx)
                                        st.session_state.obs = obs
                                        st.rerun()
