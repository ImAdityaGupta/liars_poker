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
        padding-top: 2rem !important;
        padding-bottom: 1rem !important;
        max-width: 95% !important;
    }
    header {visibility: hidden;}
    
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
        height: 55px;
        width: 100%;
        margin: 2px 0px;
        color: #333;
        font-family: 'Courier New', monospace;
        font-weight: 900;
        font-size: 1.1rem;
        box-shadow: 2px 2px 0px #e0e0e0;
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
        height: 60px;
        font-size: 1.2rem;
        font-weight: bold;
        border-radius: 8px;
        margin-top: 10px;
        box-shadow: 0px 4px 0px #cc3a3a;
    }
    div[data-testid="stButton"] button[kind="primary"]:active {
        transform: translateY(2px);
        box-shadow: 0px 2px 0px #cc3a3a;
    }

    /* 4. VISUAL CARDS */
    .card-container {
        display: flex;
        gap: 8px;
        margin-top: 5px;
    }
    .display-card {
        background: white;
        border: 2px solid #333;
        border-radius: 6px;
        width: 50px;
        height: 70px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Courier New', monospace;
        font-weight: bold;
        font-size: 20px;
        box-shadow: 3px 3px 0px #888;
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
        padding: 15px;
        height: 600px;
        overflow-y: auto;
        font-family: 'Segoe UI', sans-serif;
        font-size: 0.95rem;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .log-entry {
        padding: 8px 12px;
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
        font-size: 1.1rem; 
        font-weight: 700; 
        color: #555; 
        text-transform: uppercase; 
        letter-spacing: 1px;
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

def render_ticker(history: List[int], rules) -> str:
    # Build list of strings to avoid indentation issues in Markdown
    lines = ["<div class='ticker-box'>"]
    
    if not history:
        lines.append("<div class='log-entry system'>Game Started</div>")
    
    for i, action in enumerate(history):
        is_human = (i % 2 == 0)
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
    return rules.render_action(last_idx), 0, 0


# --- STATE MANAGEMENT ---

def init_state():
    if "env" not in st.session_state:
        st.session_state.env = None
        st.session_state.policy = None
        st.session_state.spec = None
        st.session_state.obs = None
        st.session_state.game_over = False
        st.session_state.bot_rng = random.Random(time.time())

def reset_game():
    if st.session_state.env:
        st.session_state.obs = st.session_state.env.reset()
        st.session_state.policy.begin_episode(st.session_state.bot_rng)
        st.session_state.game_over = False

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
            reset_game()
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

# --- MAIN APP ---

if not st.session_state.env:
    st.warning("Please load a policy from the sidebar to start.")
    st.stop()

env = st.session_state.env
spec = st.session_state.spec
rules = rules_for_spec(spec)
obs = st.session_state.obs
policy = st.session_state.policy

# -- GAME LOGIC --
human_turn = (env.current_player() == HUMAN_LABEL)
game_over = obs["terminal"]

if not game_over and not human_turn:
    # Bot Move
    bot_iset = env.infoset_key(BOT_LABEL)
    act = policy.sample(bot_iset, st.session_state.bot_rng)
    obs = env.step(act)
    st.session_state.obs = obs
    st.rerun()

# --- UI LAYOUT ---

# 1. HEADER
label_c1, label_c2 = st.columns([1, 1])
with label_c1:
    st.markdown("<div class='big-label'>Your Hand</div>", unsafe_allow_html=True)
with label_c2:
    lc_left, lc_right = st.columns([0.3, 0.7]) 
    with lc_left:
        st.markdown("<div class='big-label'>Bot</div>", unsafe_allow_html=True)
    with lc_right:
        reveal = st.checkbox("Reveal", value=game_over, disabled=game_over, key="reveal_chk")

hand_c1, hand_c2 = st.columns([1, 1])
with hand_c1:
    st.markdown(render_hand_html(env._p1_hand, spec), unsafe_allow_html=True)
with hand_c2:
    show_bot = reveal or game_over
    st.markdown(render_hand_html(env._p2_hand, spec, hidden=not show_bot), unsafe_allow_html=True)

st.markdown("---")

# 2. MAIN ARENA
col_log, col_game = st.columns([1, 1.3], gap="large")

# --- LEFT: LOG ---
with col_log:
    st.markdown("<div class='big-label'>Game Log</div>", unsafe_allow_html=True)
    history = list(env._history)
    st.markdown(render_ticker(history, rules), unsafe_allow_html=True)
    
    if game_over:
        winner = obs["winner"]
        res_type = "success" if winner == HUMAN_LABEL else "error"
        msg = "🎉 YOU WIN!" if winner == HUMAN_LABEL else "💀 BOT WINS"
        
        st.divider()
        if res_type == "success": st.success(msg)
        else: st.error(msg)
        
        t = get_truth_summary(spec, rules, tuple(history), env._p1_hand, env._p2_hand)
        if t:
            claim_txt, count, needed = t
            st.info(f"Claim: **{claim_txt}**\n\nActual Count: **{count}** (Needed {needed})")
            
        if st.button("🔄 Play Again", type="primary", use_container_width=True):
            reset_game()
            st.rerun()

# --- RIGHT: INPUT GRID ---
with col_game:
    if not game_over:
        st.markdown("<div class='big-label'>Input</div>", unsafe_allow_html=True)
        
        if not human_turn:
            st.info("Bot is thinking...")
        else:
            legal = set(env.legal_actions())
            claim_kinds = spec.claim_kinds

            single_kinds = [k for k in claim_kinds if k in ("RankHigh", "Pair", "Trips")]
            has_two_pair = "TwoPair" in claim_kinds

            claim_map = {}
            for idx, (k, r) in enumerate(rules.claims):
                if k in single_kinds:
                    claim_map[(k, r)] = idx

            if single_kinds:
                cols = st.columns(len(single_kinds), gap="small")
                for i, k in enumerate(single_kinds):
                    cols[i].markdown(f"**{k}**")
                
                for r in range(1, spec.ranks + 1):
                    row_cols = st.columns(len(single_kinds), gap="small")
                    for i, k in enumerate(single_kinds):
                        idx = claim_map.get((k, r))
                        if idx is not None:
                            is_legal = idx in legal
                            suffix = k[0].upper()
                            if k == "RankHigh":
                                suffix = "H"
                            btn_label = f"{r}{suffix}"
                            
                            if row_cols[i].button(btn_label, key=f"btn_{idx}", disabled=not is_legal, use_container_width=True):
                                obs = env.step(idx)
                                st.session_state.obs = obs
                                st.rerun()
                        else:
                            row_cols[i].write("")

            if has_two_pair:
                st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)
                st.markdown("**TwoPair**")
                pair_map = {}
                for idx, (k, r) in enumerate(rules.claims):
                    if k == "TwoPair":
                        low, high = rules.two_pair_ranks[r]
                        pair_map[(low, high)] = idx

                cols = st.columns(max(1, spec.ranks - 1), gap="small")
                for j in range(1, spec.ranks):
                    cols[j - 1].markdown(f"**{j}**")
                for high in range(2, spec.ranks + 1):
                    row_cols = st.columns(max(1, spec.ranks - 1), gap="small")
                    for low in range(1, spec.ranks):
                        if low >= high:
                            row_cols[low - 1].write("")
                            continue
                        idx = pair_map.get((low, high))
                        is_legal = idx in legal if idx is not None else False
                        btn_label = f"{high}-{low}"
                        if row_cols[low - 1].button(btn_label, key=f"btn_tp_{high}_{low}", disabled=not is_legal, use_container_width=True):
                            obs = env.step(idx)
                            st.session_state.obs = obs
                            st.rerun()

            st.markdown("<div style='height: 15px'></div>", unsafe_allow_html=True)
            can_call = CALL in legal
            if st.button("CALL", key="btn_call", type="primary", disabled=not can_call, use_container_width=True):
                obs = env.step(CALL)
                st.session_state.obs = obs
                st.rerun()
