import itertools
import os
import sys
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np

# --- PATH SETUP ---
# Ensure we can import liars_poker regardless of where we run this script from
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- IMPORTS ---
from liars_poker.core import card_display, card_rank, generate_deck
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL, InfoSet
# UPDATED: Use the new serialization logic
from liars_poker.serialization import load_policy 

st.set_page_config(page_title="Liar's Poker Explorer", layout="wide")

# --- CSS HACK ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Liar's Poker Policy Explorer")

# --- LABEL HELPERS ---
def format_claim_label(kind: str, value: int, rules) -> str:
    if kind == "TwoPair":
        low, high = rules.two_pair_ranks[value]
        return f"{high}-{low}"
    if kind == "FullHouse":
        trip, pair = rules.full_house_ranks[value]
        return f"{trip}{pair}-F"
    suffix_map = {"RankHigh": "H", "Pair": "P", "Trips": "T", "Quads": "Q"}
    suffix = suffix_map.get(kind, kind[:1].upper())
    return f"{value}{suffix}"

def hand_sort_key(hand: Tuple[int, ...], spec) -> Tuple[int, ...]:
    ranks_desc = sorted((card_rank(c, spec) for c in hand), reverse=True)
    cards_desc = sorted(hand, reverse=True)
    return tuple(-r for r in ranks_desc) + tuple(-c for c in cards_desc)

# --- SESSION STATE ---
if "policy_dir" not in st.session_state:
    st.session_state["policy_dir"] = ""
    st.session_state["policy_bundle"] = None
    st.session_state["history_flags"] = {}
    st.session_state["call_flag"] = False

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Controls")
    
    default_dir = os.path.join(ROOT, "artifacts", "my_dense_policy")
    policy_dir_input = st.text_input("Policy directory", value=default_dir)
    load_clicked = st.button("Load policy", use_container_width=True)

    if load_clicked and policy_dir_input:
        try:
            policy, spec = load_policy(policy_dir_input)
            st.session_state["policy_dir"] = policy_dir_input
            st.session_state["policy_bundle"] = (policy, spec)
            st.session_state["history_flags"] = {}
            st.session_state["call_flag"] = False
            for key in list(st.session_state.keys()):
                if key.startswith("history-"):
                    st.session_state.pop(key)
            st.success("Loaded!")
        except Exception as exc:
            st.error(f"Error: {exc}")

    bundle = st.session_state.get("policy_bundle")

    if not bundle:
        with st.sidebar:
            st.info("Load a policy to begin.")
        st.info("Please load a Liar's Poker policy from the sidebar.")
        st.stop()

    policy, spec = bundle
    rules = rules_for_spec(spec)
    
    st.divider()
    st.markdown(f"**Ranks:** {spec.ranks} <br> **Suits:** {spec.suits} <br> **Hand Size:** {spec.hand_size} <br> **Valid Claims:** {spec.claim_kinds}", unsafe_allow_html=True)

    # --- HISTORY BUILDER ---
    st.subheader("History")
    history_flags = st.session_state.get("history_flags", {})
    updated_flags = {}
    
    claim_indices = list(range(len(rules.claims)))
    for idx in claim_indices:
        kind, value = rules.claims[idx]
        label = format_claim_label(kind, value, rules)
        key = f"history-{idx}"
        current = history_flags.get(idx, False)
        updated_flags[idx] = st.checkbox(label, value=current, key=key)
        
    st.session_state["history_flags"] = updated_flags
    st.session_state["call_flag"] = st.checkbox("CALL", value=st.session_state.get("call_flag", False), key="history-call")

    # --- LAYOUT MODE ---
    st.divider()
    st.subheader("Range Settings")
    switch_seats = st.checkbox("Switch Seats", value=False, key="switch-seats")

    # --- LAYOUT MODE ---
    st.divider()
    st.subheader("View Settings")
    layout_mode = st.radio(
        "Layout Mode",
        options=["Landscape (Side-by-Side)", "Portrait (Stacked)"],
        index=0,
        help="Use 'Portrait' for mobile or narrow screens."
    )

# --- MAIN LOGIC ---

# 1. Reconstruct History
selected_indices = sorted(idx for idx, flag in st.session_state["history_flags"].items() if flag)
history_list = list(selected_indices)
if st.session_state["call_flag"]:
    history_list.append(CALL)
history_tuple = tuple(history_list)
is_terminal = CALL in history_tuple

pid = len(history_tuple) % 2
player_name = f"P{pid + 1}"

st.subheader(f"Analysis: {player_name}")
st.caption(f"History: {history_tuple}")
if is_terminal:
    st.warning("History ends with CALL.")

# 2. Hands and Labels
cards = generate_deck(spec)
hand_combinations = list({tuple(sorted(h)) for h in itertools.combinations(cards, spec.hand_size)})
hand_combinations.sort(key=lambda h: hand_sort_key(h, spec))

def format_hand_str(hand):
    ordered = sorted(hand, key=lambda c: (card_rank(c, spec), c), reverse=True)
    return "-".join(card_display(c, spec) for c in ordered)

hand_labels = [format_hand_str(h) for h in hand_combinations]
idx_to_label = {
    idx: format_claim_label(kind, value, rules)
    for idx, (kind, value) in enumerate(rules.claims)
}
idx_to_label[CALL] = "CALL"
claim_indices = list(range(len(rules.claims)))
claim_labels = [idx_to_label[idx] for idx in claim_indices]
all_action_labels = claim_labels + ["CALL"]

# 3. HELPER: Range Calculation
def get_actor_range(policy, spec, pid, history, hands):
    if CALL in history:
        return {h: 0.0 for h in hands}
    weights = {h: 1.0 for h in hands}
    current_history = []
    for action in history:
        actor_to_move = len(current_history) % 2
        if actor_to_move == pid:
            for h in hands:
                iset = InfoSet(pid, h, tuple(current_history))
                try:
                    dist = policy.prob_dist_at_infoset(iset)
                    prob_a = dist.get(action, 0.0)
                    weights[h] *= prob_a
                except:
                    weights[h] = 0.0
        current_history.append(action)
    total = sum(weights.values())
    if total > 0:
        for h in weights:
            weights[h] /= total
    return weights

range_pid = 1 - pid if switch_seats else pid
range_weights = get_actor_range(policy, spec, range_pid, history_tuple, hand_combinations)

# 4. Build DataFrames
strategy_records = []
range_records = []

for hand_tuple, hand_str in zip(hand_combinations, hand_labels):
    infoset = InfoSet(pid=pid, hand=hand_tuple, history=history_tuple)
    if is_terminal:
        dist = {}
    else:
        try:
            dist = policy.prob_dist_at_infoset(infoset)
        except Exception:
            dist = {}

    for action_idx in claim_indices:
        strategy_records.append({
            "Hand": hand_str,
            "Action": idx_to_label[action_idx],
            "Probability": dist.get(action_idx, 0.0)
        })
    strategy_records.append({
        "Hand": hand_str,
        "Action": "CALL",
        "Probability": dist.get(CALL, 0.0)
    })
    
    range_records.append({
        "Hand": hand_str,
        "Type": "Belief",
        "Probability": range_weights.get(hand_tuple, 0.0)
    })

df_strategy = pd.DataFrame(strategy_records)
df_range = pd.DataFrame(range_records)

# --- UNIFIED ALTAIR VISUALIZATION ---

chart_height = max(400, len(hand_labels) * 30)
is_stacked_mode = "Portrait" in layout_mode

# 1. Base Range Chart (Left)
base_range = (
    alt.Chart(df_range)
    .mark_rect()
    .encode(
        # Hide X-axis completely since it's just a single column
        x=alt.X("Type:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("Hand:N", sort=hand_labels, title="Hand"),
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="magma", domain=[0, 1]), title="Belief"),
        tooltip=["Hand", alt.Tooltip("Probability:Q", format=".1%")]
    )
    .properties(height=chart_height, title="Range")
)

# 2. Base Strategy Chart (Right)
# Conditional Y-Axis: Only show labels if stacked (Portrait)
y_axis_config = alt.Axis(labels=True) if is_stacked_mode else alt.Axis(labels=False)

base_strategy = (
    alt.Chart(df_strategy)
    .mark_rect()
    .encode(
        x=alt.X("Action:N", sort=all_action_labels, title="Next Action"),
        y=alt.Y("Hand:N", sort=hand_labels, title=None, axis=y_axis_config),
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]), title="Policy"),
        tooltip=["Hand", "Action", alt.Tooltip("Probability:Q", format=".1%")]
    )
    .properties(height=chart_height, title="Strategy")
)

# 3. Concatenation Logic
# We join them into a single Altair object. 
# resolve_scale(color='independent') ensures Belief (Magma) and Strategy (Viridis) 
# keep separate legends.

if is_stacked_mode:
    # Portrait: Stack vertical
    final_chart = alt.vconcat(base_range, base_strategy).resolve_scale(color='independent')
else:
    # Landscape: Side-by-Side (hconcat aligns the grids perfectly)
    final_chart = alt.hconcat(base_range, base_strategy).resolve_scale(color='independent')

# 4. Render
st.altair_chart(final_chart, use_container_width=True)
