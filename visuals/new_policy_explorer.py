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
from liars_poker.core import card_display, generate_deck
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL, InfoSet
# UPDATED: Use the new serialization logic
from liars_poker.serialization import load_policy 

st.set_page_config(page_title="Liar's Poker Explorer", layout="wide")

# --- CSS HACK FOR FULL WIDTH ---
st.markdown("""
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 1rem;}
    </style>
""", unsafe_allow_html=True)

st.title("Liar's Poker Policy Explorer")

# --- SESSION STATE SETUP ---
if "policy_dir" not in st.session_state:
    st.session_state["policy_dir"] = ""
    st.session_state["policy_bundle"] = None
    st.session_state["history_flags"] = {}
    st.session_state["call_flag"] = False

# !!! UPDATED: REMOVED `st.columns` FROM ROOT LAYOUT !!!
# We are no longer splitting the main app horizontally into main/controls.
# The controls are moving to the native sidebar.

# --- SIDEBAR (Controls & History) ---
# !!! UPDATED: USE `st.sidebar` !!!
# Everything that was in `control_panel` is now in `st.sidebar`.
# Streamlit's native sidebar collapses automatically on mobile/portrait.

with st.sidebar:
    st.header("Controls")
    
    default_dir = os.path.join(ROOT, "artifacts", "policy_latest")
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

    # !!! UPDATED: Changed `st.stop()` to conditional content !!!
    # If no policy is loaded, just show a prompt in the main body, not `st.stop()`
    # which can prevent the sidebar from rendering sometimes.

    if not bundle:
        with st.sidebar:
            st.info("Load a policy to begin.")
        # Display in main body as well
        st.info("Please load a Liar's Poker policy from the sidebar.")
        st.stop()

    policy, spec = bundle
    rules = rules_for_spec(spec)
    
    st.divider()
    st.markdown(f"**Ranks:** {spec.ranks} <br> **Hand Size:** {spec.hand_size}", unsafe_allow_html=True)

    # --- HISTORY BUILDER (in sidebar) ---
    st.subheader("History")
    history_flags = st.session_state.get("history_flags", {})
    updated_flags = {}
    
    for idx, (kind, value) in enumerate(rules.claims):
        suffix = "H" if kind == "RankHigh" else "P"
        label = f"{value}{suffix}"
        key = f"history-{idx}"
        current = history_flags.get(idx, False)
        updated_flags[idx] = st.checkbox(label, value=current, key=key)
        
    st.session_state["history_flags"] = updated_flags
    st.session_state["call_flag"] = st.checkbox("CALL", value=st.session_state.get("call_flag", False), key="history-call")

    # !!! UPDATED: Add Layout Mode Toggle !!!
    st.divider()
    st.subheader("Chart Settings")
    layout_mode = st.radio(
        "Layout Mode",
        options=["Side-by-Side (Landscape)", "Stacked (Portrait)"],
        index=0, # Default to landscape view
        help="Choose 'Stacked' for better visibility on narrow screens."
    )

# --- MAIN LOGIC (Now in Main Body) ---

# 1. Reconstruct History
selected_indices = sorted(idx for idx, flag in st.session_state["history_flags"].items() if flag)
history_list = list(selected_indices)
if st.session_state["call_flag"]:
    history_list.append(CALL)
history_tuple = tuple(history_list)

pid = len(history_tuple) % 2
player_name = f"P{pid + 1}"

st.subheader(f"Analysis: {player_name}")
st.caption(f"Current History Trace: {history_tuple}")

# 2. Hands and Labels
cards = generate_deck(spec)
hand_combinations = sorted({tuple(sorted(h)) for h in itertools.combinations(cards, spec.hand_size)})

def format_hand_str(hand):
    return "-".join(card_display(c, spec) for c in hand)

hand_labels = [format_hand_str(h) for h in hand_combinations]
claim_labels = [f"{k}:{v}" for (k, v) in rules.claims]
all_action_labels = claim_labels + ["CALL"]
idx_to_label = {i: l for i, l in enumerate(claim_labels)}
idx_to_label[CALL] = "CALL"

# 3. HELPER: Calculate Actor's Range (Bayesian Update)
def get_actor_range(policy, spec, pid, history, hands):
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

range_weights = get_actor_range(policy, spec, pid, history_tuple, hand_combinations)

# 4. Build DataFrames
strategy_records = []
range_records = []

for hand_tuple, hand_str in zip(hand_combinations, hand_labels):
    infoset = InfoSet(pid=pid, hand=hand_tuple, history=history_tuple)
    try:
        dist = policy.prob_dist_at_infoset(infoset)
    except:
        dist = {}

    for action_idx in range(len(rules.claims)):
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

# --- RESPONSIVE RENDERING LOGIC !!! ---

# Calculate height dynamically
chart_height = max(400, len(hand_labels) * 30)

# CHART 1: RANGE (Posterior)
chart_range = (
    alt.Chart(df_range)
    .mark_rect()
    .encode(
        x=alt.X("Type:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y("Hand:N", sort=hand_labels, title="Hand"),
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="magma", domain=[0, 1]), title="Belief"),
        tooltip=["Hand", alt.Tooltip("Probability:Q", format=".1%")]
    )
    .properties(height=chart_height) 
)

# CHART 2: STRATEGY (Heatmap)
# !!! UPDATED: Condition on Y-Axis Labels !!!

is_stacked_mode = "Stacked" in layout_mode

# Y-axis config changes based on layout
y_axis_config = None # None means show default labels
if not is_stacked_mode:
    # Hide labels for Side-by-Side view (Shared Y-alignment aesthetic)
    y_axis_config = alt.Axis(labels=False)

chart_strategy = (
    alt.Chart(df_strategy)
    .mark_rect()
    .encode(
        x=alt.X("Action:N", sort=all_action_labels, title="Action"),
        # !!! Use the conditional config here !!!
        y=alt.Y("Hand:N", sort=hand_labels, title=None, axis=y_axis_config), 
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]), title="Prob"),
        tooltip=["Hand", "Action", alt.Tooltip("Probability:Q", format=".1%")]
    )
    .properties(height=chart_height)
)

# Render based on layout choice
if not is_stacked_mode:
    # LANDSCAPE VIEW (Side-by-Side, using columns)
    col_range, col_strategy = st.columns([1, 6], gap="small")
    
    with col_range:
        st.altair_chart(chart_range, use_container_width=True)
        
    with col_strategy:
        st.altair_chart(chart_strategy, use_container_width=True)

else:
    # PORTRAIT VIEW (Stacked vertically, no columns, Strategy labels enabled)
    # This fixes the 'squished' look from image_2.png
    st.subheader("Actor's Range (Belief)")
    st.altair_chart(chart_range, use_container_width=True)
    
    st.divider()
    
    st.subheader("Actor's Strategy Heatmap")
    st.altair_chart(chart_strategy, use_container_width=True)