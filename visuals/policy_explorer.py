from __future__ import annotations

import itertools
import os
import sys
from typing import Dict, List, Tuple

import altair as alt
import pandas as pd
import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from liars_poker.core import card_display, generate_deck
from liars_poker.env import rules_for_spec
from liars_poker.infoset import CALL, InfoSet
from liars_poker.policies.base import Policy


st.set_page_config(page_title="Policy Explorer", layout="wide")
st.title("Policy Explorer")

if "policy_dir" not in st.session_state:
    st.session_state["policy_dir"] = ""
    st.session_state["policy_bundle"] = None
    st.session_state["history_flags"] = {}
    st.session_state["call_flag"] = False

left_col, right_col = st.columns([3, 1])

with right_col:
    st.header("Controls")
    policy_dir_input = st.text_input("Policy directory", value=st.session_state["policy_dir"])
    load_clicked = st.button("Load policy")

    if load_clicked and policy_dir_input:
        try:
            policy, spec = Policy.load_policy(policy_dir_input)
            st.session_state["policy_dir"] = policy_dir_input
            st.session_state["policy_bundle"] = (policy, spec)
            st.session_state["history_flags"] = {}
            st.session_state["call_flag"] = False
            for key in list(st.session_state.keys()):
                if key.startswith("history-") or key == "history-call":
                    st.session_state.pop(key)
            st.success("Policy loaded.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to load policy: {exc}")

    bundle = st.session_state.get("policy_bundle")

    if not bundle:
        st.info("Load a policy directory to begin.")
        st.stop()

    policy, spec = bundle
    rules = rules_for_spec(spec)
    st.markdown(
        f"**Spec:** ranks={spec.ranks}, suits={spec.suits}, "
        f"hand_size={spec.hand_size}, claim_kinds={spec.claim_kinds}"
    )

    st.subheader("History")
    history_flags: Dict[int, bool] = st.session_state.get("history_flags", {})
    updated_flags: Dict[int, bool] = {}
    for idx, (kind, value) in enumerate(rules.claims):
        suffix = "H" if kind == "RankHigh" else kind[0].upper()
        label = f"{value}{suffix}"
        current = history_flags.get(idx, False)
        updated_flags[idx] = st.checkbox(label, value=current, key=f"history-{idx}")
    st.session_state["history_flags"] = updated_flags
    st.session_state["call_flag"] = st.checkbox("CALL", value=st.session_state.get("call_flag", False), key="history-call")

selected_indices = sorted(idx for idx, flag in st.session_state["history_flags"].items() if flag)
history: List[int] = list(selected_indices)
if st.session_state["call_flag"]:
    history.append(CALL)
history_tuple = tuple(history)
pid = len(history_tuple) % 2

cards = generate_deck(spec)
hand_size = spec.hand_size
hand_rows = sorted({tuple(sorted(hand)) for hand in itertools.combinations(cards, hand_size)})


def format_hand(hand: Tuple[int, ...]) -> str:
    return "-".join(card_display(card, spec) for card in hand)


hand_labels = {hand: format_hand(hand) for hand in hand_rows}
row_sort_order = [hand_labels[hand] for hand in hand_rows]

claim_labels = [f"{kind}:{value}" for (kind, value) in rules.claims]
action_indices = list(range(len(rules.claims)))
call_label = "CALL"
action_columns = claim_labels + [call_label]

records = []
for hand in hand_rows:
    hand_label = hand_labels[hand]
    infoset = InfoSet(pid=pid, hand=hand, history=history_tuple)
    dist = policy.prob_dist_at_infoset(infoset)
    for idx, label in zip(action_indices, claim_labels):
        records.append(
            {
                "Hand": hand_label,
                "Action": label,
                "Probability": dist.get(idx, 0.0),
            }
        )
    records.append(
        {
            "Hand": hand_label,
            "Action": call_label,
            "Probability": dist.get(CALL, 0.0),
        }
    )

matrix_df = pd.DataFrame(records)

heatmap = (
    alt.Chart(matrix_df)
    .mark_rect()
    .encode(
        x=alt.X("Action:N", sort=action_columns, title="Action"),
        y=alt.Y("Hand:N", sort=row_sort_order, title="Hand"),
        color=alt.Color("Probability:Q", scale=alt.Scale(scheme="blues"), title="Probability"),
        tooltip=["Hand", "Action", alt.Tooltip("Probability:Q", format=".3f")],
    )
)

with left_col:
    st.header("Policy heatmap")
    st.altair_chart(heatmap, use_container_width=True)

with right_col:
    st.markdown("---")
    st.markdown(f"**Current history:** {history_tuple}")
    st.markdown(f"**Acting player id:** {pid}")
