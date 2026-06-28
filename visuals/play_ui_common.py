from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import streamlit as st

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from liars_poker.core import card_display, card_rank
from liars_poker.env import Env, rules_for_spec
from liars_poker.infoset import CALL, NO_CLAIM, InfoSet
from liars_poker.serialization import load_policy

HUMAN_DEFAULT = "P1"

SINGLE_KIND_ALIASES = {
    "RankHigh": "H",
    "Pair": "P",
    "Trips": "T",
    "Quads": "Q",
}


@dataclass
class PlayContext:
    env: Env
    spec: Any
    rules: Any
    obs: Dict[str, Any]
    policy: Any
    human_label: str
    bot_label: str
    human_turn: bool
    game_started: bool
    game_over: bool
    history: List[int]
    legal: set[int]
    human_hand: Tuple[int, ...]
    bot_hand: Tuple[int, ...]


def ensure_state() -> None:
    defaults = {
        "env": None,
        "policy": None,
        "spec": None,
        "obs": None,
        "game_started": False,
        "game_over": False,
        "human_label": HUMAN_DEFAULT,
        "score_human": 0,
        "score_bot": 0,
        "score_recorded": False,
        "reveal_bot": False,
        "bot_rng": random.Random(time.time()),
        "last_bot_action": None,
        "last_bot_top": [],
        "last_policy_dir": os.path.join(ROOT, "artifacts", "my_dense_policy"),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def other_label(label: str) -> str:
    return "P2" if label == "P1" else "P1"


def hand_for_label(env: Env, label: str) -> Tuple[int, ...]:
    return env._p1_hand if label == "P1" else env._p2_hand


def reset_game(*, begin_episode: bool = True, start: bool | None = None) -> None:
    env = st.session_state.get("env")
    policy = st.session_state.get("policy")
    if env is None:
        return
    st.session_state.obs = env.reset()
    if begin_episode and policy is not None:
        policy.begin_episode(st.session_state.bot_rng)
    st.session_state.game_over = False
    st.session_state.score_recorded = False
    st.session_state.reveal_bot = False
    st.session_state.last_bot_action = None
    st.session_state.last_bot_top = []
    if start is not None:
        st.session_state.game_started = bool(start)


def load_bot(policy_dir: str) -> None:
    policy, spec = load_policy(policy_dir)
    st.session_state.policy = policy
    st.session_state.spec = spec
    st.session_state.env = Env(spec)
    st.session_state.last_policy_dir = policy_dir
    st.session_state.score_human = 0
    st.session_state.score_bot = 0
    st.session_state.game_started = False
    reset_game(begin_episode=False, start=False)


def render_sidebar(title: str = "Bot", *, compact: bool = False) -> None:
    with st.sidebar:
        st.markdown(f"### {title}")
        policy_dir = st.text_input(
            "Policy path",
            value=st.session_state.get("last_policy_dir", ""),
            label_visibility="visible",
        )
        if st.button("Load bot", type="primary", use_container_width=True):
            try:
                load_bot(policy_dir)
                st.success("Loaded.")
            except Exception as exc:
                st.error(f"Failed to load policy: {exc}")

        spec = st.session_state.get("spec")
        if spec is not None:
            st.divider()
            if compact:
                st.caption(
                    f"{spec.ranks} ranks | {spec.suits} suits | "
                    f"{spec.hand_size} cards | {len(rules_for_spec(spec).claims)} claims"
                )
            else:
                st.markdown(
                    f"**Ranks:** {spec.ranks}  \n"
                    f"**Suits:** {spec.suits}  \n"
                    f"**Hand:** {spec.hand_size} cards  \n"
                    f"**Claims:** {', '.join(spec.claim_kinds)}  \n"
                    f"**Claim count:** {len(rules_for_spec(spec).claims)}"
                )
            st.divider()
            in_progress = (
                st.session_state.get("game_started")
                and st.session_state.get("obs") is not None
                and not st.session_state.obs["terminal"]
            )
            st.markdown("**Seat**")
            c1, c2 = st.columns(2)
            if c1.button("You P1", disabled=in_progress, use_container_width=True):
                st.session_state.human_label = "P1"
            if c2.button("You P2", disabled=in_progress, use_container_width=True):
                st.session_state.human_label = "P2"
            starter = "You" if st.session_state.human_label == "P1" else "Bot"
            st.caption(f"Starter: {starter}")


def render_loader_bar(title: str = "Play bot") -> None:
    loaded = st.session_state.get("env") is not None
    label = "Bot loaded - policy/settings" if loaded else "Load saved policy"
    with st.expander(label, expanded=not loaded):
        st.markdown(f"**{title}**")
        c1, c2, c3 = st.columns([3.0, 0.85, 1.55], gap="small")
        policy_dir = c1.text_input(
            "Policy path",
            value=st.session_state.get("last_policy_dir", ""),
            label_visibility="collapsed",
            placeholder="Path to saved policy directory",
        )
        if c2.button("Load bot", type="primary", use_container_width=True):
            try:
                load_bot(policy_dir)
                st.rerun()
            except Exception as exc:
                st.error(f"Failed to load policy: {exc}")
        spec = st.session_state.get("spec")
        if spec is None:
            c3.caption("No policy loaded")
        else:
            c3.caption(
                f"{spec.ranks} ranks | {spec.suits} suits | "
                f"{spec.hand_size} cards | {len(rules_for_spec(spec).claims)} claims"
            )

        spec = st.session_state.get("spec")
        if spec is not None:
            in_progress = (
                st.session_state.get("game_started")
                and st.session_state.get("obs") is not None
                and not st.session_state.obs["terminal"]
            )
            s1, s2, s3 = st.columns([0.55, 0.55, 2.9], gap="small")
            if s1.button("You P1", disabled=in_progress, use_container_width=True):
                st.session_state.human_label = "P1"
                st.rerun()
            if s2.button("You P2", disabled=in_progress, use_container_width=True):
                st.session_state.human_label = "P2"
                st.rerun()
            starter = "You" if st.session_state.human_label == "P1" else "Bot"
            s3.caption(f"Starter: {starter}. Seat cannot change mid-hand.")


def current_context() -> PlayContext:
    env = st.session_state.env
    spec = st.session_state.spec
    rules = rules_for_spec(spec)
    obs = st.session_state.obs
    policy = st.session_state.policy
    human_label = st.session_state.human_label
    bot_label = other_label(human_label)
    game_started = bool(st.session_state.game_started)
    game_over = bool(obs["terminal"])
    return PlayContext(
        env=env,
        spec=spec,
        rules=rules,
        obs=obs,
        policy=policy,
        human_label=human_label,
        bot_label=bot_label,
        human_turn=(env.current_player() == human_label),
        game_started=game_started,
        game_over=game_over,
        history=list(env._history),
        legal=set(env.legal_actions()),
        human_hand=hand_for_label(env, human_label),
        bot_hand=hand_for_label(env, bot_label),
    )


def record_score_if_needed(ctx: PlayContext) -> None:
    if (
        ctx.game_started
        and ctx.game_over
        and not st.session_state.get("score_recorded", False)
    ):
        winner = ctx.obs["winner"]
        if winner == ctx.human_label:
            st.session_state.score_human += 1
        elif winner == ctx.bot_label:
            st.session_state.score_bot += 1
        st.session_state.score_recorded = True


def autoplay_bot_if_needed(ctx: PlayContext) -> None:
    if not ctx.game_started or ctx.game_over or ctx.human_turn:
        return
    infoset = ctx.env.infoset_key(ctx.bot_label)
    st.session_state.last_bot_top = []
    action = ctx.policy.sample(infoset, st.session_state.bot_rng)
    st.session_state.last_bot_action = action
    st.session_state.obs = ctx.env.step(action)
    st.rerun()


def bootstrap_app() -> PlayContext:
    ensure_state()
    if st.session_state.env is None:
        st.info("Load a saved policy to play.")
        st.stop()
    ctx = current_context()
    record_score_if_needed(ctx)
    autoplay_bot_if_needed(ctx)
    return current_context()


def start_game() -> None:
    reset_game(begin_episode=True, start=True)
    st.rerun()


def play_again() -> None:
    reset_game(begin_episode=True, start=True)
    st.rerun()


def submit_action(action: int) -> None:
    st.session_state.obs = st.session_state.env.step(action)
    st.rerun()


def reset_score() -> None:
    st.session_state.score_human = 0
    st.session_state.score_bot = 0
    st.rerun()


def toggle_reveal() -> None:
    st.session_state.reveal_bot = not bool(st.session_state.reveal_bot)
    st.rerun()


def rank_text(card: int, spec: Any) -> str:
    return card_display(card, spec)


def card_html(text: str, *, hidden: bool = False, tiny: bool = False) -> str:
    cls = "lp-card hidden" if hidden else "lp-card"
    if tiny:
        cls += " tiny"
    return f"<div class='{cls}'>{'' if hidden else text}</div>"


def hand_html(hand: Sequence[int], spec: Any, *, hidden: bool = False, tiny: bool = False) -> str:
    parts = ["<div class='lp-hand'>"]
    if hidden:
        for _ in hand:
            parts.append(card_html("?", hidden=True, tiny=tiny))
    else:
        for card in hand:
            parts.append(card_html(rank_text(card, spec), tiny=tiny))
    parts.append("</div>")
    return "".join(parts)


def action_label(rules: Any, action: int) -> str:
    if action == CALL:
        return "CALL"
    kind, value = rules.claims[action]
    if kind == "RankHigh":
        return f"{value}H"
    if kind == "Pair":
        return f"{value}P"
    if kind == "Trips":
        return f"{value}T"
    if kind == "Quads":
        return f"{value}Q"
    if kind == "TwoPair":
        low, high = rules.two_pair_ranks[value]
        return f"{high}-{low}"
    if kind == "FullHouse":
        trip, pair = rules.full_house_ranks[value]
        return f"{trip}/{pair}"
    return rules.render_action(action)


def action_long_label(rules: Any, action: int) -> str:
    return "CALL" if action == CALL else rules.render_action(action)


def top_actions(dist: Dict[int, float], rules: Any, n: int = 5) -> List[Tuple[str, float]]:
    return [
        (action_long_label(rules, action), prob)
        for action, prob in sorted(dist.items(), key=lambda item: item[1], reverse=True)[:n]
    ]


def last_claim(ctx: PlayContext) -> str:
    last = InfoSet.last_claim_idx(tuple(ctx.history))
    if last == NO_CLAIM:
        return "No claim"
    return ctx.rules.render_action(last)


def actor_for_index(index: int, human_label: str) -> Tuple[str, str]:
    label = "P1" if index % 2 == 0 else "P2"
    if label == human_label:
        return "You", "human"
    return "Bot", "bot"


def log_html(ctx: PlayContext, *, max_items: int | None = None, compact: bool = False) -> str:
    history = ctx.history[-max_items:] if max_items else ctx.history
    offset = len(ctx.history) - len(history)
    parts = ["<div class='lp-log'>"]
    if not history:
        parts.append("<div class='lp-log-row system'>New hand</div>")
    for local_i, action in enumerate(history):
        index = offset + local_i
        actor, cls = actor_for_index(index, ctx.human_label)
        if compact:
            parts.append(
                f"<div class='lp-log-row {cls}'><b>{actor}</b><span>{action_label(ctx.rules, action)}</span></div>"
            )
        else:
            parts.append(
                f"<div class='lp-log-row {cls}'><b>{actor}</b><span>{action_long_label(ctx.rules, action)}</span></div>"
            )
    parts.append("</div>")
    return "".join(parts)


def timeline_html(ctx: PlayContext) -> str:
    parts = ["<div class='lp-timeline'>"]
    visible = ctx.history[-14:]
    offset = len(ctx.history) - len(visible)
    for i, action in enumerate(visible):
        actor, cls = actor_for_index(offset + i, ctx.human_label)
        parts.append(
            f"<div class='lp-chip {cls}'><span>{actor}</span><b>{action_label(ctx.rules, action)}</b></div>"
        )
    if not ctx.history:
        parts.append("<div class='lp-chip system'>No public claims yet</div>")
    parts.append("</div>")
    return "".join(parts)


def truth_summary(ctx: PlayContext) -> Tuple[str, str, str] | None:
    last_idx = InfoSet.last_claim_idx(tuple(ctx.history))
    if last_idx == NO_CLAIM:
        return None
    kind, value = ctx.rules.claims[last_idx]
    cards = ctx.env._p1_hand + ctx.env._p2_hand
    if kind in {"RankHigh", "Pair", "Trips", "Quads"}:
        needed = {"RankHigh": 1, "Pair": 2, "Trips": 3, "Quads": 4}[kind]
        count = sum(1 for c in cards if card_rank(c, ctx.spec) == value)
        return ctx.rules.render_action(last_idx), str(count), str(needed)
    if kind == "TwoPair":
        low, high = ctx.rules.two_pair_ranks[value]
        count_low = sum(1 for c in cards if card_rank(c, ctx.spec) == low)
        count_high = sum(1 for c in cards if card_rank(c, ctx.spec) == high)
        return ctx.rules.render_action(last_idx), f"{high}:{count_high}, {low}:{count_low}", "2 each"
    if kind == "FullHouse":
        trip, pair = ctx.rules.full_house_ranks[value]
        count_trip = sum(1 for c in cards if card_rank(c, ctx.spec) == trip)
        count_pair = sum(1 for c in cards if card_rank(c, ctx.spec) == pair)
        return ctx.rules.render_action(last_idx), f"{trip}:{count_trip}, {pair}:{count_pair}", "3+2"
    return ctx.rules.render_action(last_idx), "-", "-"


def single_kind_action(ctx: PlayContext, kind: str, rank: int) -> int | None:
    for idx, (claim_kind, value) in enumerate(ctx.rules.claims):
        if claim_kind == kind and value == rank:
            return idx
    return None


def two_pair_action(ctx: PlayContext, low: int, high: int) -> int | None:
    for idx, (kind, value) in enumerate(ctx.rules.claims):
        if kind == "TwoPair" and ctx.rules.two_pair_ranks[value] == (low, high):
            return idx
    return None


def full_house_action(ctx: PlayContext, trip: int, pair: int) -> int | None:
    for idx, (kind, value) in enumerate(ctx.rules.claims):
        if kind == "FullHouse" and ctx.rules.full_house_ranks[value] == (trip, pair):
            return idx
    return None


def actions_by_kind(ctx: PlayContext) -> Dict[str, List[int]]:
    grouped: Dict[str, List[int]] = {}
    for idx, (kind, _) in enumerate(ctx.rules.claims):
        grouped.setdefault(kind, []).append(idx)
    return grouped


def legal_claims(ctx: PlayContext) -> List[int]:
    return [a for a in ctx.env.legal_actions() if a != CALL]


def controls_row(ctx: PlayContext, *, compact: bool = False) -> None:
    cols = st.columns([1, 1, 1, 1], gap="small")
    if cols[0].button("Reset score", use_container_width=True):
        reset_score()
    reveal_text = "Hide bot" if st.session_state.reveal_bot else "Reveal bot"
    if cols[1].button(reveal_text, disabled=not ctx.game_started, use_container_width=True):
        toggle_reveal()
    if ctx.game_over:
        if cols[2].button("Play again", type="primary", use_container_width=True):
            play_again()
    elif not ctx.game_started:
        if cols[2].button("Start", type="primary", use_container_width=True):
            start_game()
    else:
        cols[2].button("In hand", disabled=True, use_container_width=True)
    cols[3].button(
        f"{len(ctx.rules.claims)} claims / {len(ctx.legal)} legal",
        disabled=True,
        use_container_width=True,
    )


def result_panel(ctx: PlayContext) -> None:
    winner = ctx.obs["winner"]
    if winner == ctx.human_label:
        st.success("You win")
    else:
        st.error("Bot wins")
    truth = truth_summary(ctx)
    if truth:
        claim, actual, needed = truth
        st.caption(f"Final claim: {claim} | Actual: {actual} | Needed: {needed}")


def bot_top_html(ctx: PlayContext) -> str:
    rows = st.session_state.get("last_bot_top", [])
    if not rows:
        return "<div class='lp-probs muted'>No bot action yet</div>"
    parts = ["<div class='lp-probs'>"]
    for label, prob in rows:
        pct = int(round(100 * prob))
        parts.append(
            f"<div class='lp-prob-row'><span>{label}</span><b>{pct}%</b></div>"
        )
    parts.append("</div>")
    return "".join(parts)


def call_button(ctx: PlayContext, *, key: str, label: str = "CALL") -> None:
    if st.button(
        label,
        key=key,
        type="primary",
        disabled=CALL not in ctx.legal or not ctx.human_turn or ctx.game_over,
        use_container_width=True,
    ):
        submit_action(CALL)


def button_for_action(
    ctx: PlayContext,
    action: int | None,
    *,
    key: str,
    label: str | None = None,
    help_text: str | None = None,
    primary: bool = False,
) -> None:
    disabled = (
        action is None
        or action not in ctx.legal
        or not ctx.human_turn
        or ctx.game_over
        or not ctx.game_started
    )
    if st.button(
        label if label is not None else (action_label(ctx.rules, action) if action is not None else ""),
        key=key,
        type="primary" if primary else "secondary",
        disabled=disabled,
        help=help_text,
        use_container_width=True,
    ):
        submit_action(int(action))
