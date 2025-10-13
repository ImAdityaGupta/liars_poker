from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from zoneinfo import ZoneInfo

from .core import ARTIFACTS_ROOT, GameSpec, build_deck, env_hash
from .core import decode_card
from .env import Env, rules_for_spec
from .logging import StrategyManifest, load_json, save_json, write_strategy_manifest
from .policy import (
    CommitOnceMixture,
    PerDecisionMixture,
    Policy,
    RandomPolicy,
    policy_from_json,
)


def _policy_kind(policy: Policy) -> str:
    if isinstance(policy, RandomPolicy):
        return "policy/random"
    if isinstance(policy, PerDecisionMixture):
        return "policy/mix_per_decision"
    if isinstance(policy, CommitOnceMixture):
        return "policy/mix_commit_once"
    # Fallback to class name bucket
    return f"policy/{policy.__class__.__name__.lower()}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class Run:
    spec: GameSpec
    run_dir: str
    env_hash: str
    policies_dir: str
    manifests_dir: str
    current_policy_id: Optional[str] = None

    def __post_init__(self) -> None:
        _ensure_dir(self.policies_dir)
        _ensure_dir(self.manifests_dir)
        self._avg_counter = 0
        self._br_counter = 0
        self._hydrate_from_disk()

    def _hydrate_from_disk(self) -> None:
        try:
            names = sorted(f for f in os.listdir(self.manifests_dir) if f.endswith(".json"))
        except FileNotFoundError:
            names = []
        latest_avg = None
        for name in names:
            stem = os.path.splitext(name)[0]
            if stem.startswith("A"):
                try:
                    idx = int(stem[1:])
                except ValueError:
                    continue
                self._avg_counter = max(self._avg_counter, idx + 1)
                latest_avg = stem if latest_avg is None or idx >= int(latest_avg[1:]) else latest_avg
            if stem.startswith("B"):
                try:
                    idx = int(stem[1:])
                except ValueError:
                    continue
                self._br_counter = max(self._br_counter, idx + 1)
        if latest_avg is not None:
            self.current_policy_id = latest_avg

    def _next_id(self, role: str) -> str:
        if role == "average":
            policy_id = f"A{self._avg_counter}"
            self._avg_counter += 1
            return policy_id
        if role == "best_response":
            policy_id = f"B{self._br_counter}"
            self._br_counter += 1
            return policy_id
        raise ValueError(f"Unsupported role: {role}")

    def log_policy(
        self,
        policy: Policy,
        *,
        role: str = "average",
        parents: Optional[List[Dict]] = None,
        mixing: Optional[Dict] = None,
        notes: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        policy_id = self._next_id(role)
        policy_rel_path = f"policies/{policy_id}.json"
        policy_path = os.path.join(self.run_dir, policy_rel_path)
        save_json(policy_path, policy.to_json())

        manifest = StrategyManifest(
            id=policy_id,
            role=role,
            kind=_policy_kind(policy),
            env_hash=self.env_hash,
            parents=list(parents or []),
            mixing=mixing,
            seeds={"seed": seed} if seed is not None else {},
            train={},
            artifacts={"policy": policy_rel_path},
            code_sha="unknown",
            notes=notes,
        )
        manifest_path = os.path.join(self.manifests_dir, f"{policy_id}.json")
        write_strategy_manifest(manifest_path, manifest)

        if role == "average":
            self.current_policy_id = policy_id
        return policy_id

    def current_policy(self) -> Policy:
        if not self.current_policy_id:
            raise RuntimeError("No average policy has been logged yet")
        policy_path = os.path.join(self.policies_dir, f"{self.current_policy_id}.json")
        data = load_json(policy_path)
        policy = policy_from_json(data)
        policy.bind_rules(rules_for_spec(self.spec))
        return policy


def start_run(spec: GameSpec, save_root: str = ARTIFACTS_ROOT, seed: int = 0) -> Run:
    timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}_{seed}"
    run_root = os.path.join(os.path.abspath(save_root), "runs", run_id)
    policies_dir = os.path.join(run_root, "policies")
    manifests_dir = os.path.join(run_root, "manifests")
    _ensure_dir(policies_dir)
    _ensure_dir(manifests_dir)

    env_h = env_hash(spec)
    env_spec_payload = {
        "spec": json.loads(spec.to_json()),
        "env_hash": env_h,
        "seed": seed,
    }
    save_json(os.path.join(run_root, "env_spec.json"), env_spec_payload)
    return Run(spec=spec, run_dir=run_root, env_hash=env_h, policies_dir=policies_dir, manifests_dir=manifests_dir)


def build_best_response(policy: Policy, rl_params: Dict) -> Policy:
    _ = (policy, rl_params)
    return RandomPolicy()


def mix_policies(pi: Policy, beta: Policy, mix_params: Dict) -> Policy:
    impl = mix_params.get("impl", "per_decision")
    w = float(mix_params.get("w", 0.5))
    if not 0.0 <= w <= 1.0:
        raise ValueError("Mix weight w must be in [0, 1]")
    if impl == "per_decision":
        return PerDecisionMixture(pi, beta, w)
    if impl == "commit_once":
        rng = mix_params.get("rng")

        def _flatten(policy: Policy) -> Tuple[List[Policy], List[float]]:
            if isinstance(policy, CommitOnceMixture):
                return list(policy.policies), list(policy.weights)
            return [policy], [1.0]

        left_policies, left_weights = _flatten(pi)
        right_policies, right_weights = _flatten(beta)

        combined_policies: List[Policy] = []
        combined_weights: List[float] = []

        scale_left = 1.0 - w
        scale_right = w

        for policy, weight in zip(left_policies, left_weights):
            scaled = scale_left * weight
            if scaled > 0.0:
                combined_policies.append(policy)
                combined_weights.append(scaled)

        for policy, weight in zip(right_policies, right_weights):
            scaled = scale_right * weight
            if scaled > 0.0:
                combined_policies.append(policy)
                combined_weights.append(scaled)

        return CommitOnceMixture(combined_policies, combined_weights, rng=rng)
    raise ValueError(f"Unknown mix impl: {impl}")


def _load_spec(run_dir: str) -> GameSpec:
    env_spec_path = os.path.join(run_dir, "env_spec.json")
    env_payload = load_json(env_spec_path)
    spec_payload = env_payload.get("spec")
    if spec_payload is None:
        raise ValueError(f"env_spec.json missing 'spec' section in {run_dir!r}")
    return GameSpec(
        ranks=spec_payload["ranks"],
        suits=spec_payload["suits"],
        hand_size=spec_payload["hand_size"],
        starter=spec_payload["starter"],
        claim_kinds=tuple(spec_payload.get("claim_kinds", ("RankHigh", "Pair"))),
    )


def load_policy(run_dir: str, policy_id: str) -> Policy:
    policy_path = os.path.join(run_dir, "policies", f"{policy_id}.json")
    data = load_json(policy_path)
    policy = policy_from_json(data)
    spec = _load_spec(run_dir)
    policy.bind_rules(rules_for_spec(spec))
    return policy


def play_vs_bot(
    spec: GameSpec,
    policy: Policy,
    my_cards: List[int] | str,
    bot_cards: List[int] | str,
    start: str,
) -> None:
    rng = random.Random()
    env = Env(spec)
    policy.bind_rules(env.rules)
    hands = _prepare_hands(spec, rng, my_cards, bot_cards)
    if start not in {"me", "bot", "random"}:
        raise ValueError("start must be one of 'me', 'bot', or 'random'")
    starter_override = {
        "me": "P1",
        "bot": "P2",
        "random": "random",
    }[start]
    obs = env.reset(seed=rng.randint(0, 1_000_000), hands=hands, starter=starter_override)
    policy.begin_episode(rng)
    human_view = env.observation_for("P1")
    print("Your hand:", human_view["hand"])

    while not obs["terminal"]:
        player = obs["to_play"]
        if player == "P1":
            print("To play: You")
            last = "None" if obs["last_claim_idx"] is None else env.render_action(obs["last_claim_idx"])
            print("Last claim:", last)
            print("Legal:", [env.render_action(a) for a in obs["legal_actions"]])
            move = input("Your action: ").strip()
            try:
                action = env.parse_action(move, obs["legal_actions"])
            except Exception as exc:
                print("Error:", exc)
                continue
        else:
            print("To play: Bot")
            action = policy.sample(obs["infoset_key"], rng)
            print("Bot plays:", env.render_action(action))
        obs = env.step(action)

    print("Winner:", obs["winner"])
    # Print both players' hands at the end of the game
    def _render_hand(hand: Tuple[int, ...]) -> List[str]:
        # represent suits as letters A, B, C, ... for readability
        S = spec.suits
        def _suit_letter(s: int) -> str:
            return chr(ord("A") + s) if 0 <= s < 26 else str(s)

        rendered: List[str] = []
        for c in hand:
            r, s = decode_card(c, S)
            rendered.append(f"{r}{_suit_letter(s)}")
        return rendered

    # obs might be terminal; get final observations for each player
    p1_obs = env.observation_for("P1")
    p2_obs = env.observation_for("P2")
    print("P1 hand:", _render_hand(tuple(p1_obs["hand"])))
    print("P2 hand:", _render_hand(tuple(p2_obs["hand"])))


def _prepare_hands(
    spec: GameSpec,
    rng: random.Random,
    my_cards: List[int] | str,
    bot_cards: List[int] | str,
) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    if my_cards == "random" and bot_cards == "random":
        return None

    deck = list(build_deck(spec.ranks, spec.suits))
    k = spec.hand_size

    def _consume(selection: List[int] | str) -> Tuple[int, ...] | None:
        if selection == "random":
            return None
        hand = tuple(sorted(int(c) for c in selection))
        if len(hand) != k:
            raise ValueError(f"Hand must contain exactly {k} cards")
        for card in hand:
            if card not in deck:
                raise ValueError(f"Card {card} not available in deck")
            deck.remove(card)
        return hand

    p1_hand = _consume(my_cards)
    p2_hand = _consume(bot_cards)

    if p1_hand is None:
        p1_hand = tuple(sorted(rng.sample(deck, k)))
        for card in p1_hand:
            deck.remove(card)
    if p2_hand is None:
        p2_hand = tuple(sorted(rng.sample(deck, k)))
    return p1_hand, p2_hand
