from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from collections import Counter
from math import comb
import weakref

from liars_poker.core import GameSpec, possible_starting_hands, generate_deck
from liars_poker.policies.tabular import Policy, TabularPolicy
from liars_poker.infoset import InfoSet
from liars_poker.env import resolve_call_winner, Rules


# -------------------------
# Deck-count caching (per spec instance)
# -------------------------

# Cache by id(spec) with weakref validation to avoid id reuse issues.
_DECK_COUNTS_CACHE: Dict[int, Tuple[Optional["weakref.ReferenceType[object]"], Counter]] = {}


def _cached_deck_counts(spec: GameSpec) -> Counter:
    key = id(spec)
    entry = _DECK_COUNTS_CACHE.get(key)
    if entry is not None:
        ref, counts = entry
        if ref is None:
            return counts
        obj = ref()
        if obj is spec:
            return counts

    counts = Counter(generate_deck(spec))
    try:
        ref = weakref.ref(spec)  # type: ignore[arg-type]
    except TypeError:
        ref = None
    _DECK_COUNTS_CACHE[key] = (ref, counts)
    return counts


def _hand_counter(hand: Tuple[int, ...]) -> Dict[int, int]:
    d: Dict[int, int] = {}
    for c in hand:
        d[c] = d.get(c, 0) + 1
    return d


def adjustment_factor(spec: GameSpec, my_hand: Tuple[int, ...], opp_hand: Tuple[int, ...]) -> float:
    """
    Weight for opponent hand given our hand.

    Uses cached deck_counts and count-vectors:
        weight = Π_card comb(deck_count[card] - my_count[card], opp_count[card])

    Returns 0.0 if impossible.
    """
    deck_counts = _cached_deck_counts(spec)
    my_counts = _hand_counter(my_hand)
    opp_counts = _hand_counter(opp_hand)

    weight = 1
    for card, need in opp_counts.items():
        available = deck_counts.get(card, 0) - my_counts.get(card, 0)
        if available < need:
            return 0.0
        weight *= comb(available, need)
        if weight == 0:
            return 0.0
    return float(weight)


class BestResponseComputer:
    """
    Exact best response against a fixed opponent policy using Option B:
      - Maintain and update a belief over opponent hands (Bayes updates)
      - No percolation tables; no repeated weighted_mass computations

    Compatibility / required attributes:
      - spec, rules, hands, probs
      - exploitability()
      - state_card_values (same shape as old code: history -> our_hand -> value)

    Important convention (Fix 1 philosophy):
      - We DO NOT cache terminal histories produced by OUR call action (our_action == -1),
        because those terminal histories collide across the "who started" cases.
      - We DO cache terminal histories produced by OPPONENT calling (opp_action == -1).
    """

    def __init__(self, spec: GameSpec):
        self.spec = spec
        self.rules = Rules(spec)

        # Enumerate unique multiset hands
        self.hands: Tuple[Tuple[int, ...], ...] = tuple(possible_starting_hands(self.spec))
        self._hand_to_idx: Dict[Tuple[int, ...], int] = {h: i for i, h in enumerate(self.hands)}
        self._H = len(self.hands)

        # Cached deck counts for this spec
        self._deck_counts: Counter = _cached_deck_counts(self.spec)

        # Precompute count-vectors for each hand
        self._counts_by_idx: List[Dict[int, int]] = [_hand_counter(h) for h in self.hands]

        # For averaging over our dealt hands: multiplicity of each unique multiset hand
        # weight(hand) = Π comb(deck_count[card], count_in_hand[card])
        self._deal_weight_by_idx: List[float] = []
        for hc in self._counts_by_idx:
            w = 1
            for card, need in hc.items():
                available = self._deck_counts.get(card, 0)
                if available < need:
                    w = 0
                    break
                w *= comb(available, need)
            self._deal_weight_by_idx.append(float(w))

        # Prior beliefs over opponent hands given our hand:
        # b0(opp) ∝ adjustment_factor(our_hand, opp_hand)
        self._prior_belief_by_our_idx: List[Tuple[float, ...]] = []
        for our_idx in range(self._H):
            our_counts = self._counts_by_idx[our_idx]
            row: List[float] = [0.0] * self._H
            total = 0.0
            for opp_idx in range(self._H):
                opp_counts = self._counts_by_idx[opp_idx]
                w = 1
                for card, need in opp_counts.items():
                    available = self._deck_counts.get(card, 0) - our_counts.get(card, 0)
                    if available < need:
                        w = 0
                        break
                    w *= comb(available, need)
                wf = float(w)
                row[opp_idx] = wf
                total += wf
            if total <= 0.0:
                self._prior_belief_by_our_idx.append(tuple(row))
            else:
                self._prior_belief_by_our_idx.append(tuple(x / total for x in row))

        # Outputs / caches expected by callers
        self.probs: Dict[InfoSet, Dict[int, float]] = {}  # for TabularPolicy
        self.state_card_values: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}
        self.prob_vectors: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {}  # kept for compatibility

        # Internal caches
        self._legal_actions_cache: Dict[Tuple[int, Tuple[int, ...]], Tuple[int, ...]] = {}
        self._opp_policy: Optional[Policy] = None
        self._opp_dist_cache: Dict[Tuple[int, int, Tuple[int, ...]], Dict[int, float]] = {}

    # -------------------------
    # Compatibility stub (old API had percolation; Option B doesn't need it)
    # -------------------------
    def percolate_all_hands(self, opp_policy: Policy, init_prob: float = 1.0):
        self._set_opponent_policy(opp_policy)
        return self.prob_vectors

    # -------------------------
    # Setup
    # -------------------------
    def _set_opponent_policy(self, policy: Policy) -> None:
        self._opp_policy = policy
        self._opp_dist_cache.clear()

    # -------------------------
    # Cached helpers
    # -------------------------

    def _legal_actions(self, pid: int, history: Tuple[int, ...]) -> Tuple[int, ...]:
        key = (pid, history)
        cached = self._legal_actions_cache.get(key)
        if cached is not None:
            return cached
        acts = tuple(self.rules.legal_actions_for(InfoSet(pid, (), history)))
        self._legal_actions_cache[key] = acts
        return acts

    def _opp_dist(self, opp_pid: int, opp_idx: int, history: Tuple[int, ...]) -> Dict[int, float]:
        if self._opp_policy is None:
            raise RuntimeError("Opponent policy not set. Call best_response_exact(...) first.")
        key = (opp_pid, opp_idx, history)
        cached = self._opp_dist_cache.get(key)
        if cached is not None:
            return cached
        opp_hand = self.hands[opp_idx]
        dist = self._opp_policy.prob_dist_at_infoset(InfoSet(opp_pid, opp_hand, history))
        self._opp_dist_cache[key] = dist
        return dist

    # -------------------------
    # Belief machinery (Option B)
    # -------------------------

    def _prior_belief(self, our_idx: int) -> List[float]:
        return list(self._prior_belief_by_our_idx[our_idx])

    def _update_belief_observe_action(
        self,
        belief: List[float],
        opp_pid: int,
        history_before_action: Tuple[int, ...],
        observed_action: int,
    ) -> List[float]:
        """
        Bayes update: b'(h) ∝ b(h) * pi_opp(observed_action | opp_hand, history_before_action)

        IMPORTANT (to match old semantics):
          - If the normalizer is 0, return an ALL-ZERO belief.
            This matches the old code's "denominator==0 => value 0" convention for unreachable histories.
        """
        new_belief = [0.0] * self._H
        z = 0.0
        for i, b in enumerate(belief):
            if b <= 0.0:
                continue
            p = self._opp_dist(opp_pid, i, history_before_action).get(observed_action, 0.0)
            if p <= 0.0:
                continue
            v = b * p
            new_belief[i] = v
            z += v

        if z <= 0.0:
            return [0.0] * self._H

        inv = 1.0 / z
        for i in range(self._H):
            new_belief[i] *= inv
        return new_belief

    def _belief_after_history(self, our_pid: int, our_idx: int, history: Tuple[int, ...]) -> List[float]:
        """
        Reconstruct posterior from scratch by scanning history and applying Bayes updates
        only at opponent decision points. Used by public value()/value_with_best_action().
        """
        belief = self._prior_belief(our_idx)
        n = len(history)
        if n == 0:
            return belief

        # This inference assumes we're calling this for a node where it is our turn:
        # - if history length is even, starter == our_pid
        # - if history length is odd, starter == 1-our_pid
        starting_pid = our_pid if (n % 2 == 0) else (1 - our_pid)
        opp_pid = 1 - our_pid

        for t, action in enumerate(history):
            acting_pid = starting_pid if (t % 2 == 0) else (1 - starting_pid)
            if acting_pid == opp_pid:
                belief = self._update_belief_observe_action(belief, opp_pid, history[:t], action)
            if action == -1:
                break
        return belief

    # -------------------------
    # Payoff / terminal evaluation
    # -------------------------

    def _terminal_value(
        self,
        terminal_history: Tuple[int, ...],
        our_pid: int,
        our_hand: Tuple[int, ...],
        belief: List[float],
    ) -> float:
        """
        Expected win probability given terminal history (ends with -1), our pid, our hand,
        and posterior belief over opponent hands.
        """
        v = 0.0
        for opp_idx, p_h in enumerate(belief):
            if p_h <= 0.0:
                continue
            opp_hand = self.hands[opp_idx]
            if our_pid == 0:
                p1_hand, p2_hand = our_hand, opp_hand
                win_label = "P1"
            else:
                p1_hand, p2_hand = opp_hand, our_hand
                win_label = "P2"

            winner = resolve_call_winner(self.spec, terminal_history, p1_hand, p2_hand)
            v += p_h * (1.0 if winner == win_label else 0.0)
        return v

    # -------------------------
    # Zero-belief expansion (to match old code's cache population)
    # -------------------------

    def _solve_zero_belief(self, infostate: InfoSet, record_policy: bool) -> float:
        """
        Old code still traversed and populated values even for unreachable branches
        (probability 0). To match that behavior, when belief mass is 0 we:
          - assign value 0 at this node
          - still expand all legal actions (ours and opponent) to populate downstream
            histories with 0 values
        """
        our_pid = infostate.pid
        our_hand = infostate.hand
        history = infostate.history

        # Terminal node: unreachable => value 0
        if history and history[-1] == -1:
            self.state_card_values.setdefault(history, {})[our_hand] = 0.0
            return 0.0

        legal_our = self._legal_actions(our_pid, history)

        # Record a deterministic default action (old code would pick some argmax among ties)
        if record_policy and legal_our:
            self.probs[infostate] = {legal_our[0]: 1.0}

        # Expand children to populate cache with zeros
        for our_action in legal_our:
            if our_action == -1:
                # Fix 1: do NOT cache terminal histories from OUR call
                continue

            opp_pid = 1 - our_pid
            opp_history = history + (our_action,)
            legal_opp = self._legal_actions(opp_pid, opp_history)

            for opp_action in legal_opp:
                new_history = opp_history + (opp_action,)
                if opp_action == -1:
                    # Opponent called: safe to cache terminal history; unreachable => 0
                    self.state_card_values.setdefault(new_history, {})[our_hand] = 0.0
                else:
                    child_state = InfoSet(our_pid, our_hand, new_history)
                    self._solve_zero_belief(child_state, record_policy)

        self.state_card_values.setdefault(history, {})[our_hand] = 0.0
        return 0.0

    # -------------------------
    # Core recursion (Option B)
    # -------------------------

    def _solve(
        self,
        infostate: InfoSet,
        our_idx: int,
        belief: List[float],
        record_policy: bool,
    ) -> float:
        """
        Compute value from a state where it is OUR turn (infostate.pid == our_pid),
        given our hand index and a posterior belief over opponent hands.
        """
        our_pid = infostate.pid
        our_hand = infostate.hand
        history = infostate.history

        # Terminal (call already occurred)
        if history and history[-1] == -1:
            val = self._terminal_value(history, our_pid, our_hand, belief)
            self.state_card_values.setdefault(history, {})[our_hand] = val
            return val

        # If unreachable under opponent policy updates, match old semantics:
        # value 0, but still expand to populate cache like old.
        if sum(belief) <= 0.0:
            return self._solve_zero_belief(infostate, record_policy)

        best_val = -1.0
        best_action: Optional[int] = None

        for our_action in self._legal_actions(our_pid, history):
            if our_action == -1:
                # We call now -> terminal value (DO NOT cache terminal history from our call; Fix 1)
                terminal_hist = history + (-1,)
                val = self._terminal_value(terminal_hist, our_pid, our_hand, belief)

            else:
                opp_pid = 1 - our_pid
                opp_history = history + (our_action,)

                # Mixture distribution over opponent actions under current belief.
                per_hand_dists: List[Optional[Dict[int, float]]] = [None] * self._H
                mix: Dict[int, float] = {}
                for opp_idx, b in enumerate(belief):
                    if b <= 0.0:
                        continue
                    dist = self._opp_dist(opp_pid, opp_idx, opp_history)
                    per_hand_dists[opp_idx] = dist
                    for a, p in dist.items():
                        if p <= 0.0:
                            continue
                        mix[a] = mix.get(a, 0.0) + b * p

                val = 0.0
                legal_opp = self._legal_actions(opp_pid, opp_history)

                # IMPORTANT: iterate ALL legal opponent actions (including p_a==0)
                # to populate caches for unreachable branches like the old code.
                for opp_action in legal_opp:
                    p_a = mix.get(opp_action, 0.0)

                    new_history = opp_history + (opp_action,)

                    if p_a <= 0.0:
                        # Unreachable opponent action under current belief: treat as zero-belief branch.
                        if opp_action == -1:
                            # opponent called => safe to cache terminal history as 0
                            self.state_card_values.setdefault(new_history, {})[our_hand] = 0.0
                        else:
                            child_state = InfoSet(our_pid, our_hand, new_history)
                            self._solve_zero_belief(child_state, record_policy)
                        continue

                    # Posterior after observing opp_action:
                    new_belief = [0.0] * self._H
                    inv = 1.0 / p_a
                    for opp_idx, b in enumerate(belief):
                        if b <= 0.0:
                            continue
                        dist = per_hand_dists[opp_idx]
                        if dist is None:
                            continue
                        p = dist.get(opp_action, 0.0)
                        if p <= 0.0:
                            continue
                        new_belief[opp_idx] = b * p * inv

                    if opp_action == -1:
                        # Opponent called -> terminal; safe to cache
                        child_val = self._terminal_value(new_history, our_pid, our_hand, new_belief)
                        self.state_card_values.setdefault(new_history, {})[our_hand] = child_val
                    else:
                        child_state = InfoSet(our_pid, our_hand, new_history)
                        child_val = self._solve(child_state, our_idx, new_belief, record_policy)

                    val += p_a * child_val

            if val > best_val:
                best_val = val
                best_action = our_action

        # Record and store
        self.state_card_values.setdefault(history, {})[our_hand] = best_val
        if record_policy:
            if best_action is None:
                # Shouldn't happen, but keep deterministic
                legal = self._legal_actions(our_pid, history)
                if legal:
                    best_action = legal[0]
            if best_action is not None:
                self.probs[infostate] = {best_action: 1.0}
        return best_val

    # Public API-compatible methods (signatures preserved)

    def value(self, infostate: InfoSet, our_hand: Tuple[int, ...]) -> float:
        our_idx = self._hand_to_idx[our_hand]
        belief = self._belief_after_history(infostate.pid, our_idx, infostate.history)
        return self._solve(infostate, our_idx, belief, record_policy=False)

    def value_with_best_action(self, infostate: InfoSet, our_hand: Tuple[int, ...]) -> float:
        our_idx = self._hand_to_idx[our_hand]
        belief = self._belief_after_history(infostate.pid, our_idx, infostate.history)
        return self._solve(infostate, our_idx, belief, record_policy=True)

    # -------------------------
    # Exploitability (same meaning as original)
    # -------------------------

    def exploitability(self) -> Tuple[float, float]:
        """
        Returns (p_first, p_second):
          - p_first: expected value when we are the starting player (pid=0)
          - p_second: expected value when opponent starts (we are pid=1), averaging over
                     opponent opening action and our dealt hand.
        """
        if self._opp_policy is None:
            raise RuntimeError("Opponent policy not set; exploitability() requires a computed best response.")

        # p_first: we are pid=0 at empty history
        num_first = 0.0
        den_first = 0.0
        for our_idx, our_hand in enumerate(self.hands):
            w = self._deal_weight_by_idx[our_idx]
            if w <= 0.0:
                continue
            root_hist = ()
            root_vals = self.state_card_values.get(root_hist, {})
            val = root_vals.get(our_hand)
            if val is None:
                belief = self._prior_belief(our_idx)
                val = self._solve(InfoSet(0, our_hand, ()), our_idx, belief, record_policy=False)
            num_first += w * val
            den_first += w
        p_first = 0.0 if den_first <= 0.0 else num_first / den_first

        # p_second: opponent opens (pid=0), we respond as pid=1 after observing opening action
        opening_actions = self.rules.legal_actions_from_last(None)

        num_second = 0.0
        den_second = 0.0
        for our_idx, our_hand in enumerate(self.hands):
            w_hand = self._deal_weight_by_idx[our_idx]
            if w_hand <= 0.0:
                continue

            belief0 = self._prior_belief(our_idx)

            # Mixture distribution over opponent opening actions at history=()
            mix: Dict[int, float] = {}
            per_hand_dists: List[Optional[Dict[int, float]]] = [None] * self._H
            for opp_idx, b in enumerate(belief0):
                if b <= 0.0:
                    continue
                dist = self._opp_dist(0, opp_idx, ())
                per_hand_dists[opp_idx] = dist
                for a, p in dist.items():
                    if p <= 0.0:
                        continue
                    mix[a] = mix.get(a, 0.0) + b * p

            # Expected value across opening actions
            for a in opening_actions:
                p_a = mix.get(a, 0.0)
                if p_a <= 0.0:
                    continue

                # Posterior after observing opponent opening action a
                belief1 = [0.0] * self._H
                inv = 1.0 / p_a
                for opp_idx, b in enumerate(belief0):
                    if b <= 0.0:
                        continue
                    dist = per_hand_dists[opp_idx]
                    if dist is None:
                        continue
                    pa_i = dist.get(a, 0.0)
                    if pa_i <= 0.0:
                        continue
                    belief1[opp_idx] = b * pa_i * inv

                hist = (a,)
                vals = self.state_card_values.get(hist, {})
                val = vals.get(our_hand)
                if val is None:
                    val = self._solve(InfoSet(1, our_hand, hist), our_idx, belief1, record_policy=False)

                num_second += w_hand * p_a * val
                den_second += w_hand * p_a

        p_second = 0.0 if den_second <= 0.0 else num_second / den_second
        return p_first, p_second


def best_response_exact(spec: GameSpec, policy: Policy, debug: bool = False) -> Tuple[TabularPolicy, BestResponseComputer]:
    """
    Signature preserved.

    Computes an exact best response TabularPolicy against 'policy' using Option B.
    Returns (TabularPolicy, BestResponseComputer).
    """
    br = BestResponseComputer(spec)
    br._set_opponent_policy(policy)

    if debug:
        print("Best-response solving (Option B) started.")

    # Case 1: we are pid=0 and start at empty history
    for our_hand in br.hands:
        our_idx = br._hand_to_idx[our_hand]
        belief = br._prior_belief(our_idx)
        br._solve(InfoSet(0, our_hand, ()), our_idx, belief, record_policy=True)

    # Case 2: opponent opens (pid=0), we respond as pid=1 after observing opening action
    opening_actions = br.rules.legal_actions_from_last(None)
    for our_hand in br.hands:
        our_idx = br._hand_to_idx[our_hand]
        belief0 = br._prior_belief(our_idx)
        for opp_action in opening_actions:
            belief1 = br._update_belief_observe_action(
                belief0, opp_pid=0, history_before_action=(), observed_action=opp_action
            )
            br._solve(InfoSet(1, our_hand, (opp_action,)), our_idx, belief1, record_policy=True)

    if debug:
        print("Best response calculated.")

    to_return = TabularPolicy()
    to_return.probs = br.probs
    return to_return, br
