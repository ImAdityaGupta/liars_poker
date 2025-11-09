from __future__ import annotations

from liars_poker.core import GameSpec, possible_starting_hands
from liars_poker.policies.tabular import Policy, TabularPolicy
from liars_poker.infoset import InfoSet
from liars_poker.env import resolve_call_winner, Rules

from typing import Dict, Tuple


def adjustment_factor(spec: GameSpec, my_hand, opp_hand):
    """
    Weight for opponent hand given our hand.

    Accounts for (a) actual multiplicity of each multiset hand under the deck
    and (b) the fact that our cards are removed from the deck.
    """

    from collections import Counter
    from math import comb
    from liars_poker.core import generate_deck

    deck_counts = Counter(generate_deck(spec))

    # Remove our hand from availability; if any card goes negative there is no mass.
    for card in my_hand:
        if deck_counts[card] <= 0:
            return 0.0
        deck_counts[card] -= 1

    opp_counts = Counter(opp_hand)
    weight = 1
    for card, need in opp_counts.items():
        available = deck_counts.get(card, 0)
        if available < need:
            return 0.0
        weight *= comb(available, need)
        deck_counts[card] -= need

    return float(weight)

class BestResponseComputer:
    def __init__(self, spec):
        self.spec = spec
        self.rules = Rules(spec)
        self.hands = tuple(possible_starting_hands(self.spec))
        self._adjustment_lookup: Dict[Tuple[int, ...], Dict[Tuple[int, ...], float]] = {
            my_hand: {
                opp_hand: adjustment_factor(self.spec, my_hand, opp_hand)
                for opp_hand in self.hands
            }
            for my_hand in self.hands
        }

        # These two are for calculating value of each infoset (from our PoV)
        # 1 corresponds to us guaranteed victory, 0 to us guaranteed loss.
        self.prob_vectors = {}
        self.state_card_values = {}


        # This is for eventually returning a tabular policy.
        self.probs = {}

    def _lookup_adjustment(self, our_hand: Tuple[int, ...], opp_hand: Tuple[int, ...]) -> float:
        table = self._adjustment_lookup.setdefault(our_hand, {})
        if opp_hand not in table:
            table[opp_hand] = adjustment_factor(self.spec, our_hand, opp_hand)
        return table[opp_hand]

    def _weighted_probabilities(
        self, history: Tuple[int, ...], our_hand: Tuple[int, ...]
    ) -> Dict[Tuple[int, ...], float]:
        adjusted: Dict[Tuple[int, ...], float] = {}
        for opp_hand, base_prob in self.prob_vectors.get(history, {}).items():
            weight = self._lookup_adjustment(our_hand, opp_hand)
            if weight <= 0.0:
                continue
            adjusted[opp_hand] = base_prob * weight
        return adjusted

    def _weighted_mass(self, history: Tuple[int, ...], our_hand: Tuple[int, ...]) -> float:
        return sum(self._weighted_probabilities(history, our_hand).values())


    def _record_history_prob(self, history: Tuple[int, ...], opp_hand: Tuple[int, ...], prob: float) -> None:
        bucket = self.prob_vectors.setdefault(history, {})
        bucket[opp_hand] = prob

    def _percolate_from_my_turn(
        self,
        infostate: InfoSet,
        opp_hand: Tuple[int, ...],
        opp_policy: Policy,
        init_prob: float = 1.0,
    ):
        for our_action in self.rules.legal_actions_for(infostate):
            if our_action == -1:
                continue

            half_infostate = InfoSet(1 - infostate.pid, opp_hand, infostate.history + (our_action,))
            response_vector = opp_policy.prob_dist_at_infoset(half_infostate)

            for opp_action, prob in response_vector.items():
                new_history = half_infostate.history + (opp_action,)
                self._record_history_prob(new_history, opp_hand, init_prob * prob)

                if opp_action != -1:
                    next_infostate = InfoSet(infostate.pid, infostate.hand, new_history)
                    self._percolate_from_my_turn(next_infostate, opp_hand, opp_policy, init_prob * prob)

        return self.prob_vectors

    def _percolate_opponent_openings(self, opp_policy: Policy, init_prob: float = 1.0) -> None:
        for opp_hand in self.hands:
            opp_root = InfoSet(1, opp_hand, ())
            response_vector = opp_policy.prob_dist_at_infoset(opp_root)
            for opp_action, prob in response_vector.items():
                history = (opp_action,)
                self._record_history_prob(history, opp_hand, init_prob * prob)
                if opp_action != -1:
                    my_root = InfoSet(1, (), history)
                    self._percolate_from_my_turn(my_root, opp_hand, opp_policy, init_prob * prob)

    def percolate_all_hands(self, opp_policy: Policy, init_prob=1):
        root_infostate = InfoSet(0, (), ())
        self.prob_vectors[()] = {hand: 1.0 for hand in self.hands}

        for opp_hand in self.hands:
            self._percolate_from_my_turn(root_infostate, opp_hand, opp_policy, init_prob)

        self._percolate_opponent_openings(opp_policy, init_prob)

        return self.prob_vectors

    def value(self, infostate: InfoSet, our_hand: Tuple):
        # print(infostate.history)
        dummy_hand = ()

        # Terminal case
        if len(infostate.history) > 0 and infostate.history[-1] == -1:
            temp_value = 0
            denominator = 0
            weighted = self._weighted_probabilities(infostate.history, our_hand)
            for opp_hand, prob in weighted.items():
                our_position = len(infostate.history) % 2
                their_position = 1 - our_position

                p1_hand = our_hand if our_position == 0 else opp_hand
                p2_hand = our_hand if our_position == 1 else opp_hand
                winner = resolve_call_winner(self.spec, infostate.history, p1_hand, p2_hand)
                winner = 0 if winner == 'P1' else 1

                reward = 1 if winner == our_position else 0

                temp_value += reward * prob
                denominator += prob

            temp_value = 0 if denominator == 0 else temp_value / denominator
            if infostate.history not in self.state_card_values:
                self.state_card_values[infostate.history] = dict({our_hand: temp_value})
            else:
                self.state_card_values[infostate.history][our_hand] = temp_value
            # print(temp_value, "\n")
            return temp_value

        # Recursive case
        action_values = []
        for our_action in self.rules.legal_actions_for(infostate):
            half_value = 0

            if our_action == -1:
                denominator = 0
                weighted = self._weighted_probabilities(infostate.history, our_hand)
                for opp_hand, prob in weighted.items():
                    our_position = len(infostate.history) % 2
                    their_position = 1 - our_position

                    p1_hand = our_hand if our_position == 0 else opp_hand
                    p2_hand = our_hand if our_position == 1 else opp_hand
                    winner = resolve_call_winner(self.spec, infostate.history + (our_action,), p1_hand, p2_hand)
                    winner = 0 if winner == 'P1' else 1

                    reward = 1 if winner == our_position else 0
                    half_value += reward * prob
                    denominator += prob

                half_value = 0 if denominator == 0 else half_value / denominator

            else:
                half_infostate = InfoSet(1 - infostate.pid, dummy_hand, infostate.history + (our_action,))
                for opp_action in self.rules.legal_actions_for(half_infostate):
                    new_infostate = InfoSet(infostate.pid, infostate.hand, half_infostate.history + (opp_action,))
                    numerator = self._weighted_mass(new_infostate.history, our_hand)
                    denominator = self._weighted_mass(infostate.history, our_hand)

                    prob_new_infostate = 0 if denominator == 0 else numerator / denominator

                    half_value += prob_new_infostate * self.value(new_infostate, our_hand)

            action_values.append((our_action, half_value))

        max_value = max(t[1] for t in action_values)
        if infostate.history not in  self.state_card_values:
            self.state_card_values[infostate.history] = dict({our_hand: max_value})
        else:
            self.state_card_values[infostate.history][our_hand] = max_value
        # print(action_values, max_value, "\n")
        return max_value

    def value_with_best_action(self, infostate: InfoSet, our_hand: Tuple):
        # print(infostate.history)
        dummy_hand = ()

        # Terminal case
        if len(infostate.history) > 0 and infostate.history[-1] == -1:
            temp_value = 0
            denominator = 0
            weighted = self._weighted_probabilities(infostate.history, our_hand)
            for opp_hand, prob in weighted.items():
                our_position = len(infostate.history) % 2
                their_position = 1 - our_position

                p1_hand = our_hand if our_position == 0 else opp_hand
                p2_hand = our_hand if our_position == 1 else opp_hand
                winner = resolve_call_winner(self.spec, infostate.history, p1_hand, p2_hand)
                winner = 0 if winner == 'P1' else 1

                reward = 1 if winner == our_position else 0

                temp_value += reward * prob
                denominator += prob

            temp_value = 0 if denominator == 0 else temp_value / denominator
            if infostate.history not in self.state_card_values:
                self.state_card_values[infostate.history] = dict({our_hand: temp_value})
            else:
                self.state_card_values[infostate.history][our_hand] = temp_value
            # print(temp_value, "\n")
            return temp_value

        # Recursive case
        action_values = []
        for our_action in self.rules.legal_actions_for(infostate):
            half_value = 0

            if our_action == -1:
                denominator = 0
                weighted = self._weighted_probabilities(infostate.history, our_hand)
                for opp_hand, prob in weighted.items():
                    our_position = len(infostate.history) % 2
                    their_position = 1 - our_position

                    p1_hand = our_hand if our_position == 0 else opp_hand
                    p2_hand = our_hand if our_position == 1 else opp_hand
                    winner = resolve_call_winner(self.spec, infostate.history + (our_action,), p1_hand, p2_hand)
                    winner = 0 if winner == 'P1' else 1

                    reward = 1 if winner == our_position else 0
                    half_value += reward * prob
                    denominator += prob

                half_value = 0 if denominator == 0 else half_value / denominator

            else:
                half_infostate = InfoSet(1 - infostate.pid, dummy_hand, infostate.history + (our_action,))
                for opp_action in self.rules.legal_actions_for(half_infostate):
                    new_infostate = InfoSet(infostate.pid, infostate.hand, half_infostate.history + (opp_action,))
                    numerator = self._weighted_mass(new_infostate.history, our_hand)
                    denominator = self._weighted_mass(infostate.history, our_hand)


                    prob_new_infostate = 0 if denominator == 0 else numerator / denominator

                    half_value += prob_new_infostate * self.value_with_best_action(new_infostate, our_hand)

            action_values.append((our_action, half_value))

        max_value = max(t[1] for t in action_values)
        best_action = max(action_values, key=lambda x: x[1])[0]

        if infostate.history not in  self.state_card_values:
            self.state_card_values[infostate.history] = dict({our_hand: max_value})
        else:
            self.state_card_values[infostate.history][our_hand] = max_value
        # print(action_values, max_value, "\n")
        
        self.probs[infostate] = dict({best_action: 1.0}) # can maybe change to pick all best actions with equal probability, in case of ties. fine for now.


        return max_value

    



def best_response_exact(spec: GameSpec, policy: Policy, debug=False) -> Tuple[TabularPolicy, BestResponseComputer]:
    br = BestResponseComputer(spec)
    if debug:
        print('Percolating started.')
    br.percolate_all_hands(policy)
    if debug:
        print('Percolating done.')
    for hand in br.hands:
        if debug:
            print(hand)
        br.value_with_best_action(InfoSet(0, hand, ()), hand)

    opening_actions = br.rules.legal_actions_from_last(None)
    for hand in br.hands:
        for opp_action in opening_actions:
            history = (opp_action,)
            br.value_with_best_action(InfoSet(1, hand, history), hand)
    if debug:
        print('best response calculated.')

    to_return = TabularPolicy()
    to_return.probs = br.probs

    return to_return, br
