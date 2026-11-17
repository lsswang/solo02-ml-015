"""Ordinary on-policy payoff evaluator, independent of any solver.

A self-play payoff says nothing about exploitability, so this module contains
no best-response logic.  Information sets are honored because the policy is
indexed by player and information-set name; each stored distribution is
normalized by its actual sum before use, and every information set must still
be present and complete even if play never reaches it.
"""

from game_tree import parse_game, normalize_policy


def evaluate_model(model, distributions):
    """Evaluate normalized distributions on a parsed :class:`GameModel`."""
    kind = model.kind
    values = [0.0] * model.size
    for node_i in reversed(model.order):
        if kind[node_i] == 'terminal':
            values[node_i] = model.payoff[node_i]
            continue
        if kind[node_i] == 'chance':
            weights = model.chance[node_i]
        else:
            canonical = distributions[model.infoset_keys[model.infoset[node_i]]]
            edge_pos = model.edge_pos[node_i]
            weights = [canonical[edge_pos[edge_i]]
                       for edge_i in range(len(model.children[node_i]))]
        total = 0.0
        for weight, child in zip(weights, model.children[node_i]):
            total += weight * values[child]
        values[node_i] = total
    return values[model.root]


def evaluate(game, policy):
    """Return the exact expected payoff of a supplied behavioral strategy pair.

    ``game`` is the JSON tree description; ``policy`` uses the public format
    ``policy[str(player)][infoset][str(action)]``.  Both objects are only read.
    The result is ``{'value_player0': ..., 'value_player1': ...}`` for the
    zero-sum game (player 1 receives the negative of player 0's payoff).
    """
    model = parse_game(game)
    distributions = normalize_policy(model, policy)
    value = evaluate_model(model, distributions)
    return {'value_player0': value, 'value_player1': -value}
