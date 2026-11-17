"""Brute-force oracles and reference evaluators used by tests and the demo.

The oracles use a different code path from ``solve`` / ``analyze``:

* expected payoffs are computed by a plain recursive walk over the raw JSON
  node dictionaries;
* a best response is found by enumerating *pure plans* (one action per
  information set) and evaluating each with the opponent's supplied mixed
  strategy -- no dynamic programming over information sets.

They are for small trees only.  :func:`brute_force_best_response` refuses to
materialize a pure-plan product larger than ``max_plans``.
"""

from __future__ import annotations

import itertools
import math

from game_tree import parse_game


def _normalized_weights(node, policy):
    """Per-edge weights of a chance or player node, normalized by actual sum."""
    actions = node['actions']
    if node['kind'] == 'chance':
        total = math.fsum(edge['probability'] for edge in actions)
        return [(edge['probability'] / total) for edge in actions]
    raw = policy[str(node['player'])][node['infoset']]
    total = math.fsum(float(raw[str(edge['action'])]) for edge in actions)
    return [float(raw[str(edge['action'])]) / total for edge in actions]


def reference_value(game, policy):
    """Expected player-0 payoff by a plain recursive walk over the JSON tree."""
    nodes = game['nodes']

    def visit(node_id):
        node = nodes[node_id]
        if node['kind'] == 'terminal':
            return float(node['payoff'])
        weights = _normalized_weights(node, policy)
        return math.fsum(
            weight * visit(edge['child'])
            for weight, edge in zip(weights, node['actions']))

    return visit(game['root'])


def _pure_responder_value(nodes, root, responder, choice, opponent):
    """Payoff for ``responder``: pure plan, opponent's fixed mixed strategy."""
    def visit(node_id):
        node = nodes[node_id]
        node_kind = node['kind']
        if node_kind == 'terminal':
            payoff = float(node['payoff'])
            return payoff if responder == 0 else -payoff
        if node_kind == 'chance':
            total = math.fsum(edge['probability'] for edge in node['actions'])
            return math.fsum(
                (edge['probability'] / total) * visit(edge['child'])
                for edge in node['actions'])
        if node['player'] == responder:
            action = choice[(responder, node['infoset'])]
            edge = next(edge for edge in node['actions']
                        if edge['action'] == action)
            return visit(edge['child'])
        raw = opponent[str(node['player'])][node['infoset']]
        total = math.fsum(float(raw[str(edge['action'])]) for edge in node['actions'])
        return math.fsum(
            (float(raw[str(edge['action'])]) / total) * visit(edge['child'])
            for edge in node['actions'])

    return visit(root)


def brute_force_best_response(game, policy, responder, max_plans=200000):
    """Exact best response by enumerating every pure plan of ``responder``.

    Returns ``(value, player_policy)`` where ``player_policy`` is a legal pure
    policy dictionary for ``responder`` only.  Raises ``ValueError`` when the
    plan count exceeds ``max_plans``.
    """
    model = parse_game(game)
    nodes = game['nodes']
    info_keys = [(who, name) for (who, name) in model.infoset_keys
                 if who == responder]
    actions = [model.infoset_actions[model.infoset_index[key]]
               for key in info_keys]

    total_plans = 1
    for action_list in actions:
        total_plans *= len(action_list)
    if total_plans > max_plans:
        raise ValueError(
            f'{total_plans} pure plans exceed the brute-force budget '
            f'{max_plans}; use the dynamic-programming best response')

    best_value = None
    best_choice = None
    opponent = {'0': {}, '1': {}}
    for node in nodes.values():
        if node['kind'] == 'player' and node['player'] != responder:
            opponent[str(node['player'])][node['infoset']] = dict(
                policy[str(node['player'])][node['infoset']])

    for combination in itertools.product(*actions):
        choice = dict(zip(info_keys, combination))
        value = _pure_responder_value(nodes, game['root'], responder,
                                      choice, opponent)
        if best_value is None or value > best_value:
            best_value = value
            best_choice = choice

    player_policy = {}
    for key, action in best_choice.items():
        player_policy[key[1]] = {
            str(a): (1.0 if a == action else 0.0)
            for a in model.infoset_actions[model.infoset_index[key]]
        }
    return best_value, player_policy


def pure_strategy_value(game, policy, responder):
    """Value of the pure plan contained in ``policy`` (for quality checks)."""
    model = parse_game(game)
    choice = {}
    for (who, name) in model.infoset_keys:
        if who != responder:
            continue
        raw = policy[str(who)][name]
        total = math.fsum(float(v) for v in raw.values())
        probs = {int(action): float(value) / total for action, value in raw.items()}
        best_action = max(probs, key=probs.get)
        choice[(who, name)] = best_action
    opponent = {'0': {}, '1': {}}
    for node in game['nodes'].values():
        if node['kind'] == 'player' and node['player'] != responder:
            opponent[str(node['player'])][node['infoset']] = dict(
                policy[str(node['player'])][node['infoset']])
    return _pure_responder_value(game['nodes'], game['root'], responder,
                                 choice, opponent)
