"""Validation and an index-based representation shared by analysis and solving.

The public JSON format is a rooted tree whose nodes are keyed by string IDs.
This module converts it once into integer indices so that the exact
best-response and CFR code can traverse it cheaply.  It never mutates the
caller's game object.

Representation summary
----------------------
``kind[i]``        ``'terminal'``, ``'chance'`` or ``'player'``
``children[i]``    child indices in the input edge order
``chance[i]``      normalized chance probabilities (``None`` off chance nodes)
``player[i]``      0/1 on player nodes, -1 otherwise
``infoset[i]``     infoset index on player nodes, -1 otherwise
``edge_pos[i]``    for an edge ``e`` of a player node, the position of its
                   action inside ``infoset_actions[infoset[i]]``
``edge_by_pos[i]`` inverse of ``edge_pos`` on player nodes

Actions are sorted numerically inside each information set.  Edges of a node
may appear in any order; ``edge_pos`` / ``edge_by_pos`` translate between the
two orderings.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


class GameError(ValueError):
    """The supplied game description violates the documented format."""


class PolicyError(ValueError):
    """A behavioral policy is malformed or does not cover its information sets."""


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value):
    return _is_number(value) and math.isfinite(float(value))


def _require(condition, message):
    if not condition:
        raise GameError(message)


@dataclass
class GameModel:
    ids: list
    index: dict
    root: int
    kind: list
    payoff: list
    player: list
    infoset: list
    infoset_name: list
    children: list
    chance: list
    edge_pos: list
    edge_by_pos: list
    infoset_keys: list          # sorted (player, name) tuples
    infoset_index: dict
    infoset_actions: list       # sorted integer action lists
    nodes_of_infoset: list
    player_infosets: list       # [[infoset index, ...], [...]] for players 0/1
    order: list                 # preorder traversal, parents before children

    @property
    def size(self):
        return len(self.ids)


def parse_game(game):
    """Validate ``game`` and return a :class:`GameModel`.

    The checker enforces the structural guarantees from the task: a single
    rooted tree (every node reachable exactly once, no shared child, no cycle),
    finite payoffs, strictly positive finite chance probabilities, and a
    consistent action set for every player information set.  Chance
    probabilities are normalized by their actual sum.
    """
    _require(isinstance(game, dict), 'game must be a dictionary')
    nodes = game.get('nodes')
    root_id = game.get('root')
    _require(isinstance(nodes, dict) and len(nodes) > 0,
             'game["nodes"] must be a non-empty dictionary')
    _require(isinstance(root_id, str) and root_id in nodes,
             'game["root"] must name a node')

    ids = list(nodes.keys())
    index = {node_id: i for i, node_id in enumerate(ids)}
    n = len(ids)

    kind = [None] * n
    payoff = [0.0] * n
    player = [-1] * n
    infoset_name = [None] * n
    children = [None] * n
    chance = [None] * n
    node_actions = [None] * n

    # ---- first pass: node payloads and edges -----------------------------
    for i, node_id in enumerate(ids):
        node = nodes[node_id]
        _require(isinstance(node, dict), f'node {node_id!r} must be a dictionary')
        node_kind = node.get('kind')
        _require(node_kind in ('terminal', 'chance', 'player'),
                 f'node {node_id!r} has unknown kind {node_kind!r}')
        kind[i] = node_kind

        if node_kind == 'terminal':
            value = node.get('payoff')
            _require(_finite(value), f'terminal node {node_id!r} needs a finite payoff')
            payoff[i] = float(value)
            continue

        actions = node.get('actions')
        _require(isinstance(actions, list) and len(actions) > 0,
                 f'node {node_id!r} needs a non-empty action list')
        child_ids = []
        for edge in actions:
            _require(isinstance(edge, dict), f'node {node_id!r} has a malformed edge')
            child_id = edge.get('child')
            _require(isinstance(child_id, str) and child_id in nodes,
                     f'node {node_id!r} points at unknown child {child_id!r}')
            child_ids.append(index[child_id])
        children[i] = child_ids

        if node_kind == 'chance':
            probabilities = []
            for edge in actions:
                probability = edge.get('probability')
                _require(_finite(probability) and float(probability) > 0.0,
                         f'chance node {node_id!r} needs finite positive probabilities')
                probabilities.append(float(probability))
            total = math.fsum(probabilities)
            _require(math.isfinite(total) and total > 0.0,
                     f'chance node {node_id!r} has an invalid probability sum')
            chance[i] = [p / total for p in probabilities]
        else:
            who = node.get('player')
            _require(who in (0, 1) and not isinstance(who, bool),
                     f'player node {node_id!r} must have player 0 or 1')
            player[i] = int(who)
            name = node.get('infoset')
            _require(isinstance(name, str) and name != '',
                     f'player node {node_id!r} needs a non-empty string infoset')
            infoset_name[i] = name
            action_ids = []
            for edge in actions:
                action = edge.get('action')
                _require(isinstance(action, int) and not isinstance(action, bool),
                         f'player node {node_id!r} needs integer action IDs')
                action_ids.append(action)
            _require(len(set(action_ids)) == len(action_ids),
                     f'player node {node_id!r} repeats an action ID')
            node_actions[i] = action_ids

    # ---- information sets ------------------------------------------------
    action_sets = {}
    for i in range(n):
        if kind[i] == 'player':
            key = (player[i], infoset_name[i])
            action_sets.setdefault(key, set()).update(node_actions[i])

    infoset_keys = sorted(action_sets)
    infoset_index = {key: j for j, key in enumerate(infoset_keys)}
    infoset_actions = [sorted(action_sets[key]) for key in infoset_keys]
    nodes_of_infoset = [[] for _ in infoset_keys]
    player_infosets = [[], []]
    infoset = [-1] * n
    edge_pos = [None] * n
    edge_by_pos = [None] * n

    for i in range(n):
        if kind[i] != 'player':
            continue
        key = (player[i], infoset_name[i])
        j = infoset_index[key]
        infoset[i] = j
        nodes_of_infoset[j].append(i)
        _require(set(node_actions[i]) == set(infoset_actions[j]),
                 f'information set {key!r} has inconsistent action sets')
        positions = {action: pos for pos, action in enumerate(infoset_actions[j])}
        mapping = [positions[action] for action in node_actions[i]]
        edge_pos[i] = mapping
        inverse = [0] * len(mapping)
        for edge_i, pos in enumerate(mapping):
            inverse[pos] = edge_i
        edge_by_pos[i] = inverse

    for j, key in enumerate(infoset_keys):
        player_infosets[key[0]].append(j)

    # ---- rooted-tree check ------------------------------------------------
    root = index[root_id]
    seen = [False] * n
    order = []
    stack = [root]
    while stack:
        node_i = stack.pop()
        _require(not seen[node_i],
                 f'node {ids[node_i]!r} is reached more than once '
                 '(shared child or cycle)')
        seen[node_i] = True
        order.append(node_i)
        if children[node_i] is not None:
            stack.extend(children[node_i])
    _require(all(seen), 'some nodes are unreachable from the root')

    return GameModel(ids=ids, index=index, root=root, kind=kind, payoff=payoff,
                     player=player, infoset=infoset, infoset_name=infoset_name,
                     children=children, chance=chance, edge_pos=edge_pos,
                     edge_by_pos=edge_by_pos, infoset_keys=infoset_keys,
                     infoset_index=infoset_index, infoset_actions=infoset_actions,
                     nodes_of_infoset=nodes_of_infoset,
                     player_infosets=player_infosets, order=order)


def normalize_policy(model, policy):
    """Validate a public-format policy and return normalized distributions.

    The result maps ``(player, infoset_name)`` to a list of probabilities in
    the same order as ``model.infoset_actions``.  Every information set of the
    game must be present, including information sets that the policy never
    reaches.  Each distribution must contain exactly the legal action keys,
    all probabilities must be finite and non-negative, and the distribution is
    divided by its actual sum.
    """
    if not isinstance(policy, dict):
        raise PolicyError('policy must be a dictionary')
    distributions = {}
    for j, key in enumerate(model.infoset_keys):
        who, name = key
        by_player = policy.get(str(who))
        if not isinstance(by_player, dict) or name not in by_player:
            raise PolicyError(f'missing policy for player {who} information set {name!r}')
        raw = by_player[name]
        if not isinstance(raw, dict):
            raise PolicyError(f'policy for player {who} information set {name!r} '
                              'must be a dictionary')
        actions = model.infoset_actions[j]
        expected = {str(action) for action in actions}
        if set(raw.keys()) != expected:
            raise PolicyError(
                f'policy for player {who} information set {name!r} must cover '
                f'exactly actions {sorted(actions)}')
        values = []
        for action in actions:
            value = raw[str(action)]
            if not _finite(value) or float(value) < 0.0:
                raise PolicyError(
                    f'policy probability for player {who} information set '
                    f'{name!r} action {action} must be finite and non-negative')
            values.append(float(value))
        total = math.fsum(values)
        if not (math.isfinite(total) and total > 0.0):
            raise PolicyError(
                f'policy probabilities for player {who} information set '
                f'{name!r} must have a positive finite sum')
        distributions[key] = [value / total for value in values]
    return distributions


def policy_from_distributions(model, distributions):
    """Build a public-format policy from normalized distributions."""
    policy = {'0': {}, '1': {}}
    for j, key in enumerate(model.infoset_keys):
        who, name = key
        values = distributions[key]
        policy[str(who)][name] = {
            str(action): float(values[pos])
            for pos, action in enumerate(model.infoset_actions[j])
        }
    return policy
