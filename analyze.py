"""Exact independent best-response analysis for hidden-information trees.

``analyze`` is intentionally independent of ``solve``: it accepts any legal
external behavioral policy, computes the on-policy payoff, and then computes
each player's exact best response *against the other player's supplied
strategy*.  Best responses obey information-set constraints: every node in an
information set must use the same action.

Why the result is exact
-----------------------
The game is a tree and the input has perfect recall, so a player's pure best
response decomposes over information sets.  For a responder ``r`` we compute
``reach[h]``, the product of chance probabilities and the *opponent's* action
probabilities on the path to ``h`` (the responder's own probabilities are
omitted).  Information sets are resolved from the leaves upward; the
counterfactual value of action ``a`` at information set ``I`` is

    sum_{h in I} reach[h] * value(child(h, a))

where ``value`` already uses the responder's optimal actions at deeper
information sets and the opponent's supplied strategy elsewhere.  Choosing the
maximizer at every information set of ``r`` therefore yields
``max_{tau_r} u_r(tau_r, sigma_{-r})``.  Pure actions are considered
irrespective of the input policy's probabilities, so actions the input policy
plays with probability zero are legal deviations.  Nodes that opponent/chance
play never reaches have ``reach == 0``; any legal distribution is returned
there, and those choices cannot affect the value.
"""

from __future__ import annotations

import math

from evaluate import evaluate_model
from game_tree import GameError, parse_game, normalize_policy

# Improvements are differences of quantities that can carry a few ulps of
# floating-point error.  A residual in [-CLEAN_TOL, 0] is reported as zero;
# this does not relax the separate 1e-3 NashConv *quality* target (and does not
# alter the best-response or on-policy values themselves).
CLEAN_TOL = 1e-8


class AnalysisError(RuntimeError):
    """A best response could not be determined numerically."""


def _clean_improvement(value):
    if not math.isfinite(value):
        raise AnalysisError('non-finite improvement encountered')
    if value < -CLEAN_TOL:
        # A best response cannot be worse than the supplied feasible strategy;
        # a materially negative improvement signals a numerical failure rather
        # than something to clamp.
        raise AnalysisError(f'best-response improvement {value!r} is materially negative')
    if value < 0.0:
        return 0.0
    return value


def _opponent_reach(model, distributions, responder):
    """Reach probabilities excluding the responder's own action probabilities."""
    reach = [0.0] * model.size
    reach[model.root] = 1.0
    for node_i in model.order:
        node_reach = reach[node_i]
        if node_reach == 0.0:
            continue
        kind = model.kind[node_i]
        if kind == 'terminal':
            continue
        if kind == 'chance':
            weights = model.chance[node_i]
        elif model.player[node_i] == responder:
            weights = [1.0] * len(model.children[node_i])
        else:
            infoset = model.infoset[node_i]
            canonical = distributions[model.infoset_keys[infoset]]
            edge_pos = model.edge_pos[node_i]
            weights = [canonical[edge_pos[edge_i]]
                       for edge_i in range(len(model.children[node_i]))]
        for weight, child in zip(weights, model.children[node_i]):
            reach[child] = node_reach * weight
    return reach


def best_response(model, distributions, responder):
    """Return ``(value, chosen)`` for ``responder`` against the other strategy.

    ``value`` is from the responder's perspective (expected payoff of the
    responder).  ``chosen`` maps information-set index to the canonical action
    position chosen; it is a pure strategy and is defined for every information
    set of the responder, including unreachable ones.
    """
    reach = _opponent_reach(model, distributions, responder)
    values = {}
    action_values = {}
    chosen = {}
    resolving = set()

    def resolve(infoset):
        if infoset in chosen:
            return
        if infoset in resolving:
            key = model.infoset_keys[infoset]
            raise GameError(
                f'information set {key!r} occurs twice on a path; the input '
                'does not satisfy perfect recall')
        resolving.add(infoset)
        nodes = model.nodes_of_infoset[infoset]
        n_actions = len(model.infoset_actions[infoset])
        table = {}
        for node_i in nodes:
            edge_pos = model.edge_pos[node_i]
            children = model.children[node_i]
            per_action = [0.0] * n_actions
            for edge_i, child in enumerate(children):
                per_action[edge_pos[edge_i]] = node_value(child)
            table[node_i] = per_action
        scores = [0.0] * n_actions
        for node_i in nodes:
            weight = reach[node_i]
            if weight != 0.0:
                per_action = table[node_i]
                for action_pos in range(n_actions):
                    scores[action_pos] += weight * per_action[action_pos]
        best = max(range(n_actions), key=lambda pos: scores[pos])
        chosen[infoset] = best
        action_values[infoset] = table
        resolving.remove(infoset)

    def node_value(node_i):
        cached = values.get(node_i)
        if cached is not None:
            return cached
        kind = model.kind[node_i]
        if kind == 'terminal':
            result = model.payoff[node_i] if responder == 0 else -model.payoff[node_i]
        elif kind == 'chance':
            result = 0.0
            for weight, child in zip(model.chance[node_i], model.children[node_i]):
                result += weight * node_value(child)
        elif model.player[node_i] == responder:
            infoset = model.infoset[node_i]
            resolve(infoset)
            edge_i = model.edge_by_pos[node_i][chosen[infoset]]
            result = node_value(model.children[node_i][edge_i])
        else:
            infoset = model.infoset[node_i]
            canonical = distributions[model.infoset_keys[infoset]]
            edge_pos = model.edge_pos[node_i]
            result = 0.0
            for edge_i, child in enumerate(model.children[node_i]):
                result += canonical[edge_pos[edge_i]] * node_value(child)
        if not math.isfinite(result):
            raise AnalysisError(
                f'non-finite value while evaluating node {model.ids[node_i]!r}')
        values[node_i] = result
        return result

    root_value = node_value(model.root)
    if not math.isfinite(root_value):
        raise AnalysisError('non-finite best-response value')
    return root_value, chosen, action_values


def best_response_policy(model, chosen, responder):
    """Turn a pure best-response choice into a public-format player policy."""
    policy = {'0': {}, '1': {}}
    for infoset, action_pos in chosen.items():
        who, name = model.infoset_keys[infoset]
        if who != responder:
            continue
        policy[str(who)][name] = {
            str(action): (1.0 if pos == action_pos else 0.0)
            for pos, action in enumerate(model.infoset_actions[infoset])
        }
    return policy


def analyze_model(model, distributions):
    """Core analysis on a parsed model and normalized distributions."""
    on_policy_player0 = evaluate_model(model, distributions)
    on_policy = [on_policy_player0, -on_policy_player0]

    best_values = [None, None]
    best_policies = {}
    for responder in (0, 1):
        value, chosen, _ = best_response(model, distributions, responder)
        # Recompute the value by pairing the response with the other player's
        # supplied strategy, as required for the reported number.
        paired = dict(distributions)
        for infoset, action_pos in chosen.items():
            key = model.infoset_keys[infoset]
            n_actions = len(model.infoset_actions[infoset])
            paired[key] = [1.0 if pos == action_pos else 0.0
                           for pos in range(n_actions)]
        paired_player0 = evaluate_model(model, paired)
        recomputed = paired_player0 if responder == 0 else -paired_player0
        if abs(recomputed - value) > 1e-8:
            # The dynamic program and the direct evaluation must agree; if
            # they do not, the result is not a trustworthy best response.
            raise AnalysisError(
                f'best-response value check failed for player {responder}: '
                f'{value} vs {recomputed}')
        best_values[responder] = recomputed
        policy = best_response_policy(model, chosen, responder)
        best_policies[str(responder)] = policy[str(responder)]

    raw_improvements = [best_values[0] - on_policy[0],
                        best_values[1] - on_policy[1]]
    improvements = [_clean_improvement(item) for item in raw_improvements]
    nash_conv = improvements[0] + improvements[1]

    return {
        'on_policy_values': on_policy,
        'best_response_values': best_values,
        'improvements': improvements,
        'nash_conv': nash_conv,
        'best_response_policies': {
            '0': best_policies['0'],
            '1': best_policies['1'],
        },
    }


def analyze(game, policy):
    """Analyze an externally supplied legal policy pair.

    Returns ``on_policy_values`` (``[u0, u1]``), ``best_response_values``
    (``[b0, b1]``), ``improvements`` (``[b0-u0, b1-u1]``), ``nash_conv`` (the
    sum of the two one-sided improvements, not halved), and
    ``best_response_policies``.  Each entry of ``best_response_policies`` is a
    pure behavioral policy for that player alone; player 0's entry is a best
    response to the supplied player-1 strategy and vice versa.  They are not
    meant to be played against each other.

    Raises :class:`GameError` / :class:`PolicyError` for malformed input and
    :class:`AnalysisError` if a best response cannot be computed reliably.
    """
    model = parse_game(game)
    distributions = normalize_policy(model, policy)
    return analyze_model(model, distributions)
