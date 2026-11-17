"""Hidden-information zero-sum solver with an explicit quality certificate.

Method: alternating CFR+ on the tree.

* The game is converted once into the index representation of
  :mod:`game_tree`.  Information sets are explicit, so every node sharing
  ``(player, infoset)`` is forced to use one behavioral distribution.  That is
  the *only* strategy representation available to the algorithm, so the
  information-set constraint holds by construction rather than by a
  post-processing wrapper.
* Counterfactual regrets are floored at zero after each update (CFR+).  The
  returned strategy is the average behavioral strategy: weighted action sums
  are normalized at each information set.  Unreached information sets fall
  back to a uniform legal distribution.
* Success is never inferred from the iteration count or from a self-play
  payoff.  At each checkpoint the actual average policy is materialized in the
  public format, validated, and handed to the exact independent
  :func:`analyze.analyze_model`.  The solver reports ``converged`` only when
  that analysis returns ``nash_conv <= QUALITY_TARGET`` for the returned
  policy.  Otherwise the best legal candidate found is returned as
  ``not_converged``; if no trustworthy analysis exists, ``numerical_failure``.
"""

from __future__ import annotations

import math

from analyze import AnalysisError, analyze_model
from game_tree import (GameError, PolicyError, normalize_policy, parse_game,
                       policy_from_distributions)

# Fixed quality contract from the task, checked on the returned policy.
QUALITY_TARGET = 1e-3

# Default iteration schedule: cumulative CFR+ iteration counts.  The solver
# continues the same run and re-checks the actual returned strategy at each
# point, stopping at the first one that certifies the target.  32000 is the
# default effort ceiling; on the bundled fixtures the 1000-16000 range is
# normally enough.  Raise ``checkpoints`` for unusually hard trees.
DEFAULT_CHECKPOINTS = (500, 1000, 2000, 4000, 8000, 16000, 32000)


class CFRPlus:
    """Alternating CFR+ over the parsed tree representation."""

    def __init__(self, model):
        self.model = model
        self.n_actions = [len(actions) for actions in model.infoset_actions]
        self.regret = [[0.0] * count for count in self.n_actions]
        self.strategy_sum = [[0.0] * count for count in self.n_actions]
        self.iteration = 0
        self._sig = [[1.0 / count] * count for count in self.n_actions]

    def _current_strategy(self):
        sig = self._sig
        for infoset, regret in enumerate(self.regret):
            values = sig[infoset]
            positive = 0.0
            for pos, value in enumerate(regret):
                if value > 0.0:
                    values[pos] = value
                    positive += value
                else:
                    values[pos] = 0.0
            if positive > 0.0:
                for pos in range(len(values)):
                    values[pos] /= positive
            else:
                uniform = 1.0 / len(values)
                for pos in range(len(values)):
                    values[pos] = uniform
        return sig

    def _walk(self, player):
        model = self.model
        n = model.size
        kind = model.kind
        children = model.children
        chance = model.chance
        player_of = model.player
        infoset_of = model.infoset
        edge_pos = model.edge_pos
        sig = self._sig

        # reach_opp omits this player's own action probabilities;
        # reach_own includes them and weights the average strategy.
        reach_opp = [0.0] * n
        reach_own = [0.0] * n
        reach_opp[model.root] = 1.0
        reach_own[model.root] = 1.0

        for node_i in model.order:
            node_kind = kind[node_i]
            if node_kind == 'terminal':
                continue
            ro = reach_opp[node_i]
            rp = reach_own[node_i]
            if node_kind == 'chance':
                for probability, child in zip(chance[node_i], children[node_i]):
                    reach_opp[child] = ro * probability
                    reach_own[child] = rp * probability
            else:
                positions = edge_pos[node_i]
                action_sig = sig[infoset_of[node_i]]
                if player_of[node_i] == player:
                    for edge_i, child in enumerate(children[node_i]):
                        reach_opp[child] = ro
                        reach_own[child] = rp * action_sig[positions[edge_i]]
                else:
                    for edge_i, child in enumerate(children[node_i]):
                        reach_opp[child] = ro * action_sig[positions[edge_i]]
                        reach_own[child] = rp

        value = [0.0] * n
        action_value = [[0.0] * count for count in self.n_actions]
        weight = float(self.iteration)
        for node_i in reversed(model.order):
            node_kind = kind[node_i]
            if node_kind == 'terminal':
                value[node_i] = (model.payoff[node_i] if player == 0
                                 else -model.payoff[node_i])
                continue
            if node_kind == 'chance':
                total = 0.0
                for probability, child in zip(chance[node_i], children[node_i]):
                    total += probability * value[child]
                value[node_i] = total
                continue
            infoset = infoset_of[node_i]
            action_sig = sig[infoset]
            positions = edge_pos[node_i]
            node_children = children[node_i]
            total = 0.0
            for edge_i, child in enumerate(node_children):
                total += action_sig[positions[edge_i]] * value[child]
            value[node_i] = total
            if player_of[node_i] == player:
                ro = reach_opp[node_i]
                rp = reach_own[node_i]
                incoming = action_value[infoset]
                own_sum = self.strategy_sum[infoset]
                for edge_i, child in enumerate(node_children):
                    pos = positions[edge_i]
                    incoming[pos] += ro * value[child]
                    own_sum[pos] += weight * rp * action_sig[pos]

        for infoset in model.player_infosets[player]:
            incoming = action_value[infoset]
            action_sig = sig[infoset]
            regret = self.regret[infoset]
            baseline = 0.0
            for pos in range(self.n_actions[infoset]):
                baseline += action_sig[pos] * incoming[pos]
            for pos in range(self.n_actions[infoset]):
                updated = regret[pos] + incoming[pos] - baseline
                regret[pos] = updated if updated > 0.0 else 0.0

    def run_until(self, target_iterations):
        if target_iterations < self.iteration:
            raise ValueError('target_iterations must not decrease')
        while self.iteration < target_iterations:
            self.iteration += 1
            self._current_strategy()
            self._walk(0)
            self._current_strategy()
            self._walk(1)

    def average_policy(self):
        """Return the current average strategy as normalized distributions."""
        distributions = {}
        for infoset, key in enumerate(self.model.infoset_keys):
            total = math.fsum(self.strategy_sum[infoset])
            count = self.n_actions[infoset]
            if total > 0.0:
                distributions[key] = [value / total
                                      for value in self.strategy_sum[infoset]]
            else:
                # Never reached under average play: any legal distribution.
                distributions[key] = [1.0 / count] * count
        return distributions


def _result(status, model=None, distributions=None, analysis=None,
            message=None, iterations=None):
    result = {
        'status': status,
        'policy': None,
        'value_player0': None,
        'nash_conv': None,
        'message': message,
        'iterations': iterations,
    }
    if model is not None and distributions is not None:
        result['policy'] = policy_from_distributions(model, distributions)
    if analysis is not None:
        result['value_player0'] = analysis['on_policy_values'][0]
        result['nash_conv'] = analysis['nash_conv']
        result['on_policy_values'] = analysis['on_policy_values']
        result['best_response_values'] = analysis['best_response_values']
        result['improvements'] = analysis['improvements']
    return result


def solve(game, target=QUALITY_TARGET, checkpoints=None):
    """Construct a legal behavioral policy pair with ``nash_conv <= target``.

    Returns ``status`` (``"converged"``, ``"not_converged"`` or
    ``"numerical_failure"``), ``policy``, ``value_player0`` and ``nash_conv``
    for the returned policy, plus the analysis numbers used for the decision.
    ``value_player0`` / ``nash_conv`` are ``None`` if no trustworthy candidate
    exists.

    ``target`` is the requested precision.  The task fixes the required
    quality at ``NashConv <= 1e-3``, so a looser request is clamped to that
    bound; a stricter request is attempted but is not required to succeed.
    ``checkpoints`` is the cumulative CFR+ iteration schedule (default
    :data:`DEFAULT_CHECKPOINTS`); the run stops at the first checkpoint whose
    returned average policy certifies the quality bound.
    """
    try:
        model = parse_game(game)
    except (GameError, ValueError) as error:
        return _result('numerical_failure', message=f'invalid game: {error}')

    # Never certify a policy that is looser than the fixed 1e-3 contract.
    target = min(float(target), QUALITY_TARGET)
    if not model.infoset_keys:
        # No decisions anywhere: the empty policy is legal and exact.
        distributions = {}
        try:
            analysis = analyze_model(model, distributions)
        except AnalysisError as error:
            return _result('numerical_failure',
                           message=f'analysis failed: {error}')
        return _result('converged', model, distributions, analysis,
                       iterations=0)

    schedule = tuple(sorted({int(point) for point in
                             (DEFAULT_CHECKPOINTS if checkpoints is None
                              else checkpoints)}))
    if not schedule:
        return _result('numerical_failure', message='empty iteration schedule')

    solver = CFRPlus(model)
    best = None  # (nash_conv, distributions, analysis)
    try:
        for checkpoint in schedule:
            solver.run_until(checkpoint)
            distributions = solver.average_policy()
            # Certify exactly the object that would be returned: materialize
            # the public policy, validate it, then re-normalize the way any
            # consumer would before analyzing it.
            public = policy_from_distributions(model, distributions)
            certified = normalize_policy(model, public)
            analysis = analyze_model(model, certified)
            gap = analysis['nash_conv']
            if best is None or gap < best[0]:
                best = (gap, certified, analysis)
            if gap <= target:
                return _result('converged', model, certified, analysis,
                               iterations=checkpoint)
    except AnalysisError as error:
        if best is not None:
            return _result('not_converged', model, best[1], best[2],
                           message=f'analysis failed after a usable candidate: {error}',
                           iterations=solver.iteration)
        return _result('numerical_failure', message=f'analysis failed: {error}',
                       iterations=solver.iteration)
    except (FloatingPointError, OverflowError, ValueError, PolicyError) as error:
        if best is not None:
            return _result('not_converged', model, best[1], best[2],
                           message=f'numeric issue after a usable candidate: {error}',
                           iterations=solver.iteration)
        return _result('numerical_failure', message=f'numeric failure: {error}',
                       iterations=solver.iteration)

    gap, distributions, analysis = best
    return _result('not_converged', model, distributions, analysis,
                   message=(f'NashConv {gap:.3e} did not reach target '
                            f'{target:.1e} after {solver.iteration} iterations'),
                   iterations=solver.iteration)


def solve_policy(game):
    """Return just the policy from :func:`solve` (``None`` on failure)."""
    return solve(game)['policy']
