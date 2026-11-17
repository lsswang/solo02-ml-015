"""End-to-end checks for the hidden-information analysis and solver.

Run with either ``python -m pytest tests.py -q`` or ``python tests.py``.

Small trees are checked against independent brute-force oracles:
* expected values via a plain recursive JSON walk;
* best responses via enumeration of every pure information-set plan.

Large trees are checked by validating the *actually returned* policy and
re-evaluating its payoff / best responses, without materializing a pure-plan
matrix.
"""

import itertools
import json
import math
from pathlib import Path

from analyze import analyze
from evaluate import evaluate
from game_tree import PolicyError, normalize_policy, parse_game
from oracles import (brute_force_best_response, reference_value)
from solve import QUALITY_TARGET, solve

BASE = Path(__file__).parent
FIXTURES = BASE / 'fixtures'
POLICIES = json.loads((BASE / 'external_policies.json').read_text())
TOL = 1e-8


def load(name):
    return json.loads((FIXTURES / name).read_text())


def external(game_file, name):
    for entry in POLICIES:
        if entry['game_file'] == game_file and entry['name'] == name:
            return entry['policy']
    raise KeyError((game_file, name))


def close(a, b, tol=TOL):
    return abs(a - b) <= tol


def assert_legal(game, policy):
    """Structural legality of a public-format policy."""
    model = parse_game(game)
    for key, actions in zip(model.infoset_keys, model.infoset_actions):
        who, name = key
        distribution = policy[str(who)][name]
        assert set(distribution) == {str(a) for a in actions}
        values = list(distribution.values())
        assert all(math.isfinite(v) and v >= 0.0 for v in values)
        assert abs(math.fsum(values) - 1.0) <= 1e-12
    normalize_policy(model, policy)  # the production validator agrees
    return model


# --------------------------------------------------------------------------
# analyze: brute-force cross-checks on every external policy
# --------------------------------------------------------------------------

def test_analyze_matches_oracles_for_external_policies():
    checked = 0
    for entry in POLICIES:
        game = load(Path(entry['game_file']).name)
        policy = entry['policy']
        assert_legal(game, policy)
        report = analyze(game, policy)

        reference = reference_value(game, policy)
        assert close(report['on_policy_values'][0], reference), entry
        assert close(report['on_policy_values'][1], -reference), entry

        for responder in (0, 1):
            bruteforce, pure = brute_force_best_response(game, policy, responder)
            assert close(report['best_response_values'][responder], bruteforce), \
                (entry['game_file'], entry['name'], responder)
            # Re-pair the returned pure response with the input opponent.
            if responder == 0:
                paired = {'0': pure, '1': policy['1']}
            else:
                paired = {'0': policy['0'], '1': pure}
            paired_value = evaluate(game, paired)
            got = (paired_value['value_player0'] if responder == 0
                   else paired_value['value_player1'])
            assert close(report['best_response_values'][responder], got), \
                (entry['game_file'], entry['name'], responder, got)
            # The response's own section is complete for that player.
            model = parse_game(game)
            for key, actions in zip(model.infoset_keys, model.infoset_actions):
                who, name = key
                if who == responder:
                    assert set(report['best_response_policies'][str(who)][name]) \
                        == {str(a) for a in actions}
            checked += 1

        improvements = [report['best_response_values'][0] - report['on_policy_values'][0],
                        report['best_response_values'][1] - report['on_policy_values'][1]]
        for got, expected in zip(report['improvements'], improvements):
            assert abs(got - expected) <= TOL
        assert close(report['nash_conv'],
                     report['improvements'][0] + report['improvements'][1])
    assert checked == 2 * len(POLICIES)


def test_best_response_uses_zero_probability_actions():
    # hidden_actions zero_actions: player 0 plays row=0 always; player 1's
    # best response must use col=1 although it has probability 0 on policy.
    game = load('hidden_actions.json')
    report = analyze(game, external('fixtures/hidden_actions.json', 'zero_actions'))
    assert report['best_response_policies']['1']['col']['1'] == 1.0
    assert close(report['best_response_values'][1], 1.0)

    four = load('hidden_chance_four_actions.json')
    report = analyze(four, external('fixtures/hidden_chance_four_actions.json',
                                    'zero_actions'))
    chosen = report['best_response_policies']['1']['shared_name']
    assert any(chosen[str(a)] == 1.0 for a in (7, 11, 19))
    assert chosen['3'] == 0.0


def test_unreachable_infosets_are_complete_and_legal():
    # In hidden_actions, both of player 1's nodes share an information set, so
    # there is no unreachable player infoset; in kuhn, an all-action-0 policy
    # never reaches the *pb* sets but they still need a full distribution.
    game = load('kuhn.json')
    report = analyze(game, external('fixtures/kuhn.json', 'zero_actions'))
    complete = report['best_response_policies']
    assert_legal(game, complete)
    model = parse_game(game)
    for key, actions in zip(model.infoset_keys, model.infoset_actions):
        who, name = key
        assert set(complete[str(who)][name]) == {str(a) for a in actions}
    # Delta/depth with unreached infosets in the solver output.
    for name in ('depth_twenty.json', 'kuhn.json'):
        result = solve(load(name))
        assert result['status'] == 'converged'
        assert_legal(load(name), result['policy'])


def test_shared_infoset_names_across_players():
    game = load('hidden_chance_four_actions.json')
    # Same name for both players, different distributions must stay separate.
    policy = {'0': {'shared_name': {'3': 1.0, '7': 0.0, '11': 0.0, '19': 0.0}},
              '1': {'shared_name': {'3': 0.0, '7': 0.0, '11': 0.0, '19': 1.0}}}
    report = analyze(game, policy)
    assert close(report['on_policy_values'][0], reference_value(game, policy))
    # The player-1 information set must not reuse the player-0 distribution.
    assert report['best_response_policies']['1']['shared_name']['3'] in (0.0, 1.0)
    solved = solve(game)
    assert_legal(game, solved['policy'])
    assert set(solved['policy']['0']) == {'shared_name'}
    assert set(solved['policy']['1']) == {'shared_name'}


def test_unreachable_responder_infoset_gets_full_best_response():
    game = {'root': 'r', 'nodes': {
        'r': {'kind': 'player', 'player': 0, 'infoset': 'I0', 'actions': [
            {'action': 0, 'child': 't0'}, {'action': 1, 'child': 'p1'}]},
        't0': {'kind': 'terminal', 'payoff': 1.0},
        'p1': {'kind': 'player', 'player': 1, 'infoset': 'J1', 'actions': [
            {'action': 5, 'child': 'a'}, {'action': 9, 'child': 'b'}]},
        'a': {'kind': 'terminal', 'payoff': -1.0},
        'b': {'kind': 'terminal', 'payoff': 2.0}}}
    policy = {'0': {'I0': {'0': 1.0, '1': 0.0}},
              '1': {'J1': {'5': 1.0, '9': 0.0}}}
    report = analyze(game, policy)
    # Player-1's information set is never reached, but its response is complete
    # and cannot change the value.
    assert set(report['best_response_policies']['1']['J1']) == {'5', '9'}
    assert close(report['best_response_values'][1], -1.0)

    # Player 0's own unreachable infoset after choosing action 0 is also full.
    game2 = {'root': 'r', 'nodes': {
        'r': {'kind': 'player', 'player': 0, 'infoset': 'A', 'actions': [
            {'action': 0, 'child': 'b'}, {'action': 1, 'child': 't5'}]},
        'b': {'kind': 'player', 'player': 0, 'infoset': 'B', 'actions': [
            {'action': 0, 'child': 't-10'}, {'action': 1, 'child': 't-9'}]},
        't5': {'kind': 'terminal', 'payoff': 5.0},
        't-10': {'kind': 'terminal', 'payoff': -10.0},
        't-9': {'kind': 'terminal', 'payoff': -9.0}}}
    policy2 = {'0': {'A': {'0': 0.0, '1': 1.0}, 'B': {'0': 0.5, '1': 0.5}},
               '1': {}}
    report2 = analyze(game2, policy2)
    assert set(report2['best_response_policies']['0']) == {'A', 'B'}
    assert close(report2['best_response_values'][0], 5.0)


def test_policy_normalization_and_validation():
    game = load('hidden_actions.json')
    # Unnormalized but proportional distributions are interpreted by sum.
    policy = {'0': {'row': {'0': 3.0, '1': 1.0}},
              '1': {'col': {'0': 1.0, '1': 2.0}}}
    a = analyze(game, policy)
    b = analyze(game, {'0': {'row': {'0': 0.75, '1': 0.25}},
                       '1': {'col': {'0': 1 / 3, '1': 2 / 3}}})
    assert close(a['nash_conv'], b['nash_conv'])

    # Missing information set / wrong keys / negative values are rejected.
    bad_inputs = [
        {'0': {'row': {'0': 1.0}}, '1': {'col': {'0': 0.5, '1': 0.5}}},
        {'0': {'row': {'0': 1.0, '1': 0.0, '2': 0.0}},
         '1': {'col': {'0': 0.5, '1': 0.5}}},
        {'0': {'row': {'0': -0.5, '1': 1.5}},
         '1': {'col': {'0': 0.5, '1': 0.5}}},
        {'1': {'col': {'0': 0.5, '1': 0.5}}},
    ]
    for bad in bad_inputs:
        try:
            analyze(game, bad)
        except PolicyError:
            pass
        else:
            raise AssertionError(f'accepted malformed policy {bad}')


def test_analyze_does_not_mutate_inputs():
    game = load('kuhn.json')
    policy = external('fixtures/kuhn.json', 'uniform')
    game_before = json.dumps(game, sort_keys=True)
    policy_before = json.dumps(policy, sort_keys=True)
    analyze(game, policy)
    assert json.dumps(game, sort_keys=True) == game_before
    assert json.dumps(policy, sort_keys=True) == policy_before


def test_nonconsecutive_action_ids():
    game = load('hidden_chance_four_actions.json')
    report = analyze(game, external('fixtures/hidden_chance_four_actions.json',
                                    'uniform'))
    for who in ('0', '1'):
        assert set(report['best_response_policies'][who]['shared_name']) == \
            {'3', '7', '11', '19'}
    result = solve(game)
    assert set(result['policy']['0']['shared_name']) == {'3', '7', '11', '19'}


def test_terminal_only():
    game = load('terminal_only.json')
    report = analyze(game, {'0': {}, '1': {}})
    assert report['on_policy_values'] == [2.5, -2.5]
    assert close(report['nash_conv'], 0.0)
    assert report['best_response_policies'] == {'0': {}, '1': {}}
    result = solve(game)
    assert result['status'] == 'converged'
    assert result['value_player0'] == 2.5 and result['nash_conv'] == 0.0


# --------------------------------------------------------------------------
# solve: quality and independence checks
# --------------------------------------------------------------------------

def test_solve_converges_all_fixtures_and_rechecks():
    for path in sorted(FIXTURES.glob('*.json')):
        game = json.loads(path.read_text())
        result = solve(game)
        assert result['status'] == 'converged', (path.name, result['message'])
        assert result['nash_conv'] is not None
        assert result['nash_conv'] <= QUALITY_TARGET, path.name
        policy = result['policy']
        assert_legal(game, policy)
        report = analyze(game, policy)
        assert abs(report['nash_conv'] - result['nash_conv']) <= TOL
        assert abs(report['on_policy_values'][0] - result['value_player0']) <= TOL
        assert abs(evaluate(game, policy)['value_player0']
                   - result['value_player0']) <= TOL


def test_solve_value_matches_bruteforce_on_small_games():
    for name in ('hidden_actions.json', 'hidden_chance_four_actions.json',
                 'kuhn.json', 'perfect_information.json', 'depth_twenty.json'):
        game = load(name)
        result = solve(game)
        assert result['status'] == 'converged'
        policy = result['policy']
        report = analyze(game, policy)
        # Each side is nearly best-responding to the other; the brute-force
        # oracle confirms the reported best-response values for the returned
        # policy rather than reusing analyze's internal number.
        for responder in (0, 1):
            bruteforce, _ = brute_force_best_response(game, policy, responder)
            assert close(report['best_response_values'][responder], bruteforce)
        assert report['nash_conv'] <= QUALITY_TARGET


def test_report_values_are_for_returned_policy():
    game = load('kuhn.json')
    result = solve(game)
    report = analyze(game, result['policy'])
    assert close(result['value_player0'], report['on_policy_values'][0])
    assert close(result['nash_conv'], report['nash_conv'])
    # on-policy, best-response, improvement identity
    assert close(report['nash_conv'],
                 (report['best_response_values'][0] - report['on_policy_values'][0])
                 + (report['best_response_values'][1] - report['on_policy_values'][1]))


def test_target_is_not_relaxed():
    # An unrealistically strict target must not be reported as converged.
    game = load('hidden_chance_four_actions.json')
    result = solve(game, checkpoints=(50,))
    assert result['status'] == 'not_converged'
    assert result['policy'] is not None      # best legal candidate preserved
    assert result['nash_conv'] is not None
    assert result['nash_conv'] > QUALITY_TARGET
    assert_legal(game, result['policy'])
    # The retained numbers describe that retained policy.
    report = analyze(game, result['policy'])
    assert close(report['nash_conv'], result['nash_conv'])


def test_invalid_game_reports_numerical_failure():
    broken = {'root': 'a', 'nodes': {'a': {'kind': 'player', 'player': 0,
                                           'infoset': 'x',
                                           'actions': [{'action': 0, 'child': 'b'}]},
                                     'b': {'kind': 'terminal', 'payoff': 1.0}}}
    # 'b' is reachable but 'a' has only one action: still a legal (degenerate)
    # tree, so instead break the child reference.
    broken['nodes']['a']['actions'][0]['child'] = 'missing'
    result = solve(broken)
    assert result['status'] == 'numerical_failure'
    assert result['policy'] is None
    assert result['value_player0'] is None and result['nash_conv'] is None
    assert result['message']


def test_random_trees_cross_checked():
    """Random small hidden-information trees; analyze vs. brute force."""
    import random
    rng = random.Random(20261117)
    for _ in range(25):
        game = random_tree(rng)
        policy = random_policy(game, rng)
        assert_legal(game, policy)
        report = analyze(game, policy)
        assert close(report['on_policy_values'][0], reference_value(game, policy))
        for responder in (0, 1):
            value, _ = brute_force_best_response(game, policy, responder)
            assert close(report['best_response_values'][responder], value,
                         tol=1e-7)
        result = solve(game)
        assert result['status'] == 'converged'
        assert result['nash_conv'] <= QUALITY_TARGET
        assert_legal(game, result['policy'])


def random_tree(rng, max_nodes=45):
    """Generate a random perfect-recall two-player zero-sum tree."""
    nodes = {}
    counter = itertools.count()
    action_sets = {}

    def build(depth, history):
        node_id = f'n{next(counter)}'
        if depth >= 3 or rng.random() < 0.3:
            nodes[node_id] = {'kind': 'terminal',
                              'payoff': float(rng.randint(-4, 4))}
            return node_id
        if rng.random() < 0.15:
            actions = []
            for action in range(rng.randint(2, 3)):
                actions.append({'action': action,
                                'probability': rng.choice([0.2, 0.3, 0.5, 1.7]),
                                'child': build(depth + 1, history)})
            nodes[node_id] = {'kind': 'chance', 'actions': actions}
            return node_id
        player = rng.randint(0, 1)
        # Perfect recall: the infoset is determined by the player and the
        # sequence of that player's own actions so far.  Nodes with the same
        # key therefore agree on all of that player's earlier observations.
        key = (player, history[player])
        if key not in action_sets:
            action_sets[key] = rng.choice([[0, 1], [0, 2], [5, 9], [3, 4, 8]])
        action_ids = action_sets[key]
        infoset = f'p{player}h{len(history[player])}v{history[player]}'
        actions = []
        for action in action_ids:
            child_history = (history[0], history[1])
            if player == 0:
                child_history = (history[0] + (action,), history[1])
            else:
                child_history = (history[0], history[1] + (action,))
            actions.append({'action': action,
                            'child': build(depth + 1, child_history)})
        nodes[node_id] = {'kind': 'player', 'player': player,
                          'infoset': infoset, 'actions': actions}
        return node_id

    root = build(0, ((), ()))
    return {'root': root, 'nodes': nodes}


def random_policy(game, rng):
    model = parse_game(game)
    policy = {'0': {}, '1': {}}
    for key, actions in zip(model.infoset_keys, model.infoset_actions):
        who, name = key
        values = [rng.random() for _ in actions]
        total = math.fsum(values)
        policy[str(who)][name] = {str(a): v / total
                                  for a, v in zip(actions, values)}
    return policy


if __name__ == '__main__':
    failures = 0
    for test_name, test in sorted(globals().items()):
        if test_name.startswith('test_') and callable(test):
            try:
                test()
                print(f'PASS {test_name}')
            except Exception as error:  # noqa: BLE001 - report and continue
                failures += 1
                import traceback
                print(f'FAIL {test_name}: {error}')
                traceback.print_exc()
    raise SystemExit(1 if failures else 0)
