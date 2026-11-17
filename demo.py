"""Demonstration: stable-policy construction vs. exploiting a fixed opponent.

Run ``python demo.py`` (Python 3.11 standard library only).  The script keeps
the original complete-information output and adds:

1. an external policy pair analyzed by ``analyze`` -- showing the actual
   counter-strategy, including deviations to actions the input played with
   probability zero;
2. the ``solve`` result fed straight into the same ``analyze`` -- showing
   per-player values, exploitability, and the converged/not-converged status;
3. a fixed-opponent action-bias sweep showing that the best-response value
   (and sometimes its action) moves while the game is unchanged.

Two policy pairs with the same on-policy mean are still compared by their
NashConv, never by their self-play payoff.
"""

import json
from pathlib import Path

from analyze import analyze
from evaluate import evaluate
from solve import solve

BASE = Path(__file__).parent
FIXTURES = BASE / 'fixtures'
EXTERNAL = json.loads((BASE / 'external_policies.json').read_text())


def load(name):
    return json.loads((FIXTURES / name).read_text())


def brief(policy):
    """Compact rendering of a behavioral policy."""
    return {player: {infoset: {a: round(p, 6) for a, p in dist.items()}
                     for infoset, dist in infosets.items()}
            for player, infosets in policy.items()}


def pure_choices(policy):
    """The action each information set plays with the largest probability."""
    return {player: {infoset: max(dist, key=dist.get)
                     for infoset, dist in infosets.items()}
            for player, infosets in policy.items()}


def external_examples(game_file):
    return [entry for entry in EXTERNAL if entry['game_file'] == game_file]


def main():
    # ---- 1. the original complete-information demonstration --------------
    game = load('perfect_information.json')
    result = solve(game)
    observed = evaluate(game, result['policy'])
    print('== complete-information example ==')
    print(json.dumps({'solver_status': result['status'],
                      'solver_value': result['value_player0'],
                      'evaluated': observed,
                      'policy': result['policy']}, indent=2))

    # ---- 2. analyze an externally supplied policy pair -------------------
    game = load('hidden_actions.json')
    external = {'0': {'row': {'0': 1.0, '1': 0.0}},
                '1': {'col': {'0': 1.0, '1': 0.0}}}
    print('\n== external policy: analyze (exploit a fixed opponent) ==')
    print('external policy (player 0 always plays action 0):',
          json.dumps(brief(external)))
    findings = analyze(game, external)
    print(json.dumps({
        'on_policy_values': findings['on_policy_values'],
        'best_response_values': findings['best_response_values'],
        'improvements': findings['improvements'],
        'nash_conv': findings['nash_conv'],
        'best_response_policies': brief(findings['best_response_policies']),
    }, indent=2))
    print('best-response pure choices:',
          json.dumps(pure_choices(findings['best_response_policies'])))
    print('player 1 switches to action 1, which the external policy played '
          'with probability zero.')

    # ---- 3. solve, then check the returned policy with the same analyze ---
    print('\n== solve -> same analyze (constructed stable policy) ==')
    solved = solve(game)
    checked = analyze(game, solved['policy'])
    print(json.dumps({
        'status': solved['status'],
        'value_player0': solved['value_player0'],
        'analyzed_on_policy_values': checked['on_policy_values'],
        'best_response_values': checked['best_response_values'],
        'improvements': checked['improvements'],
        'nash_conv': checked['nash_conv'],
        'policy': brief(solved['policy']),
        'best_response_policies': brief(
            checked['best_response_policies']),
    }, indent=2))

    # ---- 4. fixed-opponent bias sweep ------------------------------------
    print('\n== fixed-opponent bias sweep (game unchanged) ==')
    kuhn = load('kuhn.json')
    sweep = []
    for entry in external_examples('fixtures/kuhn.json'):
        report = analyze(kuhn, entry['policy'])
        sweep.append({
            'name': entry['name'],
            'on_policy_values': [round(v, 8)
                                 for v in report['on_policy_values']],
            'best_response_values': [round(v, 8)
                                     for v in report['best_response_values']],
            'nash_conv': round(report['nash_conv'], 8),
            'player0_best_response_actions': pure_choices(
                report['best_response_policies'])['0'],
        })
    print(json.dumps(sweep, indent=2))
    print('\nChanging player 0\'s action bias changes player 0\'s '
          'best-response value while the game is untouched.')

    # ---- 5. equal on-policy mean, different exploitability ---------------
    print('\n== same on-policy mean, different exploitability ==')
    game = load('hidden_actions.json')
    balanced = {'0': {'row': {'0': 0.5, '1': 0.5}},
                '1': {'col': {'0': 0.75, '1': 0.25}}}
    stable = {'0': {'row': {'0': 0.5, '1': 0.5}},
              '1': {'col': {'0': 0.5, '1': 0.5}}}
    pair_report = []
    for label, policy in (('balanced_vs_stable', balanced),
                          ('both_uniform', stable)):
        report = analyze(game, policy)
        pair_report.append({
            'name': label,
            'on_policy_values': report['on_policy_values'],
            'nash_conv': report['nash_conv'],
        })
    print(json.dumps(pair_report, indent=2))
    print('Both pairs have on-policy value 0, but only the second is '
          'stable; NashConv, not the self-play mean, decides.')

    # ---- 6. unreachable information sets still have full distributions ---
    print('\n== unreachable information sets ==')
    four = load('hidden_chance_four_actions.json')
    zero = next(entry for entry in EXTERNAL
                if entry['game_file'] == 'fixtures/hidden_chance_four_actions.json'
                and entry['name'] == 'zero_actions')
    report = analyze(four, zero['policy'])
    print(json.dumps({
        'input_zero_probability_actions':
            zero['policy']['0']['shared_name'],
        'player1_best_response_actions': pure_choices(
            report['best_response_policies'])['1'],
        'player1_best_response_policy': brief(
            report['best_response_policies'])['1'],
        'nash_conv': report['nash_conv'],
    }, indent=2))


if __name__ == '__main__':
    main()
