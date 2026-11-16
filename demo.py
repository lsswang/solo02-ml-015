"""Run the ordinary complete-information use of the two production helpers."""
import json
from pathlib import Path
from solve import solve
from evaluate import evaluate


if __name__ == '__main__':
    game = json.loads((Path(__file__).parent / 'fixtures/perfect_information.json').read_text())
    result = solve(game)
    observed = evaluate(game, result['policy'])
    print(json.dumps({'solver_value': result['value_player0'], 'evaluated': observed,
                      'policy': result['policy']}, indent=2))
