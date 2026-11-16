# Finite game policies

Run the existing complete-information example with Python 3.11:

```sh
python demo.py
```

`solve.py` currently computes a deterministic minimax solution when each player
information set contains one tree node. `evaluate.py` computes the expected
payoffs for an externally supplied behavioral policy pair. `demo.py` calls both
on `fixtures/perfect_information.json`. The existing sources use the Python
standard library.

Each game JSON contains a root node ID and a node dictionary. Player decision
nodes have a player number, information-set name, and action edges. Chance edges
also contain probabilities; terminal nodes contain Player 0's payoff. Player 1
receives its negative. A policy is a nested dictionary keyed by player string,
information-set string, then action string. See `external_policies.json` for
complete examples, including zero-probability actions.

The fixture collection includes a hidden-action game, Kuhn poker, a hidden chance
case with four nonconsecutive actions and the same information-set name for both
players, a depth-20 tree, a terminal-only game, and public chance mixtures of
small hidden-action games. The largest tree has 1999 nodes. These are finite game
descriptions, not precomputed strategies.

You may add numerical or game-solving dependencies. `requirements-optional.txt`
lists one available combination for an OpenSpiel/LP route; it is optional and
does not prescribe an algorithm. Record the dependencies and commands actually
used by your implementation.
