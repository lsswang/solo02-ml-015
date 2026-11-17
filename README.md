# Finite game policies

Two-player zero-sum **hidden-information** tree games, with an exact
best-response analyzer and a CFR+-based equilibrium solver.  Python 3.11
standard library only; no third-party dependency is required.

## Dependencies and commands

Only the Python 3.11 standard library is used.  `requirements.txt` stays empty;
`requirements-optional.txt` is unused by this implementation.  Reproduction:

```sh
python tests.py    # expects every line to be PASS; exits non-zero otherwise
python demo.py
```

`tests.py` is a plain script; it also works under `python -m pytest tests.py`
if pytest is installed.

## Public API

```python
from evaluate import evaluate     # expected payoff of a supplied policy pair
from analyze import analyze       # exact best responses / exploitability
from solve import solve           # construct a legal equilibrium candidate
```

### `evaluate(game, policy)` → `{'value_player0', 'value_player1'}`

Expected payoffs of the supplied behavioral policy pair. Each distribution is
normalized by its actual sum. Purely on-policy: it never searches for
deviations.

### `analyze(game, policy)` → dict

Accepts *any* legal policy, including one written by another program; it does
not need to come from `solve`.  With input strategy pair `σ = (σ₀, σ₁)` and
on-policy payoffs `u₀, u₁`:

| key | meaning |
| --- | --- |
| `on_policy_values` | `[u₀, u₁]` |
| `best_response_values` | `[b₀, b₁]`, `bᵣ = max_{τᵣ} uᵣ(τᵣ, σ₋ᵣ)` |
| `improvements` | `[b₀−u₀, b₁−u₁]` |
| `nash_conv` | `improvements[0] + improvements[1]` (not halved) |
| `best_response_policies` | `{"0": …, "1": …}`, a pure best response for each side |

Each best response is paired with the *other* supplied strategy to obtain its
reported value; the two responses are not played against each other.  Every
information set gets a complete distribution, including information sets that
the policy never reaches, and actions the input played with probability 0 are
legal deviations.  Player 0 and player 1 may use identical infoset names; the
namespaces are kept separate.

### `solve(game)` → dict

| key | meaning |
| --- | --- |
| `status` | `"converged"`, `"not_converged"`, or `"numerical_failure"` |
| `policy` | the returned behavioral policy, or `None` on failure |
| `value_player0` | payoff of the returned policy for player 0 |
| `nash_conv` | NashConv of the returned policy |
| `message`, `iterations`, `on_policy_values`, `best_response_values`, `improvements` | diagnostics |

`solve` reports `converged` only if the *actual returned policy*, run through
the same exact analysis semantics, has `nash_conv ≤ 1e-3`. The 1e-8 numeric
residual tolerance used inside `analyze` never widens that quality bound.  If
the bound is missed, the best legal candidate found is returned with
`status="not_converged"`; if no trustworthy candidate exists, all metrics are
`None` and `status="numerical_failure"` with a reason in `message`.

## Method

* `game_tree.py` validates the JSON (single rooted tree, no shared child or
  cycle; finite payoff; strictly positive chance probabilities; consistent
  actions per information set) and converts it once into an integer-indexed
  model.  Chance probabilities are normalized by their actual sum.
* **Solver:** alternating CFR+ over that tree.  Strategies are represented
  *only* as one behavioral distribution per `(player, infoset)`, so hidden
  nodes cannot be given separate choices: the information-set constraint holds
  by construction.  Counterfactual regrets are floored at zero; the output is
  the iteration-weighted average behavioral strategy (normalized at each
  information set), with a uniform fallback for information sets never
  reached.  At cumulative iteration checkpoints `(500, 1000, 2000, 4000, 8000,
  16000, 32000)` the actual policy is materialized, validated, and re-analyzed;
  the first checkpoint meeting the quality bound ends the run.  The checkpoint
  list is an implementation effort knob, not the convergence certificate.
* **Analyzer:** the game is a tree with perfect recall, so a pure best response
  decomposes over information sets.  Reach probabilities exclude the
  responder's own actions; information sets are resolved leaves-up using the
  counterfactual value `Σ_{h∈I} reach[h]·value(child(h,a))`.  The reported
  best-response value is recomputed by re-pairing the response with the other
  player's supplied strategy and re-evaluating, and the dynamic-programming
  value must agree with that within 1e-8 or analysis fails explicitly.

## Data format

`{"root": <node id>, "nodes": {<id>: node}}`. Rooted tree, all nodes reachable
exactly once.

* terminal: `{"kind": "terminal", "payoff": <player 0 payoff>}`; player 1 gets
  the negation.
* chance: `{"kind": "chance", "actions": [{"action", "probability", "child"}]}`;
  finite, strictly positive probabilities, normalized by their sum.
* player: `{"kind": "player", "player": 0|1, "infoset": <non-empty str>,
  "actions": [{"action": <int>, "child"}]}`. Nodes of the same player and
  infoset share the same action set and must choose one shared distribution.

Policy format: `policy["0"|"1"][infoset][str(action)] = probability`. Every
information set must be present with exactly its legal action keys, finite
non-negative values, and a positive sum (normalized on input). Caller objects
are never mutated.

## Files

| file | role |
| --- | --- |
| `game_tree.py` | validation, indexed model, policy normalization |
| `evaluate.py` | on-policy expected payoff |
| `analyze.py` | exact best responses, improvements, NashConv |
| `solve.py` | CFR+ solver with the quality certificate |
| `oracles.py` | independent brute-force / reference evaluators (tiny trees) |
| `demo.py` | stable-policy vs. fixed-opponent demonstrations |
| `tests.py` | fixture, oracle, random-tree, and quality checks |
| `fixtures/`, `external_policies.json` | synthetic local reproductions |

## Verification

Reproduction commands:

```sh
python tests.py    # every line should be PASS; non-zero exit on failure
python demo.py
```

`tests.py` checks, among other things:

* information-set sharing (both players' nodes constrained to one action);
* the same infoset name for both players kept in separate namespaces;
* nonconsecutive action IDs (`3, 7, 11, 19`), zero policy probabilities, and
  complete output for unreachable information sets;
* best responses that deviate from actions the input played with probability 0;
* exact agreement with brute-force pure-plan enumeration and independent
  reference evaluation on every bundled external policy and on random small
  hidden-information trees;
* that `solve`'s returned policy is legal, reaches NashConv ≤ 1e-3 on every
  bundled game, and matches a re-analysis of that same returned policy;
* that an unattainably short run reports `not_converged` (never a zero gap) and
  that malformed games report `numerical_failure`.

Larger trees are checked by validating and re-evaluating the actual returned
policy, not by materializing a pure-plan matrix.
