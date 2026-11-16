"""Ordinary on-policy payoff evaluator, independent of any training algorithm."""


def evaluate(game, policy):
    """Compute the exact expected payoff of a supplied behavioral strategy pair.

    Policies are keyed by player and information set, so even this ordinary
    evaluator does not allow state-dependent cheating. It does not find a best
    response or claim that a good self-play score proves low exploitability.
    """
    def visit(node_id):
        node = game['nodes'][node_id]
        if node['kind'] == 'terminal':
            return node['payoff']
        if node['kind'] == 'chance':
            probabilities = [edge['probability'] for edge in node['actions']]
        else:
            distribution = policy[str(node['player'])][node['infoset']]
            probabilities = [distribution[str(edge['action'])] for edge in node['actions']]
        return sum(probability * visit(edge['child'])
                   for probability, edge in zip(probabilities, node['actions']))

    value = visit(game['root'])
    return {'value_player0': value, 'value_player1': -value}
