"""Ordinary perfect-information zero-sum tree solver, authored for this baseline."""


def solve(game):
    """Return a deterministic minimax policy; repeated information sets unsupported.

    This is an honest perfect-information implementation, not an implementation
    of the new hidden-information equilibrium task. Chance is allowed here.
    """
    seen = set()
    for node in game['nodes'].values():
        if node['kind'] == 'player':
            key = (node['player'], node['infoset'])
            if key in seen:
                raise NotImplementedError('this baseline requires singleton information sets')
            seen.add(key)
    policy = {'0': {}, '1': {}}

    def visit(node_id):
        node = game['nodes'][node_id]
        if node['kind'] == 'terminal':
            return node['payoff']
        if node['kind'] == 'chance':
            return sum(edge['probability'] * visit(edge['child']) for edge in node['actions'])
        values = [(visit(edge['child']), edge['action']) for edge in node['actions']]
        value, action = (max if node['player'] == 0 else min)(values, key=lambda item: item[0])
        policy[str(node['player'])][node['infoset']] = {
            str(edge['action']): float(edge['action'] == action) for edge in node['actions']}
        return value

    value = visit(game['root'])
    return {'value_player0': value, 'policy': policy}
