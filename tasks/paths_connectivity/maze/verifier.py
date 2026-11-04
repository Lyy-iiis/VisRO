"""maze verifier: breadth-first search on the accepted maze graph."""
from collections import deque

from visro import runners

TASK = 'maze'
CORRECTNESS = 'BFS on the accepted case graph captured before rendering; check reachability, distance order or radius count.'
UNIQUENESS = 'Single reachable start, strict requested distance rank, or a determinate count/boolean.'


def _check(record, meta, original, image):
    audit = meta['audit']
    graph = {tuple(c): set() for c in audit['cells']}
    for a, b in audit['edges']:
        a, b = tuple(a), tuple(b)
        graph[a].add(b); graph[b].add(a)
    start = tuple(audit['a'] if audit['form'] == 'tf' else audit['target'])
    distance = {start: 0}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in distance:
                distance[v] = distance[u] + 1
                queue.append(v)
    gold = runners.answer(original)
    if audit['form'] == 'tf':
        expected = tuple(audit['b']) in distance
        if gold.lower() not in ('true', 'false'):
            raise ValueError('invalid maze boolean')
        if (gold.lower() == 'true') != expected:
            raise ValueError('maze reachability disagrees with answer')
        return
    starts = [(str(label), tuple(cell)) for label, cell in audit['starts']]
    distances = {label: distance.get(cell, float('inf')) for label, cell in starts}
    if audit['form'] == 'mc':
        reachable = [label for label, d in distances.items() if d < float('inf')]
        if len(reachable) != 1:
            raise ValueError('maze reaching start is not unique')
        expected = reachable[0]
    elif audit['form'] in ('closer', 'kth'):
        ranked = sorted(distances, key=distances.get)
        if len(set(distances.values())) != len(ranked) or float('inf') in distances.values():
            raise ValueError('maze distance ordering is not unique')
        expected = ranked[0 if audit['form'] == 'closer' else audit['k'] - 1]
    elif audit['form'] == 'within':
        expected = str(sum(d <= audit['radius'] for d in distances.values()))
    else:
        raise ValueError('unknown maze form')
    if gold != expected:
        raise ValueError('maze graph answer mismatch')


def verify(record, image):
    return runners.run_checks(TASK, record, image, _check)
