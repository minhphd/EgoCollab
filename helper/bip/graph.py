"""Task graphs G = (V, E): subgoals V (given) and required orderings E (u, v): v cannot start before u is done.

Graphs are frozensets of index pairs; ancestors() turns one into bitmasks, bit u of anc[v] set when u must be
done before v. The prior P(G | instruction) treats orderings as independent (log_prior); S1 works with a finite
pool of graphs sampled from it (sample_pool), to which S2 adds revisions.
"""

from __future__ import annotations

import numpy as np


def ancestors(n: int, edges) -> np.ndarray:
    """Bitmask of the transitive predecessors of every node; raises on a cycle."""
    preds = [0] * n
    for u, v in edges:
        preds[v] |= 1 << u
    anc = preds[:]
    for _ in range(n):
        new = [a | _or(anc[u] for u in range(n) if a >> u & 1) for a in anc]
        if new == anc:
            break
        anc = new
    if any(anc[v] >> v & 1 for v in range(n)):
        raise ValueError("cycle")
    return np.array(anc, dtype=np.int64)


def _or(xs) -> int:
    out = 0
    for x in xs:
        out |= x
    return out


def closure(nodes: list, edges) -> set[tuple]:
    """Every (u, v) with a path from u to v, by node names."""
    succ = {n: set() for n in nodes}
    for u, v in edges:
        succ[u].add(v)
    out = set()
    for n in nodes:
        stack, seen = list(succ[n]), set()
        while stack:
            m = stack.pop()
            if m not in seen:
                seen.add(m)
                stack.extend(succ[m])
        out |= {(n, m) for m in seen if m != n}
    return out


def log_prior(edges: dict, g: frozenset) -> float:
    """log P(G | instruction): each ordering (u, v) independent with S2's probability p_uv; orderings S2 did not list
    get 0.02."""
    return sum(np.log(p if e in g else 1 - p) for e, p in edges.items()) + np.log(0.02) * len(g - edges.keys())


def sample_pool(size: int, n: int, edges: dict, rng) -> list[frozenset]:
    """The most probable acyclic graph and up to size - 1 distinct samples from the prior."""
    def build(order, keep) -> frozenset:
        g = set()
        for e in order:
            if keep(e):
                try:
                    ancestors(n, g | {e})
                    g.add(e)
                except ValueError:  # this ordering would close a cycle
                    pass
        return frozenset(g)
    pool = {build(sorted(edges, key=edges.get, reverse=True), lambda e: edges[e] >= 0.5)}
    for _ in range(size * 20):
        if len(pool) >= size:
            break
        order = [list(edges)[i] for i in rng.permutation(len(edges))]
        pool.add(build(order, lambda e: rng.random() < edges[e]))
    return list(pool)


def repair(n: int, g: frozenset, starts: list[tuple[int, int]]) -> frozenset:
    """S2 without a model: drop every ordering into an observed start whose source was not yet done."""
    g = set(g)
    for v, sat in starts:
        anc = ancestors(n, g)
        g -= {(a, b) for a, b in g if b == v and (not sat >> a & 1 or anc[a] & ~sat)}
    return frozenset(g)
