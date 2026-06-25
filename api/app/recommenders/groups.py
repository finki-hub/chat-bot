from __future__ import annotations

import itertools
from collections.abc import Iterable, Mapping, Sequence


def cooccurrence_edges(
    groupings: Iterable[Sequence[str]],
) -> dict[frozenset[str], int]:
    edges: dict[frozenset[str], int] = {}
    for people in groupings:
        uniq = sorted({p for p in people if p})
        for a, b in itertools.combinations(uniq, 2):
            key = frozenset((a, b))
            edges[key] = edges.get(key, 0) + 1
    return edges


def detect_groups(
    edges: Mapping[frozenset[str], int],
    min_weight: int,
) -> list[list[str]]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for pair, weight in edges.items():
        if weight < min_weight:
            continue
        a, b = tuple(pair)
        union(a, b)

    components: dict[str, list[str]] = {}
    for node in parent:
        components.setdefault(find(node), []).append(node)

    groups = [sorted(members) for members in components.values() if len(members) >= 2]
    groups.sort(key=lambda members: (-len(members), members))
    return groups
