"""Manifoldness checks for the boundary of a tet mesh.

A tet mesh's boundary is a closed 2-manifold iff every boundary edge is shared
by exactly two boundary triangles *and* the triangles around every boundary
vertex form a single closed fan (otherwise two sheets touch at a pinch point).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
import igl


def nonmanifold_boundary(T):
    """Return (non-manifold boundary vertices, non-manifold boundary edges)."""
    F = igl.boundary_facets(np.asarray(T))[0]
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    bad_e = [e for e, n in Counter(map(tuple, E)).items() if n != 2]
    link = defaultdict(lambda: defaultdict(list))
    for f in F:
        for k in range(3):
            a, b = f[(k + 1) % 3], f[(k + 2) % 3]
            link[f[k]][a].append(b)
            link[f[k]][b].append(a)
    bad_v = []
    for v, adj in link.items():
        if any(len(n) != 2 for n in adj.values()):
            bad_v.append(v)
            continue
        start = next(iter(adj))
        seen, stack = {start}, [start]
        while stack:
            for y in adj[stack.pop()]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        if len(seen) != len(adj):
            bad_v.append(v)
    return bad_v, bad_e


def is_manifold(T):
    bv, be = nonmanifold_boundary(T)
    return not bv and not be


def _tet_volumes(X, T):
    a, b, c, d = (X[T[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0


def _fan_groups(T, tets):
    """Split ``tets`` (indices) into groups connected through shared faces."""
    faces = {}
    for t in tets:
        for k in range(4):
            f = tuple(sorted(np.delete(T[t], k)))
            faces.setdefault(f, []).append(t)
    parent = {t: t for t in tets}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for ts in faces.values():
        for u in ts[1:]:
            parent[find(u)] = find(ts[0])
    groups = defaultdict(list)
    for t in tets:
        groups[find(t)].append(t)
    return list(groups.values())


def make_manifold(X, T, max_rounds=50):
    """Remove the tets that pinch the boundary at non-manifold vertices/edges.

    Around every offending vertex (edge), the incident tets are split into
    groups connected through faces; the largest group (by volume) is kept and
    the others' tets at that vertex (edge) are deleted. Repeats until the
    boundary is a closed 2-manifold. Returns (T_new, removed_volume_fraction).
    """
    T = np.asarray(T).copy()
    vol0 = _tet_volumes(X, T).sum()
    for _ in range(max_rounds):
        bv, be = nonmanifold_boundary(T)
        if not bv and not be:
            break
        drop = set()
        v2t = defaultdict(list)
        for i, t in enumerate(T):
            for v in t:
                v2t[v].append(i)
        vol = _tet_volumes(X, T)
        for v in bv:
            groups = _fan_groups(T, v2t[v])
            if len(groups) > 1:
                keep = max(groups, key=lambda g: vol[g].sum())
                drop.update(t for g in groups if g is not keep for t in g)
        for a, b in be:
            around = [t for t in v2t[a] if b in T[t]]
            groups = _fan_groups(T, around)
            if len(groups) > 1:
                keep = max(groups, key=lambda g: vol[g].sum())
                drop.update(t for g in groups if g is not keep for t in g)
        if not drop:
            # a non-manifold vertex whose incident tets are face-connected
            # (a boundary pinch through the interior): drop its smallest tet
            for v in bv[:1] or [be[0][0]]:
                drop.add(min(v2t[v], key=lambda t: vol[t]))
        T = np.delete(T, sorted(drop), axis=0)
    return T, 1.0 - _tet_volumes(X, T).sum() / vol0
