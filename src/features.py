"""Member A — feature maps. Each returns phi with shape (num_states, d)."""
from __future__ import annotations

import numpy as np


def boyan_features(num_states: int = 13, spacing: int = 4) -> np.ndarray:
    """Paper Fig. 4: anchors at states 12, 8, 4, 0, linear interpolation between.

    phi(12) = [1, 0, 0, 0], phi(11) = [3/4, 1/4, 0, 0], ..., phi(0) = [0, 0, 0, 1].
    """
    if (num_states - 1) % spacing:
        raise ValueError("num_states - 1 must be a multiple of spacing")
    d = (num_states - 1) // spacing + 1
    phi = np.zeros((num_states, d))
    for s in range(num_states):
        pos = (num_states - 1 - s) / spacing
        i = int(np.floor(pos))
        frac = pos - i
        phi[s, i] = 1.0 - frac
        if frac > 0:
            phi[s, i + 1] = frac
    return phi


def random_walk_features(n: int = 19, kind: str = "aggregation", num_groups: int = 5) -> np.ndarray:
    """Features for RandomWalk(n); terminal states (0 and n+1) get all-zero rows.

    tabular:     one-hot per non-terminal state (d = n, exact).
    aggregation: contiguous groups of states share one feature (d = num_groups,
                 V^pi is NOT exactly representable -> "weaker features").
    """
    states = np.arange(1, n + 1)
    if kind == "tabular":
        phi = np.zeros((n + 2, n))
        phi[states, states - 1] = 1.0
    elif kind == "aggregation":
        phi = np.zeros((n + 2, num_groups))
        for g, group in enumerate(np.array_split(states, num_groups)):
            phi[group, g] = 1.0
    else:
        raise ValueError(f"unknown feature kind {kind!r}")
    return phi


def random_features(num_states: int, d: int, seed=0) -> np.ndarray:
    """Unit-norm Gaussian features; used only for the timing-vs-d experiment."""
    phi = np.random.default_rng(seed).normal(size=(num_states, d))
    return phi / np.linalg.norm(phi, axis=1, keepdims=True)
