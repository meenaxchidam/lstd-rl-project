"""Member C — LSTD(lambda) (paper Fig. 2), ridge LSTD and recursive LSTD.

z_t  = gamma * lam * z_{t-1} + phi(s_t)
A   += z_t (phi(s_t) - gamma * phi(s_{t+1}))^T          (phi(terminal) = 0)
b   += z_t r_t
theta = A^{-1} b        ->  ridge: (A + reg * I) theta = b
"""
from __future__ import annotations

import numpy as np


def _validate_checkpoints(checkpoints, n_episodes: int) -> list[int]:
    cps = [int(c) for c in checkpoints]
    if not cps or cps != sorted(cps) or cps[0] < 1 or cps[-1] > n_episodes:
        raise ValueError(f"checkpoints must be increasing and within [1, {n_episodes}]")
    return cps


def _accumulate(A: np.ndarray, b: np.ndarray, episode, phi: np.ndarray, gamma: float, lam: float) -> None:
    """Add one episode's contribution to A and b in place."""
    z = np.zeros(phi.shape[1])
    for s, r, s_next, done in episode:
        z = gamma * lam * z + phi[s]
        phi_next = 0.0 if done else phi[s_next]
        A += np.outer(z, phi[s] - gamma * phi_next)
        b += r * z


def lstd_statistics(episodes, phi, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, b) summed over all episodes."""
    phi = np.asarray(phi, dtype=float)
    d = phi.shape[1]
    A, b = np.zeros((d, d)), np.zeros(d)
    for episode in episodes:
        _accumulate(A, b, episode, phi, gamma, lam)
    return A, b


def solve_theta(A: np.ndarray, b: np.ndarray, reg: float = 0.0) -> np.ndarray:
    """Solve (A + reg I) theta = b.

    Falls back to the SVD-based least-squares solution (as in the paper) when the
    matrix is exactly singular, e.g. when some feature was never visited.
    """
    M = A + reg * np.eye(len(b)) if reg else A
    try:
        return np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, b, rcond=None)[0]


def lstd_lambda(episodes, phi, gamma: float, lam: float, reg: float = 0.0) -> np.ndarray:
    """Batch LSTD(lambda); shape (d,)."""
    A, b = lstd_statistics(episodes, phi, gamma, lam)
    return solve_theta(A, b, reg)


def lstd_lambda_curve(episodes, phi, gamma: float, lam: float, checkpoints, reg: float = 0.0) -> np.ndarray:
    """Accumulate A, b once and solve at each checkpoint; shape (len(checkpoints), d)."""
    phi = np.asarray(phi, dtype=float)
    cps = _validate_checkpoints(checkpoints, len(episodes))
    d = phi.shape[1]
    A, b = np.zeros((d, d)), np.zeros(d)
    out = np.empty((len(cps), d))
    k = 0
    for n, episode in enumerate(episodes[: cps[-1]], start=1):
        _accumulate(A, b, episode, phi, gamma, lam)
        while k < len(cps) and cps[k] == n:
            out[k] = solve_theta(A, b, reg)
            k += 1
    return out


def recursive_lstd(episodes, phi, gamma: float, lam: float, delta: float = 1e3) -> np.ndarray:
    """LSTD(lambda) with A^{-1} maintained by Sherman-Morrison, O(d^2) per step.

    Starts from A_0 = I / delta, so the result equals lstd_lambda(..., reg=1/delta).
    """
    phi = np.asarray(phi, dtype=float)
    d = phi.shape[1]
    B = delta * np.eye(d)
    b = np.zeros(d)
    for episode in episodes:
        z = np.zeros(d)
        for s, r, s_next, done in episode:
            z = gamma * lam * z + phi[s]
            u = phi[s] - gamma * (0.0 if done else phi[s_next])
            Bz = B @ z
            uB = u @ B
            B -= np.outer(Bz, uB) / (1.0 + uB @ z)
            b += r * z
    return B @ b
