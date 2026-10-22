"""Member B — linear TD(lambda) with accumulating traces (paper Fig. 1).

z_t     = gamma * lam * z_{t-1} + phi(s_t)
delta_t = r_t + gamma * phi(s_{t+1})^T theta - phi(s_t)^T theta     (phi(terminal) = 0)
theta  += alpha_n * delta_t * z_t

online=False (default, as in the paper): the updates of a trajectory are summed
and applied once at its end, with theta held fixed during the trajectory.
online=True: theta is updated after every transition.
"""
from __future__ import annotations

import numpy as np

SCHEDULES = ("constant", "decay")


def step_size(n: int, alpha: float, schedule: str = "constant", n0: float | None = None) -> float:
    """Step size for trajectory n (1-based).

    constant: alpha
    decay:    alpha * (n0 + 1) / (n0 + n)        # paper Eq. 8 with a0 = alpha
    """
    if schedule == "constant":
        return alpha
    if schedule == "decay":
        if n0 is None:
            raise ValueError("the decay schedule needs n0")
        return alpha * (n0 + 1.0) / (n0 + n)
    raise ValueError(f"unknown schedule {schedule!r}; expected one of {SCHEDULES}")


def _validate_checkpoints(checkpoints, n_episodes: int) -> list[int]:
    cps = [int(c) for c in checkpoints]
    if not cps or cps != sorted(cps) or cps[0] < 1 or cps[-1] > n_episodes:
        raise ValueError(f"checkpoints must be increasing and within [1, {n_episodes}]")
    return cps


def td_lambda_curve(
    episodes,
    phi,
    gamma: float,
    lam: float,
    alpha: float,
    checkpoints,
    schedule: str = "constant",
    n0: float | None = None,
    online: bool = False,
    theta0=None,
) -> np.ndarray:
    """Run TD(lambda) once and return theta after each checkpoint (number of trajectories).

    Returns shape (len(checkpoints), d). If theta diverges, the remaining rows are NaN.
    """
    phi = np.asarray(phi, dtype=float)
    cps = _validate_checkpoints(checkpoints, len(episodes))
    d = phi.shape[1]
    theta = np.zeros(d) if theta0 is None else np.array(theta0, dtype=float)
    out = np.full((len(cps), d), np.nan)
    gl = gamma * lam
    k = 0
    for n, episode in enumerate(episodes[: cps[-1]], start=1):
        a = step_size(n, alpha, schedule, n0)
        z = np.zeros(d)
        if online:
            for s, r, s_next, done in episode:
                z = gl * z + phi[s]
                v_next = 0.0 if done else phi[s_next] @ theta
                delta = r + gamma * v_next - phi[s] @ theta
                theta += a * delta * z
        else:
            v = phi @ theta
            dtheta = np.zeros(d)
            for s, r, s_next, done in episode:
                z = gl * z + phi[s]
                delta = r + gamma * (0.0 if done else v[s_next]) - v[s]
                dtheta += delta * z
            theta += a * dtheta
        if not np.all(np.isfinite(theta)):
            break
        while k < len(cps) and cps[k] == n:
            out[k] = theta
            k += 1
    return out


def td_lambda(episodes, phi, gamma: float, lam: float, alpha: float, **kwargs) -> np.ndarray:
    """theta after one pass of TD(lambda) over all episodes, shape (d,)."""
    return td_lambda_curve(episodes, phi, gamma, lam, alpha, [len(episodes)], **kwargs)[0]


def td_alpha_sweep(episodes, phi, gamma: float, lam: float, alphas, **kwargs) -> dict[float, np.ndarray]:
    """Run TD(lambda) for each step size; returns {alpha: theta}."""
    return {float(a): td_lambda(episodes, phi, gamma, lam, a, **kwargs) for a in alphas}
