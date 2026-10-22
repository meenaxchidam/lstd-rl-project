"""Member A — environments, episode generation and ground-truth values.

Each environment is defined by an explicit transition model (P, R), so the
simulator used by `step()` and the exact Bellman solve in `true_values()`
are guaranteed to describe the same chain.

Episode format (shared by every module):
    episode = [(s, r, s_next, done), ...]     # s, s_next: int, r: float, done: bool
"""
from __future__ import annotations

import numpy as np

Transition = tuple[int, float, int, bool]
Episode = list[Transition]


class MarkovChain:
    """Episodic Markov reward process with a fixed start state.

    P[s, s'] : transition probability, R[s, s'] : reward on that transition.
    Rows of P for terminal states are ignored.
    """

    gamma = 1.0

    def __init__(self, P, R, terminal, start_state: int, seed=None):
        self.P = np.asarray(P, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.num_states = self.P.shape[0]
        self.terminal_mask = np.zeros(self.num_states, dtype=bool)
        self.terminal_mask[list(terminal)] = True
        self.start_state = int(start_state)
        self._cdf = np.cumsum(self.P, axis=1)
        self.state: int | None = None
        self.seed(seed)

    def seed(self, seed=None) -> None:
        self.rng = np.random.default_rng(seed)

    def is_terminal(self, s: int) -> bool:
        return bool(self.terminal_mask[s])

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self) -> tuple[int, float, bool]:
        """Sample one transition from the current state -> (s_next, reward, done)."""
        s = self.state
        if s is None or self.is_terminal(s):
            raise RuntimeError("Episode is over; call reset() first.")
        s_next = int(np.searchsorted(self._cdf[s], self.rng.random(), side="right"))
        reward = float(self.R[s, s_next])
        self.state = s_next
        return s_next, reward, self.is_terminal(s_next)

    def transition_model(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (P, r) with r[s] = expected one-step reward from s."""
        return self.P.copy(), (self.P * self.R).sum(axis=1)


class BoyanChain(MarkovChain):
    """13-state Boyan chain (paper Fig. 4).

    States 12 -> 0, start at 12, state 0 terminal. From s >= 2 move to s-1 or
    s-2 with prob 0.5 each (reward -3); from 1 move to 0 (reward -2).
    """

    def __init__(self, seed=None):
        n = 13
        P = np.zeros((n, n))
        R = np.zeros((n, n))
        for s in range(2, n):
            P[s, s - 1] = P[s, s - 2] = 0.5
            R[s, s - 1] = R[s, s - 2] = -3.0
        P[1, 0] = 1.0
        R[1, 0] = -2.0
        super().__init__(P, R, terminal=(0,), start_state=n - 1, seed=seed)


class RandomWalk(MarkovChain):
    """n-state random walk (Sutton & Barto, Example 7.1; n = 19 by default).

    States 0..n+1; 0 and n+1 are terminal; start in the centre. Move left or
    right with prob 0.5. Reward -1 on entering state 0, +1 on entering n+1.
    True values: V(s) = 2 s / (n + 1) - 1.
    """

    def __init__(self, n: int = 19, seed=None):
        self.n = n
        size = n + 2
        P = np.zeros((size, size))
        R = np.zeros((size, size))
        for s in range(1, n + 1):
            P[s, s - 1] = P[s, s + 1] = 0.5
        R[1, 0] = -1.0
        R[n, n + 1] = 1.0
        super().__init__(P, R, terminal=(0, n + 1), start_state=(n + 1) // 2, seed=seed)


def generate_episodes(env: MarkovChain, n: int, seed=None) -> list[Episode]:
    """Sample n complete episodes; identical seeds give identical episodes."""
    env.seed(seed)
    episodes = []
    for _ in range(n):
        s = env.reset()
        episode = []
        done = False
        while not done:
            s_next, r, done = env.step()
            episode.append((s, r, s_next, done))
            s = s_next
        episodes.append(episode)
    return episodes


def true_values(env: MarkovChain, gamma: float | None = None) -> np.ndarray:
    """Exact V^pi via the Bellman equations; V = 0 at terminal states."""
    gamma = env.gamma if gamma is None else gamma
    P, r = env.transition_model()
    nt = ~env.terminal_mask
    v = np.zeros(env.num_states)
    v[nt] = np.linalg.solve(np.eye(nt.sum()) - gamma * P[np.ix_(nt, nt)], r[nt])
    return v
