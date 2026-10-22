# E1 277 Course Project Plan — Least-Squares TD Learning (Paper #22)

**Paper:** J. Boyan, *Least-Squares Temporal Difference Learning*, ICML 1999\
**Team size:** 4 &nbsp;|&nbsp; **Presentation:** Nov 15–19 (20-min slot incl. Q&A)\
**Software:** Python 3, NumPy, Matplotlib, PyYAML (all free) &nbsp;|&nbsp; **Hardware:** any laptop, CPU-only

---

## 1. Project Goal

1. **Reimplement LSTD(λ)** and reproduce the paper's main claim: LSTD gives more accurate value estimates from fewer trajectories than TD(λ), with no step size to tune.
2. **Add one small novelty** (bonus): ridge-regularized LSTD in the low-data regime.
3. **Present** — each member presents the part they built.

---

## 2. Core Equations (everyone uses these)

Features $\phi(s) \in \mathbb{R}^d$, discount $\gamma$, trace parameter $\lambda$. For each transition $(s_t, r_t, s_{t+1})$ in an episode (reset $z = 0$ at episode start):

**LSTD(λ)**

$$z_t = \gamma\lambda\, z_{t-1} + \phi(s_t)$$
$$A \mathrel{+}= z_t\,\big(\phi(s_t) - \gamma\,\phi(s_{t+1})\big)^\top \qquad b \mathrel{+}= z_t\, r_t$$
$$\theta = A^{-1} b \qquad \hat V(s) = \phi(s)^\top\theta$$

(If $s_{t+1}$ is terminal, use $\phi(s_{t+1}) = 0$.)

**TD(λ) baseline**

$$\delta_t = r_t + \gamma\,\phi(s_{t+1})^\top\theta - \phi(s_t)^\top\theta \qquad \theta \mathrel{+}= \alpha\,\delta_t\,z_t$$

**Metric:** RMS error between $\hat V = \Phi\theta$ and true $V^\pi$ over non-terminal states, plotted vs. number of trajectories.

---

## 3. Fixed Environment Spec (so nobody waits on anybody)

**Boyan chain** (✅ verified against paper Fig. 4 and Sec. 4 — frozen):

- States 12 → 0; state 0 is terminal. Start at state 12.
- From state $s \ge 2$: move to $s-1$ or $s-2$ with probability 0.5 each, reward **−3**.
- From state 1: move to 0 with probability 1, reward **−2**.
- $\gamma = 1$ (episodic).
- **Features (d = 4):** $\phi(12)=[1,0,0,0]$, $\phi(8)=[0,1,0,0]$, $\phi(4)=[0,0,1,0]$, $\phi(0)=[0,0,0,1]$; states in between are linearly interpolated (e.g. $\phi(10) = [0.5, 0.5, 0, 0]$). The true $V^\pi$ is exactly representable.

**Paper details to keep in mind** (checked against `boyanLSTDlambda.pdf`):

- $V^\pi(s) = -2s$; optimal coefficients are $\theta^* = (-24, -16, -8, 0)$ (Sec. 4).
- Paper's TD(λ) (Fig. 1) is **offline**: updates are summed over a trajectory and applied at its end. Our `td_lambda` does this by default (`online=False`); `online=True` gives per-step updates.
- Paper's step sizes (Eq. 8): $\alpha_n = a_0 \frac{n_0+1}{n_0+n}$ per trajectory $n$, with $a_0 \in \{0.1, 0.01\}$, $n_0 \in \{10^2, 10^3, 10^6\}$ — this is our `schedule="decay"`.
- Paper inverts $A$ with SVD; we use `np.linalg.solve` and fall back to SVD least-squares (`lstsq`) only when $A$ is exactly singular.
- Paper's metric is RMS error **over all states** (incl. terminal state 0); config `rmse_states: all|nonterminal` switches between paper and Sec. 2 definitions.
- Paper averages 10 trials, λ ∈ {0, 0.2, …, 1.0}, up to 10,000 trajectories; headline plot uses λ = 0.4.
- LSTD(1) = Monte-Carlo linear regression; LSTD(0) with tabular features = model-based $(N-C)^{-1}s$ (Sec. 3). Both are unit-tested.

---

## 4. Shared Interfaces (frozen in Week 1)

```python
# Data format: an episode is a list of transitions
#   (s: int, r: float, s_next: int, done: bool)
episodes: list[list[tuple[int, float, int, bool]]]

# Member A
env = BoyanChain()
phi = boyan_features()                       # np.ndarray, shape (num_states, d)
episodes = generate_episodes(env, n, seed)   # list of episodes
v_true = true_values(env)                    # np.ndarray, shape (num_states,)

# Member B
theta = td_lambda(episodes, phi, gamma, lam, alpha)    # np.ndarray, shape (d,)

# Member C
theta = lstd_lambda(episodes, phi, gamma, lam, reg=0.0)  # np.ndarray, shape (d,)
theta = recursive_lstd(episodes, phi, gamma, lam, delta=1e3)

# Member D
err = rmse(phi @ theta, v_true)
```

**Repo layout**

```
lstd-project/
├── configs/experiment.yaml
├── src/
│   ├── env.py              # A
│   ├── features.py         # A
│   ├── td_lambda.py        # B
│   ├── lstd.py             # C
│   ├── run_experiment.py   # D
│   └── plot_results.py     # D
├── tests/
│   ├── test_env.py         # A
│   ├── test_td.py          # B
│   └── test_lstd.py        # C
├── results/raw/  results/figures/
└── slides/
```

---

## 5. Work Split — All Four Tracks Run in Parallel

Each track depends only on the frozen spec (Sections 2–4), not on another member's code. Until real modules exist, use the mock episode below.

**Shared mock (copy into your tests until Member A's code lands):**

```python
# Deterministic 3-state chain: 2 -> 1 -> 0(terminal), reward -1 per step, gamma = 1
mock_phi = np.eye(3)
mock_episodes = [[(2, -1.0, 1, False), (1, -1.0, 0, True)]] * 50
mock_v_true = np.array([0.0, -1.0, -2.0])
```

### Member A — Environments, Features & Ground Truth
| Task | Done when |
|---|---|
| `BoyanChain` class (`reset`, `step`) per Section 3 | Episodes always end at state 0 |
| `boyan_features()` with linear interpolation | Rows sum to 1; anchors match spec |
| `generate_episodes(env, n, seed)` | Reproducible with same seed |
| `true_values(env)` via exact Bellman solve (`np.linalg.solve`) | Matches hand-computed values for states 1, 2 |
| Second environment: 19-state random walk (Sutton & Barto) + features | Used by D for novelty / generalization |
| `test_env.py` | All tests pass |

### Member B — TD(λ) Baseline
| Task | Done when |
|---|---|
| Linear TD(λ) with accumulating traces | Converges on mock chain to `mock_v_true` |
| Constant and decaying step-size schedules | Both selectable via argument |
| Step-size helper for sweeps (α grid) | Returns θ for each α |
| `test_td.py` (mock chain) | RMSE < 0.05 after 50 episodes |

### Member C — LSTD(λ) Core
| Task | Done when |
|---|---|
| Batch `lstd_lambda` (use `np.linalg.solve`, not `inv`) | Exact on mock chain |
| `reg` argument: solve $(A + \epsilon I)\theta = b$ | Needed for the novelty experiment |
| `recursive_lstd` using Sherman–Morrison ($A^{-1}$ updated per step, $O(d^2)$) | Matches batch LSTD to 1e-6 on mock chain |
| `test_lstd.py` | All tests pass |

### Member D — Experiments, Plots & Novelty
| Task | Done when |
|---|---|
| Config-driven runner (seeds, n_trajectories grid, λ grid, α grid) | Runs end-to-end with stub algorithms |
| `rmse` + mean ± std-error aggregation over 20–50 seeds | Saved to `results/raw/*.npy` |
| Plotting script for all figures below | PNGs saved to `results/figures/` |
| Novelty experiment script (ridge LSTD) | Ready to run once C's `reg` lands |

**Figures to produce**

1. **Headline:** RMS error vs. #trajectories — LSTD(λ) vs. TD(λ) at its best α
2. TD final error vs. α (step-size sensitivity)
3. LSTD final error vs. λ
4. Wall-clock time vs. feature dimension $d$ (batch LSTD vs. recursive LSTD vs. TD)
5. **Novelty:** ridge LSTD — error vs. $\epsilon$ for very few trajectories (1, 2, 5, 10)

---

## 6. Novelty (Bonus)

**Ridge-regularized LSTD in the low-data regime.** With very few trajectories, $A$ can be singular or ill-conditioned and plain LSTD becomes unstable. We add $\epsilon I$ and study:

- Error vs. $\epsilon$ for small sample sizes
- Whether a small $\epsilon$ lets LSTD beat TD even at 1–5 trajectories
- Behavior on the 19-state random walk with weaker features (generalization beyond the paper)

---

## 7. Timeline

| Week | Dates | Milestone |
|---|---|---|
| 1 | Oct 10–16 | Everyone reads the paper. Verify Section 3 constants; freeze Sections 2–4. Repo created. **All four tracks start.** |
| 2 | Oct 17–23 | Each member finishes their module + tests against the mock data. |
| 3 | Oct 24–30 | **Integration:** D swaps stubs for real modules. Reproduce headline figure. |
| 4 | Oct 31–Nov 6 | Sweeps + novelty experiment. Code freeze Nov 6. |
| 5 | Nov 7–13 | Each member builds their own slides; 2 full timed rehearsals. |
| — | Nov 15–19 | Presentation. |

**Sync points:** short check-in every week; integration meeting at start of Week 3.

---

## 8. Presentation Split (20 min total)

| Time | Speaker | Content (~3–4 slides each) |
|---|---|---|
| 0:00–4:00 | **A** | Motivation: policy evaluation; value functions & linear FA; Boyan chain + features |
| 4:00–8:00 | **B** | TD(λ) & eligibility traces; weaknesses (step-size sensitivity, data inefficiency); α-sensitivity plot |
| 8:00–12:00 | **C** | LSTD idea: solve the TD fixed point directly; derive $A\theta = b$; recursive version; complexity $O(d^2)$/$O(d^3)$ vs. $O(d)$ |
| 12:00–16:00 | **D** | Headline result, λ and timing plots, **novelty result**, conclusions & limitations |
| 16:00–20:00 | All | Q&A — each member handles questions on their section |

**Likely questions to prepare**

- Why is LSTD more data-efficient than TD?
- What happens when $A$ is singular?
- How does cost scale with $d$?
- What is LSTD with λ = 1 equivalent to? (Monte Carlo regression)
- Can LSTD be used for control? (Yes — LSPI)

---

## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| $A$ singular with few trajectories | Ridge $\epsilon I$ (also the novelty) |
| Unfair TD comparison | Always compare against TD's best α from the sweep |
| Noisy curves | ≥ 20 seeds, plot ± standard error |
| Integration breakage | Frozen interfaces + shared mock data + per-module tests |
| A member falls behind | Mock data means no one is blocked; reassign at weekly sync |
