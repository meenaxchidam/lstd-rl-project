# LSTD(λ) Reimplementation — E1 277 Course Project (Paper #22)

Reimplementation of J. Boyan, *Least-Squares Temporal Difference Learning*, ICML 1999 ([boyanLSTDlambda.pdf](boyanLSTDlambda.pdf)).
Full plan, interfaces and timeline: [LSTD_Project_Plan.md](LSTD_Project_Plan.md).
New to TD/LSTD? Start with [docs/MC_vs_TD_vs_LSTD.md](docs/MC_vs_TD_vs_LSTD.md) (worked example, no heavy maths).

**Claim reproduced:** on the 13-state Boyan chain, LSTD(λ) has lower value-function error than TD(λ) for every λ and every step-size schedule we tried, at every number of trajectories, and LSTD has no step size to tune.
**Novelty:** ridge-regularised LSTD, $(A + \epsilon I)\theta = b$, with 1–10 trajectories, run on the Boyan chain and on a 19-state random walk with aggregated features.

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # all module tests (~10 s)
python -m src.run_experiment --quick     # smoke run -> results/quick/
python -m src.plot_results   --quick

python -m src.run_experiment             # full run (~30 s on 8 cores) -> results/raw/*.npz
python -m src.plot_results               # -> results/figures/*.png
python -m src.run_experiment --experiment ridge   # just one: learning_curves | timing | ridge
```

All experiment settings live in [configs/experiment.yaml](configs/experiment.yaml).

## Repo layout

```
configs/experiment.yaml     all experiment settings (D)
src/env.py                  BoyanChain, RandomWalk, generate_episodes, true_values   (A)
src/features.py             boyan_features, random_walk_features, random_features    (A)
src/td_lambda.py            td_lambda, td_lambda_curve, td_alpha_sweep, step_size    (B)
src/lstd.py                 lstd_lambda, lstd_lambda_curve, recursive_lstd, ...      (C)
src/run_experiment.py       rmse, mean_se, experiment runner + CLI                   (D)
src/plot_results.py         all figures                                              (D)
tests/conftest.py           shared mock chain (plan Sec. 5) as pytest fixtures
tests/test_env.py (A)  tests/test_td.py (B)  tests/test_lstd.py (C)  tests/test_experiment.py (D)
results/raw/  results/figures/  slides/
```

Shared data format (frozen): `episodes: list[list[tuple[int, float, int, bool]]]`, i.e. `(s, r, s_next, done)`; when `done` is True, $\phi(s_{t+1}) = 0$.

---

## Per-member guide — read your section, check it, change what you need

Each member owns their files below. Before your weekly sync, check that your code matches the paper. Tweak anything you disagree with, then rerun `pytest` and the quick run.

### Member A — Environments, features, ground truth
**Files:** [src/env.py](src/env.py), [src/features.py](src/features.py), [tests/test_env.py](tests/test_env.py)

- Each environment is defined by an explicit transition model `P[s, s']`, `R[s, s']` (`MarkovChain`). Both `step()` and `true_values()` read that same model, so the simulator and the ground truth always agree.
- `BoyanChain`: the constants are checked against paper Fig. 4. The tests check $V(s) = -2s$ and that $\theta^* = (-24,-16,-8,0)$ reproduces $V^\pi$ exactly (paper Sec. 4).
- `RandomWalk(n=19)` (Sutton & Barto Ex. 7.1): reward −1 on the left end and +1 on the right end, so $V(s) = (s-10)/10$. `random_walk_features(kind="aggregation", num_groups=5)` gives the weak features used in the novelty experiment. `kind="tabular"` gives exact features.
- **To check / tweak:** the random-walk reward convention (some editions use 0 on the left end), the number of aggregation groups, and whether you want extra feature sets.
- **Slides (0:00–4:00):** policy evaluation, linear FA $\hat V = \Phi\theta$, the Boyan chain diagram, and why its features represent $V^\pi$ exactly.

### Member B — TD(λ) baseline
**Files:** [src/td_lambda.py](src/td_lambda.py), [tests/test_td.py](tests/test_td.py)

- Accumulating traces. `online=False` (the default) matches paper Fig. 1: updates are summed over a trajectory and applied at its end. `online=True` updates after every step.
- `schedule="constant"` or `schedule="decay"`. The decay schedule is paper Eq. 8: $\alpha_n = a_0 (n_0+1)/(n_0+n)$.
- `td_lambda_curve(...)` runs once and returns θ at several trajectory counts (the runner uses this). If θ diverges, the remaining rows are NaN.
- **To check / tweak:** the α grid in the config (`learning_curves.td.constant_alphas`). Also compare online vs. offline TD: set `online: true` in the config and rerun.
- **Slides (4:00–8:00):** eligibility traces, the TD update, `fig2_td_alpha.png` (step-size sensitivity), and `fig1b_paper_fig5.png` (all six schedules lose to LSTD).

### Member C — LSTD(λ) core
**Files:** [src/lstd.py](src/lstd.py), [tests/test_lstd.py](tests/test_lstd.py)

- `lstd_lambda` builds $A, b$ as in paper Fig. 2 and solves with `np.linalg.solve`. If $A$ is exactly singular, it falls back to SVD `lstsq`, as the paper does. `reg=ε` solves $(A+\epsilon I)\theta=b$.
- `recursive_lstd` maintains $A^{-1}$ with Sherman–Morrison at $O(d^2)$ per step. It starts from $A_0 = I/\delta$, so its result equals `lstd_lambda(..., reg=1/δ)` exactly. The tests check this, and also that it matches plain LSTD to 1e-6 with δ = 1e8.
- The tests also check two equivalences from the paper: **LSTD(1) = Monte-Carlo linear regression**, and **tabular LSTD(0) = the model-based solution $(N-C)^{-1}s$** (Sec. 3). Both make good Q&A material.
- **Slides (8:00–12:00):** the TD fixed point $d + C\theta = 0$, which gives $A\theta=b$; the model-based interpretation; recursive LSTD; and per-step cost: $O(d)$ for TD, $O(d^2)$ for LSTD accumulation, $O(d^3)$ per solve (`fig4_timing.png`).

### Member D — Experiments, plots, novelty
**Files:** [src/run_experiment.py](src/run_experiment.py), [src/plot_results.py](src/plot_results.py), [configs/experiment.yaml](configs/experiment.yaml), [tests/test_experiment.py](tests/test_experiment.py)

- For each seed, every algorithm sees the same episodes (common random numbers). Raw per-seed errors are saved to `results/raw/*.npz`. Plots show the mean ± 1 standard error.
- Fig. 1 uses the TD config (constant α or Eq. 8 schedule) with the lowest mean error over the whole trajectory grid (`plots.best_td_criterion: auc`). Set it to `final` to choose by the error after the last trajectory instead.
- `rmse_states: all` follows the paper ("over all states"). `nonterminal` follows plan Sec. 2.
- **To check / tweak:** number of seeds, the ε grid, and whether ε should scale with the number of transitions. $A$ is a *sum* over transitions, so a fixed ε regularises less as $n$ grows.
- **Slides (12:00–16:00):** the headline figure, λ and timing plots, the novelty result, and limitations.

| Figure | File |
|---|---|
| 1. Headline: RMS error vs #trajectories, LSTD vs best-tuned TD | `fig1_headline.png` |
| 1b. Replica of paper Fig. 5 (λ = 0.4, six Eq. 8 schedules) | `fig1b_paper_fig5.png` |
| 2. TD error vs α | `fig2_td_alpha.png` |
| 3. LSTD error vs λ | `fig3_lstd_lambda.png` |
| 3b. Replica of paper Fig. 6 | `fig3b_paper_fig6.png` |
| 4. Wall-clock time vs d | `fig4_timing.png` |
| 5. Novelty: ridge LSTD error vs ε | `fig5_ridge_boyan.png`, `fig5_ridge_random_walk.png` |

---

## First full run (20 seeds; 50 seeds for the ridge experiment) — results to discuss at the sync

- **Headline:** with λ = 0.4, LSTD error falls from about 1.6 (1 trajectory) to about 0.016 (10⁴ trajectories). The best-tuned TD (constant α = 0.3) plateaus at about 0.7. The Fig. 6 replica shows LSTD has the lowest error after 10⁴ trajectories at every λ. After 100 trajectories it beats every decaying schedule; Fig. 2 shows constant-α TD is at least 2× worse than LSTD at that point.
- **λ:** λ barely affects LSTD on this chain, which matches the paper.
- **Ridge, Boyan chain:** almost no gain (1–3 % at ε ≈ 0.03–0.1). $A$ is already well conditioned here, and larger ε shrinks θ toward 0, which hurts because $V^\pi$ goes down to −24. Plain LSTD already beats TD at n = 1–10 (e.g. 1.7 vs 6.5 at n = 1). The best TD α is at the edge of the grid (α = 1), so D should extend `ridge.td_alphas`.
- **Ridge, random walk (aggregated features):** ε ≈ 1–3 nearly halves the error at n = 1 (0.83 → 0.44). That matches the best-tuned TD (0.43) at n = 1 and slightly beats it at n = 2 and 5 (0.41 vs 0.43, 0.37 vs 0.39). At n = 10, TD is marginally better (0.33 vs 0.34).
