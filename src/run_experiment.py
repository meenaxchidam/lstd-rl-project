"""Member D — config-driven experiment runner.

Run from the repo root:
    python -m src.run_experiment                          # all experiments
    python -m src.run_experiment --experiment ridge       # one experiment
    python -m src.run_experiment --quick                  # fast smoke run -> results/quick/

Raw per-seed RMS errors are written to <output_dir>/raw/*.npz; aggregation
(mean +/- standard error) and plotting happen in src/plot_results.py.
"""
from __future__ import annotations

import argparse
import copy
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from src.env import BoyanChain, RandomWalk, generate_episodes, true_values
from src.features import boyan_features, random_features, random_walk_features
from src.lstd import lstd_lambda, lstd_lambda_curve, lstd_statistics, recursive_lstd, solve_theta
from src.td_lambda import td_lambda, td_lambda_curve

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "experiment.yaml"
EXPERIMENTS = ("learning_curves", "timing", "ridge")


# ----------------------------------------------------------------------------- metrics

def rmse(v_hat, v_true, mask=None) -> np.ndarray:
    """RMS error over states (last axis); `mask` selects which states count."""
    v_hat = np.asarray(v_hat, dtype=float)
    v_true = np.asarray(v_true, dtype=float)
    if mask is None:
        mask = np.ones(v_true.shape[-1], dtype=bool)
    diff = v_hat[..., mask] - v_true[mask]
    return np.sqrt(np.mean(diff**2, axis=-1))


def mean_se(x, axis: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Mean and standard error along `axis` (the seed axis)."""
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    mean = x.mean(axis=axis)
    se = x.std(axis=axis, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, se


def _diverged_to_inf(err: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(err), err, np.inf)


# ----------------------------------------------------------------------------- config

def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path=DEFAULT_CONFIG, quick: bool = False) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    quick_overrides = cfg.pop("quick", {}) or {}
    if quick:
        _deep_update(cfg, copy.deepcopy(quick_overrides))
    return cfg


def raw_dir(cfg: dict) -> Path:
    return ROOT / cfg["output_dir"] / "raw"


def figures_dir(cfg: dict) -> Path:
    return ROOT / cfg["output_dir"] / "figures"


def make_problem(name: str, cfg: dict) -> dict:
    """Build env, features, true values and the RMSE state mask for `name`."""
    if name == "boyan":
        env, phi = BoyanChain(), boyan_features()
    elif name == "random_walk":
        rw = cfg.get("random_walk", {})
        n = int(rw.get("n", 19))
        env = RandomWalk(n=n)
        phi = random_walk_features(n, kind=rw.get("features", "aggregation"), num_groups=int(rw.get("num_groups", 5)))
    else:
        raise ValueError(f"unknown env {name!r}")

    states = cfg.get("rmse_states", "all")
    if states == "all":
        mask = np.ones(env.num_states, dtype=bool)
    elif states == "nonterminal":
        mask = ~env.terminal_mask
    else:
        raise ValueError(f"rmse_states must be 'all' or 'nonterminal', got {states!r}")
    return {"env": env, "phi": phi, "v_true": true_values(env), "mask": mask, "gamma": env.gamma}


def td_configs(td_cfg: dict) -> list[dict]:
    """Constant-alpha grid followed by the paper's Eq. 8 decaying schedules."""
    configs = [
        {"label": f"TD α={float(a):g}", "alpha": float(a), "schedule": "constant", "n0": None}
        for a in td_cfg.get("constant_alphas", [])
    ]
    for sched in td_cfg.get("decay_schedules", []):
        a0, n0 = float(sched["a0"]), float(sched["n0"])
        label = f"TD a0={a0:g}, n0=10^{int(round(np.log10(n0)))}"
        configs.append({"label": label, "alpha": a0, "schedule": "decay", "n0": n0})
    return configs


def _seeds(cfg: dict, section: dict | None = None) -> list[int]:
    n = int((section or {}).get("n_seeds", cfg["n_seeds"]))
    return [int(cfg["seed_base"]) + i for i in range(n)]


def _parallel_map(fn, jobs: list, workers: int, name: str) -> list:
    start = time.perf_counter()
    results = []
    if workers <= 1:
        iterator = map(fn, jobs)
        for i, res in enumerate(iterator, 1):
            results.append(res)
            print(f"  [{name}] {i}/{len(jobs)} done ({time.perf_counter() - start:.1f}s)", flush=True)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, res in enumerate(pool.map(fn, jobs), 1):
            results.append(res)
            print(f"  [{name}] {i}/{len(jobs)} done ({time.perf_counter() - start:.1f}s)", flush=True)
    return results


# ----------------------------------------------------------------------------- learning curves

def _learning_curve_seed(job: tuple[dict, int]) -> tuple[np.ndarray, np.ndarray]:
    cfg, seed = job
    lc = cfg["learning_curves"]
    prob = make_problem(lc["env"], cfg)
    phi, gamma, v_true, mask = prob["phi"], prob["gamma"], prob["v_true"], prob["mask"]
    cps = [int(n) for n in lc["n_trajectories"]]
    lambdas = [float(lam) for lam in lc["lambdas"]]
    tds = td_configs(lc["td"])
    online = bool(lc["td"].get("online", False))

    episodes = generate_episodes(prob["env"], cps[-1], seed)
    lstd_err = np.empty((len(lambdas), len(cps)))
    td_err = np.empty((len(lambdas), len(tds), len(cps)))
    with np.errstate(all="ignore"):
        for li, lam in enumerate(lambdas):
            thetas = lstd_lambda_curve(episodes, phi, gamma, lam, cps)
            lstd_err[li] = rmse(thetas @ phi.T, v_true, mask)
            for ki, tc in enumerate(tds):
                thetas = td_lambda_curve(
                    episodes, phi, gamma, lam, tc["alpha"], cps,
                    schedule=tc["schedule"], n0=tc["n0"], online=online,
                )
                td_err[li, ki] = rmse(thetas @ phi.T, v_true, mask)
    return _diverged_to_inf(lstd_err), _diverged_to_inf(td_err)


def run_learning_curves(cfg: dict, out_dir: Path, workers: int = 1) -> Path:
    """Errors for LSTD(lambda) and every TD config at every checkpoint, lambda and seed."""
    lc = cfg["learning_curves"]
    seeds = _seeds(cfg, lc)
    results = _parallel_map(_learning_curve_seed, [(cfg, s) for s in seeds], workers, "learning_curves")
    tds = td_configs(lc["td"])
    path = out_dir / "learning_curves.npz"
    np.savez(
        path,
        checkpoints=np.array(lc["n_trajectories"], dtype=int),
        lambdas=np.array(lc["lambdas"], dtype=float),
        seeds=np.array(seeds),
        td_labels=np.array([tc["label"] for tc in tds]),
        td_alphas=np.array([tc["alpha"] for tc in tds]),
        td_schedules=np.array([tc["schedule"] for tc in tds]),
        td_n0=np.array([np.nan if tc["n0"] is None else tc["n0"] for tc in tds]),
        lstd_err=np.stack([r[0] for r in results]),   # (seeds, lambdas, checkpoints)
        td_err=np.stack([r[1] for r in results]),     # (seeds, lambdas, td_configs, checkpoints)
    )
    return path


# ----------------------------------------------------------------------------- timing

def _best_time(fn, repeats: int) -> float:
    best = np.inf
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run_timing(cfg: dict, out_dir: Path) -> Path:
    """Wall-clock time vs feature dimension d on the Boyan chain with random features."""
    tc = cfg["timing"]
    env = BoyanChain()
    n = int(tc["n_trajectories"])
    lam, reg = float(tc["lam"]), float(tc["reg"])
    alpha, repeats = float(tc["td_alpha"]), int(tc["repeats"])
    episodes = generate_episodes(env, n, cfg["seed_base"])
    gamma = env.gamma

    algorithms = {
        "TD(λ)": lambda phi: td_lambda(episodes, phi, gamma, lam, alpha),
        "LSTD(λ), one solve at end": lambda phi: lstd_lambda(episodes, phi, gamma, lam, reg=reg),
        "LSTD(λ), solve after every trajectory": lambda phi: lstd_lambda_curve(
            episodes, phi, gamma, lam, range(1, n + 1), reg=reg
        ),
        "Recursive LSTD(λ) (Sherman–Morrison)": lambda phi: recursive_lstd(episodes, phi, gamma, lam, delta=1.0 / reg),
    }
    dims = [int(d) for d in tc["dims"]]
    times = np.empty((len(algorithms), len(dims)))
    for j, d in enumerate(dims):
        phi = random_features(env.num_states, d, seed=cfg["seed_base"])
        for i, fn in enumerate(algorithms.values()):
            times[i, j] = _best_time(lambda: fn(phi), repeats)
        print(f"  [timing] d={d}: " + ", ".join(f"{t:.3f}s" for t in times[:, j]), flush=True)

    path = out_dir / "timing.npz"
    np.savez(path, dims=np.array(dims), labels=np.array(list(algorithms)), times=times, n_trajectories=n)
    return path


# ----------------------------------------------------------------------------- ridge novelty

def _ridge_seed(job: tuple[dict, str, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cfg, env_name, seed = job
    rc = cfg["ridge"]
    prob = make_problem(env_name, cfg)
    phi, gamma, v_true, mask = prob["phi"], prob["gamma"], prob["v_true"], prob["mask"]
    ns = [int(n) for n in rc["n_trajectories"]]
    epsilons = [float(e) for e in rc["epsilons"]]
    alphas = [float(a) for a in rc["td_alphas"]]
    lam = float(rc["lam"])

    episodes = generate_episodes(prob["env"], max(ns), seed)
    plain = np.empty(len(ns))
    ridge = np.empty((len(ns), len(epsilons)))
    td = np.empty((len(alphas), len(ns)))
    with np.errstate(all="ignore"):
        for ni, n in enumerate(ns):
            A, b = lstd_statistics(episodes[:n], phi, gamma, lam)
            plain[ni] = rmse(phi @ solve_theta(A, b), v_true, mask)
            for ei, eps in enumerate(epsilons):
                ridge[ni, ei] = rmse(phi @ solve_theta(A, b, eps), v_true, mask)
        for ai, alpha in enumerate(alphas):
            thetas = td_lambda_curve(episodes, phi, gamma, lam, alpha, ns)
            td[ai] = rmse(thetas @ phi.T, v_true, mask)
    return _diverged_to_inf(plain), _diverged_to_inf(ridge), _diverged_to_inf(td)


def run_ridge(cfg: dict, out_dir: Path, workers: int = 1) -> list[Path]:
    """Novelty: ridge LSTD error vs epsilon at very small sample sizes, per environment."""
    rc = cfg["ridge"]
    seeds = _seeds(cfg, rc)
    paths = []
    for env_name in rc["envs"]:
        jobs = [(cfg, env_name, s) for s in seeds]
        results = _parallel_map(_ridge_seed, jobs, workers, f"ridge/{env_name}")
        path = out_dir / f"ridge_{env_name}.npz"
        np.savez(
            path,
            n_trajectories=np.array(rc["n_trajectories"], dtype=int),
            epsilons=np.array(rc["epsilons"], dtype=float),
            td_alphas=np.array(rc["td_alphas"], dtype=float),
            lam=float(rc["lam"]),
            plain_err=np.stack([r[0] for r in results]),   # (seeds, n)
            ridge_err=np.stack([r[1] for r in results]),   # (seeds, n, epsilons)
            td_err=np.stack([r[2] for r in results]),      # (seeds, alphas, n)
        )
        paths.append(path)
    return paths


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment", choices=(*EXPERIMENTS, "all"), default="all")
    parser.add_argument("--quick", action="store_true", help="apply the `quick:` overrides from the config")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    cfg = load_config(args.config, quick=args.quick)
    out_dir = raw_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = EXPERIMENTS if args.experiment == "all" else (args.experiment,)

    for name in todo:
        print(f"== {name}", flush=True)
        start = time.perf_counter()
        if name == "learning_curves":
            saved = [run_learning_curves(cfg, out_dir, args.workers)]
        elif name == "timing":
            saved = [run_timing(cfg, out_dir)]
        else:
            saved = run_ridge(cfg, out_dir, args.workers)
        for p in saved:
            print(f"   saved {p}")
        print(f"   ({time.perf_counter() - start:.1f}s)")


if __name__ == "__main__":
    main()
