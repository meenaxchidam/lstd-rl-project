"""Member D — figures from results/raw/*.npz (mean +/- standard error over seeds).

    python -m src.plot_results            # -> results/figures/*.png
    python -m src.plot_results --quick    # -> results/quick/figures/*.png

Figures
    fig1_headline.png         RMS error vs #trajectories: LSTD(λ) vs TD(λ) at its best step size
    fig1b_paper_fig5.png      replica of paper Fig. 5 (λ = headline λ, six Eq. 8 schedules)
    fig2_td_alpha.png         TD error vs constant α (step-size sensitivity), one line per λ
    fig3_lstd_lambda.png      LSTD error vs λ at several sample sizes
    fig3b_paper_fig6.png      replica of paper Fig. 6 (error at n_lo -> n_hi for every λ)
    fig4_timing.png           wall-clock time vs feature dimension d
    fig5_ridge_<env>.png      novelty: ridge LSTD error vs ε for very few trajectories
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from src.run_experiment import DEFAULT_CONFIG, figures_dir, load_config, mean_se, raw_dir  # noqa: E402


def _finite(x) -> np.ndarray:
    x = np.array(x, dtype=float)
    x[~np.isfinite(x)] = np.nan
    return x


def _band(ax, x, errs, label, **kwargs):
    """Plot mean over axis 0 with a +/- standard-error band."""
    m, se = mean_se(errs)
    m, se = _finite(m), _finite(se)
    (line,) = ax.plot(x, m, label=label, **kwargs)
    ax.fill_between(x, np.clip(m - se, 1e-12, None), m + se, color=line.get_color(), alpha=0.2, lw=0)
    return line


def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"   saved {path}")
    return path


def _nearest(values, target) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - target)))


def best_td_index(errs: np.ndarray, criterion: str = "auc") -> int:
    """Index of the TD config with the lowest mean error. errs: (seeds, configs, checkpoints).

    auc:   mean error over the whole n_trajectories grid (rewards learning fast)
    final: error at the last checkpoint only
    """
    m = errs.mean(axis=0)
    m = np.where(np.isfinite(m), m, np.inf)
    score = m.mean(axis=1) if criterion == "auc" else m[:, -1]
    return int(np.argmin(score))


# ----------------------------------------------------------------------------- learning curves

def plot_learning_curves(cfg: dict, raw: Path, figs: Path) -> list[Path]:
    path = raw / "learning_curves.npz"
    if not path.exists():
        print(f"   skip learning-curve figures ({path} missing)")
        return []
    data = np.load(path)
    cps, lambdas = data["checkpoints"], data["lambdas"]
    labels, schedules, alphas = data["td_labels"], data["td_schedules"], data["td_alphas"]
    lstd, td = data["lstd_err"], data["td_err"]  # (S, L, C), (S, L, K, C)
    pc = cfg["plots"]
    li = _nearest(lambdas, pc["headline_lambda"])
    lam = lambdas[li]
    const_idx = np.where(schedules == "constant")[0]
    decay_idx = np.where(schedules == "decay")[0]
    saved = []

    # Fig 1: headline
    k = best_td_index(td[:, li], pc["best_td_criterion"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _band(ax, cps, lstd[:, li], f"LSTD(λ={lam:g})", color="k", marker="o", ms=3)
    _band(ax, cps, td[:, li, k], f"TD(λ={lam:g}), best step size: {labels[k][3:]}", color="C3", marker="s", ms=3)
    ax.set(xscale="log", yscale="log", xlabel="number of trajectories", ylabel="RMS error of V",
           title=f"Boyan chain: LSTD vs best-tuned TD ({lstd.shape[0]} seeds, ±1 SE)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    saved.append(_save(fig, figs / "fig1_headline.png"))

    # Fig 1b: paper Fig. 5 replica
    if len(decay_idx):
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for j, kk in enumerate(decay_idx):
            _band(ax, cps, td[:, li, kk], labels[kk], color=f"C{j}")
        _band(ax, cps, lstd[:, li], "Least-Squares TD", color="k", lw=2)
        ax.set(xscale="log", yscale="log", xlabel="trajectory number", ylabel="RMS error of V",
               title=f"Replica of paper Fig. 5 (λ = {lam:g})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        saved.append(_save(fig, figs / "fig1b_paper_fig5.png"))

    summary_n = [n for n in pc["summary_n"] if n <= cps[-1]]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(lambdas)))

    # Fig 2: TD step-size sensitivity
    if len(const_idx) and summary_n:
        fig, axes = plt.subplots(1, len(summary_n), figsize=(5.5 * len(summary_n), 4.2), squeeze=False)
        for ax, n in zip(axes[0], summary_n):
            c = _nearest(cps, n)
            for l, (lam_l, col) in enumerate(zip(lambdas, colors)):
                _band(ax, alphas[const_idx], td[:, l, const_idx, c], f"TD λ={lam_l:g}", color=col, marker="o", ms=3)
                ax.axhline(_finite(lstd[:, l, c].mean()), color=col, ls=":", lw=1)
            ax.set(xscale="log", yscale="log", xlabel="constant step size α", ylabel="RMS error of V",
                   title=f"after {cps[c]} trajectories (dotted = LSTD)")
            ax.grid(True, which="both", alpha=0.3)
        axes[0][0].legend(fontsize=8)
        saved.append(_save(fig, figs / "fig2_td_alpha.png"))

    # Fig 3: LSTD error vs lambda
    lambda_n = [n for n in pc["lambda_plot_n"] if n <= cps[-1]]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for j, n in enumerate(lambda_n):
        c = _nearest(cps, n)
        _band(ax, lambdas, lstd[:, :, c], f"n = {cps[c]}", color=f"C{j}", marker="o", ms=4)
    ax.set(yscale="log", xlabel="λ", ylabel="RMS error of V", title="LSTD(λ): error vs λ")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    saved.append(_save(fig, figs / "fig3_lstd_lambda.png"))

    # Fig 3b: paper Fig. 6 replica
    if len(decay_idx) and len(summary_n) >= 2:
        lo, hi = _nearest(cps, summary_n[0]), _nearest(cps, summary_n[-1])
        algos = [(labels[kk], td[:, :, kk]) for kk in decay_idx] + [("Least-Squares TD", lstd)]
        step = np.min(np.diff(lambdas)) if len(lambdas) > 1 else 1.0
        offsets = np.linspace(-0.35, 0.35, len(algos)) * step
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for j, (label, errs) in enumerate(algos):
            m = _finite(errs.mean(axis=0))  # (L, C)
            color = "k" if j == len(algos) - 1 else f"C{j}"
            ax.vlines(lambdas + offsets[j], m[:, hi], m[:, lo], color=color, lw=3, label=label)
        ax.set(yscale="log", xlabel="λ", ylabel=f"RMS error at traj #{cps[lo]} (top) → #{cps[hi]} (bottom)",
               title="Replica of paper Fig. 6")
        ax.set_xticks(lambdas)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=7)
        saved.append(_save(fig, figs / "fig3b_paper_fig6.png"))
    return saved


# ----------------------------------------------------------------------------- timing

def plot_timing(cfg: dict, raw: Path, figs: Path) -> list[Path]:
    path = raw / "timing.npz"
    if not path.exists():
        print(f"   skip timing figure ({path} missing)")
        return []
    data = np.load(path)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for label, t in zip(data["labels"], data["times"]):
        ax.plot(data["dims"], t, marker="o", label=label)
    ax.set(xscale="log", yscale="log", xlabel="feature dimension d", ylabel="wall-clock time (s)",
           title=f"Cost of {int(data['n_trajectories'])} trajectories vs d")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    return [_save(fig, figs / "fig4_timing.png")]


# ----------------------------------------------------------------------------- ridge novelty

def plot_ridge(cfg: dict, raw: Path, figs: Path) -> list[Path]:
    saved = []
    for env_name in cfg["ridge"]["envs"]:
        path = raw / f"ridge_{env_name}.npz"
        if not path.exists():
            print(f"   skip ridge figure for {env_name} ({path} missing)")
            continue
        data = np.load(path)
        ns, eps = data["n_trajectories"], data["epsilons"]
        plain, ridge, td = data["plain_err"], data["ridge_err"], data["td_err"]
        td_mean = np.where(np.isfinite(td.mean(axis=0)), td.mean(axis=0), np.inf)  # (A, N)

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for ni, n in enumerate(ns):
            color = f"C{ni}"
            _band(ax, eps, ridge[:, ni, :], f"n = {n}", color=color)
            ax.axhline(_finite(plain[:, ni].mean()), color=color, ls=":", lw=1.2)
            ax.axhline(_finite(td_mean[:, ni].min()), color=color, ls="--", lw=1.2)
        style = [Line2D([], [], color="gray", ls=s, label=l)
                 for s, l in (("-", "ridge LSTD"), (":", "plain LSTD (ε = 0)"), ("--", "TD, best α per n"))]
        first = ax.legend(loc="upper left", fontsize=8)
        ax.add_artist(first)
        ax.legend(handles=style, loc="best", fontsize=8)
        ax.set(xscale="log", yscale="log", xlabel="ridge ε", ylabel="RMS error of V",
               title=f"Ridge LSTD(λ={float(data['lam']):g}) on {env_name}, {ridge.shape[0]} seeds")
        ax.grid(True, which="both", alpha=0.3)
        saved.append(_save(fig, figs / f"fig5_ridge_{env_name}.png"))
    return saved


def plot_all(cfg: dict, raw: Path, figs: Path) -> list[Path]:
    figs.mkdir(parents=True, exist_ok=True)
    return plot_learning_curves(cfg, raw, figs) + plot_timing(cfg, raw, figs) + plot_ridge(cfg, raw, figs)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, quick=args.quick)
    plot_all(cfg, raw_dir(cfg), figures_dir(cfg))


if __name__ == "__main__":
    main()
