"""Refresh Member B's three raster slides inside the image-based PPTX.

The deck's pages are full-slide PNGs. This script replaces pages 6–8 using
the saved quick-run arrays, so updating these pages does not require a TeX
installation. The corresponding Beamer source remains in lstd_presentation.tex.
"""
from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/lstd-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "slides" / "lstd_presentation.pptx"
DATA = ROOT / "results" / "raw" / "learning_curves.npz"
PAPER_FIG = ROOT / "results" / "figures" / "fig1b_paper_fig5.png"
INK = "#1E2A3A"
BODY = "#2B2F36"
ACCENT = "#E4572E"
GREEN = "#1F7A6D"
MUTED = "#758092"
PALE = "#F2F7F6"


def _base(title: str, page: int):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor="white")
    fig.text(0.07, 0.94, "02   ━━━   TD(λ)", color=ACCENT, fontsize=11, weight="bold")
    fig.text(0.07, 0.875, title, color=INK, fontsize=23, weight="bold")
    fig.text(0.93, 0.94, "MEMBER B", color=GREEN, fontsize=10, weight="bold", ha="right")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.075, 0.075], color="#D9DDE3", lw=1))
    fig.text(0.07, 0.04, "LSTD(λ)  ·  Boyan, ICML 1999", color=MUTED, fontsize=8)
    fig.text(0.93, 0.04, f"{page} / 20", color=INK, fontsize=9, weight="bold", ha="right")
    return fig


def _card(fig, rect, title: str, color: str):
    x, y, w, h = rect
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        transform=fig.transFigure, facecolor=PALE if color == GREEN else "#FCF4F1",
        edgecolor="none", zorder=-1,
    )
    fig.add_artist(patch)
    fig.text(x + 0.025, y + h - 0.06, title, color=color, fontsize=14, weight="bold")


def slide_method() -> plt.Figure:
    fig = _base("How TD updates, and how it differs from MC", 7)
    fig.text(0.08, 0.79, "For a fixed policy π, the value of state s is its expected discounted future reward:",
             color=BODY, fontsize=11.5)
    fig.text(0.5, 0.735,
             r"$V^\pi(s)=\mathbb{E}_\pi\!\left[\sum_{k=t}^{T-1}\gamma^{k-t}r_k\mid s_t=s\right]$",
             color=INK, fontsize=15, ha="center")
    fig.text(0.5, 0.65,
             r"$\hat V(s)=\phi(s)^\top\theta,\quad "
             r"\delta_t=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t),\quad "
             r"\theta\leftarrow\theta+\alpha\delta_t\phi(s_t)$",
             color=INK, fontsize=15, ha="center")

    _card(fig, (0.08, 0.29, 0.39, 0.29), "Monte Carlo (MC)", ACCENT)
    fig.text(0.11, 0.46, r"Target: $G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k$", color=BODY, fontsize=12)
    fig.text(0.11, 0.40, "Wait until the episode ends; use the full observed return.", color=BODY, fontsize=10.5)
    fig.text(0.11, 0.33, "Unbiased return target; often higher variance.", color=ACCENT, fontsize=10, weight="bold")

    _card(fig, (0.53, 0.29, 0.39, 0.29), "Temporal Difference (TD)", GREEN)
    fig.text(0.56, 0.46, r"Target: $r_t+\gamma\hat V(s_{t+1})$", color=BODY, fontsize=12)
    fig.text(0.56, 0.40, "Can update before termination; bootstraps from its estimate.", color=BODY, fontsize=10.5)
    fig.text(0.56, 0.33, "Often lower variance; biased while estimates are wrong.", color=GREEN, fontsize=10, weight="bold")

    fig.text(0.08, 0.20,
             "Worked example (TD(0), one-hot state): current estimate = 2; reward = 1; next estimate = 4; γ = 0.9.",
             color=INK, fontsize=9.5)
    fig.text(0.08, 0.15,
             r"TD target $=1+0.9(4)=4.6$; error $\delta=4.6-2=2.6$. With $\alpha=0.1$: "
             r"new estimate $=2+0.1(2.6)=2.26$.",
             color=INK, fontsize=9.5)
    fig.text(0.08, 0.105,
             r"TD($\lambda$) trace: $z_t=\gamma\lambda z_{t-1}+\phi(s_t)$; our code sums $\delta_tz_t$ and updates at episode end.",
             color=MUTED, fontsize=8.2)
    return fig


def slide_overview() -> plt.Figure:
    fig = _base("TD: what it is and what we tested", 6)
    fig.text(0.08, 0.79,
             "Temporal Difference (TD) estimates a state's expected future reward from sampled experience. "
             "It updates after a transition using the reward and its estimate of the next state.",
             color=BODY, fontsize=12, wrap=True)
    fig.text(0.5, 0.68,
             r"TD target $=r_t+\gamma\hat V(s_{t+1})$; "
             r"error $\delta_t=\text{target}-\hat V(s_t)$",
             color=INK, fontsize=16, ha="center")

    _card(fig, (0.08, 0.35, 0.39, 0.24), "What we did", GREEN)
    fig.text(0.11, 0.47,
             "Ran TD(λ) and LSTD(λ) on the 13-state Boyan chain.\n"
             "Varied the number of trajectories and TD step-size settings.",
             color=BODY, fontsize=10.2, linespacing=1.5)
    _card(fig, (0.53, 0.35, 0.39, 0.24), "Main result", ACCENT)
    fig.text(0.56, 0.47,
             "At 100 trajectories: LSTD 0.152; best paper-style TD 0.308.\n"
             "At 10,000: LSTD 0.0157; TD 0.0362.",
             color=BODY, fontsize=10.2, linespacing=1.5)

    fig.text(0.08, 0.25,
             "We measure RMSE over all 13 states; lower is better. Full run: 20 seeds, "
             "λ = 0.4, γ = 1, up to 10,000 trajectories.",
             color=MUTED, fontsize=9.5)
    fig.text(0.08, 0.17,
             "Next: the TD update, how it differs from Monte Carlo, and a worked example.",
             color=INK, fontsize=10, weight="bold")
    return fig


def _mean_se(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    count = np.sum(np.isfinite(values), axis=0)
    se = np.nanstd(values, axis=0, ddof=1) / np.sqrt(np.maximum(count, 1))
    return mean, se


def slide_results(data: np.lib.npyio.NpzFile) -> plt.Figure:
    fig = _base("Results vs Boyan (1999), Figure 5", 8)
    cps = data["checkpoints"]
    lambdas = data["lambdas"]
    li = int(np.argmin(np.abs(lambdas - 0.4)))
    is_decay = data["td_schedules"] == "decay"
    td = data["td_err"][:, li, is_decay, :]
    lstd = data["lstd_err"][:, li, :]

    ax = fig.add_axes([0.08, 0.20, 0.52, 0.56])
    if PAPER_FIG.exists():
        ax.imshow(plt.imread(PAPER_FIG))
        ax.axis("off")
    else:
        raise SystemExit(f"Missing paper comparison plot: {PAPER_FIG}")

    fig.text(0.63, 0.78, "Experiment setup", color=INK, fontsize=11, weight="bold")
    fig.text(0.63, 0.73,
             "Boyan: 13 states, 4 features; γ=1, λ=0.4.\n"
             "Paper: 10 trials, 6 decays. Ours: 20 seeds,\n"
             "same 6 decays + 6 constant α settings; RMSE all states.",
             color=BODY, fontsize=7.6, linespacing=1.35)
    fig.text(0.63, 0.62, "Our full-run results (mean ± SE)", color=INK, fontsize=10, weight="bold")
    fig.text(0.80, 0.57, "100 traj.", color=MUTED, fontsize=8.5, ha="center")
    fig.text(0.91, 0.57, "10,000 traj.", color=MUTED, fontsize=8.5, ha="center")
    row_y = [0.51, 0.44]
    for y, label, values in zip(row_y,
                                ["Best TD decay", "LSTD(λ)"],
                                [td, lstd[:, None, :]]):
        fig.text(0.63, y, label, color=BODY, fontsize=8.7)
        vals = []
        for n in [100, 10000]:
            c = int(np.argmin(np.abs(cps - n)))
            arr = values[:, :, c].reshape(values.shape[0], -1)
            ix = int(np.nanargmin(np.nanmean(arr, axis=0))) if label.startswith("Best") else 0
            mean, se = _mean_se(arr[:, ix:ix+1])
            vals.append(f"{mean[0]:.3f}±{se[0]:.3f}")
        fig.text(0.80, y, vals[0], color=BODY, fontsize=8.8, ha="center")
        fig.text(0.91, y, vals[1], color=BODY, fontsize=8.8, ha="center")

    fig.text(0.63, 0.34, "What this means", color=GREEN, fontsize=10, weight="bold")
    fig.text(0.63, 0.29,
             "Best TD schedule is picked separately at each checkpoint (favors TD).\n"
             "LSTD is still lower at both, matching Boyan's Figure 5 conclusion.",
             color=BODY, fontsize=7.2, linespacing=1.4)
    fig.text(0.63, 0.16,
             "Mean ± SE over 20 seeds. Boyan reports 10-trial curves, not exact table values.\n"
             "We compare the same 100 and 10,000 trajectory checkpoints.",
             color=MUTED, fontsize=6.8, wrap=True)
    return fig


def main() -> None:
    if not PPTX.exists() or not DATA.exists() or not PAPER_FIG.exists():
        raise SystemExit("Run the full experiment and plot commands before updating Member B's slides.")
    data = np.load(DATA)
    figures = [slide_overview(), slide_method(), slide_results(data)]
    with tempfile.TemporaryDirectory() as temp:
        images = []
        for i, fig in enumerate(figures, start=6):
            path = Path(temp) / f"slide{i}.png"
            fig.savefig(path, dpi=150, facecolor="white")
            plt.close(fig)
            images.append(path)
        output = Path(temp) / "updated.pptx"
        with zipfile.ZipFile(PPTX, "r") as original, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as updated:
            replacements = {f"ppt/media/image{i}.png": img for i, img in zip(range(6, 9), images)}
            for item in original.infolist():
                if item.filename in replacements:
                    updated.write(replacements[item.filename], item.filename)
                else:
                    updated.writestr(item, original.read(item.filename))
        os.replace(output, PPTX)
    print(f"Updated Member B slides 6–8 in {PPTX}")


if __name__ == "__main__":
    main()
