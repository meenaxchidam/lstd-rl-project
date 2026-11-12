"""Refresh Member B's MC-to-TD slides in the full presentation."""
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

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "slides/lstd_presentation.pptx"
INK, BODY, ORANGE, GREEN, MUTED = "#1E2A3A", "#2B2F36", "#E4572E", "#1F7A6D", "#758092"


def base(title: str, page: int, number: bool = True):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor="white")
    fig.text(.07,.94,"02   ━━━   TD(λ)",color=ORANGE,fontsize=11,weight="bold")
    fig.text(.07,.875,title,color=INK,fontsize=22,weight="bold")
    fig.text(.93,.94,"MEMBER B",color=GREEN,fontsize=10,weight="bold",ha="right")
    fig.add_artist(plt.Line2D([.07,.93],[.075,.075],color="#D9DDE3",lw=1))
    fig.text(.07,.04,"TD(λ)  ·  Member B",color=MUTED,fontsize=8)
    if number:
        fig.text(.93,.04,f"{page} / 20",color=INK,fontsize=9,weight="bold",ha="right")
    return fig


def slide_problem():
    f=base("Why move from Monte Carlo to TD?",6)
    f.text(.08,.78,"Both estimate future reward from experience, but they use different targets.",fontsize=12,color=BODY)
    f.text(.08,.68,"MONTE CARLO  ·  wait for the episode",fontsize=12,color=GREEN,weight="bold")
    f.text(.08,.60,r"$G_t=r_t+\gamma r_{t+1}+\cdots$",fontsize=17,color=INK)
    f.text(.08,.52,"Uses the actual return: unbiased on average. But a long, random\nfuture makes the target noisy, and learning waits for termination.",fontsize=10.5,color=BODY,linespacing=1.4)
    f.text(.54,.68,"TEMPORAL DIFFERENCE  ·  learn each step",fontsize=12,color=ORANGE,weight="bold")
    f.text(.54,.60,r"$r_t+\gamma\hat V(s_{t+1})$",fontsize=17,color=INK)
    f.text(.54,.52,"Bootstraps from the next-state estimate. It can update before\nthe episode ends and often has lower target variance, at the cost\nof bias while that estimate is inaccurate.",fontsize=10.2,color=BODY,linespacing=1.4)
    f.text(.08,.36,"Corridor example",fontsize=11,color=GREEN,weight="bold")
    f.text(.08,.225,"A → B gives 0; B pays 1 with probability .75. In the sample, A is visited once and that\nepisode loses. MC’s one return says 0 for A. TD can use 0 + V(B), borrowing B’s average;\nas B is learned, this reduces the need to revisit A to learn its value.",fontsize=10,color=BODY,linespacing=1.4)
    f.text(.08,.13,"MC is not wrong: with enough A-visits it averages to the right value. TD’s advantage is faster credit assignment from fewer visits.",fontsize=9,color=MUTED)
    return f


def slide_method():
    f=base("TD(λ): algorithm and the role of λ",7)
    f.text(.5,.80,r"Value estimate: $\hat V(s)=\phi(s)^\top\theta$",fontsize=16,color=INK,ha="center")
    f.text(.5,.70,r"Error: $\delta_t=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t)$",fontsize=15,color=INK,ha="center")
    f.text(.5,.60,r"Trace: $z_t=\gamma\lambda z_{t-1}+\phi(s_t)$",fontsize=15,color=INK,ha="center")
    f.text(.5,.51,r"Update: $\theta\leftarrow\theta+\alpha_n\sum_t\delta_tz_t$ per trajectory",fontsize=15,color=ORANGE,ha="center")
    f.text(.08,.40,"The λ continuum (the paper’s bias–variance trade-off)",fontsize=11.5,color=GREEN,weight="bold")
    f.text(.08,.35,r"$\lambda=0$: one-step TD target     $0<\lambda<1$: mixtures of multi-step targets     $\lambda=1$: Monte Carlo return",fontsize=10.1,color=BODY)
    f.text(.08,.26,"There are no three LSTD-style “views” of TD in the paper. The key TD interpretation is this\ncontinuum: more bootstrapping at low λ; more sampled return at high λ.",fontsize=9.5,color=BODY,linespacing=1.35)
    f.text(.08,.15,r"Expected updates stop at $d+C\theta=0$. TD approaches this fixed point incrementally; its speed and accuracy depend on the step-size schedule.",fontsize=9.5,color=INK)
    f.text(.08,.095,"The next section starts from this fixed point and shows how LSTD estimates and solves it directly.",fontsize=9,color=MUTED)
    return f


def make_td_only_curve():
    data = np.load(ROOT / "results/raw/learning_curves.npz")
    checkpoints, lambdas = data["checkpoints"], data["lambdas"]
    li = int(np.argmin(abs(lambdas - 0.4)))
    indices = np.where(data["td_schedules"] == "decay")[0]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=180)
    for idx in indices:
        values = data["td_err"][:, li, idx, :]
        mean = np.mean(values, axis=0)
        se = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
        label = data["td_labels"][idx].replace("TD ", "")
        (line,) = ax.plot(checkpoints, mean, lw=1.7, label=label)
        ax.fill_between(checkpoints, np.maximum(mean-se, 1e-12), mean+se, color=line.get_color(), alpha=.16, lw=0)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of trajectories"); ax.set_ylabel("RMSE over 13 states")
    ax.set_title("TD(λ), λ = 0.4: six decaying step-size schedules")
    ax.grid(True, which="both", alpha=.22); ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    path = ROOT / "results/figures/fig_td_learning_curves.png"
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return path


def slide_results(data=None):
    f=base("TD experiment: setup, results, and computation",8)
    ax=f.add_axes([.07,.255,.48,.53]); ax.imshow(plt.imread(make_td_only_curve())); ax.axis('off')
    f.text(.59,.79,"Experimental setup",fontsize=11,color=GREEN,weight="bold")
    f.text(.59,.64,"Boyan chain: 13 states, 4 features\nγ=1, λ=.4; 20 random seeds\nRMSE across all 13 states",fontsize=9.3,color=BODY,linespacing=1.4)
    f.text(.59,.54,"Measured RMSE ± SE",fontsize=10,color=INK,weight="bold")
    f.text(.80,.495,"100 traj.",fontsize=8.5,color=MUTED,ha='center'); f.text(.92,.495,"10,000",fontsize=8.5,color=MUTED,ha='center')
    f.text(.59,.45,"Best decaying TD",fontsize=8.7,color=BODY); f.text(.80,.45,".308 ± .031",fontsize=8.2,color=BODY,ha='center'); f.text(.92,.45,".0362 ± .0039",fontsize=8.2,color=BODY,ha='center')
    f.text(.59,.41,"Best constant TD",fontsize=8.7,color=BODY); f.text(.80,.41,".379 ± .046",fontsize=8.2,color=BODY,ha='center'); f.text(.92,.41,".0413 ± .0037",fontsize=8.2,color=BODY,ha='center')
    f.text(.59,.35,"TD-only takeaway",fontsize=10,color=GREEN,weight="bold")
    f.text(.59,.275,"Best decay changed from (a₀,n₀)=(.1,100) at 100\ntrajectories to (.01,1,000) at 10,000. Best constant\nα shifted .1 → .001. Step-size choice matters.",fontsize=8.8,color=BODY,linespacing=1.4)
    f.text(.08,.17,"TD(λ) costs O(K) work per transition and O(K) weight/trace storage. Our run tests six\ndecaying schedules and six constant α settings; the best schedule changes with data budget.",fontsize=9,color=INK,linespacing=1.35)
    return f


def slide_td_paper_findings():
    f=base("What Boyan’s paper says about TD",9,number=False)
    f.text(.08,.78,"The paper studies TD(λ) as an incremental value-learning method.",fontsize=11.5,color=BODY)
    f.text(.08,.69,"CONVERGENCE",fontsize=11,color=GREEN,weight="bold")
    f.text(.08,.55,"With linear value functions, TD converges under a suitable\ndiminishing step-size schedule. The paper’s sufficient conditions\nare αₙ ≥ 0, Σαₙ = ∞, and Σαₙ² < ∞.",fontsize=9.8,color=BODY,linespacing=1.4)
    f.text(.08,.42,"LIMITATIONS",fontsize=11,color=ORANGE,weight="bold")
    f.text(.08,.29,"After each trajectory, ordinary TD keeps its weights but no\ntransition statistics; replay must revisit saved data. Results are\nsensitive to step-size choice, initialization, and feature scale.",fontsize=9.6,color=BODY,linespacing=1.4)
    f.text(.55,.69,"PAPER’S EXPERIMENT",fontsize=11,color=GREEN,weight="bold")
    f.text(.55,.55,"Six decaying step-size schedules; λ = 0, .2, .4, .6, .8, 1;\n10 trials. Schedules use αₙ = a₀(n₀+1)/(n₀+n), with\na₀ ∈ {.01, .1} and n₀ ∈ {100, 1,000, 1,000,000}.",fontsize=9.5,color=BODY,linespacing=1.4)
    f.text(.55,.42,"PAPER’S TD FINDING",fontsize=11,color=ORANGE,weight="bold")
    f.text(.55,.29,"The paper reports that TD performance depends critically\non the step-size schedule. Its tested schedules meet the\nconvergence conditions, yet their finite-data learning curves differ.",fontsize=9.5,color=BODY,linespacing=1.4)
    f.text(.08,.13,"This slide summarizes the paper’s TD analysis; the preceding experiment slide reports our TD-only 20-seed run at λ = .4.",fontsize=9,color=MUTED)
    return f


def main():
    if not PPTX.exists():
        raise SystemExit(f"Missing presentation: {PPTX}")
    figures=[slide_problem(),slide_method(),slide_results()]
    paper_fig=slide_td_paper_findings()
    paper_fig.savefig(ROOT / "slides/member_b_td_paper_findings.png",dpi=150,facecolor='white')
    plt.close(paper_fig)
    with tempfile.TemporaryDirectory() as td:
        # PDF page 8 is the section 02 divider; Member B's content is pages 9–11.
        repl={f"ppt/media/image{i}.png":None for i in range(9,12)}
        for i,fig in enumerate(figures,9):
            p=Path(td)/f"slide{i}.png"; fig.savefig(p,dpi=150,facecolor='white'); plt.close(fig); repl[f"ppt/media/image{i}.png"]=p
        out=Path(td)/"updated.pptx"
        with zipfile.ZipFile(PPTX) as src,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                if item.filename in repl: dst.write(repl[item.filename],item.filename)
                else: dst.writestr(item,src.read(item.filename))
        os.replace(out,PPTX)
    print(f"Updated Member B MC-to-TD slides on pages 9–11: {PPTX}")


if __name__=='__main__': main()
