"""Refresh Member B's raster slides from the full TD learning-curve run."""
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
DATA = ROOT / "results/raw/learning_curves.npz"
PLOT = ROOT / "results/figures/fig_td_learning_curves.png"
INK, BODY, ORANGE, GREEN, MUTED = "#1E2A3A", "#2B2F36", "#E4572E", "#1F7A6D", "#758092"


def base(title: str, page: int):
    fig = plt.figure(figsize=(13.333, 7.5), dpi=150, facecolor="white")
    fig.text(.07,.94,"02   ━━━   TD(λ)",color=ORANGE,fontsize=11,weight="bold")
    fig.text(.07,.875,title,color=INK,fontsize=22,weight="bold")
    fig.text(.93,.94,"MEMBER B",color=GREEN,fontsize=10,weight="bold",ha="right")
    fig.add_artist(plt.Line2D([.07,.93],[.075,.075],color="#D9DDE3",lw=1))
    fig.text(.07,.04,"LSTD(λ)  ·  Boyan, ICML 1999",color=MUTED,fontsize=8)
    fig.text(.93,.04,f"{page} / 20",color=INK,fontsize=9,weight="bold",ha="right")
    return fig


def slide_problem():
    f=base("Problem: estimating state values with TD",6)
    f.text(.08,.78,"Temporal Difference (TD) estimates a state's future reward from sampled transitions. It bootstraps\nfrom its current next-state estimate after each transition.",fontsize=12,color=BODY,linespacing=1.45)
    f.text(.5,.65,r"TD target $=r_t+\gamma\hat V(s_{t+1})$     TD error $\delta_t=target-\hat V(s_t)$",fontsize=17,color=INK,ha="center")
    f.text(.08,.52,"Problem",fontsize=14,color=GREEN,weight="bold")
    f.text(.08,.45,"Estimate values for the 13-state Boyan chain from sampled trajectories. Test how TD(λ)\naccuracy changes with experience and learning-rate choice.",fontsize=11,color=BODY,linespacing=1.5)
    f.text(.08,.30,"Headline result",fontsize=14,color=ORANGE,weight="bold")
    f.text(.08,.23,"Best paper-style TD schedule: RMSE 0.308 at 100 trajectories and 0.0362 at 10,000.\nThese are means over 20 seeds; lower RMSE means more accurate state-value estimates.",fontsize=11,color=BODY,linespacing=1.5)
    f.text(.08,.13,"Next: the TD(λ) update, the experiment settings, and the learning curves.",fontsize=10,color=MUTED)
    return f


def slide_method():
    f=base("Methodology: TD(λ) update and parameters",7)
    f.text(.5,.785,r"Value estimate: $\hat V(s)=\phi(s)^\top\theta$",fontsize=16,color=INK,ha="center")
    f.text(.5,.705,r"TD error: $\delta_t=r_t+\gamma\hat V(s_{t+1})-\hat V(s_t)$",fontsize=16,color=INK,ha="center")
    f.text(.5,.625,r"Trace: $z_t=\gamma\lambda z_{t-1}+\phi(s_t)$     Update: $\theta\leftarrow\theta+\alpha_n\sum_t\delta_t z_t$",fontsize=15,color=INK,ha="center")
    f.text(.08,.51,"What the symbols mean",fontsize=12,color=GREEN,weight="bold")
    f.text(.08,.455,"φ(s): four features for state s     θ: learned weights     γ = 1: discount factor\nλ = 0.4: trace decay     z: recent-state eligibility trace     αₙ: learning rate",fontsize=10.5,color=BODY,linespacing=1.5)
    f.text(.08,.365,"Experiment procedure and tested step sizes",fontsize=12,color=GREEN,weight="bold")
    f.text(.08,.315,"For each trajectory, hold θ fixed, sum δₜzₜ, then update θ once. Decay: αₙ=a₀(n₀+1)/(n₀+n),\nwith a₀∈{.01,.1}, n₀∈{100,1,000,1,000,000}. Constant α∈{.001,.003,.01,.03,.1,.3}.",fontsize=9.5,color=BODY,linespacing=1.4)
    f.text(.08,.205,"Example: current value 2, reward 1, next value 4, γ=.9 → target 4.6, error 2.6.\nWith α=.1 and a one-hot feature, the new value is 2 + .1×2.6 = 2.26.",fontsize=9.5,color=INK,linespacing=1.4)
    return f


def mean_se(a):
    return np.nanmean(a,axis=0),np.nanstd(a,axis=0,ddof=1)/np.sqrt(np.sum(np.isfinite(a),axis=0))


def make_curve(data):
    PLOT.parent.mkdir(parents=True,exist_ok=True)
    cps=data['checkpoints']; li=int(np.argmin(abs(data['lambdas']-.4)))
    ix=np.where(data['td_schedules']=='decay')[0]
    vals=data['td_err'][:,li,ix,:]
    m,se=mean_se(vals)
    fig,ax=plt.subplots(figsize=(8.2,4.6),dpi=180)
    colors=plt.cm.tab10(np.linspace(0,1,len(ix)))
    for j,k in enumerate(ix):
        ax.plot(cps,m[j],lw=1.7,color=colors[j],label=data['td_labels'][k].replace('TD ','').replace('10^','10^'))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Number of trajectories'); ax.set_ylabel('RMSE over 13 states')
    ax.set_title('TD(λ), λ = 0.4: six decaying step-size schedules')
    ax.grid(True,which='both',alpha=.22); ax.legend(fontsize=7,ncol=2,frameon=False)
    fig.tight_layout(); fig.savefig(PLOT,bbox_inches='tight'); plt.close(fig)
    return PLOT


def slide_results(data):
    f=base("Results: TD(λ) on the Boyan chain",8)
    ax=f.add_axes([.07,.23,.53,.54]); ax.imshow(plt.imread(PLOT)); ax.axis('off')
    f.text(.63,.78,"Setup",fontsize=12,color=INK,weight="bold")
    f.text(.63,.725,"13 states; start at 12, terminal state 0\n4 interpolated features; γ=1, λ=.4\n20 seeds; RMSE over all states",fontsize=9,color=BODY,linespacing=1.45)
    f.text(.63,.59,"Mean RMSE ± standard error",fontsize=10,color=INK,weight="bold")
    f.text(.79,.545,"100 traj.",fontsize=8,color=MUTED,ha='center'); f.text(.91,.545,"10,000",fontsize=8,color=MUTED,ha='center')
    cps=data['checkpoints']; li=int(np.argmin(abs(data['lambdas']-.4)))
    for row,(sched,label,y) in enumerate([('decay','Best decay',.49),('constant','Best constant',.425)]):
        settings=data['td_schedules']==sched; results=[]; configs=[]
        for n in [100,10000]:
            ci=int(np.where(cps==n)[0][0]); a=data['td_err'][:,li,settings,ci]
            mn,se=mean_se(a); j=int(np.argmin(mn)); results.append(f"{mn[j]:.3f} ± {se[j]:.3f}"); configs.append(np.where(settings)[0][j])
        f.text(.63,y,label,fontsize=8.7,color=BODY)
        f.text(.79,y,results[0],fontsize=8.1,color=BODY,ha='center'); f.text(.91,y,results[1],fontsize=8.1,color=BODY,ha='center')
    f.text(.63,.335,"Best settings",fontsize=10,color=GREEN,weight='bold')
    f.text(.63,.29,"Decay: (a₀,n₀)=(.1,100) → (.01,1,000)\nConstant α: .1 → .001",fontsize=8.4,color=BODY,linespacing=1.5)
    f.text(.63,.16,"Paper: 10 trials, six decay schedules. Our run: 20 seeds,\nsame six decays plus six constant rates.",fontsize=7.6,color=MUTED,linespacing=1.4)
    return f


def main():
    if not PPTX.exists() or not DATA.exists():
        raise SystemExit("Run python3 -m src.run_experiment --experiment learning_curves first")
    data=np.load(DATA); make_curve(data)
    figures=[slide_problem(),slide_method(),slide_results(data)]
    with tempfile.TemporaryDirectory() as td:
        repl={f"ppt/media/image{i}.png":None for i in range(6,9)}
        for i,fig in enumerate(figures,6):
            p=Path(td)/f"slide{i}.png"; fig.savefig(p,dpi=150,facecolor='white'); plt.close(fig); repl[f"ppt/media/image{i}.png"]=p
        out=Path(td)/"updated.pptx"
        with zipfile.ZipFile(PPTX) as src,zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as dst:
            for item in src.infolist():
                if item.filename in repl: dst.write(repl[item.filename],item.filename)
                else: dst.writestr(item,src.read(item.filename))
        os.replace(out,PPTX)
    print(f"Updated Member B slides and saved TD plot: {PPTX}")


if __name__=='__main__': main()
