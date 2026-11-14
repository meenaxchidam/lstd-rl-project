# Speaker script: Meena (Member D), Experiments & novelty

**DRAFT.** Finalise once everyone has frozen their slides.

- **Deck:** `lstd_presentation.pptx`. My slides are **18–24**.
- **Time:** **6 minutes**, as part of an 18-minute talk (Dinesh 4.5, Nadhiya 3, Sebin 4.5, Meena 6). Q&A is separate.
- **If we must cut to 5 minutes:** skip the lines marked ✂.
- **Story thread (🏥):** one running example — *each trajectory is one patient's record, which is expensive to collect*. It is used only in the 🏥 lines; the plots and equations carry the method.
  Keep it to **evaluating a fixed treatment plan** (predicting outcomes), which is what our experiment does — not choosing treatments (that would be control/LSPI).
- **Before me:** Sebin's "Experimental setup" (slide 17).
- **After me:** the "Questions?" slide.
- **Team:** Dinesh (A, setup and environments), Nadhiya (B, TD(λ)), Sebin (C, LSTD(λ)).
- **My code:** `configs/experiment.yaml`, `src/run_experiment.py`, `src/plot_results.py`.
  Together these run every experiment and draw every figure.

**How to read every plot in my section:**
- **x-axis:** number of trajectories, i.e. how many games the method has seen.
- **y-axis:** RMS error, i.e. how far the guessed values are from the true values.
- Both axes are log scale. **Lower is better.**

---

## Slide 18: Section divider (0:20)

> "Thanks, Sebin. Now I'll show what happens when we run TD and LSTD side by side on the same data, and then our own extension.
>
> 🏥 Think of each trajectory as one patient's record — expensive to collect. Which method learns more from each patient?"

---

## Slide 19: Headline result (1:10)

**On screen:** learning curves on the left, four numbers on the right.

> "This is our main result on the Boyan chain, with λ = 0.4 as in the paper, averaged over 20 seeds.
>
> The black line is LSTD. Its error keeps falling, from 1.64 after one trajectory to 0.016 after ten thousand.
>
> The red solid line is TD with its best fixed step size, α = 0.3. It learns quickly, then gets stuck at about 0.7, because every new noisy episode keeps pushing it around.
>
> The red dashed line is TD with its best α picked separately at each data size, out of 12 settings. You can't do that in practice, so it's generous to TD.
>
> Even so, LSTD has about half its error everywhere, needs 3 to 4 times fewer trajectories for the same accuracy, and has no step size to tune.
>
> 🏥 In hospital terms: the same accuracy from 3 to 4 times fewer patients."

**If you blank:** *"LSTD beats even the best-tuned TD at every data size, with nothing to tune."*

---

## Slide 20: Decaying step sizes and every λ (0:45)

> "On the left is the paper's Figure 5, with λ = 0.4. A schedule is just a rule for how TD's step size changes over time. Here TD uses six decaying schedules: a₀ is the starting step size, and n₀ is how long before α starts decaying: α has halved after about n₀ trajectories. Some shrink too fast, others stay noisy. LSTD in black is below all six.
>
> On the right, Figure 6 repeats this for every λ, with the same colours. Each bar is one algorithm: the top is the error after 100 trajectories, the bottom after 10,000. The black LSTD bar is lowest at every λ.
>
> Same conclusion as the paper: LSTD is lowest for every schedule and every λ."

**Remember:**
- **α** is how big each step is; it shrinks over time here.
- **a₀** = the starting step size. **n₀** = roughly how many trajectories until the step has halved. With n₀ = 10⁶ it barely shrinks within 10⁴ trajectories, so it acts almost like a fixed α.
- **λ** is what TD learns from (0 = one-step TD, 1 = Monte Carlo). It is fixed at 0.4 on the left and varied on the right.
- If asked for the formula: α_n = a₀ (n₀ + 1) / (n₀ + n), the paper's Eq. 8.
- The values are in the left plot's legend: a₀ ∈ {0.1, 0.01}, n₀ ∈ {10², 10³, 10⁶}. The right plot has no legend because its colours are the same.

---

## Slide 21: Effect of λ (0:40)

> "Here we vary λ, the trace parameter Nadhiya introduced.
>
> LSTD's error hardly changes with λ: about 0.15 after 100 trajectories and 0.016 after ten thousand. The Boyan features are exact, so every λ converges to the same answer, θ* = (−24, −16, −8, 0).
>
> ✂ TD★ is the best of the 12 TD settings for each λ and data size.
>
> And LSTD is 2 to 3.5 times better than TD★, in every cell of the table."

**If asked "are the features really exact?":** yes. V(s) = −2s is a straight line, and the features linearly interpolate between the anchor states 12, 8, 4, 0, so θ* = (−24, −16, −8, 0) reproduces every true value (fit error ≈ 10⁻¹⁴). The paper says the same in Section 4: this "guarantees that LSTD(λ) will converge with probability 1 to the optimal θ for any setting of λ". Our runs agree: with 10⁴ trajectories, λ = 0, 0.4 and 1 all give θ ≈ (−23.97, −15.99, −7.97, 0.00).

**If asked:** TD★ does best at a middle λ ≈ 0.4, matching Lecture 20's result that a middle n beats one-step TD and Monte Carlo.

---

## Slide 22: Novelty: ridge LSTD (1:20)

**On screen, top to bottom:** Problem → Method → Questions Q1–Q3 → random-walk diagram → Setup.

> "Now our own contribution.
>
> 🏥 Imagine a new clinic with only a handful of patients. **The problem:** LSTD solves Aθ = b, and with only 1 to 10 trajectories, A can be singular or badly conditioned, so the solution blows up.
>
> **Our method** is ridge regularisation: add a small ε to the diagonal and solve (A + εI)θ = b. That always has a unique solution. ✂ It is the same as starting Sebin's recursive LSTD from B₀ = I/ε.
>
> **We asked three questions.** Q1: how much ε should we add? Q2: with so little data, can it match the best-tuned TD? Q3: does it help on a harder problem with weak features?
>
> **For Q3 we use the random walk from Lecture 20** *(point at the diagram)*: start in the middle, step left or right at random, −1 at the left end, +1 at the right. The colours are the weak features: five groups, each sharing one value, so the true values can't be represented exactly.
>
> ✂ **Setup:** both environments, 50 seeds, 1 to 10 trajectories, ε from 10⁻⁴ to 100, with every method seeing the same episodes, against TD with its best of five step sizes."

**Note:** say the bold words (Problem, method, questions, setup) slightly louder, so the structure is audible.

**If you blank:** *"Problem: A can't be solved with tiny data. Fix: add εI. Three questions: how much ε, does it match TD, does it help on a harder problem."*

---

## Slide 23: Novelty: results (1:15)

**On screen:** the two plots (Boyan on the left, 19-state random walk on the right), with answers Q1–Q3 underneath.

> "The x-axis is ε, and each colour is a number of trajectories. If the solid ridge curve dips below the dotted plain-LSTD line, ridge helped.
>
> **Q1, how much ε:** there's a sweet spot, around 0.03 to 0.1 on Boyan and 1 to 3 on the random walk. Too much ε shrinks every value toward zero and hurts.
>
> **Q2, can it match TD:** nearly. On the random walk, after one trajectory, ridge reaches 0.44 against the best TD's 0.43, and it beats TD with 2 and 5 trajectories. ✂ On Boyan the question doesn't really arise, because plain LSTD already beats TD: 1.7 versus 6.5.
>
> **Q3, the harder problem:** this is where ridge helps most. The error after one trajectory is halved, from 0.83 to 0.44. On Boyan the gain is at most 3 percent, because A is already well conditioned.
>
> 🏥 That's the new clinic: few patients, coarse groups, exactly where ridge helps."

**If you blank:** *"Ridge helps when data are scarce and the features are weak; it does little when LSTD is already well posed."*

**Random walk, checked (in case anyone asks):**
- It is Sutton & Barto's Example 7.1 (also Lecture 20): 19 states plus two ends, start in the middle (state 10), step left or right with probability ½, reward −1 at the left end and +1 at the right end, γ = 1. True values are −0.9, −0.8, …, +0.9. Episodes last about 100 steps on average.
- Weak features: the 19 states are split into 5 neighbouring groups (sizes 4, 4, 4, 4, 3); each group shares one value. This is state aggregation (S&B §9.3).
- The **best possible** error with these features is about **0.10**, so the true values really can't be represented exactly.
- For scale: **predicting 0 everywhere gives 0.52**. Plain LSTD after one trajectory (0.83) is *worse* than that; ridge (0.44) and TD (0.43) both beat it. Since the true values are centred on 0, shrinking toward 0 is a safe default here, which is why ridge helps.
- The error is measured over all 21 states, including the two ends, whose value 0 is always predicted correctly.

**Backup facts (do not say unless asked):**
- 82% of single random-walk episodes miss at least one feature group, which makes A singular. On Boyan, 0% do.
- Boyan's values go down to −24, so shrinking toward 0 costs a lot there.
- Ridge vs TD on the random walk: TD slightly better at 1 trajectory (0.43 vs 0.44) and 10 (0.33 vs 0.34); ridge better at 2 (0.41 vs 0.43) and 5 (0.37 vs 0.39).
- Both ε for ridge and α for TD are picked as the best value after the fact, so the comparison treats them alike.
- Why A is well conditioned on Boyan: every episode passes through all the features.
- Plot legend: solid = ridge LSTD, dotted = plain LSTD, dashed = best-tuned TD.

---

## Slide 24: Conclusions and limitations (0:45)

> "To conclude: we reproduced the paper. LSTD beats TD at every data size, λ and step size, with 2 to 3.5 times lower error. It needs no step size, while a bad α for TD costs 10 to 100 times more error. The price is computation: O(K²) per step instead of O(K). And ridge helps exactly when data are scarce and features are weak.
>
> 🏥 So when data is expensive, like patient records, LSTD is the better choice.
>
> The main limitations: these are toy problems with exact features, the best case for LSTD, and a fixed ε has less effect as data grow.
>
> Thank you."

**If asked about the other limitations on the slide:**
- TD was tuned over a 12-point grid only.
- Ridge shrinks toward zero, not toward a sensible prior.

---

## 🏥 Story mapping (reference, not to read out)

| Our experiment | Hospital story |
|---|---|
| State | Stage of a patient's treatment |
| Value V(s) | Expected outcome from that stage |
| One trajectory | One patient's full record |
| n trajectories | Number of patients we have data on |
| TD with α | Adjust predictions a little after each patient; the step size must be tuned |
| LSTD | Use all records at once and solve for the best predictions |
| Weak features | Only coarse patient groups (e.g. age bands) |
| Ridge with n = 1–10 | A new clinic with a handful of patients; ridge stops over-reacting to a few records |

---

## Likely questions for me

| Question | Short answer |
|---|---|
| Is the hospital example something you ran? | No — it is only an analogy for *expensive data*. All results are on the Boyan chain and the 19-state random walk. In practice, evaluating a fixed treatment plan from patient records is a policy-evaluation problem of the same kind. |
| Why does LSTD need less data? | With the **same** data, LSTD is more accurate. TD takes one small step per sample and gradually forgets old samples. LSTD keeps every sample in A and b and solves exactly. TD needs about 5–10× more games to match LSTD's error at 100 games. |
| Is the comparison fair to TD? | Yes, generous to TD. In each seed, every algorithm sees the **same episodes**. We also show TD with its best α picked at every data size. |
| How do you know α = 0.3 is the best fixed step size? | It's the best of the **12 TD settings we tried** (6 fixed α from 0.001 to 0.3, plus 6 decaying schedules). "Best" means the **lowest average error over all data sizes** (1 to 10⁴ trajectories, 20 seeds), as set in `configs/experiment.yaml` (`best_td_criterion: auc`). Average errors: α = 0.3 → 2.38, 0.1 → 3.54, 0.03 → 5.16, … 0.001 → 9.95; the best decaying schedule → 3.47. |
| α = 0.3 is the largest α tried. Would a bigger α do better? | **Partly.** α = 0.5 learns faster early, so its average error is lower (≈ 2.0 vs 2.38). But it levels off **higher** in the long run (≈ 1.0 vs 0.7), and α = 1.0 is worse on both counts (≈ 2.7 average, ≈ 2.0 long run). Either way, LSTD is still lower at every n (e.g. after 1 trajectory: LSTD 1.64 vs TD with α = 1.0 at 6.4, its best early value). |
| Is "2×" true at every point? | It's conservative. The ratio is 1.6–2.3× from 10 trajectories on, and 5–7× with 5 or fewer. |
| Why is TD's error about 11 after one trajectory? | TD starts at θ = 0 and the true values go down to −24, and one small step can't get there. LSTD solves directly, so it doesn't depend on a starting point. |
| Why doesn't λ matter for LSTD here? | The Boyan features represent V exactly, so every λ has the same answer. λ only changes the noise a little. |
| What does the shaded band mean? | ±1 standard error over the seeds, i.e. how uncertain the average curve is. |
| What if A is singular? | The paper uses an SVD pseudo-inverse. We use `solve` with an SVD fallback, or ridge (A + εI). |
| How would you choose ε in practice? | Cross-validation, or scale ε with the amount of data, since A grows with n. |
| Why does ridge's error blow up at large ε? | A large ε shrinks θ to about 0, so every value is predicted as 0. On Boyan that gives an error of about 14, the RMS of the true values. |
| What is the 19-state random walk? | Sutton & Barto's example (also in Lecture 20). You start in the middle and move left or right at random. The left end gives −1 and the right end +1. We used 5 grouped features. |
| How long do the experiments take? | The full sweep takes about 30 s on a laptop, CPU only: `python -m src.run_experiment && python -m src.plot_results`. |
| Which code is yours? | `configs/experiment.yaml`, `src/run_experiment.py` and `src/plot_results.py`: the experiment runner and every figure. |

---

## 30-second version (if time is cut)

> "LSTD reaches about half the error of even the best-tuned TD at every data size, needs 3 to 4 times fewer trajectories, and has no step size to tune. We reproduced the paper's Figures 5 and 6. λ barely matters for LSTD when the features are exact. Our extension, ridge LSTD, halves the error after a single trajectory when the features are weak, matching the best-tuned TD, but adds little when LSTD is already well posed."
