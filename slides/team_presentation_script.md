# LSTD(λ): coordinated team presentation script

**Audience:** IISc M.Tech reinforcement learning course.  
**Deck:** the current 31-slide presentation. Slide numbers below are physical PDF/PowerPoint page numbers, including dividers.  
**Session:** 20 minutes, including questions. Finish prepared remarks by 16:00.

## Timing and delivery

The planned script ends at **15:30**, leaving 30 seconds for pauses or slower explanations and at least four minutes for questions. These are rehearsal targets, not measured speaking times. Read only the blockquotes aloud. Bracketed cues, implementation anchors, and Q&A notes are preparation material.

| Speaker | Slides | Speaking allocation | Presentation clock |
|---|---|---:|---:|
| Dinesh Chand — A | 1–7 | 3:45 | 0:00–3:45 |
| Nadhiya R S — B | 8–12 | 3:45 | 3:45–7:30 |
| Sebin Francis Kannampuzha — C | 13–18 | 4:00 | 7:30–11:30 |
| Meena Chidambaram — D | 19–26 | 4:00 | 11:30–15:30 |
| Shared timing margin | — | 0:30 | 15:30–16:00 |
| Questions | 27–31 only if useful | At least 4:00 | Start as soon as the talk ends |

The prepared remarks contain approximately **1,560 spoken words**. At 110–120 words per minute, speech takes about 13–14.5 minutes, leaving room within the allocation for pointing, equation pauses, slide changes, and handoffs. Do not stretch a section merely to use all its time; the timed rehearsal determines the actual duration.

Use a steady technical speaking pace. Pause after the Bellman equation, the TD fixed point, and the LSTD system. Point to the term or curve being discussed; do not read every equation, table entry, or legend label. Advance through each divider without a separate introduction. Each outgoing speaker's final sentence introduces the next speaker's question.

## A — Dinesh: problem, representation, and experimental foundation

### Slide 1 — Title · 0:12

> Good morning. We examine Boyan’s Least-Squares Temporal Difference Learning, our implementation, and a regularisation extension. How effectively can we learn from limited trajectories?

### Slide 2 — Part 1 divider · 0:03

[Advance directly to slide 3.]

### Slide 3 — Problem statement and motivation · 0:30

> Our task is policy evaluation: predicting future reward under a fixed policy. With known transition probabilities, this becomes a linear system. Here, the learner receives sampled trajectories. We reproduce Boyan’s TD–LSTD comparison, then investigate regularisation with very limited data.

### Slide 4 — Policy evaluation · 0:45

[Point first to the Bellman equation, then to the linear approximation.]

> The value of a state is its expected discounted return. The Bellman equation splits this into an immediate reward and the discounted next-state value.
>
> We approximate values using fixed features phi and learned weights theta. The approximation is linear in the weights; the features can be nonlinear functions of state.
>
> Our simulator knows the transition model, but the learners only receive transitions. We use the model separately to compute reference values and prediction error.

### Slide 5 — Monte Carlo, TD, and LSTD · 0:35

> These methods share the same representation. Monte Carlo fits complete observed returns. TD bootstraps using an estimate of the next state’s value.
>
> LSTD instead accumulates transition statistics and solves for weights whose aggregate TD correction is zero. Boyan extends the earlier LSTD-zero algorithm to arbitrary lambda. At lambda one, it becomes regression on complete returns. With tabular features, LSTD-zero also matches solving an empirical Markov chain. We will explain these connections.

### Slide 6 — Boyan chain and the paper’s experiment · 1:05

[Trace one transition, then point to the feature vectors.]

> Each episode in this thirteen-state chain starts at twelve and ends at zero. From state two or above, it moves down one or two states with equal probability and reward minus three. From state one, it terminates with reward minus two. Gamma is one.
>
> The true value is minus twice the state number. Four features interpolate between anchor states twelve, eight, four, and zero. State eleven, for example, mixes the first two anchors with weights three quarters and one quarter, giving minus twenty-two.
>
> The true value is therefore exactly representable, separating learning error from representation error. Tests check the transitions, features, and reference values.

### Slide 7 — Parameters and variables · 0:35

> Each implemented observation contains the state, reward, next state, and termination flag. Gamma discounts rewards, lambda controls traces, alpha is TD’s step size, and epsilon controls regularisation.
>
> We generate seeded episodes once and give competing methods the same data. At termination, learners explicitly use zero next-state value.
>
> Nadhiya will now show how TD turns these transitions into weight updates, and why the step-size choice matters.

**Implementation anchors, not spoken:** `BoyanChain`, `generate_episodes`, `true_values`, and `boyan_features`. Be ready to explain the shared transition tuple and the distinction between the simulator's model and the learner's inputs.

## B — Nadhiya: TD targets, traces, and step-size sensitivity

### Slide 8 — Part 2 divider · 0:03

[Advance directly to slide 9.]

### Slide 9 — Why move from Monte Carlo to TD? · 0:55

> With the representation established, what target should we learn from?
>
> Monte Carlo uses the complete return: an unbiased sample of value, potentially with high variance. TD bootstraps from an immediate reward and a next-state estimate.
>
> In this corridor, A leads to B without reward. B pays one or zero, then terminates. Our only episode through A received zero, so Monte Carlo estimates A as zero. Across eight B visits, six paid one. Observations starting at B can improve B’s estimate. Revisiting A or replaying the data lets TD transfer that information back. Sebin will revisit this example.

### Slide 10 — TD(λ): the update and the role of λ · 1:10

[Point to delta, z, and the trajectory-level update in that order.]

> Delta is the one-step prediction error: reward plus discounted next-state value, minus current value. The eligibility trace z determines which recently active features share that error.
>
> At lambda zero, only current features are eligible. Increasing lambda carries credit farther back. At lambda one, the forward-view target is the complete Monte Carlo return.
>
> Our default follows the paper’s episode-wise update: freeze theta, accumulate delta times z, then apply the step-size-scaled correction at episode end. Immediate online updates are supported but are not the default.
>
> We reset z each episode and use zero terminal bootstrap value. A hand-calculated update checks the code. Lambda controls the target mixture; alpha controls correction size.

### Slide 11 — TD experiment: setup, results, and computation · 0:55

[Point to the separation between the six TD curves, then the table.]

> Here we isolate TD at lambda point four over twenty seeds. Lower error is better. These six decaying step-size schedules produce substantially different learning curves.
>
> The best decay setting changes with the data budget: at one hundred episodes, it starts at point one and decays relatively quickly; at ten thousand, a smaller initial step with slower decay works best.
>
> We also test six constant step sizes. The table selects the best tested settings using true-value evaluation. TD costs order K computation per transition and order K storage.

### Slide 12 — What Boyan’s paper says about TD · 0:42

> This sensitivity is consistent with convergence theory. Under the paper’s assumptions, step sizes must sustain learning while their squared sum remains finite to control noise.
>
> Convergence does not identify the best schedule at a finite data budget. We separate the step-size function from the update to compare schedules on identical trajectories.
>
> We have seen how TD moves. Sebin will now ask where those updates are trying to settle, and whether we can estimate that destination directly.

**Implementation anchors, not spoken:** `td_lambda_curve`, `step_size`, `online=False`, `dtheta`, and terminal handling. Do not describe the default experiment as changing theta after every transition.

## C — Sebin: from TD’s expected update to LSTD

### Slide 13 — Part 3 divider · 0:03

[Advance directly to slide 14.]

### Slide 14 — Motivation for LSTD · 0:48

> Nadhiya showed how TD moves. Where should those updates settle?
>
> With linear features, the expected episode correction is alpha times d plus C theta. Here, d summarises rewards; C captures current and next-state feature contributions.
>
> At the fixed point, d plus C theta equals zero. This is an average balance: individual observed TD errors need not vanish.
>
> Boyan estimates this system from experience and solves it directly, targeting TD’s expected balance without a step-size schedule.

### Slide 15 — The LSTD(λ) algorithm · 1:05

[Follow the pseudocode from z to A and b, then the solve.]

> We initialise A and b once, resetting only z each episode. Each transition updates the trace, adds an outer product to A, and adds the trace-weighted reward to b.
>
> A uses current features minus discounted next features. C uses the opposite subtraction. After n episodes, A divided by n estimates minus C, and b divided by n estimates d. Substitution gives A theta equals b.
>
> Our accumulator uses NumPy’s outer product. We use a direct linear solve, with an SVD-based least-squares fallback if it reports singularity. Batch LSTD does not explicitly invert A.
>
> A and b retain each transition’s contribution across episodes, rather than a recoverable copy of each trajectory.

### Slide 16 — Why use LSTD instead of TD? · 0:50

> Return to Nadhiya’s corridor: A was visited once, with return zero. B paid one on six of eight visits.
>
> LSTD-zero solves B’s value as six eighths, or point seven five. A gives no reward before reaching B, so its value is also point seven five.
>
> Starting from zero, TD initially has no useful B estimate to pass to A. Replay propagates B’s later improvement. LSTD-zero solves both empirical relationships together after one pass.
>
> This illustrates information sharing; the paper’s data-efficiency evidence comes from the Boyan-chain experiments.

### Slide 17 — Recursive LSTD · 0:40

> Frequent predictions can make repeated full solves costly. Recursive LSTD maintains B, the inverse of the accumulated system matrix.
>
> Each transition changes A by an outer product. Sherman–Morrison updates the inverse using that rank-one change. We update b as before and obtain theta from B times b.
>
> We initialise a regularised matrix because zero has no inverse. Tests compare batch and recursive solutions with matching regularisation. Boyan mentions this option; our code implements it explicitly.

### Slide 18 — Computational cost · 0:34

> TD costs order K per transition; LSTD costs order K squared, plus a cubic batch solve. Recursive LSTD avoids repeated full solves.
>
> Blue is TD; orange uses one final solve. Green solves per episode; red updates the inverse per transition. Green can be faster at small dimensions because it performs fewer updates.
>
> Meena will evaluate the resulting accuracy–computation tradeoff and our regularisation extension.

**Implementation anchors, not spoken:** `_accumulate`, `lstd_statistics`, `solve_theta`, `lstd_lambda_curve`, and `recursive_lstd`. For the recursive initialisation, B-zero equals delta times identity, corresponding to batch regularisation epsilon equals one over delta.

## D — Meena: experimental evidence and regularisation extension

### Slide 19 — Part 4 divider · 0:03

[Advance directly to slide 20.]

### Slide 20 — Headline result · 0:45

[Identify black, solid red, and dashed red before discussing the result.]

> We now compare the accuracy gained from the same observations. Our runner reuses each seed’s episodes across methods and reports mean RMSE over states with standard-error bands.
>
> Black is LSTD. Solid red is constant-step TD. Dashed red selects the best of twelve TD configurations separately at each budget, using ground-truth evaluation.
>
> LSTD’s error falls from one point six four to point zero one six, below even that optimistic TD reference over the displayed budgets. These are our reproduction results.

### Slide 21 — Decaying step sizes and every λ · 0:35

> These reproduce the comparisons in the paper’s Figures five and six. Left: lambda point four, six decaying TD schedules, and LSTD in black.
>
> Right: lambda varies; each segment connects errors at one hundred and ten thousand episodes. Colours match the left plot.
>
> LSTD is lower across these settings, supporting the paper’s finding for this domain and comparison grid, rather than universal superiority over TD.

### Slide 22 — Effect of λ · 0:23

> LSTD changes little with lambda here: about point one five error at one hundred episodes.
>
> Exact features allow all lambda settings to approach the true value. With imperfect features, lambda can change the limiting approximation. TD-star again selects the best tested configuration.

### Slide 23 — Novelty: ridge LSTD for very few trajectories · 0:50

> Our extension examines one to ten episodes and coarse features.
>
> We solve A plus epsilon identity times theta equals b. This diagonal regularisation trades sensitivity to limited data against bias.
>
> Alongside Boyan, we test a nineteen-state random walk, starting centrally with boundary rewards minus one and plus one. Five groups share predictions, preventing exact representation of its true values.
>
> We compare thirteen regularisation strengths over fifty seeds on identical episodes. TD selects its best of five constant step sizes.

### Slide 24 — Novelty: results · 0:48

[Point to the dip in the random-walk curves and the Q1–Q3 answers.]

> The axis is epsilon; colours indicate episode budgets. Intermediate regularisation helps, while excessive regularisation biases predictions toward zero.
>
> With one random-walk episode, error falls from point eight three to point four four, close to TD’s point four three. Ridge outperforms TD at two and five episodes here.
>
> Boyan improves by at most three percent. The benefit is strongest in our scarce-data, coarse-feature setting. Useful epsilon values differ between environments: regularisation introduces a parameter that still needs selection.

### Slide 25 — Conclusions and limitations · 0:32

> We reproduce Boyan’s main empirical finding: LSTD uses these trajectories more effectively than our tested TD schedules, at greater computational cost.
>
> Our extension helps with limited data, especially in the coarse-feature random walk.
>
> Our limits are small chains, finite TD tuning, and exact main-benchmark features. Fixed epsilon also weakens relative to growing data. Broader evaluation remains necessary.

### Slide 26 — Questions · 0:04

> Thank you. We welcome your questions.

**Implementation anchors, not spoken:** `run_learning_curves`, `run_ridge`, `run_timing`, `rmse`, `mean_se`, and `plot_all`. The main comparison uses twenty seeds; the regularisation experiment uses fifty. Error is measured over all states. Standard-error bands are not 95% confidence intervals.

## Rehearsal and recovery plan

1. Rehearse once with the actual slides and a single running timer. Include clicks, equations, and handoffs. Do not estimate time from reading silently.
2. Check the clock at 3:45, 7:30, and 11:30. Finish by 16:00 even if an earlier section runs long.
3. If a speaker is about 15 seconds behind, use the relevant short version below. Keep the main algorithm explanation and Meena's limitations.
4. Do not present the backup slides during prepared remarks. Navigate to them only for a question.

| Speaker | Short version if behind |
|---|---|
| A | On slide 7, say: “The implementation uses shared transition tuples, fixed features, and seeded datasets. Nadhiya will now explain the TD update.” |
| B | On slide 11, state the schedule sensitivity and best-setting change; omit the individual parameter values. |
| C | On slide 18, state the three computational costs and hand off; omit the graph-colour walkthrough. |
| D | On slide 22, state only that lambda sensitivity is weak because the Boyan features are exact, then advance. |

Handoffs are part of the allocated time. The next speaker should begin with the next question in the argument, without repeating the preceding introduction or thanking each team member.

## Q&A preparation — outside the timed script

Keep the first answer to 20–30 seconds. The named owner answers first; another member adds detail only if needed. Never defend a stronger claim merely because a slide uses shorthand.

### A: Why is this reinforcement learning if the policy is fixed?

We study policy evaluation, a core subproblem in RL. The learner estimates long-term returns from sampled transitions. Action selection and policy improvement are outside this project; a fixed policy induces the Markov reward process used here.

### A: Does knowing the environment model make the comparison model-based?

The simulator and evaluator know the model. TD and LSTD receive only transitions and fixed features. `true_values` solves the Bellman equations to produce the evaluation target; that solution is never supplied as training data. Separately, Boyan gives a model-based interpretation of tabular LSTD(0), which is a statement about its empirical solution.

### A: How can the stored terminal feature be nonzero when the algorithm requires zero?

Boyan's stored feature row for state zero is `[0, 0, 0, 1]`. Both learners inspect `done` and explicitly substitute zero for the terminal next-state value/features in their updates. Do not say the stored Boyan terminal feature vector itself is zero. The true terminal value is zero, and the exact weights predict zero there; a finite-sample learned weight vector need not do so when evaluated directly against that stored row.

### A: What does the shared feature representation buy us?

It reduces thirteen state values to four parameters and shares information between states. In Boyan, interpolation between anchor weights exactly represents `V(s) = -2s`. In the random walk, the five aggregation groups deliberately introduce representation error.

### B: Are you implementing online TD or the paper’s episode-wise version?

The default is `online=False`: theta stays fixed within the episode, step contributions accumulate in `dtheta`, and the update is applied once at the end. `online=True` is an additional option. The conceptual one-step target is available before termination, but our default experiment still applies the weight change episode-wise.

### B: Why does the corridor’s later B data not immediately change A in ordinary TD?

In the described ordering, A is visited only in the first episode and later episodes start at B. For tabular TD(0), those later B updates do not change A's separate parameter. A must be revisited or the earlier data replayed. With shared features or nonzero traces, the details can differ. The example is specifically the tabular, lambda-zero comparison.

### B: Does TD fail to converge?

That is not our claim. Under appropriate assumptions and a suitable diminishing step-size schedule, linear TD converges. We compare finite-data learning performance. A constant step size can retain sampling fluctuations, and a convergent schedule can still learn slowly at the budgets studied. Do not generalise these on-policy results to arbitrary off-policy TD with function approximation.

### C: Is LSTD ordinary regression on the one-step TD targets?

Not generally. Those targets contain the unknown weights, and the next-state features are sampled. LSTD solves the empirical eligibility-weighted TD balance `sum(z_t * delta_t) = 0`. Its matrix need not be symmetric. At lambda one, complete episodic data with zero terminal bootstrap yield ordinary regression of observed returns on features. Backup slide 28 gives the telescoping argument.

### C: How exactly do C and d become A and b?

For one episode, the expected correction is proportional to `d + C theta`. Across n sampled episodes, `b/n` estimates d and `A/n` estimates minus C, since A reverses the feature subtraction. Thus `d + C theta = 0` becomes `b - A theta = 0`, or `A theta = b`. Here, bold d is a reward vector; K, also labelled d on the cost graph, is the feature count.

### C: Does one solve give the true value?

It solves the system estimated from the available data, subject to rank, regularisation, and numerical accuracy. It does not remove sampling error or representation error. The corridor's 0.75 is the empirical TD(0) solution; the true transition law is not inferred perfectly from arbitrary finite data.

### C: What happens if A is singular or nearly singular?

The batch solver first calls `numpy.linalg.solve`. If it raises `LinAlgError`, `numpy.linalg.lstsq` supplies an SVD-based solution. A near-singular matrix may not raise that exception, so this fallback is not a full conditioning safeguard. Diagonal regularisation is the separate extension evaluated by Meena.

### C: Why use recursive LSTD, and when does it equal batch LSTD?

It is useful when updated weights are needed frequently. Starting with `B0 = delta I` represents `A0 = I/delta`, so the matching batch system uses `reg = 1/delta`. Equivalence assumes the inverse updates are well-defined and numerically accurate. Recursive LSTD is not automatically faster than one batch solve at the end. Our implementation returns `B @ b` at the end; intermediate predictions could use that same multiplication.

### C: What are the model-based and Monte Carlo views in the paper?

With tabular features and gamma one, LSTD(0) builds the same equations as counting visits, transitions, and reward sums, then solving the empirical chain's Bellman equations. LSTD(1) becomes least-squares Monte Carlo regression. These are special cases, not a claim that every lambda and arbitrary feature map exactly reconstructs a transition model. Backup slide 30 gives the two interpretations.

### D: Is the TD comparison fair?

Each algorithm sees the same episodes for each seed. We compare a finite, declared grid of six constant and six decaying schedules; the dashed reference selects the best tested configuration separately at each budget using ground-truth evaluation. That is an optimistic benchmark, not a practical hyperparameter-selection method or proof against every possible TD tuning strategy. The ridge experiment uses a different reference grid: five constant step sizes.

### D: Why does lambda barely matter for LSTD here?

The Boyan value is exactly representable, and the episodic assumptions give a common true-value solution under appropriate coverage. Finite-sample estimates still vary with lambda. With imperfect features, different lambda values can yield different projected solutions, so the observed insensitivity should not be generalised.

### D: Is your ridge extension new to the literature, and does it guarantee invertibility?

We present it as a project extension to this reproduction, not a new invention in RL. The implemented method adds epsilon identity directly to A. Since a general LSTD matrix is not a symmetric Gram matrix, do not claim that every positive epsilon guarantees invertibility for every possible dataset. Also do not replace the implemented system with `(A^T A + epsilon I) theta = A^T b`; that is a different regularisation procedure.

### D: Why does regularisation help more in the random walk?

With few episodes, some feature groups may be poorly covered, and the empirical system can be unstable or underdetermined. Diagonal regularisation changes that system and can reduce estimation variability at the cost of bias. The values in this symmetric random walk are centred around zero, so shrinkage toward zero can be favourable. It is not a universally neutral choice or a general guarantee for weak features.

### D: How do we know the code matches the mathematics?

Tests check environment transitions and features against the paper, an episode-wise TD update by hand, and A and b against hand calculations. They compare recursive and batch LSTD at matching regularisation, verify the Monte Carlo and tabular model-based special cases, and exercise the full experiment-to-plot pipeline. These checks support implementation consistency; they do not replace the paper's assumptions or establish results for new domains.

## Source and notation guide

The script was checked against the current deck, Boyan's supplied paper, the implementation, configuration, and existing tests. The prepared talk covers the main story; the Q&A notes retain details that should not be squeezed into the four-minute sections.

- Paper: [Boyan, Least-Squares Temporal Difference Learning (1999)](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/boyanLSTDlambda.pdf), especially the TD/LSTD derivation, Figures 1–2, the model-based interpretation, and Figures 4–6.
- Current slide source: [lstd_presentation.tex](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/slides/lstd_presentation.tex).
- A implementation: [env.py](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/src/env.py) and [features.py](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/src/features.py).
- B implementation: [td_lambda.py](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/src/td_lambda.py).
- C implementation: [lstd.py](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/src/lstd.py).
- D implementation and settings: [run_experiment.py](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/src/run_experiment.py), [plot_results.py](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/src/plot_results.py), and [experiment.yaml](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/configs/experiment.yaml).
- Corridor provenance: [MC_vs_TD_vs_LSTD.md](/Users/skannamp/Personal/m-tech/sem-courses/04-reinforcement-learning/lstd-rl-project/docs/MC_vs_TD_vs_LSTD.md). This is an illustrative teaching example, not Boyan's experimental domain.

Say “lambda zero,” “lambda one,” “eligibility trace,” “outer product,” and “Sherman–Morrison.” On the slides, C/d describe the expected system, A/b describe the sampled system, and capital B denotes the maintained inverse only in recursive LSTD. Lowercase b is the reward-statistics vector. The original paper uses undiscounted episodes; our code and displayed equations include gamma, set to one in these environments.

This coordinated script replaces the earlier timing plans in the individual draft scripts for rehearsal purposes. It does not change the slides, algorithms, or experimental results.
