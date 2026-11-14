# Member B speaker script: why MC leads to TD

**Target length:** about 4–5 minutes. The sequence hands off to Member C’s LSTD section.

## Slide 6 — Why move from Monte Carlo to TD?

“Monte Carlo and temporal difference both learn values from sampled experience. Monte Carlo waits for the episode to end and uses the actual return. This is unbiased on average, but the return can be noisy because it includes a long sequence of random outcomes. It also cannot update until the episode finishes.

In our corridor example, A always leads to B with zero reward, and B pays one with probability 0.75. In our small data set, A appears once and that episode loses. Monte Carlo’s one observed return for A is zero. That is a valid sample, but a poor estimate from one visit. Meanwhile B has eight observed plays and an average return of 0.75.

TD replaces the unseen rest of the episode with the current estimate of the next state. For A, its target is zero plus the current estimate of B. Once B has been learned, TD can pass that value back to A through the transition. This is bootstrapping. It lowers target variance and can use visits to later states to improve earlier-state estimates. The trade-off is that TD is biased while its next-state estimate is inaccurate. MC would also approach 0.75 with enough visits to A; TD’s advantage here is that it needs fewer direct A samples.”

## Slide 7 — TD(λ) algorithm and the role of λ

“We approximate value with features times weights. On each transition, TD computes the error: observed reward plus discounted estimated next value, minus the current value. The eligibility trace z records which features were recently active. Then the error is assigned to those features, weighted by the trace and the learning rate alpha.

In our paper-style implementation, we hold the weights fixed during one trajectory, sum the trace-weighted errors, and apply the total update at the end of that trajectory. The trace decays by gamma times lambda, so recent states receive more credit than older states.

The paper does not give TD three separate views like its later section does for LSTD. TD’s key interpretation is the lambda continuum: lambda zero uses a one-step bootstrapped target; lambda one uses the full Monte Carlo return; values between them combine multi-step targets. This trades off bootstrapping bias against return variance. Averaged over data, TD approaches a fixed point described by d plus C theta equals zero. How quickly it gets there depends on the step-size schedule.”

## Slide 8 — TD experiment, results, and computation

“We evaluate TD on the 13-state Boyan chain, with four linear features, gamma one, lambda 0.4, and 20 random seeds. We report mean RMSE plus or minus standard error over all 13 states. At 100 trajectories, the best tested decaying schedule gets 0.308 plus or minus 0.031, and the best constant rate gets 0.379 plus or minus 0.046. At 10,000 trajectories, those errors are 0.0362 plus or minus 0.0039 and 0.0413 plus or minus 0.0037. The best constant alpha shifts from 0.1 at the smaller data budget to 0.001 at the larger one, showing why tuning matters.

These are TD-only results from our TD(lambda) implementation. TD takes order K work per transition and stores order K weights and trace. In our run, the best decaying schedule changed from a-zero 0.1 and n-zero 100 at 100 trajectories to a-zero 0.01 and n-zero 1,000 at 10,000. The best constant alpha shifted from 0.1 to 0.001. That shows how the useful learning rate depends on how much data TD has seen.”

## Supplement — What the paper says about TD

“One extra paper detail: TD’s convergence is conditional. With linear value functions, Boyan cites convergence for a suitable decreasing step-size schedule. The sufficient conditions used for the paper’s schedules are that the stepsizes are nonnegative, their sum diverges, and the sum of their squares converges.

The paper also points out TD’s limitations: each trajectory updates the weights but leaves no transition statistics behind; replay can reuse the data, but requires repeated passes. Performance can be sensitive to the step-size schedule, initial weights, and feature scaling.

For its TD experiments, the paper tests six decaying schedules across lambda values zero, point two, point four, point six, point eight, and one, averaging ten trials. Its conclusion about TD is that performance depends critically on the step-size schedule. The next section then introduces LSTD as a different algorithm.”

## Delivery cues

- State MC’s limitation as variance and waiting for the episode, not that MC is incorrect.
- Use “the lambda continuum,” not “three views of TD.” The paper’s three views are specific to LSTD.
- Keep the TD experiment slide TD-only; Member C presents LSTD and its computational cost.
- The extra paper-context slide is unnumbered so it does not change the deck’s frame numbering.
