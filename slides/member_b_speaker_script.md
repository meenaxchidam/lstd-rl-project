# Member B presentation script: TD, our experiment, and results

**Target length:** about 3 minutes. Read the quoted text aloud; bracketed text is a presentation cue.

## Slide 6 — TD: what it is and what we tested

“Temporal Difference, or TD, is a way to estimate how much future reward a state is worth. It learns from sampled experience. After each transition, it uses the reward just received and its current estimate of the next state’s value. This use of its own estimate is called bootstrapping.

Our experiment asks: on the Boyan chain, how does TD lambda compare with LSTD in how quickly they estimate state values? We run both methods, vary the amount of experience and the TD step-size settings, and measure their root mean squared error across all 13 states. Lower error means a more accurate estimate.

Our full run shows LSTD with lower error at both checkpoints: at 100 trajectories, 0.152 compared with 0.308 for the best paper-style TD schedule; at 10,000 trajectories, 0.0157 compared with 0.0362. I’ll explain the update next, then show the full setup and comparison with the paper.”

## Slide 7 — How TD updates, and how it differs from Monte Carlo

“For a fixed policy, the value of a state is its expected discounted future reward. In our code, we estimate that value using state features multiplied by learned weights.

Monte Carlo waits until an episode ends and uses the full observed reward sequence as its target. TD can update after a single transition. Its target is the reward just received plus its current estimate of the next state’s value. The difference between that target and the current estimate is the TD error. Monte Carlo uses no next-state estimate; TD bootstraps from one. MC’s target is unbiased but often has higher variance. TD often has lower variance, but is biased while its estimates are wrong.

Here is a small TD zero example. The current estimate is 2, the reward is 1, the next-state estimate is 4, and gamma is 0.9. The TD target is 1 plus 0.9 times 4, or 4.6. Subtract the current estimate, 2, and the TD error is 2.6. With a learning rate alpha of 0.1, the estimate moves from 2 to 2.26.

TD lambda adds an eligibility trace, a short memory of recently visited states. The trace lets an error update the current state and recent states. In our default code, those updates are summed during an episode and applied at the end.”

**Point to the contrast:** MC waits for the full return; TD uses one reward and its estimate of the next state.

## Slide 8 — Results vs Boyan (1999), Figure 5

“We use the 13-state Boyan chain with four features, gamma equal to 1, and lambda equal to 0.4. The paper used 10 trials and six decaying TD step-size schedules, with up to 10,000 trajectories. Our full run uses the same chain, features, lambda, six schedules, and trajectory limit. It averages over 20 random seeds and adds six constant step-size settings.

This plot shows our results for the six paper-style TD schedules and LSTD. Lower on the vertical axis means lower error. The table reports the mean plus or minus the standard error over our 20 seeds.

At 100 trajectories, the best of the six TD schedules has RMSE 0.308 plus or minus 0.031; LSTD has 0.152 plus or minus 0.022. At 10,000 trajectories, TD has 0.0362 plus or minus 0.0039; LSTD has 0.0157 plus or minus 0.0018. We choose TD’s best schedule separately at each checkpoint, which favors TD. LSTD is still lower at both.

This agrees with the paper’s conclusion that LSTD learns faster than the six TD schedules and has lower final error. The paper reports the averages as curves rather than an exact data table, so we say our result is consistent with the paper, not that the exact numbers match.”

## Rehearsal cues

- Begin with the order shown: what TD is, what we tested, and the headline result.
- On slide 7, explain the example one step at a time: target, error, then update.
- On slide 8, say “lower is better” before describing the graph.
- Call these results the **20-seed full run** and say **consistent with the paper** rather than claiming an exact numerical replication.
