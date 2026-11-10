# Member B presentation script: TD, experiment, and results

**Target length:** about 3 minutes. Bracketed text is a presentation cue.

## Slide 6 — Problem: estimating state values with TD

“Temporal Difference learning, or TD, estimates how much future reward we can expect from a state. It learns from transitions we observe. After moving from one state to the next, TD combines the reward we just received with its current estimate of the next state’s value. The difference from the current state estimate is the TD error.

Our problem is to estimate the values of states in the Boyan chain using a compact set of features. We run TD lambda while changing the number of trajectories and the step-size settings. We measure root mean squared error over all 13 states, so a smaller number means a more accurate estimate.

In the full run, the best of the paper’s six decaying step-size schedules reaches RMSE 0.308 at 100 trajectories and 0.0362 at 10,000 trajectories. These are averages over 20 random seeds. Next I’ll show exactly how the update is calculated.”

## Slide 7 — Methodology: TD(lambda) update and parameters

“We represent the value estimate as the feature vector for a state multiplied by a weight vector: V-hat of s equals phi of s transposed times theta.

For each transition, the TD target is the reward plus gamma times the estimated value of the next state. We subtract the current estimate to get delta, the TD error. The eligibility trace is updated as z equals gamma lambda times the previous trace plus the current state’s feature vector. Then we add alpha times the TD error times the trace to the weights. In the paper-style offline implementation, we sum those weight changes during one trajectory and apply the total when that trajectory ends.

Here, phi is the four-feature state representation, theta is the learned weight vector, gamma is the discount factor and is 1, lambda is 0.4, and alpha is the learning rate. The trace z lets an error affect recently visited states, with older states fading by gamma lambda. Our decaying schedule is alpha sub n equals a-zero times n-zero plus 1, divided by n-zero plus n. We test a-zero values 0.01 and 0.1, crossed with n-zero values 100, 1,000, and 1 million. We also test constant alpha values 0.001, 0.003, 0.01, 0.03, 0.1, and 0.3.

For a one-step example, suppose the current estimate is 2, the reward is 1, the next-state estimate is 4, and gamma is 0.9. The target is 1 plus 0.9 times 4, or 4.6. The error is 4.6 minus 2, or 2.6. With alpha equal to 0.1 and a one-hot state feature, the new estimate is 2 plus 0.1 times 2.6, which is 2.26.”

## Slide 8 — Results: TD(lambda) on the Boyan chain

“The experiment uses the 13-state Boyan chain. Each trajectory starts at state 12 and ends at state 0. There are four linearly interpolated state features. We use gamma equal to 1, lambda equal to 0.4, and evaluate RMSE over all 13 states. The code evaluates 20 random seeds and tracks performance from 1 up to 10,000 trajectories.

The graph shows the six decaying schedules used in Boyan’s Figure 5. The table gives the best decaying schedule and best constant learning rate at two checkpoints. At 100 trajectories, the best decay is 0.308 plus or minus 0.031, compared with 0.379 plus or minus 0.046 for the best constant rate. At 10,000 trajectories, the best decay is 0.0362 plus or minus 0.0039, and the best constant rate is 0.0413 plus or minus 0.0037.

The best decaying schedule changes from a-zero 0.1 and n-zero 100 at the earlier checkpoint to a-zero 0.01 and n-zero 1,000 at the later checkpoint. This shows that step-size choice affects the result. Overall, TD’s error drops as it sees more trajectories. The paper reports 10 trials; our run uses 20 seeds and includes six extra constant-rate settings, so these are our measured values under a closely matched setup.”

## Rehearsal cues

- Follow the order: problem, update method, then measured results.
- On slide 7, explain the target first, then the error, trace, and weight update.
- On slide 8, say “lower RMSE means a more accurate estimate” before describing the numbers.
- Describe the paper connection as a closely matched setup; the values in the table come from our own 20-seed run.
