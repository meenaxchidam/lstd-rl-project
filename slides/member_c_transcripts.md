# Member C — LSTD presentation transcripts

These scripts follow the five Member C slides in `lstd_presentation.tex` and `lstd_presentation.pptx`. Allow roughly five minutes at a comfortable pace. The quoted text is intended to be spoken aloud; transitions connect the slides.

## 1. Motivation for LSTD

> Nadhiya explained how TD learns by gradually adjusting its weights using observed rewards and value predictions.
>
> My starting question is: **what weights is TD trying to reach?**
>
> It is trying to reach a point where its corrections balance on average. Individual observations may still suggest changes, but there should be no average push to change the weights.
>
> With linear features, we can express this average correction as **d plus C theta**. Here, d summarizes the rewards, and C theta describes the contribution from our value predictions.
>
> Setting this correction to zero gives the equation shown here.
>
> LSTD’s idea is to estimate this equation from experience and solve it directly, instead of approaching its solution through many small updates.

**Transition:** “Let’s see how LSTD builds that equation from data.”

## 2. The LSTD(λ) algorithm

> LSTD collects two quantities: a matrix A and a vector b. We initialize both to zero.
>
> At the beginning of each episode, we reset the eligibility trace z. This trace remembers previously visited features, with older features receiving less weight according to gamma and lambda.
>
> For every transition, we first update the trace. We then add the trace times the current-minus-discounted-next feature difference to A. We also add the trace times the reward to b.
>
> Across episodes, we keep accumulating A and b. Only the trace is reset.
>
> These quantities connect directly to the previous slide: **b divided by the number of episodes estimates d, while A divided by that number estimates minus C**. The minus sign comes from reversing the feature subtraction.
>
> Therefore, d plus C theta equals zero becomes **A theta equals b**.
>
> Whenever we need predictions, we solve this system. There is no TD learning rate or initial weight guess to choose.

**Transition:** “Why might this be useful compared with ordinary TD?”

## 3. Why use LSTD instead of TD?

> The main benefits are effective use of the collected data and avoiding a TD step-size schedule. In the paper’s Boyan-chain experiments, LSTD learned from fewer trajectories than the tested TD schedules.
>
> This corridor example illustrates how it combines information.
>
> A leads to B with zero reward. B then gives either zero or one reward, and the episode ends. Here, we use gamma equal to one.
>
> We have eight episodes. One starts at A and finishes with reward zero. The other seven start at B: six give one, and one gives zero.
>
> Across all eight visits to B, the average reward is six divided by eight, or **0.75**.
>
> LSTD(0) solves both relationships together: B has value 0.75, and A gives zero reward before reaching B, so A also has value 0.75.
>
> Starting from zero, ordinary TD initially updates A using B’s zero estimate. Replaying the data lets B’s improved estimate gradually reach A. LSTD obtains the empirical solution through one pass and a solve.

**Transition:** “If we need updated predictions frequently, we can use recursive LSTD.”

## 4. Recursive LSTD

> Batch LSTD accumulates A and b, then solves the system when weights are needed.
>
> Recursive LSTD maintains a matrix B representing the inverse of A.
>
> Each transition changes A by an outer product, written here as z times u transpose. Because this is a rank-one change, the Sherman–Morrison formula lets us update the inverse using the previous inverse.
>
> We still update b using the observed reward. We then obtain the weights by multiplying B and b.
>
> The key idea is: **update the inverse as data arrives, instead of performing a fresh full solve each time**.
>
> In our implementation, we start with a regularized matrix so that the initial inverse exists.

**Transition:** “This brings us to the computational tradeoff.”

## 5. Computational cost

> Here, K is the number of features, and one step means one transition.
>
> TD updates vectors, so its cost per step is order K. LSTD updates a K-by-K matrix, giving order K squared per step. A full batch solve costs order K cubed.
>
> Recursive LSTD also costs order K squared per step, but avoids repeated full solves.
>
> The graph measures the time to process 200 episodes as the feature count increases. Lower means faster.
>
> Blue is TD. Orange is batch LSTD with one final solve. Green solves after every episode. Red performs a recursive inverse update after every transition.
>
> At smaller feature counts, green can be below red because one small solve per episode can be cheaper than several inverse updates.
>
> The tradeoff is **learning accuracy per observation versus computation**. LSTD is attractive when observations are costly and the feature count is manageable. TD has cheaper individual updates.

**Handoff:** “Meena will now show how these tradeoffs appear in our experimental results.”
