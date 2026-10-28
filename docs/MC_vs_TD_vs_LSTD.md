# MC vs TD vs LSTD: explained with one tiny example

This walkthrough uses one tiny example (a corridor and a slot machine) to explain:

1. What a **value** is
2. **Monte Carlo (MC)** vs **Temporal Difference (TD)**
3. How **TD learns** step by step, and why it needs a step size
4. How **LSTD** (our paper) gets the answer in one shot

---

## 0. The setup: a corridor and a slot machine

```
   A (corridor) ──(reward 0)──► B (slot machine) ──(reward 1 or 0)──► END (go home)
```

- **Room A is a corridor.** Walking through it gives you nothing, and it **always** leads to Room B.
- **Room B has a slot machine.** It pays **1** with 75% probability and **0** with 25% probability. Then the episode ends.

### What is a value?

The **value** of a state, $V(s)$, is the **average total reward you will collect from that state until the episode ends**. Informally: "how good is it to be here?"

A value is an **average over many trips, not a promise about any single trip.**

**V(B):**

$$V(B) = 0.75 \times 1 + 0.25 \times 0 = 0.75$$

**V(A):** you get 0, then you are always at B:

| Trip from A | Chance | Total reward |
|---|---|---|
| A → B → machine pays 1 | 75% | **1** |
| A → B → machine pays 0 | 25% | **0** |

$$V(A) = 0 + V(B) = 0.75$$

A trip from A has exactly the same outcomes as a trip from B, because A adds nothing and just hands you over to B. So "being in A is as good as being in B" holds **on average**, not on every trip.

These are the **true** values. We can compute them only because we know the probabilities. **MC and TD don't know the probabilities; they only see experience.**

---

## 1. The data: 8 episodes

| Episode | What happened |
|---|---|
| 1 | start in corridor: A → B (reward 0), then B → END (reward **0**). Unlucky, the 25% case |
| 2–7 | start at the machine: B → END (reward **1**), six times |
| 8 | start at the machine: B → END (reward **0**) |

---

## 2. Monte Carlo vs TD

Both methods do the same thing: **nudge a guess toward a target.**

$$\text{new guess} = \text{old guess} + \alpha \times (\text{target} - \text{old guess})$$

The **only** difference is the target:

| Method | Target | In words |
|---|---|---|
| **MC** | $G_t$ = actual total reward to the end of the episode | "What actually happened, all the way to the end" |
| **TD** | $r_t + V(s_{t+1})$ | "What happened on the next step, plus **what I already believe** about where I landed" |

Where the TD target comes from: the return splits into "first reward" + "everything after":

$$G_t = \underbrace{r_t}_{\text{seen}} + \underbrace{G_{t+1}}_{\text{rest of episode}}
\quad\longrightarrow\quad
\text{TD target} = r_t + \underbrace{V(s_{t+1})}_{\text{my current guess}}$$

MC **waits and observes** the rest of the episode, which is one random sample. TD **doesn't wait**; it plugs in its current estimate. Updating a guess using another guess is called **bootstrapping**.

### Estimating V(B)
B was visited 8 times with returns 0, 1, 1, 1, 1, 1, 1, 0, so V(B) = 6/8 = **0.75**. MC and TD agree.

### Estimating V(A), where they disagree

**MC:** "The one time I started in A, I went home with **0**. So V(A) = 0."

**TD:** "Walking through A gives me nothing, and it **always** puts me in front of the machine. I've played that machine 8 times and it pays 75% of the time. So V(A) = 0 + 0.75 = **0.75**."

| | True value | MC estimate | TD estimate |
|---|---|---|---|
| V(B) | 0.75 | 0.75 | 0.75 |
| V(A) | 0.75 | **0** | **0.75** |

### Why TD is right here

Split the one episode from A into its pieces:

| Part of episode | Reward | Whose "fault"? |
|---|---|---|
| A → B | 0 | **A's doing.** Always 0, nothing random. |
| B → END | 0 | **B's luck.** The machine happened to lose. |

- **MC** treats the whole episode as one number and **blames A for B's bad luck**.
- **TD** keeps A's own part and, for the B part, uses **B's average over 8 plays** instead of one unlucky play.

### They agree eventually
If you started in A many times, MC would also average to 0.75. The difference is **speed**:
- MC needs many visits to **A itself**.
- TD learns V(A) from **one visit**, by borrowing what it knows about the next state.

### Trade-off

| | MC | TD |
|---|---|---|
| Target | One long random sample | One step + a guess |
| Bias | None | Yes, while the guesses are wrong |
| Variance | High | Low |
| Must wait until the episode ends? | Yes | No |

---

## 3. How TD actually learns, step by step

So far we showed TD's **final** answer (0.75). In practice TD gets there through small updates.

**The notebook analogy:** you keep a notebook with two numbers, "how good is the corridor?" and "how good is the machine?". Both start at **0** (no idea).

**The update rule with α = 0.5:** after each step, move **halfway** from your old belief toward the new evidence:

$$\text{new} = \text{old} + 0.5 \times (\text{target} - \text{old})$$

- **Machine B:** the episode ends after B, so the **target is just the payout** (1 for a win, 0 for a loss).
- **Corridor A:** the episode continues, so the **target is reward + the notebook's current value for B**.

### Reading the diary once, in order

**Day 1 (start in the corridor):**
- Walk through A and arrive at the machine. Update A: target = 0 + B = 0 + **0** → new A = 0 + 0.5 × (0 − 0) = **0**.
  *(The notebook knew nothing about the machine yet, so A borrowed a useless guess.)*
- Play and **lose**. Target = 0 → B = 0 + 0.5 × (0 − 0) = **0**.

**Days 2–8 (start at the machine):**

| Day | Old B | Target | Gap (target − old) | Half the gap | New B |
|---|---|---|---|---|---|
| 2 (win) | 0 | 1 | 1 | 0.5 | **0.5** |
| 3 (win) | 0.5 | 1 | 0.5 | 0.25 | **0.75** |
| 4 (win) | 0.75 | 1 | 0.25 | 0.125 | **0.875** |
| 5 (win) | 0.875 | 1 | 0.125 | 0.0625 | **0.9375** |
| 6 (win) | 0.9375 | 1 | 0.0625 | 0.03125 | **0.96875** |
| 7 (win) | 0.96875 | 1 | 0.03125 | 0.015625 | **0.984** |
| 8 (lose) | 0.984 | 0 | −0.984 | −0.492 | **0.492** |

- Each win closes **half the remaining distance to 1**.
- One loss moves B **halfway toward 0**, a big drop from 0.984 to 0.492.

**After one read:** A = **0**, B = **0.49**. Both are wrong; the truth is 0.75.

### What went wrong?

1. **The corridor was visited too early.** On day 1 the notebook knew nothing about the machine, so A borrowed 0 and never got updated again. **A never heard the good news.**
2. **The mood swings are too big.** Moving halfway every time meant one loss cut B in half.

### How TD fixes it: reread many times, calmly
- **Replay the diary many times**, so A is updated *after* B has been learned.
- **Use a small α** (e.g. 0.01) so a single result doesn't swing the estimate.

After about **2,000 rereads with α = 0.01**, the notebook settles at **A = 0.75, B = 0.75**. That's correct, but slow, and **you had to choose α**.

### What α means

| α | Behaviour |
|---|---|
| 1 | Jump all the way to the latest target; only the last result counts (maximum mood swing) |
| 0.5 | Move halfway (the example above) |
| 0.01 | Move 1% of the way: calm, but needs many updates |

**Choosing α is the step-size problem the paper says TD suffers from.**

---

## 4. LSTD: tally the diary once, then solve

Instead of rereading and nudging, LSTD **summarises the whole diary once**:

- "The corridor gave 0 and **always** led to the machine."
- "The machine paid out **6 times in 8** plays, so 0.75 on average."

Then it **solves directly**:

$$V(B) = 0.75 \qquad V(A) = 0 + V(B) = 0.75$$

| | TD (notebook, nudging) | LSTD (tally, then solve) |
|---|---|---|
| After reading the diary once | A = 0, B = 0.49 | **A = 0.75, B = 0.75** |
| Step size α needed? | Yes | **No** |
| Does day order matter? | Yes | **No** |
| Reaches the right answer | After about 2,000 replays | **Immediately** |

> **The paper's claim:** LSTD computes, in one shot, the answer TD would reach if you replayed the data forever with a perfectly tuned step size. Same answer, less data, nothing to tune.

### How this connects to the paper's maths
- The "tally" is the matrix $A$ and vector $b$, built from the data (paper Fig. 2, our `src/lstd.py`).
- "Solve" means $\theta = A^{-1} b$.
- Paper Sec. 3: with one feature per state, this tally is **exactly** "count transitions, build an empirical model of the chain, solve it". LSTD is secretly **model-based**.
- In the real problem there are too many states for a table, so $V(s) = \phi(s)^\top \theta$ (features). The idea doesn't change.

### The λ knob
TD(λ) and LSTD(λ) slide between the two extremes:
- **λ = 0:** pure TD, one step plus a guess → V(A) = 0.75
- **λ = 1:** pure MC, the full actual return → V(A) = 0

---

## 5. Verify it yourself with our code

```bash
source .venv/bin/activate
python -c "
import numpy as np
from src.lstd import lstd_lambda
from src.td_lambda import td_lambda
A, B, END = 2, 1, 0
eps = [[(A,0.0,B,False),(B,0.0,END,True)]] + [[(B,1.0,END,True)]]*6 + [[(B,0.0,END,True)]]
phi = np.eye(3)   # one feature per state = a lookup table
print('LSTD(0)  V(A),V(B) =', lstd_lambda(eps, phi, 1.0, 0.0)[[A,B]].round(3))
print('LSTD(1)  V(A),V(B) =', lstd_lambda(eps, phi, 1.0, 1.0)[[A,B]].round(3))
print('TD(0) 2000 passes  =', td_lambda(eps*2000, phi, 1.0, 0.0, 0.01)[[A,B]].round(3))
print('TD(1) 2000 passes  =', td_lambda(eps*2000, phi, 1.0, 1.0, 0.01)[[A,B]].round(3))
"
```

Output:
```
LSTD(0)  V(A),V(B) = [0.75 0.75]    <- one shot, TD-style answer
LSTD(1)  V(A),V(B) = [0.   0.75]    <- lambda = 1 behaves like Monte Carlo
TD(0) 2000 passes  = [0.75 0.75]    <- TD needs 2000 replays to get there
TD(1) 2000 passes  = [0.   0.75]
```

---

## 6. Exercise for the team

Redo the TD diary in Section 3 with **day 1 moved to the end**, so the corridor is visited *after* all the machine days.

- What is V(A) after one read?
- Why is it no longer 0?
- What does this tell you about TD's sensitivity to the order of data, and why doesn't LSTD have this problem?

---

**Further reading:** Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.):
- §6.1–6.3: TD prediction; Example 6.4 is this example
- §7.1: n-step TD
- §9.1–9.4: linear function approximation
- §12.1–12.2: TD(λ)
- §9.8: LSTD
