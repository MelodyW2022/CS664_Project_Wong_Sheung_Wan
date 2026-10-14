# CS664 presentation script

## 0:00–0:40 — Opening and start the full run

Screen:

Show `# Flight Booking Decision Assistant — completed experiment`. From the project root, launch the full run by 0:20:

```sh
time .venv/bin/python run_notebook.py
```

Then show `# 1A — imports and official settings`.

Say:

**This project uses Q-learning to study when to book a simulated flight at a lower expected cost.**

I will start the whole project now and explain the code while it runs.
The outputs currently visible are from an earlier saved run.
After the runner finishes, I will reload the Notebook and show new outputs.
This is a small machine learning application. It uses simulated prices, not
real flight data.

## 0:40–1:55 — Environment and simulation rules

Screen:

Show `## 2. Price simulator`, then `# 2A — one price transition` and `# 2B — episode and terminal rules`. Point to probabilities, clipping, rewards, and the deadline/termination branches.

Say:

**The environment makes every episode end with one purchase, within the allowed price range.**

The state contains time remaining and current price. There are seven buying
opportunities, from time six to zero. Prices are 120, 180, 240, 300, or 360
dollars. Each starting price is equally likely.

BOOK pays the current price. Its reward is the negative price, and the episode
ends. WAIT gives zero reward and moves to the next day. It still leads to a
later purchase cost.

At times six to three, prices rise, stay, or fall with probabilities 30, 30,
and 40 percent. At times two and one, they are 60, 20, and 20 percent.
These probabilities are assumptions.

Clipping keeps prices between 120 and 360 dollars. At time zero, the program
forces BOOK, even if WAIT is requested. Another action after purchase raises
an error. These checks test rewards, the deadline, and termination.

## 1:55–3:20 — Q-learning code, updates, and exploration

Screen:

Show `# 3B — one real Q-value update` and `# 3D — display actual updates and BOOK/WAIT predictions`. Point to terminal versus WAIT targets. Then show `# 3A — indexing, exploration, and greedy prediction` and the update/break in `# 3C — training loop`.

Say:

**Q-learning improves an action estimate by moving its old value toward a target.**

The table starts at zero. Each update adds alpha times target minus old value.
Alpha is 0.1. Gamma is one, so the final purchase cost is not discounted.

Here is an actual logged update. In the first episode, the state was six and
120 dollars, and the action was BOOK. The old value was zero. The terminal
target was negative 120. Zero plus 0.1 times negative 120 gives negative 12.

A terminal purchase has no future value. For WAIT, the target includes the
best legal next-state Q-value. At the deadline, that means BOOK only.
The logged WAIT example had old value zero, reward zero, and target negative
30. Its updated value was negative three. That was an estimate during learning.

Training uses epsilon-greedy exploration. Epsilon falls from one to 0.05 over
40,000 episodes, then stays there. The original training has 50,000 episodes.
Testing uses only greedy actions, with no exploration or updates.

## 3:20–4:20 — Fair comparison and validation questions

Screen:

Show `## 4. Fair evaluation on 10,000 fresh trajectories`, `# 4B — state-only policies`, and `# 4C — stop-on-purchase evaluator`. Point to `policy(state)` and the return on BOOK.

Say:

**A fair comparison gives every strategy the same test paths and no future-price information.**

The four rules are: buy on the first day; buy at 120 dollars or at the deadline;
buy at 180 dollars or below, or at the deadline; and follow Q-learning.

All four use the same 10,000 fresh test trajectories. Training seed 66401 and
test seed 66402 are separate. The evaluator stores each path, but the decision
function receives only the current state. It stops calling the policy after
purchase. The trained Q-table stays unchanged during evaluation.

There are three validation questions. Does Q-learning lower cost? How reliable
are the estimated savings? How close is it to optimal, and does the program
follow the rules?

## 4:20–4:50 — Confirm full execution and reload

Screen:

Show the terminal. Require `PASS: 80 code cells executed in a fresh kernel; no cell errors.`, the saved-Notebook message, and a returned prompt. Reload the main Notebook from disk before showing fresh results. Say the following only after success.

Say:

**The fresh full run has completed, and these reloaded outputs come from that run.**

The runner executed all 80 code cells, including both supplementary studies,
without errors. It preserved the original numerical evidence.
I have reopened the saved Notebook.

## 4:50–5:40 — Cost results and one complete trace

Screen:

Show the fresh table in `# 4E — official comparison and paired cost differences`. Then display all rows of `# 5A — actual states, actions, and final cost`.

Say:

**The original learned policy had the lowest sampled mean cost among the four strategies.**

The means were 239 dollars and 58 cents for immediate booking, 223 dollars
and 48 cents for the 120-dollar rule, and 219 dollars and 74 cents for the
180-dollar rule. Q-learning averaged 214 dollars and 85 cents.

These are sample means, not exact model expectations.
This complete trace starts at 360 dollars. The policy waits until the deadline
and buys for 360 dollars. This example shows every decision and termination.
It also shows why average savings do not mean savings on every trip.

## 5:40–6:30 — Reliability and paired confidence intervals

Screen:

Show `# 8B — save all three paired savings intervals`. Point to the $180 baseline row and its interval.

Say:

**The paired intervals support positive mean savings for this fixed trained policy.**

For each trip, saving means baseline cost minus Q-learning cost. Pairing keeps
the comparison on the same price path.

Against the 180-dollar rule, mean saving was about four dollars and 88 cents.
Its 95 percent interval was about four dollars and three cents to five dollars
and 74 cents. All three intervals are above zero.

These intervals describe uncertainty in mean savings. They do not contain
95 percent of individual-trip savings. They also do not measure variation
between training seeds. The three intervals are individual, not simultaneous.

## 6:30–7:50 — Optimality, policy, and verification

Screen:

Show the recurrence in `# 6B — backward induction and exact policy evaluation`, the output of `# 6D — save analytical comparison and per-state references`, and the figure in `# 7A — draw and save the learned policy heatmap`. Finish with the PASS output in `# 9C — save actual run summary and trace`.

Say:

**The implementation passed its checks, but the original learned policy is still suboptimal.**

Backward induction works from the deadline toward the first day. At each
state, it compares buying now with the expected cost of waiting.
This reference uses the known simulator probabilities. The learning agent
does not receive them.

The optimal expected cost is about 207 dollars and 27 cents. The original
learned policy's expected cost is about 215 dollars and 71 cents. Its gap is
about eight dollars and 43 cents. These are analytical expectations, not the
sample means shown earlier.

The heatmap shows BOOK or WAIT for each time and price. At the deadline, every
price means BOOK. This is a decision policy, not a price forecast.

The program checked 90 transitions, all seven purchase days, reward targets,
boundaries, forced booking, and termination. It also checked future-price
isolation and independently verified the exact reference. Passing these finite
checks supports correctness; it does not prove convergence or cover every bug.

## 7:50–8:50 — One minute on supplementary findings

Screen:

Show `# S3C — paired gap changes and mean/range summaries`, then `# L4B — means and ranges across the five seeds`. Spend about one minute on these supplementary findings.

Say:

**The supplementary studies show that more training alone did not help, while the tested learning rates improved final performance.**

The first study restarted five seeds at 50,000 and 200,000 episodes, with the
original settings. Every seed had a higher final expected cost at 200,000.
Two budgets do not prove a continuous trend.

The next study compared constant alpha 0.1, constant 0.03, and visit-based
decay. Each condition used five seeds and 200,000 episodes. Mean exact costs
were about 220 dollars and 98 cents, 213 dollars and 32 cents, and 209 dollars
and 16 cents.

Both alternatives improved all five final policies relative to alpha 0.1.
Every final policy still had a positive optimality gap. These exploratory
results show sensitivity, not convergence or a unique cause. Seed variation
is different from test-trip uncertainty. The original policy was not replaced.

## 8:50–9:35 — Limits and answers to the three questions

Screen:

Show `## 10. Interpretation and limits` and `# 10A — evidence-based interpretation`.

Say:

**The project supports savings inside this simulator, with clear limits on reliability and optimality.**

For the first question, the original policy lowered sampled mean cost against
all three baselines. For the second, the paired intervals support positive
mean savings for that policy. Training-seed variation needs separate evidence.
For the third, checks passed, but the analytical gap shows room for improvement.

The model uses hypothetical probabilities and a small price grid. It leaves
out refunds, cancellations, and sold-out flights. The tested rate settings are
not proven optimal. The remaining gap is not fully explained. None of these
results proves savings on real flight prices.

## 9:35–10:00 — Close and leave the completed demo visible

Screen:

Return to the fresh table in `# 4E — official comparison and paired cost differences`. Keep your camera visible. Use any remaining time for a clear closing.

Say:

**The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.**

This project demonstrates a complete Python Q-learning workflow, from the
environment to evaluation and verification.
Thank you for watching.
