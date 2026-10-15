# CS664 presentation script

## 0:00–0:40 — Opening

Purpose [not spoken]: Introduce the project and the earlier successful run.

Narration:

Hello, my name is Sheung Wan Wong. My project is called Flight Booking Decision Assistant.
(Show `# Flight Booking Decision Assistant`.)

**The goal is to learn when to book a simulated flight at a lower average cost.**

I previously ran the full Notebook successfully. I’ll walk through the code and the saved results from that run.
(Keep the main Notebook open with its saved outputs.)

## 0:40–1:15 — Simulated prices

Purpose [not spoken]: Explain how the simulator creates bounded daily prices.

Narration:

Let’s start with the prices. The model uses five prices, from 120 to 360 dollars.
(Show `# 1A — imports and official settings`; point to `PRICES`.)

Each day, the price can rise by 60 dollars, stay the same, or fall by 60 dollars.
(Show `# 2A — one price transition`; point to `rng.choice([60, 0, -60], p=probabilities)`.)

Earlier in the trip, the chance of a fall is 40 percent. Close to the deadline, it is 20 percent.
(Point to `probabilities`: the early and late tuples, in the order rise, unchanged, fall.)

Clipping keeps the price inside the allowed range. These probabilities are assumptions, not patterns measured from real flights.
(Point to `np.clip` in the same cell.)

## 1:15–1:55 — One booking episode

Purpose [not spoken]: Explain state, actions, rewards, the deadline, and termination.

Narration:

Let’s look at how one booking episode works. The state contains the current price and time remaining. There are seven opportunities, from time six to zero. Each starting price is equally likely.
(Show `# 2B — episode and terminal rules`; point to `reset` and its initial state.)

The first check rejects any action after a purchase. At time zero, the program forces BOOK, even if the agent requests WAIT.
(Follow `step` downward: point to `if self.done`, then `effective_action`.)

**Booking pays the current price and ends the episode.** Its reward is the negative price, so a cheaper purchase gives a better reward.
(Point to `self.done = True` and `return None, -price, True, effective_action` in the BOOK branch.)

If the agent waits, time goes down by one, a new price is sampled, and the reward is zero.
(Point to the two lines in the WAIT branch at the bottom of the same cell.)

## 1:55–2:30 — Choosing actions

Purpose [not spoken]: Distinguish exploratory training from greedy evaluation.

Narration:

During training, the agent needs to try both actions. Here, a random choice lets it explore instead of always following its current table.
(Show `# 3A — indexing, exploration, and greedy prediction`; point to `choose_training_action`, `rng.random() < epsilon`, and `rng.integers(2)`.)

The chance of a random choice starts at one. It falls to five percent over the first 40,000 episodes, then stays there.
(Point to `epsilon_for` in the same cell.)

**During testing, the agent uses what it learned without exploring.** It chooses the action with the higher Q-value, and the deadline always means BOOK.
(Point to `best_action`: the time-zero branch and `np.argmax`.)

## 2:30–3:20 — Learning and one actual update

Purpose [not spoken]: Explain the Q update, the training loop, and a logged numerical example.

Narration:

Here is how the table learns. If BOOK ends the episode, the target is just the reward. For WAIT, the target adds the best estimated value at the next state. Only BOOK is allowed when that next state is the deadline.
(Show `# 3B — one real Q-value update`; point to the `if done` and `else` target assignments.)

The update moves the old value part of the way toward that target. Gamma is one, so future cost is not discounted.
(Point to `q[row] = old + alpha * (target - old)` in the same cell.)

The original training starts with zeros and repeats this process for 50,000 episodes. Each episode stops after purchase.
(Show `# 3C — training loop`; point to `q = np.zeros`, the `update_q` call, and `if done` / `break`.)

Here is an actual logged example from the first episode. The agent booked at 120 dollars. The old value was zero, the target was negative 120, and the learning rate was 0.1. **The updated Q-value was negative 12.**
(Show `# 3D — display actual updates and BOOK/WAIT predictions`; point to the first observed update’s state, action, old, target, alpha, and updated value.)

## 3:20–4:20 — Fair evaluation and three questions

Purpose [not spoken]: Explain how the comparison avoids unequal test conditions and future-price access.

Narration:

For the comparison, all four strategies receive the same 10,000 new test paths. Testing uses a separate random seed from training.
(Show `# 4A — generate independent shared test trajectories`; point to `test_paths`, `TEST_TRAJECTORIES`, and `TEST_SEED`.)

The rules are to buy immediately, buy at 120 dollars or the deadline, buy at 180 dollars or below or the deadline, and follow Q-learning. The learned rule makes no random choices during testing.
(Show `# 4B — state-only policies`; point to `immediate_policy`, `threshold_policy`, and `learned_policy`.)

The evaluator stores the whole path, but each decision receives only today’s price and time remaining. It stops calling the policy after a purchase.
(Show `# 4C — stop-on-purchase evaluator`; point to `policy(state)`, then the return inside `if action == BOOK`.)

**No strategy can use tomorrow’s price to decide what to do today.**
(Point again to the current-state argument in `policy(state)`.)

There are three questions: Does Q-learning lower costs? How reliable are the savings? How close is the policy to the best possible one, and does the program follow the rules?
(Stay on the evaluator; pause before switching to the terminal.)

## 4:20–4:50 — Saved execution evidence

Purpose [not spoken]: Distinguish the earlier successful execution from this walkthrough.

Narration:

The Notebook contains the saved outputs from the earlier run. This walkthrough is not starting a new training run.
(Show `# 9C — save actual run summary and trace`; point to the saved PASS message and artifact summary.)

The saved output reports that the core checks passed. The final supplementary cell has also been executed.
(Show `# L5C — append actual findings, links, and limits to RESULTS.md`; point to its execution count, 80.)

## 4:50–5:20 — Four-strategy results

Purpose [not spoken]: Answer whether the original learned policy reduced average test cost.

Narration:

Here are the four averages. Buying immediately cost $239.58. The 120-dollar rule cost $223.48, and the 180-dollar rule cost $219.74.
(Show the saved output of `# 4E — official comparison and paired cost differences`; follow the first three rows in `Mean purchase cost`.)

Q-learning cost $214.85. **It had the lowest average purchase cost in this test.** Another test sample could give slightly different averages.
(Point to the Q-learning row in the same column.)

## 5:20–5:40 — One complete trip

Purpose [not spoken]: Connect the learned decisions with one final purchase.

Narration:

This table follows one trip from start to finish. The agent starts at $360, keeps waiting, and finally books at the deadline for $360. The last row ends the episode.
(Show `# 5A — actual states, actions, and final cost`; keep all seven rows visible and follow the state, action, and reward columns to the final BOOK.)

**Average savings do not mean savings on every trip.** This example explains the decisions; it does not estimate overall performance.
(Point to `Final purchase cost: $360` below the table.)

## 5:40–6:30 — How reliable are the savings?

Purpose [not spoken]: Interpret a paired confidence interval for average savings.

Narration:

Here, I’m comparing Q-learning with the rule that buys at 180 dollars or below. On the same trips, Q-learning saved about $4.88 on average.
(Show `# 8B — save all three paired savings intervals`; point to the $180 baseline row and its `Mean Q-learning saving` value, $4.884.)

The 95 percent confidence interval runs from about $4.03 to $5.74. Both ends are above zero, which supports positive average savings in this model.
(Point to that row’s `Paired 95% CI`, [$4.029, $5.739].)

This interval is about the average, not what happens on each trip. It also does not measure how much the policy would change with another training seed. The 95 percent level applies to each comparison separately.
(Stay on the same row.)

The other two intervals are also above zero. **All three comparisons support positive average savings for this trained policy.**
(Point to the lower endpoints in the other two rows.)

## 6:30–7:10 — How close is the policy to optimal?

Purpose [not spoken]: Distinguish calculated model expectations from sampled costs.

Narration:

Because we know the simulator’s probabilities, we can calculate expected costs without sampling test paths. The calculation starts at the deadline and works backward, comparing buying now with waiting. The learning agent does not receive this probability table.
(Show `# 6B — backward induction and exact policy evaluation`; point to `values[0] = PRICES`, then `waiting_cost = TRANSITIONS[t] @ values[t - 1]` and the action comparison.)

The best possible policy costs about $207.27 on average. The original learned policy costs about $215.71. That leaves a gap of about $8.43.
(Show `# 6D — save analytical comparison and per-state references`; follow `Optimal MDP`, `Learned policy`, and `Gap above optimum` in the displayed table.)

**The learned policy still leaves room for improvement.** These are calculated model expectations, which is why they differ from the earlier test averages.
(Keep the same table visible.)

## 7:10–7:50 — Policy and implementation checks

Purpose [not spoken]: Read the policy heatmap and explain what the executed checks establish.

Narration:

Here, each square tells us whether to BOOK or WAIT at a given price and time. Every square at the deadline says BOOK.
(Show `# 7A — draw and save the learned policy heatmap`; point to the price columns, time rows, and mandatory BOOK deadline row.)

The checks passed for price changes, boundaries, rewards, deadline booking, and stopping after purchase. They also checked that changing prices after purchase did not change the decisions.
(Show `# 9C — save actual run summary and trace`; point to the output `PASS: 400 future-suffix invariance cases; all core checks passed.`)

**Passing these checks supports correctness, but cannot rule out every bug or prove convergence.**
(Keep the PASS output visible.)

## 7:50–8:20 — Supplementary training budgets

Purpose [not spoken]: Separate training length from variation across training seeds.

Narration:

First, this study changes the amount of training. It uses five training seeds to see how results vary between runs. Each seed starts from scratch at both 50,000 and 200,000 episodes.
(Show `# S3C — paired gap changes and mean/range summaries`; point to the seed column and the two gap columns in the first table.)

With the original settings, all five final costs were higher at 200,000. **More training alone did not improve these five final policies.** Two budgets do not tell us what happens at every training length.
(Point to the five negative `gap_reduction_50000_minus_200000` entries; then the two budget summary rows.)

## 8:20–8:50 — Supplementary learning rates

Purpose [not spoken]: Explain the demonstrated rate effect without claiming a complete cause of the remaining gap.

Narration:

The next study keeps training at 200,000 episodes and compares three learning-rate settings across five seeds. Average model costs were about $220.98 with 0.1, $213.32 with 0.03, and $209.16 with visit-based decay.
(Show `# L4B — means and ranges across the five seeds`; follow the three condition rows in `mean_exact_cost`.)

**Both alternatives improved all five final policies under these tested conditions.** Every policy still cost more than the optimum. We do not yet have a complete explanation for the remaining gap.
(Show `# L5A — print actual outcomes and preserve the official policy`; point to both `improves 5/5 seeds` messages and the printed cost ranges.)

## 8:50–9:35 — Limitations

Purpose [not spoken]: Keep the findings within what this simulation and these experiments support.

Narration:

The prices and probabilities are made up for this experiment. The model also leaves out refunds, cancellations, and sold-out flights.
(Show `# 10A — evidence-based interpretation`; point to the line about hypothetical probabilities and no real-flight recommendation.)

**These results apply to simulated prices, not real flight prices.** The later experiments improved results with different learning rates, but they do not prove those settings are best or that training will reach the optimum.
(Show `### L5. Cautious exploratory interpretation`; point to the limits on the chosen rates and convergence.)

## 9:35–10:00 — Closing

Purpose [not spoken]: Bring the three validation questions together.

Narration:

The original policy had the lowest test average. The intervals support average savings, while the exact reference shows a remaining gap. The checks support the implementation, within the cases tested.
(Return to the saved table in `# 4E — official comparison and paired cost differences`; leave the Q-learning row visible.)

**The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.** Thank you for watching.
(Keep the completed results and camera visible.)

## Requirements-to-script checklist [not spoken]

- [ ] Approximately ten minutes: rehearse the estimated timestamps, including navigation; they are not measured durations.
- [ ] Camera on: keep it visible throughout the recording.
- [ ] Explain Python code: prices and episodes at 0:40–1:55; action choice and Q-learning at 1:55–3:20; evaluation at 3:20–4:20.
- [ ] Whole-project demo run: the instructor PDF asks for a demo run of the whole project showing results. This script explains code and saved outputs from the earlier successful run; it does not show a new run being launched or completed.
- [ ] Show saved results: means, trace, intervals, exact reference, checks, and brief supplementary findings at 4:50–8:50.
- [ ] Upload the video separately, as the instructor PDF specifies.
