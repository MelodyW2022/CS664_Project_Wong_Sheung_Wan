# CS664 presentation script

## 0:00–0:40 — Opening and start the full run

Conclusion:

**This project uses Q-learning to study when to book a simulated flight at a lower expected cost.**

Screen / Point to:

- `# Flight Booking Decision Assistant — completed experiment`: project title.
- From the project root, launch the full run by 0:20:

```sh
time .venv/bin/python run_notebook.py
```

- `# 1A — imports and official settings`: `PRICES`, `TRAIN_EPISODES`, and `TEST_TRAJECTORIES`.

Say:

I’ll run the full Notebook, explain the code while it runs, and reload the results when it finishes. The currently displayed outputs are from an earlier run.

## 0:40–1:15 — State, actions, and rewards

Conclusion:

**The environment makes every episode end with one purchase, within the allowed price range.**

Screen / Point to:

- `# 1A — imports and official settings`: the five prices in `PRICES`.
- `# 2B — episode and terminal rules`: `reset`, `(time_remaining, price)`, `effective_action`, `-price`, and `self.done = True` in the BOOK branch; WAIT’s `return self.state, 0, False, effective_action`.

Say:

Here, the state tells the agent the price and how much time is left. There are seven chances to buy, from time six to zero. Each starting price is equally likely. BOOK pays the current price and ends the episode. Its reward is the negative price, so a cheaper purchase gives a better reward. WAIT gives zero reward and moves forward one day. At time zero, the program forces a purchase.

## 1:15–1:55 — Daily price changes and boundaries

Conclusion:

**Daily prices can rise, stay the same, or fall.**

Screen / Point to:

- `# 2A — one price transition`: `probabilities`, `rng.choice([60, 0, -60], p=probabilities)`, and `np.clip`.
- Point to the early tuple `(0.3, 0.3, 0.4)`, the late tuple `(0.6, 0.2, 0.2)`, and the `time_remaining >= 3` condition.
- `# 2B — episode and terminal rules`: the `if self.done` error before any action.

Say:

Here, the program randomly chooses a price increase, no change, or a decrease. The chances change as the deadline gets closer. These are assumptions for the experiment, not patterns measured from real flights. Clipping stops prices from going below 120 or above 360 dollars. The other check stops the agent from taking another action after it has bought.

## 1:55–2:45 — One Q-value update and legal targets

Conclusion:

**Each Q update adjusts what the agent expects from an action.**

Screen / Point to:

- `# 3B — one real Q-value update`: `old`, `if done`, the two `target` assignments, and `q[row] = old + alpha * (target - old)`.
- `# 3D — display actual updates and BOOK/WAIT predictions`: the first observed update at `(6, 120)`, BOOK; point to old, target, alpha, and updated value.
- `# 3A — indexing, exploration, and greedy prediction`: `best_action` returns BOOK at time zero, including when used in the next-state target.

Say:

Here is an actual update from the first training episode. The agent booked at 120 dollars. The old Q-value was zero, and the target was negative 120. The learning rate was 0.1, so the new value became negative 12. Booking ends the trip, so there is nothing more to add. For WAIT, the target also includes the best estimated value at the next state. At the deadline, only BOOK is allowed. Gamma is one, so future cost is not discounted.

## 2:45–3:20 — Exploration during training

Conclusion:

**Training tries different actions, while testing uses what the agent learned.**

Screen / Point to:

- `# 3A — indexing, exploration, and greedy prediction`: `epsilon_for`, `rng.random() < epsilon`, and `best_action`.
- `# 3C — training loop`: zero-initialized `q`, the `update_q` call, and `if done` / `break`.
- `# 4B — state-only policies`: `learned_policy` calls `best_action(Q, state)`.

Say:

Training starts with a zero-filled table and runs for 50,000 episodes. The agent first chooses actions randomly. Over the first 40,000 episodes, the chance of a random choice falls to five percent and stays there. This helps it try both actions. During testing, it always picks the action with the higher Q-value. It makes no random choices and does not update the table.

## 3:20–4:20 — Fair comparison and validation questions

Conclusion:

**A fair comparison gives every strategy the same test paths and no future-price information.**

Screen / Point to:

- `# 1A — imports and official settings`: training seed 66401 and test seed 66402.
- `# 4A — generate independent shared test trajectories`: `test_paths` and the separate `TEST_SEED`.
- `# 4C — stop-on-purchase evaluator`: `policy(state)` and the immediate return on BOOK; the policy never receives `path`.
- `# 4E — official comparison and paired cost differences`: all four rows and `assert np.array_equal(Q, q_before_evaluation)`. Outputs are still earlier evidence until reloaded.

Say:

Here, every strategy faces the same 10,000 new price paths. Training and testing use separate random seeds. The program stores the whole path, but each booking decision receives only today’s price and time remaining. It cannot look ahead, and it stops after buying. This lets us ask three questions: Does Q-learning lower the cost? How reliable are the savings? How close is it to the best possible policy, and does the code follow the rules?

## 4:20–4:50 — Confirm full execution and reload

Conclusion:

**The fresh full run has completed, and these reloaded outputs come from that run.**

Screen / Point to:

- Terminal: require `PASS: 80 code cells executed in a fresh kernel; no cell errors.`, the unchanged-numerical-evidence message, the saved-Notebook message, and a returned prompt.
- Reload `CS664_Flight_Booking.ipynb` from disk before presenting fresh outputs; do not save the stale viewer.
- Only after success and reload, say the conclusion and narration below; otherwise use the checklist’s slow-run contingency.

Say:

The runner reports that all 80 code cells finished without errors. That includes the original experiment and both supplementary studies. It also checked that the original numerical results stayed the same. I have now reopened the saved Notebook, so these are the new outputs.

## 4:50–5:20 — Four-strategy mean costs

Conclusion:

**Q-learning had the lowest average purchase cost in this test.**

Screen / Point to:

- `# 4E — official comparison and paired cost differences`: fresh table, `Mean purchase cost` column.
- Point down the rows: Immediate booking; Minimum-price rule ($120); Fixed-threshold rule ($180); Q-learning (greedy).

Say:

Here are the four averages. Buying immediately cost $239.58. Buying at $120, or at the deadline, cost $223.48. Buying at $180 or below, or at the deadline, cost $219.74. Q-learning cost $214.85. These are averages from the test paths, so another test sample could give slightly different numbers.

## 5:20–5:40 — One complete decision trace

Conclusion:

**Average savings do not mean savings on every trip.**

Screen / Point to:

- `# 5A — actual states, actions, and final cost`: keep all seven rows visible.
- Point from the first `(6, $360)` state down the `Action` column to the final `(0, $360)` BOOK row, reward −360, and `Final purchase cost: $360`.

Say:

This table follows one trip from start to finish. The agent starts at $360, keeps waiting, and finally books at the deadline for $360. The last row ends the episode. One trip cannot tell us which strategy is cheaper on average.

## 5:40–6:30 — Paired confidence intervals

Conclusion:

**These intervals support average savings against all three baselines.**

Screen / Point to:

- `# 8B — save all three paired savings intervals`: `Fixed-threshold rule ($180)` row, `Mean Q-learning saving` and `Paired 95% CI` columns.
- Point to $4.884 and [$4.029, $5.739], then the positive lower endpoints in the other two rows.

Say:

Here, I’m comparing Q-learning with the rule that buys at $180 or below. On the same trips, Q-learning saved about $4.88 on average. The 95 percent confidence interval runs from about $4.03 to $5.74. Both ends are above zero, which supports positive average savings in this model. This interval is about the average, not what happens on each trip. It also does not tell us how much a different training seed would change the policy. The 95 percent level applies to each comparison separately.

## 6:30–7:10 — Exact optimal reference

Conclusion:

**The learned policy still leaves room for improvement.**

Screen / Point to:

- `# 6B — backward induction and exact policy evaluation`: `values[0] = PRICES`, `waiting_cost = TRANSITIONS[t] @ values[t - 1]`, and the BOOK/WAIT comparison.
- `# 6D — save analytical comparison and per-state references`: `Optimal MDP`, `Learned policy`, and `Gap above optimum` rows in the displayed table.
- The same cell’s printed comparison gives $207.274224, $215.709120, and $8.434896.

Say:

Because we know the simulator’s probabilities, we can calculate expected costs without sampling test paths. The calculation starts at the deadline and works backward, comparing buying now with waiting. The best possible policy costs about $207.27 on average. The original learned policy costs about $215.71. That leaves a gap of about $8.43. These are calculated model expectations, which is why they differ from the earlier test averages. The learning agent does not receive the probability table.

## 7:10–7:50 — Policy heatmap and implementation checks

Conclusion:

**The checks passed, but they cannot prove the policy will reach the optimum.**

Screen / Point to:

- `# 7A — draw and save the learned policy heatmap`: price columns, time-remaining rows, BOOK/WAIT labels, and the mandatory BOOK deadline row.
- `# 2D — rewards, deadline, and termination checks`: the PASS output for 90 transitions and seven purchase days.
- `# 9C — save actual run summary and trace`: the PASS output for 400 future-suffix invariance cases.

Say:

Here, each heatmap square tells us whether to BOOK or WAIT at a given price and time. Every square at the deadline says BOOK. The checks below test price changes, boundaries, rewards, forced booking, and stopping after purchase. They also check that changing prices after purchase cannot change the decision. The checks passed, but testing these cases cannot rule out every possible bug.

## 7:50–8:20 — Supplementary training budgets

Conclusion:

**More training alone did not improve these five final policies.**

Screen / Point to:

- `# S3C — paired gap changes and mean/range summaries`: all five `gap_reduction_50000_minus_200000` entries are negative.
- Point to the two summary rows for 50,000 and 200,000 episodes; these are analytical costs across seeds, not test-trip confidence intervals.

Say:

First, this study changes the amount of training. It uses five training seeds to see how results vary between runs. Each seed starts from scratch at both 50,000 and 200,000 episodes. With the original settings, all five final costs were higher at 200,000. Two budgets do not tell us what happens at every training length.

## 8:20–8:50 — Supplementary learning rates

Conclusion:

**The tested alternative learning rates improved final costs, but did not reach the optimum.**

Screen / Point to:

- `# L4B — means and ranges across the five seeds`: `condition`, `mean_exact_cost`, `min_exact_cost`, and `max_exact_cost` columns.
- Point to constant_0.1, constant_0.03, and visit_decay rows: mean costs $220.978560, $213.3248832, and $209.164944.
- `# L5A — print actual outcomes and preserve the official policy`: both alternatives improve 5/5 seeds.
- In `# L4B — means and ranges across the five seeds`, each `min_exact_cost` remains above the $207.274224 optimum from `# 6D — save analytical comparison and per-state references`.

Say:

The next study keeps training at 200,000 episodes and compares three learning-rate settings across five seeds. Average model costs were about $220.98 with 0.1, $213.32 with 0.03, and $209.16 with visit-based decay. Both alternatives improved all five final policies. That demonstrates an effect under these tested conditions. Every policy still cost more than the optimum. We do not yet have a complete explanation for the remaining gap.

## 8:50–9:35 — Limitations

Conclusion:

**The project supports savings inside this simulator, not a claim about real flight prices.**

Screen / Point to:

- `# 10A — evidence-based interpretation`: printed limits on hypothetical probabilities, fixed learning rate, and finite training.
- `### L5. Cautious exploratory interpretation`: no unique cause, no proven optimal rate, and no convergence claim.

Say:

The prices and probabilities are made up for this experiment. The model also leaves out refunds, cancellations, and sold-out flights. The original comparison supports lower average costs, and the intervals describe uncertainty for that trained policy. The exact reference still shows a gap. The later experiments improved results with different learning rates, but they do not prove those settings are best or that training will reach the optimum.

## 9:35–10:00 — Closing

Conclusion:

**The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.**

Screen / Point to:

- `# 4E — official comparison and paired cost differences`: leave the fresh four-strategy table visible.
- Keep your camera on and pause briefly over the Q-learning row before closing.

Say:

This project connects a booking problem with a complete learning and evaluation process. It shows why a good average result still needs uncertainty estimates and a check against the best possible policy. Thank you for watching.
