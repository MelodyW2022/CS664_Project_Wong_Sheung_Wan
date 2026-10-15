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

The state is time remaining and price, with seven opportunities from time six to zero and five possible starting prices chosen uniformly. BOOK gives a negative-price reward and ends the episode; WAIT gives zero reward and advances one day, while the deadline forces BOOK.

## 1:15–1:55 — Daily price changes and boundaries

Conclusion:

**The simulator uses assumed daily price changes and keeps fares within bounds.**

Screen / Point to:

- `# 2A — one price transition`: `probabilities`, `rng.choice([60, 0, -60], p=probabilities)`, and `np.clip`.
- Point to the early tuple `(0.3, 0.3, 0.4)`, the late tuple `(0.6, 0.2, 0.2)`, and the `time_remaining >= 3` condition.
- `# 2B — episode and terminal rules`: the `if self.done` error before any action.

Say:

Cell 2A samples a rise, no change, or fall and keeps fares between 120 and 360 dollars. These probabilities are hypothetical; Cell 2B rejects any further action after purchase.

## 1:55–2:45 — One Q-value update and legal targets

Conclusion:

**Q-learning improves an action estimate by moving its old value toward a target.**

Screen / Point to:

- `# 3B — one real Q-value update`: `old`, `if done`, the two `target` assignments, and `q[row] = old + alpha * (target - old)`.
- `# 3D — display actual updates and BOOK/WAIT predictions`: the first observed update at `(6, 120)`, BOOK; point to old, target, alpha, and updated value.
- `# 3A — indexing, exploration, and greedy prediction`: `best_action` returns BOOK at time zero, including when used in the next-state target.

Say:

In the logged BOOK update, the old value is zero, the target is negative 120, and alpha is 0.1, giving an updated value of negative 12. A terminal target is reward alone; WAIT adds the best legal future estimate, with gamma one.

## 2:45–3:20 — Exploration during training

Conclusion:

**Training explores actions, while evaluation follows the learned policy greedily.**

Screen / Point to:

- `# 3A — indexing, exploration, and greedy prediction`: `epsilon_for`, `rng.random() < epsilon`, and `best_action`.
- `# 3C — training loop`: zero-initialized `q`, the `update_q` call, and `if done` / `break`.
- `# 4B — state-only policies`: `learned_policy` calls `best_action(Q, state)`.

Say:

The original training runs 50,000 episodes, with epsilon falling from one to 0.05 over 40,000 episodes and then staying there. Evaluation uses the greedy action without exploration or Q updates.

## 3:20–4:20 — Fair comparison and validation questions

Conclusion:

**A fair comparison gives every strategy the same test paths and no future-price information.**

Screen / Point to:

- `# 1A — imports and official settings`: training seed 66401 and test seed 66402.
- `# 4A — generate independent shared test trajectories`: `test_paths` and the separate `TEST_SEED`.
- `# 4C — stop-on-purchase evaluator`: `policy(state)` and the immediate return on BOOK; the policy never receives `path`.
- `# 4E — official comparison and paired cost differences`: all four rows and `assert np.array_equal(Q, q_before_evaluation)`. Outputs are still earlier evidence until reloaded.

Say:

All four strategies use the same 10,000 fresh paths from a separate test seed, and each decision sees only its current state. The three questions are whether costs fall, how reliable the savings are, and how close the policy is to optimal with a correct implementation.

## 4:20–4:50 — Confirm full execution and reload

Conclusion:

**The fresh full run has completed, and these reloaded outputs come from that run.**

Screen / Point to:

- Terminal: require `PASS: 80 code cells executed in a fresh kernel; no cell errors.`, the unchanged-numerical-evidence message, the saved-Notebook message, and a returned prompt.
- Reload `CS664_Flight_Booking.ipynb` from disk before presenting fresh outputs; do not save the stale viewer.
- Only after success and reload, say the conclusion and narration below; otherwise use the checklist’s slow-run contingency.

Say:

The runner executed all 80 code cells, including both supplementary studies, without errors. The numerical checks passed, and the saved Notebook has been reopened.

## 4:50–5:20 — Four-strategy mean costs

Conclusion:

**The original learned policy had the lowest sampled mean cost among the four strategies.**

Screen / Point to:

- `# 4E — official comparison and paired cost differences`: fresh table, `Mean purchase cost` column.
- Point down the rows: Immediate booking; Minimum-price rule ($120); Fixed-threshold rule ($180); Q-learning (greedy).

Say:

The sample means were $239.58 for buying immediately, $223.48 for buying at $120 or the deadline, $219.74 for buying at $180 or below or the deadline, and $214.85 for Q-learning. These are sampled costs, not analytical expectations.

## 5:20–5:40 — One complete decision trace

Conclusion:

**Average savings do not mean savings on every trip.**

Screen / Point to:

- `# 5A — actual states, actions, and final cost`: keep all seven rows visible.
- Point from the first `(6, $360)` state down the `Action` column to the final `(0, $360)` BOOK row, reward −360, and `Final purchase cost: $360`.

Say:

This trace waits from a starting price of $360 until the deadline, then books for $360. The final BOOK ends the episode; the trace is an illustration, not a performance estimate.

## 5:40–6:30 — Paired confidence intervals

Conclusion:

**The paired intervals support positive mean savings for this fixed trained policy.**

Screen / Point to:

- `# 8B — save all three paired savings intervals`: `Fixed-threshold rule ($180)` row, `Mean Q-learning saving` and `Paired 95% CI` columns.
- Point to $4.884 and [$4.029, $5.739], then the positive lower endpoints in the other two rows.

Say:

Against the $180 rule, paired mean saving was about $4.88, with a 95% interval of about $4.03 to $5.74. These individual intervals concern uncertainty in mean savings for one fixed policy, not individual-trip ranges or training-seed variation, and are not simultaneous intervals.

## 6:30–7:10 — Exact optimal reference

Conclusion:

**The original learned policy is still suboptimal.**

Screen / Point to:

- `# 6B — backward induction and exact policy evaluation`: `values[0] = PRICES`, `waiting_cost = TRANSITIONS[t] @ values[t - 1]`, and the BOOK/WAIT comparison.
- `# 6D — save analytical comparison and per-state references`: `Optimal MDP`, `Learned policy`, and `Gap above optimum` rows in the displayed table.
- The same cell’s printed comparison gives $207.274224, $215.709120, and $8.434896.

Say:

Backward induction compares buying now with waiting, using known simulator probabilities that the learner never receives. The analytical optimum is about $207.27, compared with $215.71 for the original learned policy, leaving a gap of about $8.43.

## 7:10–7:50 — Policy heatmap and implementation checks

Conclusion:

**The executed checks support correctness, without proving convergence.**

Screen / Point to:

- `# 7A — draw and save the learned policy heatmap`: price columns, time-remaining rows, BOOK/WAIT labels, and the mandatory BOOK deadline row.
- `# 2D — rewards, deadline, and termination checks`: the PASS output for 90 transitions and seven purchase days.
- `# 9C — save actual run summary and trace`: the PASS output for 400 future-suffix invariance cases.

Say:

The heatmap maps current price and time to an action, with BOOK at every deadline price. Checks passed for transitions, boundaries, rewards, deadline booking, termination, and future-price isolation; finite checks cannot rule out every bug.

## 7:50–8:20 — Supplementary training budgets

Conclusion:

**More training alone did not improve these five final policies.**

Screen / Point to:

- `# S3C — paired gap changes and mean/range summaries`: all five `gap_reduction_50000_minus_200000` entries are negative.
- Point to the two summary rows for 50,000 and 200,000 episodes; these are analytical costs across seeds, not test-trip confidence intervals.

Say:

Each seed restarted at 50,000 and 200,000 episodes with the original settings, and every final expected cost rose at the larger budget. This shows training-seed variation, while two budgets do not establish a continuous trend.

## 8:20–8:50 — Supplementary learning rates

Conclusion:

**The tested alternative learning rates improved final costs, but did not reach the optimum.**

Screen / Point to:

- `# L4B — means and ranges across the five seeds`: `condition`, `mean_exact_cost`, `min_exact_cost`, and `max_exact_cost` columns.
- Point to constant_0.1, constant_0.03, and visit_decay rows: mean costs $220.978560, $213.3248832, and $209.164944.
- `# L5A — print actual outcomes and preserve the official policy`: both alternatives improve 5/5 seeds.
- In `# L4B — means and ranges across the five seeds`, each `min_exact_cost` remains above the $207.274224 optimum from `# 6D — save analytical comparison and per-state references`.

Say:

Across five seeds at 200,000 episodes, mean exact costs were about $220.98 with alpha 0.1, $213.32 with alpha 0.03, and $209.16 with visit-based decay. Both alternatives improved all five final policies, but positive remaining gaps leave the cause unresolved and do not prove convergence.

## 8:50–9:35 — Limitations

Conclusion:

**The project supports savings inside this simulator, not a claim about real flight prices.**

Screen / Point to:

- `# 10A — evidence-based interpretation`: printed limits on hypothetical probabilities, fixed learning rate, and finite training.
- `### L5. Cautious exploratory interpretation`: no unique cause, no proven optimal rate, and no convergence claim.

Say:

The model assumes its price probabilities and omits refunds, cancellations, and sold-out events. Supplementary results show sensitivity to tested rates, not an optimal setting or a unique explanation for the remaining gap.

## 9:35–10:00 — Closing

Conclusion:

**The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.**

Screen / Point to:

- `# 4E — official comparison and paired cost differences`: leave the fresh four-strategy table visible.
- Keep your camera on and pause briefly over the Q-learning row before closing.

Say:

This project demonstrates a complete Python Q-learning workflow, from the environment to evaluation and verification. Thank you for watching.
