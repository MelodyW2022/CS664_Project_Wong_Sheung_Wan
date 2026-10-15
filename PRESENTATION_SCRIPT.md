# Flight Booking Decision Assistant

## Opening (0:00–0:40, approximate)

Hello, my name is Sheung Wan Wong. My project is called Flight Booking Decision Assistant.
(Show the Notebook title; keep your camera on.)

**The goal is to learn when to book a simulated flight at a lower average cost.** I’ll start the full run now.
(From the project root, enter `time .venv/bin/python run_notebook.py` by about 0:20.)

I previously ran the full Notebook successfully. I’ll walk through the code and the saved results from that run while the fresh run executes.
(Keep the main Notebook visible; the outputs currently shown are from the earlier run.)

## 1. Setup and fixed experiment settings (0:40–0:55, approximate)

The model uses five ticket prices, from 120 to 360 dollars. Training and testing use different random seeds.
(Point to `PRICES`, `TRAIN_SEED`, and `TEST_SEED` in Cell 1A.)

## 2. Price simulator (0:55–1:55, approximate)

Let’s start with how prices change. Each day, the price can rise by 60 dollars, stay the same, or fall by 60 dollars.
(Point to `rng.choice([60, 0, -60], p=probabilities)` in Cell 2A.)

Earlier in the trip, the chance of a fall is 40 percent. Near the deadline, it is 20 percent. These are assumptions, not patterns measured from real flights.
(Point to the two `probabilities` tuples in Cell 2A.)

The price stays between 120 and 360 dollars.
(Point to `np.clip` in Cell 2A.)

Let’s look at one booking episode. The state contains price and time remaining. There are seven opportunities, from time six to zero. Each starting price is equally likely.
(Point to `reset` in Cell 2B.)

The program rejects another action after purchase and forces BOOK at time zero.
(Point to `if self.done`, then `effective_action`, in Cell 2B.)

**Booking pays the current price and ends the episode.** Its reward is the negative price, so a cheaper purchase gives a better reward.
(Point to `self.done = True` and `return None, -price, True, effective_action` in Cell 2B.)

WAIT gives zero reward and moves forward one day.
(Point to the final two lines of Cell 2B.)

### 2C. Verify the simulator before training (1:55–2:10, approximate)

These checks cover the price boundaries, rewards, forced booking, and stopping after purchase. This is saved evidence from the earlier run.
(Point to the assertions and saved PASS output in Cells 2C and 2D.)

## 3. Tabular Q-learning (2:10–3:00, approximate)

Here, the agent chooses the action with the higher Q-value. The deadline always means BOOK.
(Point to `best_action` in Cell 3A.)

Training also needs exploration. The chance of a random choice falls from one to five percent over the first 40,000 episodes, then stays there.
(Point to `epsilon_for`, then the random branch in `choose_training_action`, in Cell 3A.)

During testing, there are no random choices or Q updates.
(Keep Cell 3A visible.)

Here is how the table learns. For BOOK, the target is just the reward. For WAIT, it also includes the best estimated value at the next state. Only BOOK is allowed at a deadline next state.
(Point to the two target assignments in Cell 3B.)

The update moves the old value toward the target. Gamma is one, so future cost is not discounted.
(Point to `old + alpha * (target - old)` in Cell 3B.)

### 3C. Train 50,000 episodes (3:00–3:50, approximate)

The original training starts with a zero-filled table and runs 50,000 episodes. Each episode stops after purchase.
(Point to `q = np.zeros`, the `update_q` call, and `if done` / `break` in Cell 3C.)

Here is an actual logged example from the first episode. The agent booked at 120 dollars. The old value was zero, the target was negative 120, and the learning rate was 0.1. **The updated Q-value was negative 12.**
(Point to the first observed update in Cell 3D.)

(At this section boundary, check the terminal for `PASS: 80 code cells executed in a fresh kernel; no cell errors.`, unchanged numerical evidence, the saved-Notebook message, and a returned prompt. Only after success, reload the Notebook from disk without saving the old viewer.)

The full run has now finished, including both supplementary studies. I’ve reopened the saved Notebook. The results from this point are from the completed fresh run.
(Say this only after success and reload; otherwise wait and follow the separate checklist.)

## 4. Fair evaluation on 10,000 fresh trajectories (3:50–4:45, approximate)

For the comparison, all four strategies receive the same 10,000 new test paths. Testing uses a separate random seed from training.
(Point to `test_paths` and `TEST_SEED` in Cell 4A.)

The rules are to buy immediately, buy at 120 dollars or the deadline, buy at 180 dollars or below or the deadline, and follow Q-learning.
(Point to `immediate_policy`, `threshold_policy`, and `learned_policy` in Cell 4B.)

The evaluator stores the whole path, but each decision receives only today’s price and time remaining. It stops calling the policy after purchase.
(Point to `policy(state)` and the return inside the BOOK branch in Cell 4C.)

There are three questions: Does Q-learning lower costs? How reliable are the savings? How close is it to the best possible policy, and does the program follow the rules?
(Keep Cell 4C visible.)

Here are the averages. Buying immediately cost $239.58. The 120-dollar rule cost $223.48, and the 180-dollar rule cost $219.74. Q-learning cost $214.85.
(Follow the four rows in the `Mean purchase cost` column of Cell 4E.)

**Q-learning had the lowest average purchase cost in this test.** Another test sample could give slightly different averages.
(Point to the Q-learning row in Cell 4E.)

## 5. One complete Q-learning decision trace (4:45–5:10, approximate)

This table follows one trip from start to finish. The agent starts at $360, keeps waiting, and finally books at the deadline for $360. The last row ends the episode.
(Follow all seven rows in Cell 5A, from the first state to the final BOOK and purchase cost.)

Average savings do not mean savings on every trip. This example explains the decisions; it does not estimate overall performance.
(Keep the complete trace visible.)

## 6. Exact finite-horizon MDP reference (5:10–5:55, approximate)

Because we know the simulator’s probabilities, we can calculate expected costs without sampling test paths. The calculation starts at the deadline and works backward, comparing buying now with waiting. The learning agent does not receive this probability table.
(Point to `values[0] = PRICES`, then `waiting_cost`, in Cell 6B.)

The best possible policy costs about $207.27 on average. The original learned policy costs about $215.71. That leaves a gap of about $8.43.
(Follow the `Optimal MDP`, `Learned policy`, and `Gap above optimum` rows in Cell 6D.)

**The learned policy still leaves room for improvement.** These are calculated model expectations, which is why they differ from the earlier test averages.
(Keep the same table visible.)

## 7. BOOK/WAIT policy heatmap (5:55–6:15, approximate)

Here, each square tells us whether to BOOK or WAIT at a given price and time. Every square at the deadline says BOOK.
(Point to the price columns, time rows, and mandatory BOOK deadline row in Cell 7A.)

## 8. Paired 95% confidence intervals (6:15–7:00, approximate)

Here, I’m comparing Q-learning with the rule that buys at 180 dollars or below. On the same trips, Q-learning saved about $4.88 on average.
(Point to the $180 baseline row and `Mean Q-learning saving`, $4.884, in Cell 8B.)

The 95 percent confidence interval runs from about $4.03 to $5.74. Both ends are above zero, which supports positive average savings in this model.
(Point to that row’s `Paired 95% CI`, [$4.029, $5.739].)

This interval is about the average, not each trip. It does not measure how much another training seed would change the policy. The 95 percent level applies to each comparison separately.
(Stay on the same row.)

**All three comparisons support positive average savings for this trained policy.**
(Point to the positive lower endpoints in the other two rows of Cell 8B.)

## 9. Save actual results and verification evidence (7:00–7:25, approximate)

The program saves the tables, price paths, learned values, and complete trace for review.
(Point to the array and CSV exports in Cell 9B.)

It also checked that changing prices after purchase did not change the decisions. The core checks passed, but testing these cases cannot rule out every possible bug.
(Point to `PASS: 400 future-suffix invariance cases; all core checks passed.` in Cell 9C.)

## 10. Interpretation and limits (7:25–7:55, approximate)

The original comparison supports lower average costs, and the intervals describe uncertainty for that trained policy. The exact reference still shows a gap.
(Point to the printed interpretation in Cell 10A.)

The prices and probabilities are made up for this experiment. The model leaves out refunds, cancellations, and sold-out flights. **These results apply to simulated prices, not real flight prices.**
(Point to the lines about hypothetical probabilities and no real-flight recommendation in Cell 10A.)

## 11. Essential cells to personally run and explain before recording (7:55–8:00, approximate)

(Scroll past this preparation checklist.)

## Appendix. Optional threshold demonstration (8:00–8:05, approximate)

(Leave the optional threshold unchanged and continue downward.)

## Supplementary Validation and Optimality Investigation (8:05–8:08, approximate)

Two supplementary studies looked at the remaining gap.
(Keep this section heading visible.)

### S1. Verify the implementation (8:08–8:11, approximate)

(Point briefly to the isolated checks in Cells S1A and S1B.)

### S2. Inspect original experience coverage and action optimality (8:11–8:17, approximate)

Seven states had worse-than-optimal decisions, despite visits to both actions. Visits alone do not explain the cause.
(Point to the suboptimal-state table in Cell S2C.)

### S3. Bounded five-seed/two-budget experiment (8:17–8:32, approximate)

The first study restarted five training seeds at 50,000 and 200,000 episodes. With the original settings, all five final costs were higher at 200,000. More training alone did not help here; two budgets do not establish a continuous trend.
(Point to the five negative paired gap changes in Cell S3C.)

### S4. Review learning rate and exploration (8:32–8:35, approximate)

(Point to the unchanged alpha and epsilon checks in Cell S4A.)

### Supplementary conclusion (8:35–8:37, approximate)

(Continue downward past the budget-study conclusion.)

## Learning Dynamics and Learning-Rate Comparison (8:37–8:40, approximate)

The next study changed the learning rate.
(Keep this section heading visible.)

### L1. Prespecify fifteen runs (8:40–8:45, approximate)

It kept 200,000 episodes per run and used five seeds for each setting.
(Point to the recorded conditions and episode budget in Cell L1A.)

### L2. Verify rates, counters, legal updates, and checkpoint isolation (8:45–8:47, approximate)

(Point briefly to the counter and checkpoint checks in Cells L2B and L2D.)

### L3. Train and record every run (8:47–8:49, approximate)

(Scroll past the fifteen completed run cells without rerunning them.)

### L4. Display all results and state-level associations (8:49–9:05, approximate)

Both alternatives improved all five final policies.
(Point to the five positive reductions for each alternative in Cell L4A.)

Average model costs were about $220.98 with 0.1, $213.32 with 0.03, and $209.16 with visit-based decay. Every final policy still cost more than the optimum.
(Point to the mean and minimum cost columns in Cell L4B.)

### Observed state-level examples from this execution (9:05–9:08, approximate)

(Continue past the recorded state-level examples.)

### L5. Cautious exploratory interpretation (9:08–9:20, approximate)

**The learning-rate changes helped under these tested conditions.** We still do not have a complete explanation for the remaining gap. These experiments do not prove the settings are best or that training will reach the optimum.
(Point to the cautious interpretation and the two `improves 5/5 seeds` messages in Cell L5A.)

## Closing (9:20–10:00, approximate)

The original policy had the lowest test average. The intervals support average savings, while the exact reference shows a remaining gap. The checks support the implementation within the cases tested.
(Stay at the end of the Notebook; do not jump back to earlier tables.)

**The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.** Thank you for watching.
(Keep your camera visible.)
