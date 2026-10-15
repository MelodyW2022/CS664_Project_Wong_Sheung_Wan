# Flight Booking Decision Assistant

## Opening (0:00–0:40, approximate)

(Show the Notebook title and saved outputs; keep your camera on. Use the completed command-line run’s results throughout the recording.) Hello, my name is Sheung Wan Wong. My project is Flight Booking Decision Assistant.

The goal is to lower average booking cost. I ran the full Notebook once from the terminal; these are its saved results.

## 1. Setup and fixed experiment settings (0:40–0:55, approximate)

(Cell 1A: `TRAIN_SEED, TEST_SEED`.) First, let’s check the experiment settings.

Training uses seed six six four zero one. Testing uses seed six six four zero two.

## 2. Price simulator (0:55–1:55, approximate)

(Cell 2A: `probabilities`.) Next, let’s see how the simulator generates prices.

Prices can rise, stay unchanged, or fall. The chance of falling changes from forty percent earlier to twenty percent near the deadline.

(Cell 2A: `np.clip`.) Prices stay between one hundred twenty and three hundred sixty dollars.

(Cell 2B: `reset`.) The state contains time remaining and price. There are seven opportunities.

(Cell 2B: `effective_action`.) BOOK gives the negative fare as reward and ends the episode. WAIT gives zero reward and advances one day. The deadline forces BOOK.

### 2C. Verify the simulator before training (1:55–2:10, approximate)

(Cell 2D: `print`; output: rewards, deadline, termination.) These checks cover boundaries, rewards, deadline booking, and rejection of actions after purchase.

## 3. Tabular Q-learning (2:10–3:00, approximate)

(Cell 3A: `best_action`.) (Cell 3A: `epsilon_for`.) (Cell 3A: `rng.random() < epsilon`.) Now, let’s look at how Q-learning updates its estimates.

Over the first forty thousand episodes, the random choice rate falls to five percent, then stays there. Testing chooses the higher Q-value without exploration or updates.

(Cell 3B: `target`.) For BOOK, the target is reward alone. For WAIT, it includes the best estimated next action value. Gamma is one, so future cost is not discounted.

### 3C. Train 50,000 episodes (3:00–3:50, approximate)

(Cell 3C: `q = np.zeros`.) (Cell 3C: `range(episodes)`.) Training starts with a zero Q-table and runs fifty thousand episodes.

(Cell 3D: `print`; output: first observed update.) In the first episode, BOOK had an old value of zero and a target of negative one hundred twenty. With a learning rate of zero point one, **the updated value became negative twelve.**

## 4. Fair evaluation on 10,000 fresh trajectories (3:50–4:45, approximate)

(Cell 4A: `test_paths`.) Now, let’s compare the four strategies.

All four share ten thousand new test paths. The three rule-based baselines require no training.

(Cell 4B: `immediate_policy`.) (Cell 4B: `price <= threshold`.) (Cell 4B: `best_action(Q, state)`.) They buy immediately, at one hundred twenty, at one hundred eighty or below, or follow Q-learning. Both price rules buy at the deadline if necessary.

(Cell 4C: `policy(state)`.) (Cell 4C: `return state[1], trace`.) Each strategy sees only the current state and stops after purchase.

(Cell 4E: `display(Markdown(table))`; output: Mean purchase cost, four rows.) **Q-learning has the lowest test average**, about two hundred fourteen dollars and eighty five cents.

## 5. One complete Q-learning decision trace (4:45–5:10, approximate)

(Cell 5A: `display(Markdown(trace_table))`; output: seven rows, final BOOK.) (Cell 5A: `print`; output: final purchase cost.) Let’s follow one complete trip.

The agent waits until the deadline, then buys for three hundred sixty dollars.

## 6. Exact finite-horizon MDP reference (5:10–5:55, approximate)

(Cell 6B: `waiting_cost`.) Next, let’s calculate the best possible policy.

Known simulator probabilities let us calculate expected costs without sampling test paths. We compare buying with waiting backward from the deadline. The learner never receives these probabilities.

(Cell 6D: `print`; output: optimum, learned cost, gap.) The learned policy’s expected cost is about eight dollars and forty three cents higher than the optimum. These are calculated expectations, not test averages.

## 7. BOOK/WAIT policy heatmap (5:55–6:15, approximate)

(Cell 7A: `ax.set_xticks`.) (Cell 7A: `display`; output: axes and deadline row.) Now, let’s read the learned policy.

Each square shows BOOK or WAIT for time remaining and price. Every deadline square says BOOK.

## 8. Paired 95% confidence intervals (6:15–7:00, approximate)

(Cell 8B: `display(Markdown(ci_table))`; output: $180 row, mean saving.) Let’s examine uncertainty in the average savings.

Compared with the threshold rule, average saving is about four dollars and eighty eight cents.

(Cell 8B: `display(Markdown(ci_table))`; output: interval and lower endpoints.) The ninety five percent interval is about four dollars and three cents to five dollars and seventy four cents. It concerns average savings, not individual trips or different training seeds. All three intervals are above zero.

## 9. Save actual results and verification evidence (7:00–7:25, approximate)

(Cell 9C: `print`; output: all core checks passed.) Let’s check the verification evidence.

Checks passed, though they cannot rule out every bug.

## 10. Interpretation and limits (7:25–7:55, approximate)

(Cell 10A: `print`; output: simulation limits.) Next, let’s review the model’s limits.

**These results concern simulated prices, not real flights.** The model omits refunds, cancellations, and sold-out flights.

## Appendix. Optional threshold demonstration (8:00–8:05, approximate)

(Leave the optional threshold unchanged.) Here is an optional threshold exercise.

## Supplementary Validation and Optimality Investigation (8:05–8:08, approximate)

(Continue into the supplementary investigation.) Let’s investigate the remaining gap.

### S1. Verify the implementation (8:08–8:11, approximate)

(Cell S1A: `assert`.) (Cell S1B: `assert following is None`.)

### S2. Inspect original experience coverage and action optimality (8:11–8:17, approximate)

(Cell S2C: `display_supplementary_table`; output: actions and visits.)

### S3. Bounded five-seed/two-budget experiment (8:17–8:32, approximate)

(Cell S3C: `display_supplementary_table`; output: five gap changes.) Each seed was trained separately for fifty thousand and two hundred thousand episodes. Across five seeds, more training worsened every final policy under the original settings.

### S4. Review learning rate and exploration (8:32–8:35, approximate)

(Cell S4A: `assert`.)

### Supplementary conclusion (8:35–8:37, approximate)

(Continue downward.)

## Learning Dynamics and Learning-Rate Comparison (8:37–8:40, approximate)

(Continue to the learning-rate study.) Now, let’s compare different learning rates.

### L1. Prespecify fifteen runs (8:40–8:45, approximate)

(Cell L1A: `DYNAMICS_CONDITIONS`.)

### L2. Verify rates, counters, legal updates, and checkpoint isolation (8:45–8:47, approximate)

(Cell L2D: `np.array_equal`.)

### L3. Train and record every run (8:47–8:49, approximate)

(Scroll past the completed runs without rerunning them.)

### L4. Display all results and state-level associations (8:49–9:05, approximate)

(Cell L4A: `display_supplementary_table`; output: positive paired reductions.) (Cell L4B: `display_supplementary_table`; output: mean and minimum costs.) The settings are zero point one, zero point zero three, and visit based decay. Both alternatives improved final costs for all five seeds, but none reached the optimum.

### Observed state-level examples from this execution (9:05–9:08, approximate)

(Continue past the recorded examples.)

### L5. Cautious exploratory interpretation (9:08–9:20, approximate)

(Cell L5A: `print`; output: improves 5/5.) (Cell L5A: `print`; output: remaining gap unresolved.) The remaining gap is not fully explained. These runs do not prove convergence.

## Closing (9:20–10:00, approximate)

(Stay at the end of the Notebook with your camera visible.) To finish, **the policy saved money, uncertainty remains, and the exact reference shows room for improvement.** Thank you.
