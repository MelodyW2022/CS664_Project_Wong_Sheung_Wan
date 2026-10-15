# Flight Booking Decision Assistant

## Opening (0:00–0:40, approximate)

Hello, my name is Sheung Wan Wong. My project is Flight Booking Decision Assistant.
(Show the Notebook title; keep your camera on.)

I’ll start the full run now.
(From the project root, enter `time .venv/bin/python run_notebook.py` by about 0:20.)

The goal is to learn when to book at a lower average cost. I previously ran the full Notebook successfully. These displayed outputs are from that earlier run.
(Keep the main Notebook visible while the fresh run executes.)

## 1. Setup and fixed experiment settings (0:40–0:55, approximate)

Training and testing use different random seeds.
(Point to `TRAIN_SEED` and `TEST_SEED` in Cell 1A.)

## 2. Price simulator (0:55–1:55, approximate)

Daily prices can rise, stay unchanged, or fall. These probabilities are assumptions. The chance of falling changes from forty percent earlier to twenty percent near the deadline.
(Point to `probabilities` and `rng.choice` in Cell 2A.)

Clipping keeps prices between one hundred twenty and three hundred sixty dollars.
(Point to `np.clip` in Cell 2A.)

The state contains price and time remaining. There are seven opportunities.
(Point to `reset` in Cell 2B.)

BOOK gives a negative price reward and ends the episode. WAIT gives zero reward and advances one day. The deadline forces BOOK.
(Point to `effective_action`, the BOOK return, and the WAIT return in Cell 2B.)

### 2C. Verify the simulator before training (1:55–2:10, approximate)

These checks cover boundaries, rewards, deadline booking, and rejection of actions after purchase.
(Point to the saved PASS output in Cells 2C and 2D.)

## 3. Tabular Q-learning (2:10–3:00, approximate)

Training explores both actions. Over the first forty thousand episodes, the random choice rate falls to five percent, then stays there. Testing chooses the higher Q-value without exploration or updates.
(Point to `best_action`, `epsilon_for`, and `choose_training_action` in Cell 3A.)

For BOOK, the target is reward alone. For WAIT, it includes the best estimated next action value. Deadline WAIT is excluded. Gamma is one, so future cost is not discounted.
(Point to the two target branches in Cell 3B.)

### 3C. Train 50,000 episodes (3:00–3:50, approximate)

Training starts with zeros and runs fifty thousand episodes.
(Point to `q = np.zeros` and the episode loop in Cell 3C.)

In the first episode, BOOK had an old value of zero and a target of negative one hundred twenty. With a learning rate of zero point one, **the updated value became negative twelve.**
(Point to the first observed update in Cell 3D.)

(Check the terminal for all eighty code cells completed, unchanged numerical evidence, the saved-path message, and a returned prompt. Reload from disk before continuing; never save the stale viewer.)

The full run has finished, including both supplementary studies. These reopened outputs are fresh.
(Say this only after successful completion and reload; otherwise wait.)

## 4. Fair evaluation on 10,000 fresh trajectories (3:50–4:45, approximate)

All four strategies share ten thousand new test paths.
(Point to `test_paths` in Cell 4A.)

They buy immediately, at one hundred twenty, at one hundred eighty or below, or follow Q-learning. Both price rules buy at the deadline if necessary.
(Point to the policy functions in Cell 4B; the threshold includes equality.)

Each decision sees only current state and stops after purchase. It never receives tomorrow’s price.
(Point to `policy(state)` and the BOOK return in Cell 4C.)

We ask whether costs fall, how reliable savings are, and how close the policy is to optimal with correct code.
(Keep Cell 4C visible.)

**Q-learning has the lowest test average**, about two hundred fourteen dollars and eighty five cents.
(Point down all four `Mean purchase cost` rows in Cell 4E.)

## 5. One complete Q-learning decision trace (4:45–5:10, approximate)

This complete trace waits until the deadline, then buys for three hundred sixty dollars. Average savings do not guarantee savings on every trip.
(Follow all seven state/action rows to the final BOOK and purchase cost in Cell 5A.)

## 6. Exact finite-horizon MDP reference (5:10–5:55, approximate)

Known simulator probabilities let us calculate expected costs without sampling test paths. We work backward, comparing buying with waiting. The learner never receives these probabilities.
(Point to `waiting_cost` in Cell 6B.)

The learned policy costs about eight dollars and forty three cents above the optimum. These are model expectations, not test averages.
(Point to the optimum, learned-policy cost, and gap in Cell 6D.)

## 7. BOOK/WAIT policy heatmap (5:55–6:15, approximate)

Each square shows BOOK or WAIT for a price and time. Every deadline square says BOOK.
(Point to the axes and deadline row in Cell 7A.)

## 8. Paired 95% confidence intervals (6:15–7:00, approximate)

Compared with the threshold rule, average saving is about four dollars and eighty eight cents.
(Point to the $180 row in Cell 8B.)

The ninety five percent interval is about four dollars and three cents to five dollars and seventy four cents. It concerns average savings, not individual trips or different training seeds. All three intervals stay above zero for this trained policy.
(Point to that interval and the other positive lower endpoints in Cell 8B.)

## 9. Save actual results and verification evidence (7:00–7:25, approximate)

The checks passed, but cannot rule out every bug.
(Point to the core-check PASS message in Cell 9C.)

## 10. Interpretation and limits (7:25–7:55, approximate)

**These results concern simulated prices, not real flights.** The model omits refunds, cancellations, and sold-out flights.
(Point to the limitations in Cell 10A.)

## 11. Essential cells to personally run and explain before recording (7:55–8:00, approximate)

(Scroll past this preparation checklist.)

## Appendix. Optional threshold demonstration (8:00–8:05, approximate)

(Leave the optional threshold unchanged.)

## Supplementary Validation and Optimality Investigation (8:05–8:08, approximate)

(Continue into the supplementary investigation.)

### S1. Verify the implementation (8:08–8:11, approximate)

(Point briefly to the isolated checks in Cells S1A and S1B.)

### S2. Inspect original experience coverage and action optimality (8:11–8:17, approximate)

(Point to the visit counts and suboptimal decisions in Cell S2C.)

### S3. Bounded five-seed/two-budget experiment (8:17–8:32, approximate)

Each seed restarts at fifty thousand and two hundred thousand episodes. Across five seeds, more training worsened every final policy under the original settings.
(Point to all five paired budget changes in Cell S3C.)

### S4. Review learning rate and exploration (8:32–8:35, approximate)

(Point to the unchanged settings in Cell S4A.)

### Supplementary conclusion (8:35–8:37, approximate)

(Continue downward.)

## Learning Dynamics and Learning-Rate Comparison (8:37–8:40, approximate)

(Continue to the learning-rate study.)

### L1. Prespecify fifteen runs (8:40–8:45, approximate)

(Point to the three conditions and fixed episode budget in Cell L1A.)

### L2. Verify rates, counters, legal updates, and checkpoint isolation (8:45–8:47, approximate)

(Point briefly to the checkpoint checks.)

### L3. Train and record every run (8:47–8:49, approximate)

(Scroll past the completed runs without rerunning them.)

### L4. Display all results and state-level associations (8:49–9:05, approximate)

The settings are zero point one, zero point zero three, and visit based decay. Changing learning rates improved all five final policies, but none reached the optimum. The original policy remains unchanged.
(Point to the positive paired reductions in Cell L4A, then costs in Cell L4B.)

### Observed state-level examples from this execution (9:05–9:08, approximate)

(Continue past the recorded examples.)

### L5. Cautious exploratory interpretation (9:08–9:20, approximate)

The rate effect is demonstrated here; the remaining gap is not fully explained. These runs do not prove convergence.
(Point to the interpretation and improvement messages in Cell L5A.)

## Closing (9:20–10:00, approximate)

**The policy saved money, uncertainty remains, and the exact reference shows room for improvement.** Thank you.
(Stay at the end of the Notebook with your camera visible.)
