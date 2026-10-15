# Flight Booking Decision Assistant

CS664 Term Project | Prepared for Sheung Wan Wong | October 14, 2026

DRAFT — awaiting student review.

## 1. Project goal and simulator

This Python tabular Q-learning application chooses BOOK or WAIT for a simulated traveler with a fixed departure date. It tests whether current fare and time remaining can reduce expected purchase cost relative to simple booking rules. The proposal specified a seven-opportunity simulator, 50,000 training episodes, four strategies on 10,000 shared test paths, paired uncertainty estimates, an exact MDP reference, a policy heatmap, a decision trace, and verification [1, 2]. Supplementary experiments examine training budget and learning rate separately from the original comparison.

State is (time remaining, current price). Seven daily purchase opportunities run from time 6 to 0; fares are $120, $180, $240, $300, and $360. The starting fare is uniform over these five levels. WAIT uses the probability regime for the time before the action:

| Time before WAIT | +$60 | Unchanged | −$60 |
| --- | --- | --- | --- |
| 6–3 | 30% | 30% | 40% |
| 2–1 | 60% | 20% | 20% |

WAIT gives reward 0 and reduces time by one. Moves beyond $120–$360 stay at the boundary. BOOK gives the negative current fare as reward and terminates. At time 0, BOOK is mandatory and a requested WAIT becomes BOOK. Actions after purchase raise an error until reset. Refunds, cancellations, and sold-out events are excluded. All probabilities are hypothetical.

## 2. Training and an observed Q-value update

The agent learns from observed transitions without receiving their probability table [3]. Maximizing undiscounted negative purchase reward minimizes cost. Its zero-initialized Q-table has shape 7 × 5 × 2, with 65 legal state/action pairs; deadline WAIT is excluded and greedy ties choose BOOK.

| Original setting | Value |
| --- | --- |
| Episodes / learning rate / gamma | 50,000 / 0.1 / 1.0 |
| Training / test seeds | 66401 / 66402 |
| Epsilon schedule | 1.0 on episode 1 to 0.05 on episode 40,000; then 0.05 |
| Evaluation | Greedy; epsilon 0; no Q updates |

Q_new = Q_old + alpha × (target − Q_old). Terminal BOOK uses target = reward; otherwise target = reward + gamma × max Q(next state, legal action). Only BOOK is legal in a deadline next state.

Actual logged example: episode 1, state (6, $120), BOOK. Old Q = 0, reward and terminal target = -120, alpha = 0.1. Thus Q_new = 0 + 0.1 × (−120 − 0) = -12. The saved update is in [run_summary.json](results/run_summary.json).

## 3. Original sampled four-strategy comparison

All four strategies used the same 10,000 fresh seven-price paths, generated with test seed 66402 independently of training seed 66401 [5]. Only the evaluator sees complete paths: each policy receives its current state, stops at purchase, and must book at the deadline. Evaluation has no exploration or Q updates. The $180 threshold was specified from analytical model knowledge in the proposal, without searching the test paths.

| Strategy | Rule | Sample mean cost |
| --- | --- | --- |
| Immediate booking | BOOK first day | $239.58 |
| Minimum-price rule ($120) | BOOK at $120; else deadline | $223.48 |
| Fixed-threshold rule ($180) | BOOK at $180 or below; else deadline | $219.74 |
| Q-learning (greedy) | Original greedy policy | $214.85 |

| Baseline | Mean paired saving | Individual paired 95% CI |
| --- | --- | --- |
| Immediate booking | $24.73 | [$23.84, $25.61] |
| Minimum-price rule ($120) | $8.62 | [$7.24, $10.00] |
| Fixed-threshold rule ($180) | $4.88 | [$4.03, $5.74] |

Saving is baseline cost minus Q-learning cost on the same path. Each interval is mean(d) ± t(0.975, 9999) × sample_sd(d)/sqrt(10000), using SciPy's Student-t quantile [4]. All three intervals lie above zero. They estimate mean savings conditional on this trained policy; they are neither simultaneous intervals nor ranges for individual-trip savings. Training-seed variation is assessed separately below.

Evidence: [four-strategy means](results/mean_cost_comparison.csv), [paired savings and intervals](results/paired_savings_95ci.csv), and [all 10,000 paired purchase costs](results/paired_purchase_costs.csv). Monetary summaries are rounded to cents only after calculations from full-precision values; saved artifacts retain their precision.

## 4. Exact model expectations

Backward induction uses V*(0,p) = p and V*(t,p) = min[p, sum P(p′|t,p)V*(t−1,p′)]. Fixed-policy evaluation follows the chosen action instead of minimizing. Averaging over the five uniform starting fares gives full-model expectations, distinct from sampled means and estimated Q-values.

| Analytical reference | Expected cost |
| --- | --- |
| Optimal finite-horizon policy | $207.274224 |
| Original learned policy | $215.709120 |
| Learned minus optimal | $8.434896 |

The original policy beats the simple rules in the sample but remains suboptimal. At (6, $180), for example, it BOOKs while the optimal action is WAIT. Its sampled mean of $214.85 differs from its exact expectation of $215.709120 because the test sample is finite. See [exact policy costs](results/exact_mdp_comparison.csv) and [all state values](results/exact_state_values.csv).

## 5. Original learned policy and complete decision trace

![Figure 1. Original greedy policy: BOOK or WAIT at each state. Time 0 is mandatory BOOK.](results/policy_heatmap.png)

The trace is trajectory index 1, the first test example with at least three decisions. It shows every observed decision through purchase; unseen post-purchase prices are excluded.

| Day | Time remaining | Price | Action | Reward |
| --- | --- | --- | --- | --- |
| 1 | 6 | $360 | WAIT | 0 |
| 2 | 5 | $300 | WAIT | 0 |
| 3 | 4 | $360 | WAIT | 0 |
| 4 | 3 | $360 | WAIT | 0 |
| 5 | 2 | $360 | WAIT | 0 |
| 6 | 1 | $360 | WAIT | 0 |
| 7 | 0 | $360 | BOOK | -360 |

The episode ends after 7 decisions with a $360 purchase. It illustrates policy behavior, not average performance: positive mean savings can coexist with expensive trips. Evidence: [complete trace](results/q_learning_trace.csv) and [all policy actions](results/learned_policy_actions.csv).

## 6. Executed verification

| Core check | Cases | Outcome |
| --- | --- | --- |
| simulator transitions | 90 | PASS |
| episode rules | 7 | PASS |
| Q-value updates | 2 | PASS |
| exact MDP reference | 18,225 | PASS |
| policy actions | 5 | PASS |
| paired intervals | 2 | PASS |
| future-price isolation | 400 | PASS |

The exact reference was independently checked by enumerating 3,645 latent-move paths for each of five policies (18,225 comparisons). Supplementary checks passed for nonzero-old-value Q updates, deadline selection, all-state rewards and invalid actions, tie-aware penalties, independent training, and preservation of the original model. Equal-cost optimal actions are treated as ties.

Instrumentation verified update counters, applied learning rates, terminal/legal targets, and checkpoint isolation. All 300 checkpoints preserved Q, counts, and training RNG state. Constant alpha 0.1 exactly reproduced the ten earlier 50,000/200,000 endpoint Q-tables, visit counts, and analytical costs. The saved main Notebook has 80 successfully executed code cells. These finite checks support implementation behavior; they do not prove convergence.

Complete executed records: [core checks](results/verification_results.csv), [supplementary checks](results/supplementary_verification.csv), and [learning-dynamics checks](results/learning_dynamics_verification.csv).

## 7. Supplementary training-budget experiment

Seeds 66401, 66411, 66421, 66431, and 66441 each restarted from zero at 50,000 and 200,000 episodes (ten runs). Only episode budget changed: simulator, alpha 0.1, gamma 1, initialization, and the original epsilon schedule stayed fixed. Final greedy policies were evaluated exactly under the uniform initial-fare distribution.

| Episodes | Mean exact cost | Across-seed cost range |
| --- | --- | --- |
| 50,000 | $216.87 | [$215.45, $219.97] |
| 200,000 | $220.98 | [$219.52, $221.91] |

All five seeds worsened at 200,000 episodes; seed 66401 increased by $4.26. All ten models cost less than the immediate and $120 baselines in exact evaluation; seven cost less than the $180 baseline. The original policy had seven strictly suboptimal states, although all 65 legal pairs were visited and none of those states met the descriptive lower-quartile coverage flag. Coverage alone does not explain the optimality gap.

Full evidence: [all ten runs](results/supplementary_training_runs.csv), [paired budget changes](results/supplementary_paired_gap_changes.csv), [coverage diagnostics](results/supplementary_suboptimal_states.csv), and the [executed Notebook](CS664_Flight_Booking.ipynb).

## 8. Supplementary learning-rate experiment

The same five seeds independently restarted for 200,000 episodes under three prespecified conditions: constant alpha 0.1; constant alpha 0.03; and alpha(s,a) = 0.1/(1 + N_previous(s,a)/1000). N_previous counts updates to that pair before the current update. At 0, 1,000, and 3,000 prior updates, alpha is 0.1, 0.05, and 0.025. Scale 1,000 was chosen before the experiment. All other settings and the legal greedy evaluator stayed fixed.

Fifteen runs recorded 300 checkpoints and 2,100 diagnostics for the seven originally suboptimal states. Equal seeds do not ensure identical experience across conditions: changed policies can alter later transitions and RNG consumption. Every run, including equal-cost policies, remains in the evidence.

| Condition | Mean exact cost | Across-seed cost range | Mean optimality gap |
| --- | --- | --- | --- |
| Constant 0.1 | $220.98 | [$219.52, $221.91] | $13.70 |
| Constant 0.03 | $213.32 | [$209.96, $215.93] | $6.05 |
| Visit decay | $209.16 | [$207.93, $209.67] | $1.89 |

Both alternatives lowered every seed's final exact cost relative to alpha 0.1. For seed 66401, reductions were $10.01 with alpha 0.03 and $10.29 with visit decay. Every final policy still had a positive optimality gap. Checkpoint costs fluctuated in all conditions; action switches accompanied performance changes, but sparse checkpoints do not isolate their cause. No best-checkpoint selection was performed.

Complete evidence: [all fifteen final runs](results/learning_dynamics_final_comparison.csv), [paired rate comparisons](results/learning_dynamics_paired_comparisons.csv), [all checkpoints](results/learning_dynamics_checkpoints.csv), [state diagnostics](results/learning_dynamics_tracked_states.csv), and the [saved dynamics figure](results/learning_dynamics_checkpoint_gaps.png). Methods and outputs are also in the [main Notebook](CS664_Flight_Booking.ipynb).

## 9. Limitations

The model uses hypothetical probabilities, five fare levels, a short horizon, and no sold-out events; it provides no evidence of real-flight savings. Paired intervals quantify test-path uncertainty for one policy, not training uncertainty or individual-trip outcomes. Across-seed ranges describe five selected seeds, not confidence intervals; two budgets do not establish a continuous trend. Supplementary analyses are exploratory, not independent confirmation, and do not establish universal robustness or convergence. Fixed-step target variability, decay, learning speed, and coverage may matter, but their causal contributions were not isolated. No simultaneous inference or additional tuning is claimed.

## 10. Reproducibility and references

Follow [README setup and execution instructions](README.md) for the pinned environment, project interpreter, Notebook kernel, and fresh-kernel runner. [RESULTS.md](RESULTS.md) indexes the complete saved outputs. Full per-seed results and full-precision artifacts are retained; this report revision uses saved evidence without rerunning experiments.

[1] Kalathur, S. (2026). CS664 Term Project. Instructor-provided metcs664_TermProject.pdf; official project requirements.

[2] [Original project proposal](CS664_Project_Wong_Sheung_Wan_Proposal.md); [executed Notebook](CS664_Flight_Booking.ipynb). Numerical audit: [report_number_audit.json](report_assets/report_number_audit.json).

[3] Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. Machine Learning, 8, 279–292. [Publisher reference](https://doi.org/10.1007/BF00992698). Algorithm background; no convergence theorem is claimed for this experiment.

[4] SciPy contributors. [scipy.stats.t documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.t.html). Student-t quantile; consulted October 14, 2026.

[5] NumPy contributors. [Random Generator documentation](https://numpy.org/doc/stable/reference/random/generator.html). Seeded random streams; consulted October 14, 2026.
