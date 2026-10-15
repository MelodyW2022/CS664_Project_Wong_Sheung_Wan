# Actual results — Flight Booking Decision Assistant

This report is generated from the executed [main Notebook](CS664_Flight_Booking.ipynb).
The instructor PDF supplies the official rubric; the unchanged proposal is the implementation plan.

## Simulation and training
Seven opportunities (time 6 to 0); fares $120/$180/$240/$300/$360; uniform initial fare.
WAIT changes fare by +$60/$0/−$60 with probabilities 30%/30%/40% at times 6–3,
and 60%/20%/20% at times 2–1. Boundary moves stay at $120 or $360.
BOOK rewards negative fare and terminates; WAIT rewards 0; deadline BOOK is mandatory.
These are hypothetical assumptions; there are no refunds, cancellations, or sold-out events.
Training: 50,000 episodes, alpha 0.1, gamma 1.0, seed 66401;
epsilon 1.0 to 0.05 over 40,000 episodes, then 0.05. Test seed: 66402.
See [assumptions](results/simulation_assumptions.json), [parameters CSV](results/training_parameters.csv),
and [run summary](results/run_summary.json).

## Sampled evaluation — 10,000 shared fresh trajectories
Every policy receives only current state. Q-learning evaluation is greedy with no updates.
| Strategy | Test trajectories | Mean purchase cost | Mean paired saving from Q |
|---|---:|---:|---:|
| Immediate booking | 10,000 | $239.58 | $24.73 |
| Minimum-price rule ($120) | 10,000 | $223.48 | $8.62 |
| Fixed-threshold rule ($180) | 10,000 | $219.74 | $4.88 |
| Q-learning (greedy) | 10,000 | $214.85 | $0.00 |

| Baseline | Mean Q-learning saving | Paired 95% CI |
|---|---:|---:|
| Immediate booking | $24.726 | [$23.838, $25.614] |
| Minimum-price rule ($120) | $8.622 | [$7.241, $10.003] |
| Fixed-threshold rule ($180) | $4.884 | [$4.029, $5.739] |

[Mean-cost CSV](results/mean_cost_comparison.csv) · [paired savings/CI CSV](results/paired_savings_95ci.csv) ·
[10,000 paired costs](results/paired_purchase_costs.csv) · [CI figure](results/paired_savings_95ci.png).
Positive saving means baseline cost minus learned-policy cost. Lowest sampled mean:
Q-learning (greedy) ($214.854). These are sampled results, not analytical expectations.

## Analytical finite-horizon reference — full model expectation
Optimal expected cost: **$207.274224**.
Learned-policy expected cost: **$215.709120**.
Learned policy minus optimum: **$8.434896**.
Values average uniformly over the five initial fares and use full backward induction;
“exact” means analytical expectation to floating-point precision. The trained policy is suboptimal.
[Exact comparison CSV](results/exact_mdp_comparison.csv) · [per-state values CSV](results/exact_state_values.csv).
These reference probabilities are never supplied to the learning agent or used to tune its policy.

## Policy, trace, and verification
[BOOK/WAIT heatmap](results/policy_heatmap.png) · [policy actions CSV](results/learned_policy_actions.csv).
[Complete trace CSV](results/q_learning_trace.csv) · [trace figure](results/q_learning_trace.png) ·
[trace table](results/q_learning_trace.md). Illustrated trajectory 1 ends at $360;
it was selected as the first example with at least three decisions, not to estimate performance.
[Verification CSV](results/verification_results.csv) records actually executed checks:
90 transitions, all seven purchase days, rewards, deadline booking, termination,
Q updates, 400 future-suffix checks, and independent 3,645-path expectations for five reference policies.

## Limitations
Individual paired 95% Student-t intervals are large-sample approximations conditional on this trained policy;
they are not simultaneous intervals and do not quantify training-seed variation or real-flight uncertainty.
The test trajectories are independent; cost differences retain within-trajectory pairing.
Fixed-rate finite training does not guarantee convergence. Average savings do not guarantee savings on each trip.

## Supplementary Validation and Optimality Investigation

Implementation checks passed with no correction required. Original model: 7 strictly suboptimal states; 0 have a legal action at/below the 326-visit lower quartile. This association does not establish causation.

Analytical action penalties use optimal continuation; equally optimal actions are accepted. [Coverage CSV](results/supplementary_original_coverage.csv) · [strictly suboptimal states](results/supplementary_suboptimal_states.csv) · [executed checks](results/supplementary_verification.csv).

| training_seed | episodes | exact_expected_cost | gap_from_optimal |
| --- | --- | --- | --- |
| 66401 | 50000 | 215.709120 | 8.434896 |
| 66401 | 200000 | 219.965568 | 12.691344 |
| 66411 | 50000 | 219.965568 | 12.691344 |
| 66411 | 200000 | 221.913600 | 14.639376 |
| 66421 | 50000 | 217.266816 | 9.992592 |
| 66421 | 200000 | 221.913600 | 14.639376 |
| 66431 | 50000 | 215.453952 | 8.179728 |
| 66431 | 200000 | 219.522432 | 12.248208 |
| 66441 | 50000 | 215.933952 | 8.659728 |
| 66441 | 200000 | 221.577600 | 14.303376 |

| training_seed | gap_50000 | gap_200000 | gap_reduction_50000_minus_200000 | improved |
| --- | --- | --- | --- | --- |
| 66401 | 8.434896 | 12.691344 | -4.256448 | False |
| 66411 | 12.691344 | 14.639376 | -1.948032 | False |
| 66421 | 9.992592 | 14.639376 | -4.646784 | False |
| 66431 | 8.179728 | 12.248208 | -4.068480 | False |
| 66441 | 8.659728 | 14.303376 | -5.643648 | False |

| episodes | seeds | mean_exact_cost | min_exact_cost | max_exact_cost | mean_gap | min_gap | max_gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50000 | 5 | 216.865882 | 215.453952 | 219.965568 | 9.591658 | 8.179728 | 12.691344 |
| 200000 | 5 | 220.978560 | 219.522432 | 221.913600 | 13.704336 | 12.248208 | 14.639376 |

| training_seed | episodes | Immediate booking | Minimum-price rule ($120) | Fixed-threshold rule ($180) |
| --- | --- | --- | --- | --- |
| 66401 | 50000 | True | True | True |
| 66401 | 200000 | True | True | True |
| 66411 | 50000 | True | True | True |
| 66411 | 200000 | True | True | False |
| 66421 | 50000 | True | True | True |
| 66421 | 200000 | True | True | False |
| 66431 | 50000 | True | True | True |
| 66431 | 200000 | True | True | True |
| 66441 | 50000 | True | True | True |
| 66441 | 200000 | True | True | False |

[All ten runs](results/supplementary_training_runs.csv) · [paired gap changes](results/supplementary_paired_gap_changes.csv) · [mean/range summary](results/supplementary_budget_summary.csv) · [all baseline comparisons](results/supplementary_baseline_comparisons.csv) · [endpoint figure](results/supplementary_seed_budget_gaps.png) · [saved Q-tables/visits](results/supplementary_trained_models.npz).

0/5 seeds improved at 200,000 episodes. All results are saved; the original policy is not replaced. These are exact finite-horizon expected costs under the uniform initial distribution, distinct from sampled test means. Two budgets do not establish continuous improvement.

Only episode budget varied. Alpha remains 0.1; epsilon decays from 1 to 0.05 over the first 40,000 episodes, then holds. [Configuration](results/supplementary_experiment_config.json) · [actual schedule checkpoints](results/supplementary_schedule_checkpoints.csv) · [summary](results/supplementary_summary.json).

Observed training-seed variability differs from the original paired 95% intervals for test-trajectory variation conditional on one policy. Constant-alpha estimation variability and exploration/coverage mechanisms are hypotheses, not tested causes. The remaining gap mechanism is unresolved; no additional tuning occurred. Checks support correctness in this teaching model, not convergence, universal seed robustness, per-trip savings, or performance on real flight prices.

## Learning Dynamics and Learning-Rate Comparison

Exploratory follow-up to the earlier deterioration; fifteen independently initialized runs under three prespecified rates, with unchanged simulator/gamma/epsilon. No winners replace the original policy. [Configuration](results/learning_dynamics_config.json) · [executed checks](results/learning_dynamics_verification.csv).

| training_seed | condition | exact_expected_cost | gap_from_optimal |
| --- | --- | --- | --- |
| 66401 | constant_0.1 | 219.965568 | 12.691344 |
| 66401 | constant_0.03 | 209.959872 | 2.685648 |
| 66401 | visit_decay | 209.670624 | 2.396400 |
| 66411 | constant_0.1 | 221.913600 | 14.639376 |
| 66411 | constant_0.03 | 212.737440 | 5.463216 |
| 66411 | visit_decay | 207.930624 | 0.656400 |
| 66421 | constant_0.1 | 221.913600 | 14.639376 |
| 66421 | constant_0.03 | 212.284032 | 5.009808 |
| 66421 | visit_decay | 208.882224 | 1.608000 |
| 66431 | constant_0.1 | 219.522432 | 12.248208 |
| 66431 | constant_0.03 | 215.709120 | 8.434896 |
| 66431 | visit_decay | 209.670624 | 2.396400 |
| 66441 | constant_0.1 | 221.577600 | 14.303376 |
| 66441 | constant_0.03 | 215.933952 | 8.659728 |
| 66441 | visit_decay | 209.670624 | 2.396400 |

| training_seed | cost_alpha_0.1 | cost_alpha_0.03 | cost_visit_decay | reduction_0.1_minus_0.03 | reduction_0.1_minus_decay | alpha_0.03_improved | visit_decay_improved |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 66401 | 219.965568 | 209.959872 | 209.670624 | 10.005696 | 10.294944 | True | True |
| 66411 | 221.913600 | 212.737440 | 207.930624 | 9.176160 | 13.982976 | True | True |
| 66421 | 221.913600 | 212.284032 | 208.882224 | 9.629568 | 13.031376 | True | True |
| 66431 | 219.522432 | 215.709120 | 209.670624 | 3.813312 | 9.851808 | True | True |
| 66441 | 221.577600 | 215.933952 | 209.670624 | 5.643648 | 11.906976 | True | True |

| condition | seeds | mean_exact_cost | min_exact_cost | max_exact_cost | mean_gap | range_width |
| --- | --- | --- | --- | --- | --- | --- |
| constant_0.1 | 5 | 220.978560 | 219.522432 | 221.913600 | 13.704336 | 2.391168 |
| constant_0.03 | 5 | 213.324883 | 209.959872 | 215.933952 | 6.050659 | 5.974080 |
| visit_decay | 5 | 209.164944 | 207.930624 | 209.670624 | 1.890720 | 1.740000 |

| condition | improved_intervals | worsened_intervals | unchanged_intervals |
| --- | --- | --- | --- |
| constant_0.1 | 39 | 48 | 8 |
| constant_0.03 | 46 | 40 | 9 |
| visit_decay | 36 | 26 | 33 |

Alpha 0.03 improved 5/5 final policies; visit decay improved 5/5, relative to alpha 0.1 for each seed. Each checkpoint cost is an analytical model expectation under the uniform initial-price distribution, not a sampled test mean. These are seed-paired comparisons, not test-trip paired confidence intervals.

[All 300 checkpoints](results/learning_dynamics_checkpoints.csv) · [2100 tracked-state diagnostics](results/learning_dynamics_tracked_states.csv) · [final fifteen runs](results/learning_dynamics_final_comparison.csv) · [paired reductions](results/learning_dynamics_paired_comparisons.csv) · [across-seed summaries](results/learning_dynamics_across_seed_summary.csv) · [checkpoint plots](results/learning_dynamics_checkpoint_gaps.png).

[Every checkpoint cost/action change](results/learning_dynamics_checkpoint_changes.csv) · [change summary](results/learning_dynamics_change_summary.csv) · [all final tracked states](results/learning_dynamics_final_tracked_states.csv) · [concise state summaries](results/learning_dynamics_state_summary.csv) · [labeled Q-tables and counts](results/learning_dynamics_final_models.npz) · [structured findings](results/learning_dynamics_summary.json).

Instrumentation checks executed the decay examples at 0/1000/3000 previous pair updates, verified pre-update counts and one increment per update, actual Q rates, terminal/legal behavior, independent run state, and unchanged arrays/RNG/subsequent behavior during checkpoint evaluation. Alpha 0.1 reproduces all ten earlier endpoint Q-tables, visit counts, and exact costs.

Tracked Q margins and action switches describe associations with checkpoint cost changes; untracked states can also change. Checkpoints do not reveal every intervening decision. Equal seeds across different-rate conditions need not produce identical training trajectories because different actions change subsequent experiences and RNG consumption.

The final comparisons and fluctuations support sensitivity to these tested settings where outcomes differ, not a sole-cause explanation. The remaining gap mechanism is unresolved. Neither 0.03 nor visit decay with scale 1000 is proven optimal; no extra schedules, grid search, or tuning occurred. Fifteen finite runs do not prove convergence or necessary improvement with more training. Hypothetical transitions do not establish real-flight performance.
