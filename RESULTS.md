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

## Limitations and remaining submission work
Individual paired 95% Student-t intervals are large-sample approximations conditional on this trained policy;
they are not simultaneous intervals and do not quantify training-seed variation or real-flight uncertainty.
The test trajectories are independent; cost differences retain within-trajectory pairing.
Fixed-rate finite training does not guarantee convergence. Average savings do not guarantee savings on each trip.
Practice is paused and the separate practice Notebook is unchanged. Agent execution is not personal practice.
Still required: final PDF/Word proposal/writeup, project zip with README, and an approximately
10-minute camera-on presentation with code explanation and a full demo, per the instructor PDF.
