# Flight Booking Decision Assistant

## Target users and problem
Travelers with a fixed flight departure date who are unsure whether to book now or wait for a lower fare. Waiting can save money, but the target price may never appear.

## Goal
Use Python tabular Q-learning to learn BOOK/WAIT decisions from simulated experiences. Investigate whether a policy using both current price and time remaining reduces average ticket cost compared with simple purchase rules.

## Simulated environment
- Seven daily purchase opportunities, with time remaining indexed from 6 to 0; day 0 is the mandatory booking deadline. This is a small teaching model, not a recommended real-world booking window.
- Price levels: $120, $180, $240, $300, $360. Choose the initial price uniformly.
- State: (time remaining, current price level).
- BOOK: pay the current price, receive reward = negative price, and end the episode. A $120 purchase scores -120, which is better than -300; maximizing reward therefore minimizes purchase cost.
- WAIT: reward = 0, advance one day, and sample the next price. Waiting's value includes the eventual purchase cost.

| Time remaining before WAIT | Increase $60 | Unchanged | Decrease $60 |
|---|---:|---:|---:|
| 6–3 | 30% | 30% | 40% |
| 2–1 | 60% | 20% | 20% |

Prices stay within $120–$360; attempted moves beyond a boundary stay at that boundary. No refunds, cancellations, or sold-out events. These probabilities are hypothetical assumptions, not measured airfare patterns. The learning agent sees current state and observed outcomes, not future prices or the transition probability table.

## Four strategies
| Strategy | Purchase rule |
|---|---|
| Immediate booking | Buy on the first day |
| Minimum-price rule | Buy at $120; otherwise buy at the deadline |
| Fixed-threshold rule | Buy at $180 or below; otherwise buy at the deadline |
| Q-learning | Use the learned BOOK/WAIT action for the current state |

### Why the threshold is $180
Exact policy evaluation under the seven-opportunity model gives these expected purchase costs, averaged over the five equally likely initial prices:

| Threshold | Expected cost |
|---|---:|
| $120 | $224.17 |
| $180 | $220.82 |
| $240 | $220.97 |
| $300 | $229.00 |
| $360 | $240.00 |

$180 has the lowest expected cost among fixed thresholds in this model, so it provides a strong baseline. The threshold selection uses known simulator probabilities, not test trajectories. These are analytical reference values, not Q-learning experiment results. Recompute them if the environment changes.

## Learning
Start with 50,000 training episodes, learning rate 0.1, and gamma = 1. Use epsilon-greedy exploration: decay epsilon from 1.0 to 0.05 over the first 40,000 episodes, then hold at 0.05. Terminal updates use only the observed reward. These are initial implementation settings, not guarantees of convergence.

## Three evaluation questions
1. **Does it save money?** Evaluate all four strategies on the same 10,000 fresh price trajectories with separate fixed training/test seeds. Disable exploration during testing. Report mean purchase cost and paired cost differences. All decisions use only currently available information.
2. **How uncertain is the estimated saving?** Report 95% confidence intervals for mean paired savings versus each baseline. This measures test-sample uncertainty for the evaluated policy, not a guarantee of savings on every trip or robustness across all training runs.
3. **Did it learn a good policy, and does the program follow the rules?** Compare the learned policy's expected cost with an exact finite-horizon MDP optimum. Show a BOOK/WAIT heatmap and one decision sequence. Separately check transitions, bounds, rewards, mandatory deadline booking, and episode termination.

Do not assume Q-learning outperforms every baseline. A result showing that a simple rule performs similarly is also a valid finding.

## Scope
Use a Jupyter Notebook with short sections. No real dataset, external APIs, neural network, or web app. Aim to complete the coding core in approximately three hours; actual progress depends on execution and debugging. Follow the supplied `metcs664_TermProject.pdf` for submission requirements. Add actual results, interpretation, and limitations to the final writeup after implementation.
