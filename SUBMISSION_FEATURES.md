# Submission analyses — separate from guided practice

The main Notebook now includes these proposal features in the requested order:

1. **Exact finite-horizon MDP reference solution** — use the simulator's stated
   probabilities, boundary stays, and mandatory deadline BOOK. Compute the
   optimal expected purchase cost and the learned policy's exact expected cost,
   averaging uniformly over the five initial prices. This reference may know
   the probabilities; the Q-learning agent still learns only from experience.
2. **Policy heatmap** — show greedy BOOK/WAIT actions for every time/price state,
   clearly marking mandatory deadline booking.
3. **Confidence intervals** — compute 95% intervals for mean paired savings
   versus each baseline using the saved shared test trajectories and purchase
   costs. Explain that these quantify test-sample uncertainty for this trained
   policy, not variability across training seeds.

Status: **implemented and executed** in the main Notebook. Actual evidence and
limitations are documented in `RESULTS.md`; tables and PNG figures are in
`results/`. Backward-induction costs were checked by full latent-path enumeration.

Guided practice is **paused**. The separate practice Notebook is unchanged.
These completed analyses are not personal practice. Remaining official
submission items, per the instructor PDF, are the final PDF/Word proposal and
writeup, project zip with README, and approximately 10-minute camera-on recording.
