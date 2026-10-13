# Submission analyses — separate from guided practice

The verified working core Notebook and official results remain intact. These
remaining proposal features must be completed before final submission, in order:

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

Status: **pending**, not implemented or executed. Add these analyses to a
separate submission analysis Notebook and preserve the working core Notebook.
Do not place them inside the guided practice exercises or count them as the
user's personal practice. The current guided interaction starts with topic 1
and waits for the user's attempt before advancing.
