# Practice for the approximately 10-minute presentation

Target: understand the project, record the demo, and upload by **Wednesday,
October 14, 2026, 9 PM MST (UTC−7)**. This guide and the practice files are
preparation materials, not submission deliverables.

## Start here

Open `CS664_Flight_Booking_Practice.ipynb` and select
**Python (CS664 Flight Booking)** with interpreter **this project's
`.venv/bin/python`**. Start Jupyter from the project folder:

```sh
.venv/bin/jupyter lab CS664_Flight_Booking_Practice.ipynb
```

Run **1A**, **P0-DATA**, and **P0-FORMAT**, then begin **Part 1**. All four parts
are prepared now; you do not need to wait for chat answers. The old P1-TODO
and P1-CHECK cells are retained with their source and any saved outputs.

For each exercise: predict first, attempt the small TODO, use the hint, rerun
the edited definition, run the labeled check, then compare with the separate
solution. Say the purpose and result aloud. An incomplete TODO deliberately
raises `NotImplementedError` when called. This is an exercise stop, not a
broken project. **Do not use Run All until your TODOs are complete.**

The setup and surrounding code are supplied. Solution cells use different
names from your functions and do not fill your TODOs. Setup/solution outputs
were verified by the agent; they do not establish your personal completion.
Interpretation answers are models to compare with your own explanation.

## Recommended learning order

1. **Part 1 — Environment (about 15 minutes):** finish the deadline line;
   predict and trace BOOK and WAIT; explain state, reward, bounds, and termination.
2. **Part 2 — Q-learning (about 20 minutes):** calculate terminal and WAIT
   updates; explain old/target/alpha/future value; complete one epsilon-greedy
   line and distinguish training exploration from greedy evaluation.
3. **Part 3 — Fair comparison (about 15 minutes):** follow four policies on
   three actual test paths; calculate one paired saving and its small-sample
   mean. Explain current-state-only inputs, separate seeds, and shared paths.
4. **Part 4 — Validation (about 20 minutes):** interpret official means and
   intervals, calculate one backward step, read four heatmap states, and
   explain what verification establishes and what it does not establish.
5. **Rehearse (10 minutes):** follow the timed checklist at the Notebook's end.
   Use the main Notebook for the actual recorded demo. Allow additional time
   for recording, reviewing it, preparing the writeup, and confirming uploads.

## Goal, process, and the three validation questions

The goal is to **investigate whether a policy using current fare and time
remaining lowers simulated purchase cost compared with simple rules**. It
does not require proving Q-learning always wins or predicting real airfares.

Work backward from that goal: build and check the environment → train a
tabular policy from observed rewards → freeze it → compare four policies on
the same fresh paths → estimate paired mean savings and uncertainty → compare
the learned policy with the exact model optimum and inspect its decisions.

- **Does it save money?** Explain the four official mean costs and paired savings.
- **How uncertain is the estimated saving?** Explain the paired 95% intervals,
  conditional on one trained policy; distinguish mean uncertainty from trip variation.
- **Is it good and correct?** Explain the analytical optimality gap, learned
  policy heatmap and trace, and actual program checks. Good average performance
  against baselines can coexist with a suboptimal policy.

You do not need to narrate 10,000 raw rows. The three-trip teaching sample is
deliberately selected for readable mixed outcomes and is not the official
experiment. The official means/intervals are sampled; the MDP expectations are
analytical. No statistical functions, full solver, plotting, or file-export
implementation is assigned as practice.

## Verification and remaining submission work

The preparation verifier can be rerun with:

```sh
.venv/bin/python verify_practice.py
```

It executes supplied setup and solutions in a fresh kernel, verifies the
exercise checks against separate solution functions, checks that unfinished
TODOs stop clearly, and saves evidence in `practice_results/`. It preserves
student cells and hashes the main Notebook and official files to check that
they remain unchanged. It does not grade existing attempts or mark learning
complete. The verification-only cells are not added to your practice Notebook.

The main implementation and results are complete. Remaining official items
are the final PDF/Word proposal and writeup, the project ZIP with README, and
the approximately 10-minute camera-on presentation with code explanation and
full execution, followed by upload. Follow the instructor PDF for the rubric.

**Exclude from the submission ZIP:** `CS664_Flight_Booking_Practice.ipynb`,
`PRACTICE_GUIDE.md`, `practice_results/`, and `verify_practice.py`. Also exclude
environments, caches, `.git/`, local tools, credentials, and instructor-provided
materials. Keep the main Notebook with saved outputs, main runner, requirements,
README, proposal, RESULTS.md, and official `results/`. **No ZIP has been created.**
