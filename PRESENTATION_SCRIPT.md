# CS664 presentation script and screen-sharing plan

Use the main `CS664_Flight_Booking.ipynb` as the shared screen. Keep your camera
on throughout. Screen directions below are not spoken narration. Do not display
the practice Notebook or read the report page by page.

This script uses saved, verified evidence from the instructor PDF, report draft,
main Notebook, README, and `results/`. It does not establish that you personally
completed practice. Say only what you can explain. Codex assistance is disclosed
in the report and in the closing narration.

## Full-run plan before recording

The timestamps are recording targets, not measured speaking times. Rehearse
navigation and speech with a stopwatch. Use short pauses to point at code.

The saved Notebook's first-cell-busy to last-cell-idle interval is 182.36 seconds,
about three minutes. Kernel startup and runner postchecks are excluded. Complete
CLI wall time was not measured. This plan does not promise a runtime.

Start the unchanged full runner by 0:20. Explain code while it runs. Aim to
confirm completion at 4:20, leaving roughly four minutes for execution including
startup. The 4:20–4:50 slot is a completion/reload buffer. Relative to the recorded
cell interval, a start at 0:20 leaves about 88 seconds for overhead or variation
before 4:50. This is an allocated buffer, not a measured overhead estimate.

Keep a terminal panel beside or below the main Notebook. From the project root:

```sh
time .venv/bin/python run_notebook.py
```

`time` measures this new command's elapsed time. Report that value only after it
actually finishes. The portable runner uses a fresh kernel, executes all cells,
checks earlier numerical evidence, and saves the Notebook only after success.
It runs the original experiment, both supplementary investigations, and the
existing optional threshold cell. No budgets or cells are reduced or skipped.

During execution, the open Notebook displays earlier saved outputs. Say so.
Read code without editing or executing cells in that viewer. Turn off editor
Auto Save and keep the Notebook free of unsaved edits before starting. Do not
save the stale viewer over the runner's new file.

On success, the terminal must show:

```text
PASS: 80 code cells executed in a fresh kernel; no cell errors.
Original numerical evidence, including Q-values and visits, is unchanged.
Saved executed Notebook: .../CS664_Flight_Booking.ipynb
```

Wait for the command prompt to return with no traceback. Close and reopen the
main Notebook from disk, or accept the editor's Reload from Disk option. Do not
choose to overwrite its new contents. Show the freshly reloaded outputs.
Check that the fresh displayed values match the verified numbers below before
reading them aloud. If they differ, explain the discrepancy and investigate;
do not attribute the earlier numbers to different new outputs.

If the runner has not finished by 4:50, the planned fresh-results segment cannot
start on time. Do not call a running or failed job a completed full demo. Keep
recording until completion and explain the delay; an overrun may conflict with
the approximately ten-minute requirement. Alternatively, end that recording,
resolve any setup issue, and record a new complete take with an early full run.
If repeated full runs cannot fit alongside the explanations, ask the instructor
how to accommodate the conflict. Do not assume editing, speed-up, omitted runs,
or time compression is permitted. No rehearsal or new full run was performed
as part of preparing this script.

## 0:00–0:40 — Opening, goal, and start the full run

Screen instructions:

- Show `# Flight Booking Decision Assistant — completed experiment`.
- Within the first 20 seconds, open the terminal panel and launch the command
  above. Show it starting, then return to the main Notebook.
- Navigate to `## 1. Setup and fixed experiment settings` and
  `# 1A — imports and official settings`.

Spoken narration:

**This project uses Q-learning to study when to book a simulated flight at a lower expected cost.**

I will start the whole project now and explain the code while it runs.
The outputs currently visible are from an earlier saved run.
After the runner finishes, I will reload the Notebook and show new outputs.
This is a small machine learning application. It uses simulated prices, not
real flight data.

## 0:40–1:55 — Environment and simulation rules

Screen instructions:

- Show `## 2. Price simulator`.
- Display `# 2A — one price transition`: point at `probabilities`, `rng.choice`,
  and `np.clip`.
- Display `# 2B — episode and terminal rules`: point at `step`, `effective_action`,
  `self.done`, and the two return statements.
- Briefly show the assertions in
  `# 2D — rewards, deadline, and termination checks`. Earlier outputs are still
  earlier evidence until the runner completes.

Spoken narration:

**The environment makes every episode end with one purchase, within the allowed price range.**

The state contains time remaining and current price. There are seven buying
opportunities, from time six to zero. Prices are 120, 180, 240, 300, or 360
dollars. Each starting price is equally likely.

BOOK pays the current price. Its reward is the negative price, and the episode
ends. WAIT gives zero reward and moves to the next day. It still leads to a
later purchase cost.

At times six to three, prices rise, stay, or fall with probabilities 30, 30,
and 40 percent. At times two and one, they are 60, 20, and 20 percent.
These probabilities are assumptions.

Clipping keeps prices between 120 and 360 dollars. At time zero, the program
forces BOOK, even if WAIT is requested. Another action after purchase raises
an error. These checks test rewards, the deadline, and termination.

## 1:55–3:20 — Q-learning code, updates, and exploration

Screen instructions:

- Show `## 3. Tabular Q-learning` and
  `# 3B — one real Q-value update`. Point at the `if done` branch and the update.
- Show `# 3D — display actual updates and BOOK/WAIT predictions`. Identify this
  as the earlier logged example unless the viewer has already been reloaded.
- Show `# 3A — indexing, exploration, and greedy prediction`: point at
  `epsilon_for`, `choose_training_action`, and the deadline branch.
- Briefly show `# 3C — training loop`: point at the update call and `if done: ... break`.

Spoken narration:

**Q-learning improves an action estimate by moving its old value toward a target.**

The table starts at zero. Each update adds alpha times target minus old value.
Alpha is 0.1. Gamma is one, so the final purchase cost is not discounted.

Here is an actual logged update. In the first episode, the state was six and
120 dollars, and the action was BOOK. The old value was zero. The terminal
target was negative 120. Zero plus 0.1 times negative 120 gives negative 12.

A terminal purchase has no future value. For WAIT, the target includes the
best legal next-state Q-value. At the deadline, that means BOOK only.
The logged WAIT example had old value zero, reward zero, and target negative
30. Its updated value was negative three. That was an estimate during learning.

Training uses epsilon-greedy exploration. Epsilon falls from one to 0.05 over
40,000 episodes, then stays there. The original training has 50,000 episodes.
Testing uses only greedy actions, with no exploration or updates.

## 3:20–4:20 — Fair comparison and validation questions

Screen instructions:

- Show `## 4. Fair evaluation on 10,000 fresh trajectories`.
- Display `# 4A — generate independent shared test trajectories` and
  `# 4B — state-only policies`.
- Display `# 4C — stop-on-purchase evaluator`. Point at `policy(state)` and the
  immediate return on BOOK. The evaluator stores the path; the policy receives
  only the current state.
- Display `# 4E — official comparison and paired cost differences`, pointing
  at the unchanged-Q assertion. Do not yet label its visible table fresh.

Spoken narration:

**A fair comparison gives every strategy the same test paths and no future-price information.**

The four rules are: buy on the first day; buy at 120 dollars or at the deadline;
buy at 180 dollars or below, or at the deadline; and follow Q-learning.

All four use the same 10,000 fresh test trajectories. Training seed 66401 and
test seed 66402 are separate. The evaluator stores each path, but the decision
function receives only the current state. It stops calling the policy after
purchase. The trained Q-table stays unchanged during evaluation.

There are three validation questions. Does Q-learning lower cost? How reliable
are the estimated savings? How close is it to optimal, and does the program
follow the rules?

## 4:20–4:50 — Confirm full execution and reload

Screen instructions:

- Show terminal progress. Require the final `PASS: 80 code cells...` line,
  the saved-path line, and returned prompt before declaring success.
- Close and reopen the main Notebook from disk without saving its old contents.
- Navigate to `# 4E — official comparison and paired cost differences`.
- This slot allows waiting and navigation. If completion is late, follow the
  timing-conflict instructions above rather than using the success narration.

Spoken narration, only after actual success:

**The fresh full run has completed, and these reloaded outputs come from that run.**

The runner executed all 80 code cells, including both supplementary studies,
without errors. It preserved the original numerical evidence.
I have reopened the saved Notebook.

## 4:50–5:40 — Cost results and one complete trace

Screen instructions:

- Show the fresh output of
  `# 4E — official comparison and paired cost differences`.
- Then show `## 5. One complete Q-learning decision trace` and the whole table
  from `# 5A — actual states, actions, and final cost`.
- Leave all seven trace rows visible. Read the final purchase rather than
  narrating every row.

Spoken narration:

**The original learned policy had the lowest sampled mean cost among the four strategies.**

The means were 239 dollars and 58 cents for immediate booking, 223 dollars
and 48 cents for the 120-dollar rule, and 219 dollars and 74 cents for the
180-dollar rule. Q-learning averaged 214 dollars and 85 cents.

These are sample means, not exact model expectations.
This complete trace starts at 360 dollars. The policy waits until the deadline
and buys for 360 dollars. This example shows every decision and termination.
It also shows why average savings do not mean savings on every trip.

## 5:40–6:30 — Reliability and paired confidence intervals

Screen instructions:

- Show `## 8. Paired 95% confidence intervals` and the fresh table in
  `# 8B — save all three paired savings intervals`.
- Point to the $180 baseline row. Optionally show the figure from
  `# 8C — save the paired confidence-interval figure` if navigation is quick.

Spoken narration:

**The paired intervals support positive mean savings for this fixed trained policy.**

For each trip, saving means baseline cost minus Q-learning cost. Pairing keeps
the comparison on the same price path.

Against the 180-dollar rule, mean saving was about four dollars and 88 cents.
Its 95 percent interval was about four dollars and three cents to five dollars
and 74 cents. All three intervals are above zero.

These intervals describe uncertainty in mean savings. They do not contain
95 percent of individual-trip savings. They also do not measure variation
between training seeds. The three intervals are individual, not simultaneous.

## 6:30–7:50 — Optimality, policy, and verification

Screen instructions:

- Show `## 6. Exact finite-horizon MDP reference`, then the recurrence in
  `# 6B — backward induction and exact policy evaluation` and the fresh output
  of `# 6D — save analytical comparison and per-state references`.
- Show `## 7. BOOK/WAIT policy heatmap` and the figure from
  `# 7A — draw and save the learned policy heatmap`.
- Point at price columns, time rows, and mandatory BOOK at time zero.
- Show the fresh verification output under
  `# 9C — save actual run summary and trace`, plus
  `# 2D — rewards, deadline, and termination checks` if needed for the PASS line.

Spoken narration:

**The implementation passed its checks, but the original learned policy is still suboptimal.**

Backward induction works from the deadline toward the first day. At each
state, it compares buying now with the expected cost of waiting.
This reference uses the known simulator probabilities. The learning agent
does not receive them.

The optimal expected cost is about 207 dollars and 27 cents. The original
learned policy's expected cost is about 215 dollars and 71 cents. Its gap is
about eight dollars and 43 cents. These are analytical expectations, not the
sample means shown earlier.

The heatmap shows BOOK or WAIT for each time and price. At the deadline, every
price means BOOK. This is a decision policy, not a price forecast.

The program checked 90 transitions, all seven purchase days, reward targets,
boundaries, forced booking, and termination. It also checked future-price
isolation and independently verified the exact reference. Passing these finite
checks supports correctness; it does not prove convergence or cover every bug.

## 7:50–8:50 — One minute on supplementary findings

Screen instructions:

- Show `### S3. Bounded five-seed/two-budget experiment` and the fresh paired
  table and summaries in `# S3C — paired gap changes and mean/range summaries`.
- Then show `# L4B — means and ranges across the five seeds` under
  `## Learning Dynamics and Learning-Rate Comparison`.
- If there is time, point briefly to
  `# L4E — all-seed checkpoint gap plots with consistent axes`. Do not walk
  through every run or plot line. All runs executed; only narration is selective.

Spoken narration:

**The supplementary studies show that more training alone did not help, while the tested learning rates improved final performance.**

The first study restarted five seeds at 50,000 and 200,000 episodes, with the
original settings. Every seed had a higher final expected cost at 200,000.
Two budgets do not prove a continuous trend.

The next study compared constant alpha 0.1, constant 0.03, and visit-based
decay. Each condition used five seeds and 200,000 episodes. Mean exact costs
were about 220 dollars and 98 cents, 213 dollars and 32 cents, and 209 dollars
and 16 cents.

Both alternatives improved all five final policies relative to alpha 0.1.
Every final policy still had a positive optimality gap. These exploratory
results show sensitivity, not convergence or a unique cause. Seed variation
is different from test-trip uncertainty. The original policy was not replaced.

## 8:50–9:35 — Limits and answers to the three questions

Screen instructions:

- Show `## 10. Interpretation and limits` and
  `# 10A — evidence-based interpretation`.
- Briefly show `### L5. Cautious exploratory interpretation` for supplementary
  limits. Keep the main Notebook as the shared screen.

Spoken narration:

**The project supports savings inside this simulator, with clear limits on reliability and optimality.**

For the first question, the original policy lowered sampled mean cost against
all three baselines. For the second, the paired intervals support positive
mean savings for that policy. Training-seed variation needs separate evidence.
For the third, checks passed, but the analytical gap shows room for improvement.

The model uses hypothetical probabilities and a small price grid. It leaves
out refunds, cancellations, and sold-out flights. The tested rate settings are
not proven optimal. The remaining gap is not fully explained. None of these
results proves savings on real flight prices.

## 9:35–10:00 — Close and leave the completed demo visible

Screen instructions:

- Return to the fresh `# 4E — official comparison and paired cost differences`
  output. Keep your camera visible.
- Use this slot as final navigation/breathing buffer; do not add more findings.

Spoken narration:

**The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.**

This project demonstrates a complete Python Q-learning workflow, from the
environment to evaluation and verification. Codex assisted with implementation,
verification, and report preparation, as disclosed in the report.
Thank you for watching.

## Recording requirements mapped to the plan

| Requirement | Evidence in the recording |
|---|---|
| Approximately ten minutes | Timestamp plan; stopwatch rehearsal; full-run buffer and explicit slow-run contingency |
| Camera on | Visible throughout, including terminal launch and fresh results |
| Explain key code segments | 0:40 environment `2A`/`2B`; 1:55 updater `3B`, selector `3A`, loop `3C`; 3:20 evaluator `4C`; 6:30 exact recurrence `6B` |
| Demo run of the whole project | Start the unchanged portable runner by 0:20; wait for all 80 cells and success; reload from disk at 4:20–4:50 |
| Show actual completed results | Fresh means/trace at 4:50, intervals at 5:40, reference/heatmap/checks at 6:30, supplementary tables at 7:50 |
| Explain findings and limitations | Three questions at 3:20 and 8:50; about one minute on supplements at 7:50 |
| Distinguish saved and new output | Opening explicitly labels earlier outputs; completion declaration is conditional on actual success and reload |
| Separate upload | Record video separately from the project ZIP and reviewed report, following course-portal instructions |

## Camera, microphone, screen, and execution setup

- [ ] Keep `PRESENTATION_SCRIPT.md` on an unshared screen or printed page. Share
  the main Notebook window and its terminal panel, not your entire desktop.
- [ ] Enable the camera and microphone. Test a short recording for sound and
  readable code. Confirm the camera overlay does not cover tables or plots.
- [ ] Turn off notifications. Close personal tabs and the practice Notebook.
  Avoid showing credentials or unrelated files.
- [ ] Open the project folder. Select `.venv/bin/python` and kernel
  `Python (CS664 Flight Booking)` (`cs664-flight-booking`).
- [ ] Use readable zoom. Collapse long outputs until needed. Use the Notebook
  outline for headings and Find for the exact cell labels above. Cell labels,
  not changing execution counters, identify what to display. The appendix also
  has a `6A` label; do not confuse it with the analytical `6A` cell.
- [ ] Check the main Notebook has no unsaved edits. Disable Auto Save in the
  viewer. Keep all budgets, seeds, parameters, and cells unchanged.
- [ ] Keep the terminal ready in the project root, with the full runner command
  typed but not yet executed. Do not run a second kernel or a second training
  job concurrently. Use the runner rather than Run All in the open viewer.
- [ ] Review the success lines and safe reload steps. Know how to reopen the
  Notebook without overwriting the new saved file.
- [ ] Rehearse narration and navigation. If you choose a full rehearsal run,
  measure its actual command time; it is not measured by this script-writing
  task. Do not describe rehearsal as personal mastery without understanding it.
- [ ] Start recording before launching the runner. Keep camera on. Confirm the
  whole run succeeds before saying that results are fresh.
- [ ] Watch the finished video for sound, readability, full-run evidence, and
  duration before uploading. No submission or recording is performed here.

## Sentences to rehearse

These repeat the bold spoken sentences above in plain text. Memorize those
anchors if helpful, and read the remaining narration naturally.

1. This project uses Q-learning to study when to book a simulated flight at a lower expected cost.
2. The environment makes every episode end with one purchase, within the allowed price range.
3. Q-learning improves an action estimate by moving its old value toward a target.
4. A fair comparison gives every strategy the same test paths and no future-price information.
5. The fresh full run has completed, and these reloaded outputs come from that run.
6. The original learned policy had the lowest sampled mean cost among the four strategies.
7. The paired intervals support positive mean savings for this fixed trained policy.
8. The implementation passed its checks, but the original learned policy is still suboptimal.
9. The supplementary studies show that more training alone did not help, while the tested learning rates improved final performance.
10. The project supports savings inside this simulator, with clear limits on reliability and optimality.
11. The main lesson is to compare learning with fair baselines, uncertainty, and an exact reference.

## Verification notes (not spoken)

Spoken costs are rounded to cents from the saved CSVs. For example, $223.476
rounds to $223.48 and $214.854 to $214.85. Rounded means must not be used to
recalculate precise savings: the $180-rule saving is $4.884 before rounding.
The CI quoted aloud rounds [$4.0293664935, $5.7386335065] to [$4.03, $5.74].
The analytical gap is $8.434896, rounded to $8.43. The three rate means round
from $220.978560, $213.3248832, and $209.164944 to $220.98, $213.32, and $209.16.

Source map: `results/run_summary.json` (rules, seeds, settings, logged updates,
trace, checks); `results/mean_cost_comparison.csv`; `results/paired_savings_95ci.csv`;
`results/exact_mdp_comparison.csv`; `results/verification_results.csv`;
`results/supplementary_paired_gap_changes.csv`;
`results/learning_dynamics_across_seed_summary.csv`;
`results/learning_dynamics_paired_comparisons.csv`;
`results/learning_dynamics_final_comparison.csv`;
and the main Notebook's execution timestamps and labels.

This preparation document is outside the existing submission ZIP. No project
implementation, official result, report, proposal, or practice attempt was changed
for this task. A new demo run during your recording will generate fresh outputs;
the existing ZIP is not automatically rebuilt by the runner.
